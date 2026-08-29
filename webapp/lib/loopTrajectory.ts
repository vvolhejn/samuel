/** A model response turned into something that can be read at any loop phase.
 *
 * The trajectory is stored per channel and indexed by phase in [0, 1) rather
 * than by frame, which is what makes the loop follow tempo for free: at half
 * the tempo the same articulation is read out half as fast, and since pitch is
 * a separate parameter rather than a property of a resampled waveform, nothing
 * transposes.
 *
 * A take is recorded with a pad of extra audio either side of the bar, so the
 * response is longer than the loop and the loop is a window that slides over
 * it. Sliding that window is how input latency is compensated for: the take is
 * cut and predicted once, and the "record offset" control then re-aligns the
 * predicted parameters against the grid, to one frame (about 12 ms). The window
 * is copied out on each move rather than read through an index, so the seam and
 * `sample` stay as they were. */

import type { SynthResponse } from "@/lib/audio";
import { CHANNELS, Channel, ChannelValues, REST_VALUES } from "@/lib/tractParams";

/** Frames over which the tail is bent to meet the head, so phase 1 and phase 0
 * are the same pose and the loop point doesn't jump. Three frames is ~35 ms at
 * the model's 86 fps: long enough to swallow the discontinuity, short enough
 * that it doesn't audibly clip the end of the phrase. */
const SEAM_FRAMES = 3;

export interface LoopTrajectory {
  /** Frames in the loop, which is what `sample` reads over. */
  nFrames: number;
  /** Seconds of audio the loop itself came from, for the UI. */
  sourceSeconds: number;
  /** Where the window is now. */
  offsetSeconds: number;
  /** Slide the window. Positive reaches earlier into the take, the same
   * direction the recorded window used to be shifted. Clamped to the pad. */
  setOffsetSeconds(seconds: number): void;
  /** Read the pose at `phase` into `out`, without allocating. */
  sample(phase: number, out: ChannelValues): void;
}

export interface LoopTrajectoryOptions {
  /** Length of the loop the take was cut for. Defaults to the whole take. */
  loopSeconds?: number;
  /** Extra audio recorded either side of the loop. */
  padSeconds?: number;
  offsetSeconds?: number;
  /** Frames over which the tail meets the head; 0 to hear the take raw. */
  seam?: number;
}

/** Build a looping trajectory from one model response. */
export function loopTrajectory(
  response: SynthResponse,
  {
    loopSeconds,
    padSeconds = 0,
    offsetSeconds = 0,
    seam = SEAM_FRAMES,
  }: LoopTrajectoryOptions = {},
): LoopTrajectory {
  const frameRate = response.frame_rate;
  const takeFrames = response.params.voiceness.length;
  if (takeFrames < 2) throw new Error("trajectory too short to loop");

  // Older checkpoints predate lipDiameter; leave it wherever rest puts it.
  const take = {} as Record<Channel, Float32Array>;
  for (const channel of CHANNELS) {
    const source = response.params[channel];
    take[channel] = source
      ? Float32Array.from(source)
      : new Float32Array(takeFrames).fill(REST_VALUES[channel]);
  }

  const nFrames =
    loopSeconds === undefined
      ? takeFrames
      : Math.max(2, Math.min(takeFrames, Math.round(loopSeconds * frameRate)));
  const slack = takeFrames - nFrames;

  const view = {} as Record<Channel, Float32Array>;
  for (const channel of CHANNELS) view[channel] = new Float32Array(nFrames);

  const trajectory: LoopTrajectory = {
    nFrames,
    sourceSeconds: nFrames / frameRate,
    offsetSeconds: 0,

    setOffsetSeconds(seconds) {
      const start = Math.min(
        slack,
        Math.max(0, Math.round((padSeconds - seconds) * frameRate)),
      );
      trajectory.offsetSeconds = seconds;
      for (const channel of CHANNELS) {
        const values = view[channel];
        values.set(take[channel].subarray(start, start + nFrames));
        crossfadeSeam(values, Math.min(seam, Math.floor(nFrames / 4)));
      }
    },

    sample(phase, out) {
      // Wrapped so the last frame interpolates into the first rather than
      // holding; after crossfadeSeam those two are the same pose anyway.
      const x = (phase - Math.floor(phase)) * nFrames;
      const i0 = Math.floor(x) % nFrames;
      const i1 = (i0 + 1) % nFrames;
      const frac = x - Math.floor(x);
      for (const channel of CHANNELS) {
        const values = view[channel];
        out[channel] = values[i0] + (values[i1] - values[i0]) * frac;
      }
    },
  };

  trajectory.setOffsetSeconds(offsetSeconds);
  return trajectory;
}

/** Ramp the last `n` frames toward the first, in place. */
function crossfadeSeam(values: Float32Array, n: number) {
  if (n < 1) return;
  const head = values[0];
  const start = values.length - n;
  for (let i = 0; i < n; i++) {
    const w = (i + 1) / (n + 1);
    values[start + i] += (head - values[start + i]) * w;
  }
}
