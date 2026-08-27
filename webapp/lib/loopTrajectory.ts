/** A model response turned into something that can be read at any loop phase.
 *
 * The trajectory is stored per channel and indexed by phase in [0, 1) rather
 * than by frame, which is what makes the loop follow tempo for free: at half
 * the tempo the same articulation is read out half as fast, and since pitch is
 * a separate parameter rather than a property of a resampled waveform, nothing
 * transposes. */

import type { SynthResponse } from "@/lib/audio";
import { CHANNELS, Channel, ChannelValues, REST_VALUES } from "@/lib/tractParams";

/** Frames over which the tail is bent to meet the head, so phase 1 and phase 0
 * are the same pose and the loop point doesn't jump. Three frames is ~35 ms at
 * the model's 86 fps: long enough to swallow the discontinuity, short enough
 * that it doesn't audibly clip the end of the phrase. */
const SEAM_FRAMES = 3;

export interface LoopTrajectory {
  nFrames: number;
  /** Seconds of audio this was recorded from, for the UI. */
  sourceSeconds: number;
  channels: Record<Channel, Float32Array>;
  /** Read the pose at `phase` into `out`, without allocating. */
  sample(phase: number, out: ChannelValues): void;
}

/** Build a looping trajectory from one model response.
 *
 * `seam` bends the last few frames toward the first so the loop is continuous;
 * pass 0 to hear the raw recording, seam and all. */
export function loopTrajectory(
  response: SynthResponse,
  { seam = SEAM_FRAMES }: { seam?: number } = {},
): LoopTrajectory {
  const nFrames = response.params.voiceness.length;
  if (nFrames < 2) throw new Error("trajectory too short to loop");

  const channels = {} as Record<Channel, Float32Array>;
  for (const channel of CHANNELS) {
    const source = response.params[channel];
    // Older checkpoints predate lipDiameter; leave it wherever rest puts it.
    const values = source
      ? Float32Array.from(source)
      : new Float32Array(nFrames).fill(REST_VALUES[channel]);
    crossfadeSeam(values, Math.min(seam, Math.floor(nFrames / 4)));
    channels[channel] = values;
  }

  return {
    nFrames,
    sourceSeconds: response.duration_s,
    channels,
    sample(phase, out) {
      // Wrapped so the last frame interpolates into the first rather than
      // holding; after crossfadeSeam those two are the same pose anyway.
      const x = (phase - Math.floor(phase)) * nFrames;
      const i0 = Math.floor(x) % nFrames;
      const i1 = (i0 + 1) % nFrames;
      const frac = x - Math.floor(x);
      for (const channel of CHANNELS) {
        const values = channels[channel];
        out[channel] = values[i0] + (values[i1] - values[i0]) * frac;
      }
    },
  };
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
