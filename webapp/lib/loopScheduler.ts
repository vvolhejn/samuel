/** Plays a looping parameter trajectory, letting MIDI take channels off it.
 *
 * Web Audio has no "call me every frame" hook that runs early enough to
 * schedule automation, so this is the usual two-clock arrangement: a coarse
 * timer wakes up often and writes every parameter frame that falls inside a
 * short lookahead window, each as a ramp landing at an exact audio-clock time.
 * The timer may be late without anything being heard, as long as it is never
 * later than the window.
 *
 * ## Handing over between the trajectory and your hands
 *
 * Every channel keeps a weight `w`: 0 is purely the recording, 1 is purely
 * manual, and the value written is the crossfade between them. Nothing is
 * smoothed on the way to the parameter, so at w = 0 the recorded articulation
 * reaches the synth exactly as the model predicted it — a filter over the
 * output would round off the consonants. The ramps are on `w` instead, which
 * is what makes releasing a note glide back into the phrase rather than snap.
 *
 * Pitch in `replace` mode is the exception, and it is the exception on purpose.
 * A held note's frequency does not depend on the recording at all, so it is
 * written the instant the note arrives instead of waiting for the next
 * scheduled frame, and the scheduler stops writing that parameter until the
 * note lifts. That takes the lookahead window out of note-on latency, which is
 * the difference between a keyboard you can play and one you can't. `offset`
 * mode cannot do this — a transposition has to keep tracking the contour it is
 * transposing — so there the note goes through the scheduler like everything
 * else and costs one window of delay. */

import { LoopClock } from "@/lib/loopClock";
import { LoopTrajectory } from "@/lib/loopTrajectory";
import { MidiEngine, noteToHz } from "@/lib/midi";
import {
  CHANNELS,
  CHANNEL_COMBINE,
  Channel,
  ChannelValues,
  DEFAULT_RANGES,
  REST_VALUES,
  TractTargets,
  allParams,
  cancelAndHold,
  clamp,
  pin,
  writeFrame,
} from "@/lib/tractParams";

/** How far ahead automation is written. Every millisecond here is a
 * millisecond of latency on anything MIDI does through the scheduler, so it is
 * as short as a main-thread timer can be trusted to beat. */
const LOOKAHEAD_S = 0.03;
const TICK_MS = 10;

/** Parameter frames per second. The model's own control rate (44100/512), so
 * a loop played at the tempo it was recorded at is written frame for frame. */
const DEFAULT_WRITE_HZ = 44100 / 512;

/** Crossfade times between the recording and your hands. The attack is short
 * enough to feel like a note starting; the release is long enough that falling
 * back into the phrase is a slur, not a step. */
const ATTACK_S = 0.01;
const RELEASE_S = 0.04;

/** Portamento between notes, and the smoothing on the immediate pitch path. */
const GLIDE_S = 0.012;

const GATE_FADE_S = 0.05;

export type OverrideMode = "auto" | "replace" | "offset";

export interface ChannelOverride {
  mode: OverrideMode;
  /** Controller driving this channel, or null if none is assigned. Unused for
   * `frequency`, which the keyboard drives. */
  cc: number | null;
  /** Full-scale travel in `offset` mode: a multiplier for loudness, native
   * units for everything else. Unused for pitch, whose offset is the interval
   * you played and nothing else. */
  depth: number;
}

/** The note `offset` mode transposes relative to: play this one and the pitch
 * contour comes out where it was recorded. Middle C. */
export const DEFAULT_ROOT_NOTE = 60;

function defaultOverride(channel: Channel): ChannelOverride {
  const [lo, hi] = DEFAULT_RANGES[channel];
  return {
    mode: "auto",
    cc: null,
    // Half the parameter's range either way, which is as much as a bipolar
    // controller can push without the clamp doing most of the work.
    depth: CHANNEL_COMBINE[channel] === "add" ? (hi - lo) / 2 : 1,
  };
}

export interface LoopSchedulerOptions {
  targets: TractTargets;
  clock: LoopClock;
  midi: MidiEngine;
  /** Parameter ranges from the loaded checkpoint (GET /api/health). */
  ranges?: Record<Channel, [number, number]>;
  writeHz?: number;
  /** Called once per loop turn, with the boundary's audio-clock time. */
  onLoop?: (boundaryTime: number) => void;
}

export class LoopScheduler {
  readonly clock: LoopClock;
  readonly overrides: Record<Channel, ChannelOverride>;
  rootNote = DEFAULT_ROOT_NOTE;

  private readonly targets: TractTargets;
  private readonly midi: MidiEngine;
  private readonly ranges: Record<Channel, [number, number]>;
  private readonly frameSeconds: number;
  private readonly onLoop?: (boundaryTime: number) => void;

  private timer: ReturnType<typeof setInterval> | null = null;
  private running = false;
  /** Output held down while the loop keeps turning underneath it. */
  private muted = false;
  /** Audio-clock time of the next parameter frame to write. */
  private nextTime = 0;
  private lastPhase = 0;

  private trajectory: LoopTrajectory | null = null;
  /** Swapped in at the next loop boundary, so a new take never lands mid-phrase. */
  private pending: LoopTrajectory | null = null;

  private readonly weights: Record<Channel, number>;
  /** Pitch is being written straight to the parameter; the scheduler is off it. */
  private immediatePitch = false;
  private lastImmediateHz = 0;
  private readonly base: ChannelValues = { ...REST_VALUES };
  private readonly out: ChannelValues = { ...REST_VALUES };
  private readonly skip = new Set<Channel>();

  constructor(opts: LoopSchedulerOptions) {
    this.targets = opts.targets;
    this.clock = opts.clock;
    this.midi = opts.midi;
    this.ranges = opts.ranges ?? DEFAULT_RANGES;
    this.frameSeconds = 1 / (opts.writeHz ?? DEFAULT_WRITE_HZ);
    this.onLoop = opts.onLoop;
    this.overrides = Object.fromEntries(
      CHANNELS.map((c) => [c, defaultOverride(c)]),
    ) as Record<Channel, ChannelOverride>;
    this.weights = Object.fromEntries(CHANNELS.map((c) => [c, 0])) as Record<
      Channel,
      number
    >;
  }

  isRunning(): boolean {
    return this.running;
  }

  hasTrajectory(): boolean {
    return this.trajectory !== null;
  }

  /** Loop phase right now, for the playhead. */
  phase(): number {
    return this.clock.phaseAt(this.targets.ctx.currentTime);
  }

  start() {
    if (this.running) return;
    this.running = true;
    const now = this.targets.ctx.currentTime;
    this.nextTime = now;
    this.lastPhase = this.clock.phaseAt(now);
    for (const param of allParams(this.targets)) pin(param, now);
    const gate = this.targets.gate;
    cancelAndHold(gate, now);
    gate.setTargetAtTime(this.muted ? 0 : 1, now, GATE_FADE_S / 3);
    this.timer = setInterval(this.tick, TICK_MS);
  }

  stop() {
    if (!this.running) return;
    this.running = false;
    if (this.timer !== null) clearInterval(this.timer);
    this.timer = null;
    const now = this.targets.ctx.currentTime;
    for (const param of allParams(this.targets)) cancelAndHold(param, now);
    const gate = this.targets.gate;
    cancelAndHold(gate, now);
    gate.setTargetAtTime(0, now, GATE_FADE_S / 3);
    this.immediatePitch = false;
  }

  dispose() {
    this.stop();
  }

  /** Install a take. It takes over at the next loop boundary unless there is
   * nothing playing yet, in which case it starts immediately. */
  setTrajectory(trajectory: LoopTrajectory | null) {
    if (trajectory === null) {
      this.trajectory = null;
      this.pending = null;
      return;
    }
    if (!this.trajectory) this.trajectory = trajectory;
    else this.pending = trajectory;
  }

  /** Silence the output without stopping the loop. The trajectory keeps
   * playing against the clock, so unmuting drops you back into the phrase
   * where it has got to rather than where you left it. Used while a take is
   * recorded: the take is of you, not of the loop and the room. */
  setMuted(muted: boolean) {
    if (muted === this.muted) return;
    this.muted = muted;
    if (!this.running) return;
    const now = this.targets.ctx.currentTime;
    const gate = this.targets.gate;
    cancelAndHold(gate, now);
    gate.setTargetAtTime(muted ? 0 : 1, now, GATE_FADE_S / 3);
  }

  isMuted(): boolean {
    return this.muted;
  }

  /** Re-align every take the scheduler holds against the grid, in seconds of
   * input latency. Applies to the pending take too: it is the same hands and
   * the same interface, so it was recorded through the same latency. */
  setTakeOffsetSeconds(seconds: number) {
    this.trajectory?.setOffsetSeconds(seconds);
    this.pending?.setOffsetSeconds(seconds);
  }

  setOverride(channel: Channel, patch: Partial<ChannelOverride>) {
    const override = this.overrides[channel];
    Object.assign(override, patch);
    // Dropping back to auto while a note is down would otherwise leave the
    // channel stuck at full manual weight until the note lifts.
    if (channel === "frequency" && override.mode !== "replace") {
      this.releaseImmediatePitch();
    }
  }

  /** Fed from MidiEngine.onNote. `note` is null when the last key lifted. */
  onNote(note: number | null) {
    const override = this.overrides.frequency;
    if (override.mode !== "replace") return;
    const ctx = this.targets.ctx;
    const now = ctx.currentTime;
    if (note === null) {
      this.releaseImmediatePitch();
      return;
    }
    if (!this.running) return;
    const hz = clamp(
      noteToHz(note + this.midi.bendSemitones),
      this.ranges.frequency,
    );
    // Drop the frames already queued for this parameter — they are the
    // trajectory's, and the note supersedes them.
    cancelAndHold(this.targets.frequency, now);
    this.targets.frequency.setTargetAtTime(hz, now, GLIDE_S);
    this.lastImmediateHz = hz;
    this.immediatePitch = true;
    this.weights.frequency = 1;
  }

  private releaseImmediatePitch() {
    if (!this.immediatePitch) return;
    this.immediatePitch = false;
    // Leave the parameter where the note actually left it, so the scheduler's
    // first ramp back into the phrase starts from there and not from whatever
    // event the glide was heading away from.
    pin(this.targets.frequency, this.targets.ctx.currentTime);
  }

  private tick = () => {
    const ctx = this.targets.ctx;
    const now = ctx.currentTime;
    // A timer that overslept, or a context that was suspended: give up on the
    // frames that went by rather than writing a burst of them into the past.
    if (this.nextTime < now) this.nextTime = now;
    const horizon = now + LOOKAHEAD_S;
    while (this.nextTime <= horizon) {
      this.writeFrameAt(this.nextTime);
      this.nextTime += this.frameSeconds;
    }
    this.trackBend(now);
  };

  /** Pitch bend arrives as a stream of controller messages rather than as
   * note events, so the immediate path re-aims at each tick. */
  private trackBend(now: number) {
    if (!this.immediatePitch) return;
    const hz = this.midi.noteFrequency();
    if (hz === null) return;
    const target = clamp(hz, this.ranges.frequency);
    // A cent either way is inaudible and this runs a hundred times a second.
    if (Math.abs(Math.log2(target / this.lastImmediateHz)) < 1 / 1200) return;
    this.targets.frequency.setTargetAtTime(target, now, GLIDE_S);
    this.lastImmediateHz = target;
  }

  private writeFrameAt(t: number) {
    const phase = this.clock.phaseAt(t);
    if (phase < this.lastPhase) {
      // Wrapped: this frame is the first of a new turn.
      if (this.pending) {
        this.trajectory = this.pending;
        this.pending = null;
      }
      this.onLoop?.(t);
    }
    this.lastPhase = phase;

    if (this.trajectory) this.trajectory.sample(phase, this.base);
    else Object.assign(this.base, REST_VALUES);

    this.skip.clear();
    if (this.immediatePitch) this.skip.add("frequency");

    for (const channel of CHANNELS) {
      const w = this.advanceWeight(channel);
      const base = this.base[channel];
      this.out[channel] =
        w === 0
          ? base
          : clamp(
              blend(channel, base, this.manualValue(channel, base), w),
              this.ranges[channel],
            );
    }

    writeFrame(this.targets, this.out, t, this.skip);
  }

  /** Step a channel's crossfade one frame toward where it should be. */
  private advanceWeight(channel: Channel): number {
    const target = this.weightTarget(channel);
    const current = this.weights[channel];
    if (current === target) return current;
    const seconds = target > current ? ATTACK_S : RELEASE_S;
    const step = this.frameSeconds / seconds;
    const next =
      target > current
        ? Math.min(target, current + step)
        : Math.max(target, current - step);
    this.weights[channel] = next;
    return next;
  }

  private weightTarget(channel: Channel): number {
    const override = this.overrides[channel];
    if (override.mode === "auto") return 0;
    if (channel === "frequency") {
      // Pitch is gated by the keyboard: no key down, no override. Anything
      // else and letting go would leave the last note hanging over the phrase.
      return this.midi.notes.length > 0 ? 1 : 0;
    }
    return override.cc === null ? 0 : 1;
  }

  /** What this channel would read if the override had it entirely. */
  private manualValue(channel: Channel, base: number): number {
    const override = this.overrides[channel];
    const range = this.ranges[channel];
    const combine = CHANNEL_COMBINE[channel];

    if (channel === "frequency") {
      if (override.mode === "replace") {
        return this.midi.noteFrequency() ?? base;
      }
      const interval = this.midi.noteInterval(this.rootNote);
      if (interval === null) return base;
      // Transpose: the recorded contour, moved by the interval you played, so
      // playing the root gives back exactly what was recorded.
      return base * Math.pow(2, interval / 12);
    }

    if (override.cc === null) return base;

    if (override.mode === "replace") {
      const fraction = this.midi.ccFraction(override.cc, 0.5);
      return range[0] + fraction * (range[1] - range[0]);
    }

    if (combine === "scale") {
      // A fader: untouched sits at unity, so assigning a controller that
      // hasn't moved yet doesn't silence the loop.
      return base * this.midi.ccFraction(override.cc, 1) * override.depth;
    }
    // Bipolar around the centre detent.
    return base + (this.midi.ccFraction(override.cc, 0.5) * 2 - 1) * override.depth;
  }
}

/** Crossfade one channel between the recording and the manual value. Pitch
 * interpolates in the log domain: halfway between 100 Hz and 400 Hz is 200 Hz,
 * not 250 Hz, and anything else makes a glide bend the wrong way. */
function blend(
  channel: Channel,
  base: number,
  manual: number,
  w: number,
): number {
  if (CHANNEL_COMBINE[channel] === "semitones") {
    const from = Math.max(base, 1e-3);
    const to = Math.max(manual, 1e-3);
    return from * Math.pow(to / from, w);
  }
  return base + (manual - base) * w;
}
