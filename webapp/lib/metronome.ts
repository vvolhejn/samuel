/** A click on every beat of the loop, on the audio clock.
 *
 * The same two-clock arrangement as the scheduler: a coarse timer wakes up
 * often and schedules the clicks that fall inside a short lookahead, each at an
 * exact time. Beat times are re-derived from the clock every wake-up rather
 * than accumulated, so a tempo change or a re-anchored downbeat moves the
 * clicks with it instead of drifting away from the loop. */

import { LoopClock } from "@/lib/loopClock";

const LOOKAHEAD_S = 0.1;
const TICK_MS = 25;

/** Click envelope: a short blip, loud enough to hear over the tract. */
const CLICK_S = 0.04;
const CLICK_GAIN = 0.18;
const ACCENT_HZ = 1600;
const BEAT_HZ = 1000;

export class LoopMetronome {
  private readonly ctx: AudioContext;
  private readonly clock: LoopClock;
  private readonly out: GainNode;
  private timer: ReturnType<typeof setInterval> | null = null;
  private audible = false;
  /** Last beat already scheduled, on the audio clock. */
  private lastTime = 0;

  constructor(ctx: AudioContext, clock: LoopClock) {
    this.ctx = ctx;
    this.clock = clock;
    this.out = ctx.createGain();
    this.out.gain.value = CLICK_GAIN;
    this.out.connect(ctx.destination);
  }

  isAudible(): boolean {
    return this.audible;
  }

  setAudible(audible: boolean) {
    if (audible === this.audible) return;
    this.audible = audible;
    if (!audible) {
      this.stop();
      return;
    }
    // Start counting from now: a metronome switched on mid-loop owes you no
    // clicks for the beats that already went by.
    this.lastTime = this.ctx.currentTime;
    this.timer ??= setInterval(this.tick, TICK_MS);
  }

  stop() {
    if (this.timer !== null) clearInterval(this.timer);
    this.timer = null;
  }

  dispose() {
    this.stop();
    this.out.disconnect();
  }

  private tick = () => {
    if (!this.clock.isRunning()) return;
    const now = this.ctx.currentTime;
    const horizon = now + LOOKAHEAD_S;
    let from = Math.max(now, this.lastTime);
    for (let i = 0; i < 64; i++) {
      const beat = this.clock.nextBeat(from);
      if (beat.time > horizon) break;
      this.click(beat.time, beat.index % this.clock.beatsPerBar === 0);
      this.lastTime = beat.time;
      from = beat.time;
    }
  };

  private click(time: number, accent: boolean) {
    const osc = this.ctx.createOscillator();
    const gain = this.ctx.createGain();
    osc.frequency.value = accent ? ACCENT_HZ : BEAT_HZ;
    gain.gain.setValueAtTime(0, time);
    gain.gain.linearRampToValueAtTime(accent ? 1 : 0.55, time + 0.001);
    gain.gain.exponentialRampToValueAtTime(1e-4, time + CLICK_S);
    osc.connect(gain);
    gain.connect(this.out);
    osc.start(time);
    osc.stop(time + CLICK_S + 0.01);
  }
}
