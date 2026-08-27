/** Where in the loop we are, on the audio clock.
 *
 * Everything downstream asks this one question — "what phase is it at time
 * t?" — and the two clock sources answer it the same way, so nothing else has
 * to know whether the tempo is ours or the DAW's.
 *
 * External sync deliberately does not chase every MIDI tick. USB clock jitter
 * is around a millisecond and correcting the phase mid-loop would be audible
 * as a lurch in the articulation, so the tempo is averaged continuously and
 * the phase is re-anchored only at loop boundaries — resync at the bar, the
 * way every hardware sequencer does it. */

import { TICKS_PER_BEAT, type ClockTick } from "@/lib/midi";

export type ClockSource = "internal" | "external";

export class LoopClock {
  source: ClockSource = "internal";
  bpm = 120;
  beatsPerBar = 4;
  bars = 2;

  /** Audio-clock time at which the loop last passed phase 0. */
  private anchor = 0;
  private anchored = false;
  /** Seconds per quarter note as last measured from the external clock. */
  private externalBeat = 0.5;
  private running = true;
  /** Phase held while an external transport is stopped. */
  private frozenPhase = 0;

  constructor(ctx: AudioContext) {
    this.anchor = ctx.currentTime;
    this.anchored = true;
  }

  beatSeconds(): number {
    return this.source === "external" ? this.externalBeat : 60 / this.bpm;
  }

  loopSeconds(): number {
    return this.beatSeconds() * this.beatsPerBar * this.bars;
  }

  effectiveBpm(): number {
    return 60 / this.beatSeconds();
  }

  isRunning(): boolean {
    return this.source === "internal" || this.running;
  }

  phaseAt(t: number): number {
    if (!this.isRunning()) return this.frozenPhase;
    const phase = (t - this.anchor) / this.loopSeconds();
    return phase - Math.floor(phase);
  }

  /** Audio-clock time of the first loop boundary strictly after `t`. */
  nextBoundary(t: number): number {
    const loop = this.loopSeconds();
    const turns = Math.floor((t - this.anchor) / loop) + 1;
    return this.anchor + turns * loop;
  }

  /** Loop boundaries are numbered from the anchor, so a recording armed in one
   * loop can name the boundary it means and be recognised later. */
  boundaryIndexAt(t: number): number {
    return Math.floor((t - this.anchor) / this.loopSeconds());
  }

  /** Re-time without moving the loop: the current phase is kept, and the tempo
   * change takes effect from `now` on. Without this, nudging the tempo slider
   * would jump the playhead to a different point in the phrase. */
  setBpm(bpm: number, now: number) {
    const phase = this.phaseAt(now);
    this.bpm = bpm;
    this.reanchor(phase, now);
  }

  /** Change the loop's length in bars or its metre, keeping the downbeat where
   * it is — a 2-bar loop becoming 4 bars grows forward from the same
   * downbeat rather than landing somewhere new. */
  setShape({ bars, beatsPerBar }: { bars?: number; beatsPerBar?: number }, now: number) {
    const boundary = this.nextBoundary(now) - this.loopSeconds();
    if (bars !== undefined) this.bars = bars;
    if (beatsPerBar !== undefined) this.beatsPerBar = beatsPerBar;
    this.anchor = boundary;
    this.anchored = true;
  }

  setSource(source: ClockSource, now: number) {
    if (source === this.source) return;
    const phase = this.phaseAt(now);
    this.source = source;
    // An external clock re-anchors itself on its next boundary; until it does,
    // carry on from where the internal one had got to rather than snapping.
    this.reanchor(phase, now);
    if (source === "external") this.anchored = false;
  }

  /** Restart the loop from phase 0 at `now`. */
  reset(now: number) {
    this.anchor = now;
    this.anchored = true;
    this.frozenPhase = 0;
  }

  /** One MIDI clock pulse. Only meaningful in external mode, but it is fed in
   * regardless so switching to external picks up the tempo already measured. */
  onTick(tick: ClockTick) {
    this.externalBeat = tick.beatSeconds;
    this.running = tick.running;
    if (this.source !== "external") return;

    const ticksPerLoop = TICKS_PER_BEAT * this.beatsPerBar * this.bars;
    const position = ((tick.index % ticksPerLoop) + ticksPerLoop) % ticksPerLoop;
    if (position === 0 || !this.anchored) {
      // On a boundary, that tick *is* phase 0. Off one (which only happens
      // before the first boundary has gone by) place the anchor where it must
      // have been for this tick to land where it does.
      this.anchor =
        tick.audioTime - (position / ticksPerLoop) * this.loopSeconds();
      this.anchored = true;
    }
  }

  onTransport(running: boolean, now: number) {
    if (this.source !== "external") return;
    if (!running && this.running) this.frozenPhase = this.phaseAt(now);
    this.running = running;
    // Start re-anchors on its first tick; Continue keeps the position pointer.
    if (running) this.anchored = false;
  }

  private reanchor(phase: number, now: number) {
    this.anchor = now - phase * this.loopSeconds();
    this.anchored = true;
    this.frozenPhase = phase;
  }
}
