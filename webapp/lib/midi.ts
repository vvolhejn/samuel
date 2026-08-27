/** Web MIDI input: notes, controllers, and the transport clock.
 *
 * Everything here is read synchronously by the scheduler, so the state lives in
 * plain mutable fields rather than React state — MIDI arrives far faster than
 * anything should re-render. The page subscribes to `onChange` for the parts it
 * displays, which is throttled by rendering, not by the message rate.
 *
 * Support is Chromium-wide, Firefox behind a permission prompt, and absent in
 * Safari. It also needs a secure context, same as the microphone. */

/** MIDI clock is 24 pulses per quarter note, fixed by the spec. */
export const TICKS_PER_BEAT = 24;

/** Sane inter-tick intervals, in ms: outside this a message is a hiccup in
 * delivery rather than a tempo, and taking it would lurch the loop. Covers
 * roughly 12–500 bpm. */
const MIN_TICK_MS = 5;
const MAX_TICK_MS = 200;

/** Weight of each new interval in the tempo average. Low, because USB clock
 * jitter is on the order of a millisecond and the loop length is derived from
 * this — it should crawl toward a tempo change rather than chase every tick. */
const TEMPO_EMA = 0.08;

export type MidiSupport = "ok" | "unsupported" | "denied";

export interface MidiInputInfo {
  id: string;
  name: string;
}

export interface ClockTick {
  /** Ticks since the last Start (or Song Position), so `index % (24 * beats)`
   * is the position within a bar. */
  index: number;
  /** When it happened, on the AudioContext clock. */
  audioTime: number;
  beatSeconds: number;
  running: boolean;
}

export interface MidiEngineOptions {
  /** The clock the rest of the app schedules against; MIDI timestamps are
   * converted into it so the two can be compared. */
  ctx: AudioContext;
  onTick?: (tick: ClockTick) => void;
  /** Start/Continue/Stop, i.e. the transport moving, not a tick. */
  onTransport?: (running: boolean) => void;
  /** A note was pressed or released. `note` is null when the last one lifted. */
  onNote?: (note: number | null, velocity: number, audioTime: number) => void;
  /** Anything the UI shows changed: device list, held notes, a controller. */
  onChange?: () => void;
}

export class MidiEngine {
  support: MidiSupport = "ok";
  error: string | null = null;
  inputs: MidiInputInfo[] = [];
  /** Which input we listen to; null means all of them. */
  selectedId: string | null = null;

  /** Held notes, most recent last: mono, last-note priority, the way a
   * monophonic synth has always done it. One tract, one glottis. */
  readonly notes: number[] = [];
  velocity = 0;
  /** Pitch bend in semitones, already scaled by `bendRange`. */
  bendSemitones = 0;
  bendRange = 2;
  /** Last value seen for each controller number, 0–127. */
  readonly cc = new Uint8Array(128);
  /** Controllers that have moved since the page loaded, for the CC picker's
   * "learn" affordance and so the UI can list only what your gear sends. */
  readonly seenCc = new Set<number>();
  /** The most recent controller to move, for MIDI learn. */
  lastCc: number | null = null;

  clockRunning = false;
  /** Ticks since Start. -1 before the first one arrives. */
  tickIndex = -1;
  /** Seconds per quarter note, from the external clock. */
  beatSeconds = 0.5;
  /** Has an external clock ticked recently enough to trust? */
  clockPresent = false;

  private access: MIDIAccess | null = null;
  private pendingLearn: ((cc: number) => void) | null = null;
  private lastTickAt = 0;
  private readonly opts: MidiEngineOptions;
  private disposed = false;

  constructor(opts: MidiEngineOptions) {
    this.opts = opts;
  }

  async open(): Promise<MidiSupport> {
    if (!navigator.requestMIDIAccess) {
      this.support = "unsupported";
      this.error =
        "This browser has no Web MIDI (Safari doesn't; Chrome and Edge do).";
      this.opts.onChange?.();
      return this.support;
    }
    try {
      // sysex would make the permission prompt scarier and buys nothing here.
      this.access = await navigator.requestMIDIAccess({ sysex: false });
    } catch (e) {
      this.support = "denied";
      this.error = e instanceof Error ? e.message : String(e);
      this.opts.onChange?.();
      return this.support;
    }
    if (this.disposed) {
      this.access = null;
      return this.support;
    }
    this.access.onstatechange = () => this.rebind();
    this.rebind();
    return this.support;
  }

  dispose() {
    this.disposed = true;
    if (!this.access) return;
    for (const input of this.access.inputs.values()) input.onmidimessage = null;
    this.access.onstatechange = null;
    this.access = null;
  }

  selectInput(id: string | null) {
    this.selectedId = id;
    this.rebind();
  }

  /** The frequency of the note being held, bend included, or null if none is. */
  noteFrequency(): number | null {
    const note = this.notes[this.notes.length - 1];
    if (note === undefined) return null;
    return noteToHz(note + this.bendSemitones);
  }

  /** Semitones from `root` of the note being held, bend included. */
  noteInterval(root: number): number | null {
    const note = this.notes[this.notes.length - 1];
    if (note === undefined) return null;
    return note + this.bendSemitones - root;
  }

  /** Hand the next controller that moves to `callback` instead of treating it
   * as an ordinary message — MIDI learn, so a mapping can be made by reaching
   * for the knob rather than by looking its number up. */
  learnNextCc(callback: (cc: number) => void) {
    this.pendingLearn = callback;
  }

  cancelLearn() {
    this.pendingLearn = null;
  }

  /** Press and release notes from something that isn't a MIDI keyboard — the
   * computer keyboard, so the looper can be played without hardware. They join
   * the same held-note stack, so everything downstream is none the wiser. */
  pressNote(note: number, velocity = 100) {
    this.noteOn(note, velocity, 0);
  }

  releaseNote(note: number) {
    this.noteOff(note, 0);
  }

  /** A controller as a fraction in [0, 1]. `fallback` is what an untouched
   * controller reads as, so a mapping doesn't slam to zero the moment it is
   * assigned to a knob that hasn't moved yet. */
  ccFraction(number: number, fallback = 1): number {
    if (!this.seenCc.has(number)) return fallback;
    return this.cc[number] / 127;
  }

  private rebind() {
    if (!this.access) return;
    this.inputs = [];
    for (const input of this.access.inputs.values()) {
      this.inputs.push({ id: input.id, name: input.name ?? input.id });
      const listen = this.selectedId === null || this.selectedId === input.id;
      input.onmidimessage = listen
        ? (event) => this.handle(event as MIDIMessageEvent)
        : null;
    }
    this.opts.onChange?.();
  }

  /** MIDI timestamps share performance.now()'s origin; the audio clock does
   * not, so shift by the difference between the two read back to back. Good to
   * about a millisecond, which is under the jitter we are correcting for. */
  private audioTimeOf(timeStamp: number): number {
    const now = performance.now();
    // Some drivers report 0, and none should report the future.
    if (!timeStamp || timeStamp > now) return this.opts.ctx.currentTime;
    return this.opts.ctx.currentTime - (now - timeStamp) / 1000;
  }

  private handle(event: MIDIMessageEvent) {
    const data = event.data;
    if (!data || data.length === 0) return;
    const status = data[0];

    // System real-time messages (0xF8..0xFF) can arrive in the middle of
    // anything and carry no channel.
    if (status >= 0xf8) {
      this.system(status, event.timeStamp);
      return;
    }
    if (status === 0xf2) {
      // Song Position Pointer, in sixteenth notes since the start of the song.
      const sixteenths = data[1] | (data[2] << 7);
      this.tickIndex = sixteenths * (TICKS_PER_BEAT / 4);
      return;
    }

    const kind = status & 0xf0;
    if (kind === 0x90 && data[2] > 0) this.noteOn(data[1], data[2], event.timeStamp);
    else if (kind === 0x80 || (kind === 0x90 && data[2] === 0))
      this.noteOff(data[1], event.timeStamp);
    else if (kind === 0xb0) this.controller(data[1], data[2]);
    else if (kind === 0xe0) this.pitchBend(data[1] | (data[2] << 7));
  }

  private system(status: number, timeStamp: number) {
    switch (status) {
      case 0xf8: {
        const at = performance.now();
        const interval = at - this.lastTickAt;
        this.lastTickAt = at;
        if (interval >= MIN_TICK_MS && interval <= MAX_TICK_MS) {
          const beat = (interval * TICKS_PER_BEAT) / 1000;
          this.beatSeconds = this.clockPresent
            ? this.beatSeconds + (beat - this.beatSeconds) * TEMPO_EMA
            : beat;
          this.clockPresent = true;
        }
        this.tickIndex = this.tickIndex < 0 ? 0 : this.tickIndex + 1;
        this.opts.onTick?.({
          index: this.tickIndex,
          audioTime: this.audioTimeOf(timeStamp),
          beatSeconds: this.beatSeconds,
          running: this.clockRunning,
        });
        break;
      }
      case 0xfa: // Start: the next tick is the downbeat.
        this.tickIndex = -1;
        this.clockRunning = true;
        this.opts.onTransport?.(true);
        this.opts.onChange?.();
        break;
      case 0xfb: // Continue: resume from wherever the position pointer left off.
        this.clockRunning = true;
        this.opts.onTransport?.(true);
        this.opts.onChange?.();
        break;
      case 0xfc:
        this.clockRunning = false;
        this.opts.onTransport?.(false);
        this.opts.onChange?.();
        break;
    }
  }

  private noteOn(note: number, velocity: number, timeStamp: number) {
    const at = this.notes.indexOf(note);
    if (at !== -1) this.notes.splice(at, 1);
    this.notes.push(note);
    this.velocity = velocity;
    this.opts.onNote?.(note, velocity, this.audioTimeOf(timeStamp));
    this.opts.onChange?.();
  }

  private noteOff(note: number, timeStamp: number) {
    const at = this.notes.indexOf(note);
    if (at === -1) return;
    const wasTop = at === this.notes.length - 1;
    this.notes.splice(at, 1);
    // Lifting a note you were holding underneath changes nothing you can hear.
    if (!wasTop) return;
    const next = this.notes[this.notes.length - 1] ?? null;
    this.opts.onNote?.(next, this.velocity, this.audioTimeOf(timeStamp));
    this.opts.onChange?.();
  }

  private controller(number: number, value: number) {
    this.cc[number] = value;
    this.lastCc = number;
    if (this.pendingLearn) {
      const learn = this.pendingLearn;
      this.pendingLearn = null;
      this.seenCc.add(number);
      learn(number);
      this.opts.onChange?.();
      return;
    }
    const known = this.seenCc.has(number);
    this.seenCc.add(number);
    // A knob sweep is hundreds of messages; only tell the UI about the first,
    // which is the one that changes what it can offer you.
    if (!known) this.opts.onChange?.();
  }

  private pitchBend(raw: number) {
    // 14-bit, centred at 8192.
    this.bendSemitones = ((raw - 8192) / 8192) * this.bendRange;
  }
}

export function noteToHz(note: number): number {
  return 440 * Math.pow(2, (note - 69) / 12);
}

const NOTE_NAMES = [
  "C",
  "C#",
  "D",
  "D#",
  "E",
  "F",
  "F#",
  "G",
  "G#",
  "A",
  "A#",
  "B",
];

export function noteName(note: number): string {
  return `${NOTE_NAMES[note % 12]}${Math.floor(note / 12) - 1}`;
}
