/** The looper: record a bar, hear it back forever, play over it.
 *
 * Holds the four moving parts together — the mic buffer, the model round trip,
 * the loop clock, and the scheduler that writes the tract — and exposes the
 * handful of things the page actually renders. Everything that ticks faster
 * than a render (MIDI messages, parameter frames, the playhead) stays out of
 * React state; `version` is bumped only for the things you can see change. */

import { useCallback, useEffect, useRef, useState } from "react";
import { synthesizeTake, fetchHealth } from "@/lib/audio";
import { LoopClock, type ClockSource } from "@/lib/loopClock";
import { LoopRecorder } from "@/lib/loopRecorder";
import { LoopScheduler, type ChannelOverride } from "@/lib/loopScheduler";
import { LoopMetronome } from "@/lib/metronome";
import { loopTrajectory } from "@/lib/loopTrajectory";
import { MidiEngine, type MidiSupport } from "@/lib/midi";
import {
  listTakes,
  saveTake,
  deleteTake as deleteStoredTake,
  takeResponse,
  type StoredTake,
} from "@/lib/takeStore";
import { MicProcessing, MIC_PROCESSING_DEFAULTS } from "@/lib/micProcessing";
import { micErrorMessage } from "@/lib/secureContext";
import { useMirroredState } from "@/lib/useMirroredState";
import { CHANNELS, Channel, DEFAULT_RANGES } from "@/lib/tractParams";
import type { PinkTromboneHandle } from "@/lib/usePinkTrombone";

/** Where a take is in its life. `thinking` is the model round trip, which at a
 * couple of bars takes a fraction of the loop it will drop into. */
export type TakeState = "idle" | "armed" | "recording" | "thinking";

/** When the click is audible. `recording` covers the count-in too, so the
 * beats leading up to the downbeat are the ones you come in on. */
export type MetronomeMode = "off" | "recording" | "always";

/** How often the take poller looks at the clock. Fine enough to notice a
 * boundary promptly, coarse enough not to matter. */
const POLL_MS = 20;

/** Give the mic buffer this long past the end of a window before deciding the
 * audio is never coming. */
const EXTRACT_TIMEOUT_S = 1;

/** Default compensation for microphone input latency, in ms. There is no API
 * that reports it, so this is a guess at a typical desktop figure and the UI
 * lets you nudge it: too low and the take sits late against the grid, too
 * early and the loop comes in ahead of the beat. */
const DEFAULT_RECORD_OFFSET_MS = 30;

/** How far the offset can travel either way. The take is padded by this much
 * on each side, so the whole range can be scrubbed without recording again. */
export const RECORD_OFFSET_LIMIT_MS = 200;

/** Extra audio recorded either side of the bar. The limit plus a margin, so
 * the extreme ends of the slider are still inside the take. */
const TAKE_PAD_S = (RECORD_OFFSET_LIMIT_MS + 30) / 1000;

/** How early the loop is muted before the pad starts, so the pad holds your
 * voice and not the tail of a fading loop. */
const MUTE_LEAD_S = 0.1;

export interface LooperHandle {
  /** Bumped whenever something the UI shows has changed underneath it. */
  version: number;
  error: string | null;
  setError: (message: string | null) => void;

  running: boolean;
  hasLoop: boolean;
  take: TakeState;
  /** Seconds of audio in the last take, for the readout. */
  takeSeconds: number | null;

  midi: MidiEngine | null;
  midiSupport: MidiSupport | null;
  clock: LoopClock | null;
  scheduler: LoopScheduler | null;
  levelStore: LoopRecorder["levelStore"] | null;
  /** Takes saved in this browser, newest first. */
  takes: StoredTake[];

  micProcessing: MicProcessing;
  recordOffsetMs: number;
  metronomeMode: MetronomeMode;
  /** Is the click sounding right now, for the UI's beat indicator? */
  metronomeAudible: boolean;
  muteWhileRecording: boolean;

  /** Build the looper against the synth, once it is up. */
  init: () => Promise<void>;
  start: () => Promise<void>;
  stop: () => void;
  record: () => void;
  cancelRecord: () => void;
  clear: () => void;
  resetPhase: () => void;
  setClockSource: (source: ClockSource) => void;
  setBpm: (bpm: number) => void;
  setShape: (shape: { bars?: number; beatsPerBar?: number }) => void;
  setOverride: (channel: Channel, patch: Partial<ChannelOverride>) => void;
  setRootNote: (note: number) => void;
  setRecordOffsetMs: (ms: number) => void;
  setMetronomeMode: (mode: MetronomeMode) => void;
  setMuteWhileRecording: (mute: boolean) => void;
  loadTake: (id: number) => void;
  removeTake: (id: number) => void;
  toggleMicProcessing: (key: keyof MicProcessing) => void;
  selectMidiInput: (id: string | null) => void;
}

export function useLooper(trombone: PinkTromboneHandle): LooperHandle {
  const [version, setVersion] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [running, setRunning] = useState(false);
  const [hasLoop, setHasLoop] = useState(false);
  const [take, setTake] = useState<TakeState>("idle");
  const [takeSeconds, setTakeSeconds] = useState<number | null>(null);
  const [midiSupport, setMidiSupport] = useState<MidiSupport | null>(null);
  const [micProcessing, setMicProcessingState] = useState<MicProcessing>({
    ...MIC_PROCESSING_DEFAULTS,
    // The take's level is what the model reads as loudness, and AGC would ride
    // it up and down under a loop that is already playing.
    autoGainControl: false,
  });
  const [recordOffsetMs, setRecordOffsetMsState] = useState(
    DEFAULT_RECORD_OFFSET_MS,
  );
  const [metronomeMode, setMetronomeMode] = useState<MetronomeMode>("recording");
  // The take is of you, not of the loop: what is already there would come back
  // through the mic and be re-synthesised on top of itself.
  const [muteWhileRecording, setMuteWhileRecording] = useState(true);
  const [takes, setTakes] = useState<StoredTake[]>([]);

  // The moving parts. Mirrored rather than held in refs alone: the page
  // renders from them, and a ref read during render is exactly what the
  // compiler will not let us do. Their identities never change once built, so
  // what actually re-renders the page is `version`.
  const [midi, midiRef, setMidi] = useMirroredState<MidiEngine | null>(null);
  const [clock, clockRef, setClock] = useMirroredState<LoopClock | null>(null);
  const [scheduler, schedulerRef, setScheduler] =
    useMirroredState<LoopScheduler | null>(null);
  const [recorder, recorderRef, setRecorder] =
    useMirroredState<LoopRecorder | null>(null);
  const [metronome, metronomeRef, setMetronome] =
    useMirroredState<LoopMetronome | null>(null);
  /** The window a take will be, or is being, cut from. */
  const windowRef = useRef<{ from: number; to: number } | null>(null);
  /** Bumped by anything that supersedes a take in flight, so a response that
   * arrives after you hit Stop or armed another one is dropped. */
  const takeGenRef = useRef(0);
  const offsetRef = useRef(recordOffsetMs / 1000);
  /** Checkpoint the backend is serving, stored with a take so takes made by
   * different models can be told apart. */
  const fingerprintRef = useRef<string | null>(null);
  const takeRef = useRef<TakeState>("idle");

  const bump = useCallback(() => setVersion((v) => v + 1), []);

  const setTakeState = useCallback((next: TakeState) => {
    takeRef.current = next;
    setTake(next);
  }, []);

  useEffect(
    () => () => {
      takeGenRef.current++;
      schedulerRef.current?.dispose();
      recorderRef.current?.stop();
      metronomeRef.current?.dispose();
      midiRef.current?.dispose();
    },
    [schedulerRef, recorderRef, metronomeRef, midiRef],
  );

  useEffect(() => {
    void listTakes().then(setTakes);
  }, []);

  /** Send a finished take to the model and install what comes back.
   *
   * Deliberately not run inside the poller effect below: that effect is keyed
   * on the take state, so moving to "thinking" tears it down — and a
   * cancellation flag scoped to it would throw away the very response it was
   * waiting for. */
  const finishTake = useCallback(
    async (audio: Float32Array, sampleRate: number, loopSeconds: number) => {
      const generation = ++takeGenRef.current;
      const current = () => takeGenRef.current === generation;
      setTakeState("thinking");
      try {
        const { response } = await synthesizeTake(audio, sampleRate);
        if (!current()) return;
        const trajectory = loopTrajectory(response, {
          loopSeconds,
          padSeconds: TAKE_PAD_S,
          offsetSeconds: offsetRef.current,
        });
        schedulerRef.current?.setTrajectory(trajectory);
        setHasLoop(true);
        setTakeSeconds(trajectory.sourceSeconds);
        const clock = clockRef.current;
        void saveTake({
          id: Date.now(),
          loopSeconds,
          padSeconds: TAKE_PAD_S,
          bars: clock?.bars ?? 1,
          beatsPerBar: clock?.beatsPerBar ?? 4,
          bpm: clock?.effectiveBpm() ?? 120,
          modelFingerprint: fingerprintRef.current,
          frameRate: response.frame_rate,
          params: Object.fromEntries(
            CHANNELS.filter((channel) => response.params[channel]).map(
              (channel) => [channel, Float32Array.from(response.params[channel])],
            ),
          ),
        }).then(setTakes);
      } catch (e) {
        if (current()) setError(e instanceof Error ? e.message : String(e));
      } finally {
        if (current()) setTakeState("idle");
        bump();
      }
    },
    [bump, setTakeState, schedulerRef, clockRef],
  );

  /** Build the machinery, once the synth's AudioContext exists. Deliberately
   * asks for no permissions: the MIDI prompt belongs in the Start gesture, and
   * everything here only has to be true before the controls can be shown. */
  const init = useCallback(async () => {
    const targets = trombone.tract();
    if (!targets || schedulerRef.current) return;

    const ctx = targets.ctx;
    const clock = new LoopClock(ctx);
    setClock(clock);

    const midi = new MidiEngine({
      ctx,
      onTick: (tick) => clock.onTick(tick),
      onTransport: (isRunning) => clock.onTransport(isRunning, ctx.currentTime),
      onNote: (note) => schedulerRef.current?.onNote(note),
      onChange: bump,
    });
    setMidi(midi);

    // The checkpoint's own parameter ranges and control rate, so a manual
    // override lands inside what the model was trained to emit.
    const health = await fetchHealth();
    fingerprintRef.current = health?.model_fingerprint ?? null;
    const ranges = { ...DEFAULT_RANGES };
    for (const channel of CHANNELS) {
      const reported = health?.param_ranges?.[channel];
      if (reported) ranges[channel] = [reported[0], reported[1]];
    }

    setScheduler(
      new LoopScheduler({
        targets,
        clock,
        midi,
        ranges,
        writeHz: health?.frame_rate,
      }),
    );
    setRecorder(new LoopRecorder());
    setMetronome(new LoopMetronome(ctx, clock));
  }, [
    trombone,
    bump,
    schedulerRef,
    setClock,
    setMidi,
    setScheduler,
    setRecorder,
    setMetronome,
  ]);

  const start = useCallback(async () => {
    setError(null);
    try {
      await trombone.resume();
      await init();
      const targets = trombone.tract();
      const scheduler = schedulerRef.current;
      const recorder = recorderRef.current;
      if (!targets || !scheduler || !recorder)
        throw new Error("Synth is not ready yet");
      // Both prompt, so both wait for this gesture. MIDI is allowed to fail —
      // the loop plays without it, you just can't reach into it.
      void midiRef.current?.open().then(setMidiSupport);
      if (!recorder.listening) await recorder.start(targets.ctx, micProcessing);
      scheduler.start();
      setRunning(true);
      bump();
    } catch (e) {
      setError(micErrorMessage(e));
    }
  }, [trombone, init, micProcessing, bump, midiRef, recorderRef, schedulerRef]);

  const stop = useCallback(() => {
    takeGenRef.current++;
    schedulerRef.current?.stop();
    recorderRef.current?.stop();
    windowRef.current = null;
    setTakeState("idle");
    setRunning(false);
    bump();
  }, [bump, setTakeState, recorderRef, schedulerRef]);

  /** Arm a take: it begins at the next loop boundary and runs one loop. */
  const record = useCallback(() => {
    const scheduler = schedulerRef.current;
    const clock = clockRef.current;
    const targets = trombone.tract();
    if (!scheduler || !clock || !targets) return;
    if (!recorderRef.current?.listening) {
      setError("The microphone isn't open — press Start first.");
      return;
    }
    const now = targets.ctx.currentTime;
    const from = clock.nextBoundary(now);
    const loop = clock.loopSeconds();
    // The window reaches back a whole loop from its end, plus the wait between
    // arming and the downbeat and the pad either side of it.
    recorderRef.current.retain(loop * 2 + 2 + 2 * TAKE_PAD_S);
    windowRef.current = { from, to: from + loop };
    setTakeState("armed");
    bump();
  }, [trombone, bump, setTakeState, clockRef, recorderRef, schedulerRef]);

  const cancelRecord = useCallback(() => {
    takeGenRef.current++;
    windowRef.current = null;
    setTakeState("idle");
    bump();
  }, [bump, setTakeState]);

  const clear = useCallback(() => {
    schedulerRef.current?.setTrajectory(null);
    setHasLoop(false);
    setTakeSeconds(null);
    bump();
  }, [bump, schedulerRef]);

  /** Put a saved take back on the loop. The metre it was recorded in comes
   * back with it — a two-bar take dropped into a four-bar loop would play at
   * half speed — but not the tempo, which the loop follows anyway. */
  const loadTake = useCallback(
    (id: number) => {
      const targets = trombone.tract();
      const take = takes.find((candidate) => candidate.id === id);
      if (!take || !targets) return;
      takeGenRef.current++;
      clockRef.current?.setShape(
        { bars: take.bars, beatsPerBar: take.beatsPerBar },
        targets.ctx.currentTime,
      );
      const trajectory = loopTrajectory(takeResponse(take), {
        loopSeconds: take.loopSeconds,
        padSeconds: take.padSeconds,
        offsetSeconds: offsetRef.current,
      });
      schedulerRef.current?.setTrajectory(trajectory);
      setHasLoop(true);
      setTakeSeconds(trajectory.sourceSeconds);
      setTakeState("idle");
      bump();
    },
    [trombone, takes, bump, setTakeState, clockRef, schedulerRef],
  );

  const removeTake = useCallback((id: number) => {
    void deleteStoredTake(id).then(() => listTakes().then(setTakes));
  }, []);

  const resetPhase = useCallback(() => {
    const targets = trombone.tract();
    if (!targets || !clockRef.current) return;
    clockRef.current.reset(targets.ctx.currentTime);
    bump();
  }, [trombone, bump, clockRef]);

  // The take poller: watches the clock cross into and out of the armed window,
  // then pulls the audio out of the mic buffer and sends it off.
  useEffect(() => {
    if (take !== "armed" && take !== "recording") return;
    const targets = trombone.tract();
    const recorder = recorderRef.current;
    if (!targets || !recorder) return;

    let cancelled = false;
    const timer = setInterval(() => {
      const window = windowRef.current;
      if (!window || cancelled) return;
      const now = targets.ctx.currentTime;
      // A take is the bar plus a pad either side, and the recording state
      // starts with the pad: it is muted audio the loop must not bleed into.
      const from = window.from - TAKE_PAD_S;
      const to = window.to + TAKE_PAD_S;
      if (takeRef.current === "armed") {
        if (now >= from - MUTE_LEAD_S) setTakeState("recording");
        return;
      }
      if (now < to) return;
      // Past the end of the window; the last blocks may still be in flight.
      const audio = recorder.extract(from, to);
      if (!audio) {
        if (now > to + EXTRACT_TIMEOUT_S) {
          windowRef.current = null;
          setTakeState("idle");
          setError("Lost the recording — the mic buffer didn't reach the end of the bar.");
        }
        return;
      }
      windowRef.current = null;
      void finishTake(audio, recorder.sampleRate, window.to - window.from);
    }, POLL_MS);

    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, [take, trombone, finishTake, setTakeState, recorderRef]);

  /** Whether the click should be sounding: always, or only around a take. */
  const metronomeAudible =
    running &&
    (metronomeMode === "always" ||
      (metronomeMode === "recording" &&
        (take === "armed" || take === "recording")));

  useEffect(() => {
    metronome?.setAudible(metronomeAudible);
  }, [metronome, metronomeAudible]);

  // Mute the loop for the take itself, but not for the count-in: you need to
  // hear what you are playing over until the moment you replace it.
  useEffect(() => {
    scheduler?.setMuted(muteWhileRecording && take === "recording");
  }, [scheduler, muteWhileRecording, take]);

  const setClockSource = useCallback(
    (source: ClockSource) => {
      const targets = trombone.tract();
      if (!targets || !clockRef.current) return;
      clockRef.current.setSource(source, targets.ctx.currentTime);
      bump();
    },
    [trombone, bump, clockRef],
  );

  const setBpm = useCallback(
    (bpm: number) => {
      const targets = trombone.tract();
      if (!targets || !clockRef.current) return;
      clockRef.current.setBpm(bpm, targets.ctx.currentTime);
      bump();
    },
    [trombone, bump, clockRef],
  );

  const setShape = useCallback(
    (shape: { bars?: number; beatsPerBar?: number }) => {
      const targets = trombone.tract();
      if (!targets || !clockRef.current) return;
      clockRef.current.setShape(shape, targets.ctx.currentTime);
      bump();
    },
    [trombone, bump, clockRef],
  );

  const setOverride = useCallback(
    (channel: Channel, patch: Partial<ChannelOverride>) => {
      schedulerRef.current?.setOverride(channel, patch);
      bump();
    },
    [bump, schedulerRef],
  );

  const setRootNote = useCallback(
    (note: number) => {
      if (schedulerRef.current) schedulerRef.current.rootNote = note;
      bump();
    },
    [bump, schedulerRef],
  );

  /** Re-align the take against the grid. The audio was recorded with a pad
   * either side, so this slides the predicted parameters over what is already
   * there — no re-recording, and the loop you are listening to moves as you
   * scrub. */
  const setRecordOffsetMs = useCallback(
    (ms: number) => {
      const clamped = Math.max(
        -RECORD_OFFSET_LIMIT_MS,
        Math.min(RECORD_OFFSET_LIMIT_MS, ms),
      );
      offsetRef.current = clamped / 1000;
      schedulerRef.current?.setTakeOffsetSeconds(offsetRef.current);
      setRecordOffsetMsState(clamped);
    },
    [schedulerRef],
  );

  const toggleMicProcessing = useCallback(
    (key: keyof MicProcessing) => {
      setMicProcessingState((current) => {
        const next = { ...current, [key]: !current[key] };
        void recorderRef.current?.setProcessing(next).catch((e) => {
          setError(micErrorMessage(e));
        });
        return next;
      });
    },
    [recorderRef],
  );

  const selectMidiInput = useCallback(
    (id: string | null) => {
      midiRef.current?.selectInput(id);
      bump();
    },
    [bump, midiRef],
  );

  return {
    version,
    error,
    setError,
    running,
    hasLoop,
    take,
    takeSeconds,
    midi,
    midiSupport,
    clock,
    scheduler,
    levelStore: recorder?.levelStore ?? null,
    takes,
    micProcessing,
    recordOffsetMs,
    metronomeMode,
    metronomeAudible,
    muteWhileRecording,
    init,
    start,
    stop,
    record,
    cancelRecord,
    clear,
    resetPhase,
    setClockSource,
    setBpm,
    setShape,
    setOverride,
    setRootNote,
    setRecordOffsetMs,
    setMetronomeMode,
    setMuteWhileRecording,
    loadTake,
    removeTake,
    toggleMicProcessing,
    selectMidiInput,
  };
}
