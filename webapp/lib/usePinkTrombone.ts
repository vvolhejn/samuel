import { useCallback, useMemo, useRef } from "react";
import type {
  PinkTromboneConstriction,
  PinkTromboneElement,
} from "@/types/pink-trombone";
import type { SynthResponse } from "@/lib/audio";
import {
  startVideoRecording,
  type VideoRecorder,
  type VideoRecording,
} from "@/lib/videoRecorder";

/** Headroom between scheduling and curve start; also the voicing fade-in. */
const LEAD_S = 0.15;
const FADE_OUT_S = 0.05;
/** Smoothing for direct (scrub/stop) parameter jumps. */
const SCRUB_TAU_S = 0.02;
/** Headroom when re-scheduling mid-utterance (setPlaybackSpeed). Short enough
 * to feel instant, long enough that the audio thread hasn't rendered past it. */
const RESCHEDULE_LEAD_S = 0.05;

export interface SpeakOptions {
  /** Playback speed (1 = real time, 0.5 = half speed; pitch is unaffected —
   * the parameter trajectories just move slower). */
  speed?: number;
  /** Position to start from, as a fraction of the utterance in [0, 1). */
  startFrac?: number;
  /** Called with the current position fraction while playing. */
  onProgress?: (frac: number) => void;
}

export interface PinkTromboneHandle {
  /** Load the synth + visualization. Safe to call at page load: the
   * AudioContext starts suspended, but the tract UI still renders. */
  init: (element: PinkTromboneElement) => Promise<void>;
  /** Resume the AudioContext; must be called from a user gesture. */
  resume: () => Promise<void>;
  /** Schedule a model response onto the synth. Resolves when playback ends
   * or when stop()/scrub() interrupts it. */
  speak: (response: SynthResponse, options?: SpeakOptions) => Promise<void>;
  /** Change the speed of the utterance in progress, taking effect right away.
   * No-op when nothing is playing (the next speak() takes its own speed). */
  setPlaybackSpeed: (speed: number) => void;
  /** Truncate the current utterance's automation and fade the voicing. */
  stop: () => void;
  /** Hold the synth at one frame of a response: params jump there and the
   * voicing sustains, so the tract pose can be inspected audibly. Interrupts
   * any playback in progress. */
  scrub: (response: SynthResponse, frac: number) => void;
  /** Fade the voicing after scrubbing (the tract pose stays in place). */
  endScrub: () => void;
  /** Start capturing the tract visualization and the synth's output. No-op if
   * the browser can't record, or if a capture is already running. */
  startRecording: () => void;
  /** Finish the capture started by startRecording(); null if there wasn't one
   * or the browser produced nothing. */
  stopRecording: () => Promise<VideoRecording | null>;
  ready: () => boolean;
}

/** Truncate a param's automation at ``t``, holding its current value.
 * cancelScheduledValues cannot cut short an in-progress setValueCurveAtTime
 * (see scheduleCurve); cancelAndHoldAtTime can, but is Chrome/Safari-only —
 * the fallback leaves a running curve playing to its end. */
function cancelAndHold(param: AudioParam, t: number) {
  const p = param as AudioParam & {
    cancelAndHoldAtTime?: (when: number) => AudioParam;
  };
  if (p.cancelAndHoldAtTime) p.cancelAndHoldAtTime(t);
  else param.cancelScheduledValues(t);
}

function canCancelAndHold(param: AudioParam) {
  return "cancelAndHoldAtTime" in param;
}

/** The AudioParams speak() automates, in curveValues() order. */
function automatedParams(
  element: PinkTromboneElement,
  constriction: PinkTromboneConstriction,
  lipConstriction: PinkTromboneConstriction,
): AudioParam[] {
  return [
    element.frequency,
    element.tenseness,
    element.loudness,
    element.tongue.index,
    element.tongue.diameter,
    constriction.index,
    constriction.diameter,
    lipConstriction.diameter,
    element.intensity,
  ];
}

/** Per-frame values for each automated param, mirroring the Python synth
 * (pink_trombone.py): tenseness = voiceness, loudness = voiceness ** 0.25. */
function curveValues(response: SynthResponse): Float32Array[] {
  const { params } = response;
  const voiceness = Float32Array.from(params.voiceness);
  return [
    Float32Array.from(params.frequency),
    voiceness,
    voiceness.map((v) => Math.pow(v, 0.25)),
    Float32Array.from(params.tongueIndex),
    Float32Array.from(params.tongueDiameter),
    Float32Array.from(params.constrictionIndex),
    Float32Array.from(params.constrictionDiameter),
    // Older checkpoints predate the lip constriction; leave it open then.
    params.lipDiameter
      ? Float32Array.from(params.lipDiameter)
      : new Float32Array(params.voiceness.length).fill(3),
    Float32Array.from(params.intensity),
  ];
}

/** Schedule one param's trajectory. The caller must first clear any events in
 * [t0, t0 + duration] — setValueCurveAtTime throws on an overlap. */
function scheduleCurve(
  param: AudioParam,
  values: Float32Array,
  t0: number,
  duration: number,
) {
  param.setValueAtTime(values[0], t0 - 0.01);
  param.setValueCurveAtTime(values, t0, duration);
}

/** The utterance currently scheduled, mutated in place by setPlaybackSpeed so
 * that speak()'s progress polling follows a mid-flight re-schedule. */
interface Playback {
  values: Float32Array[];
  nFrames: number;
  frameRate: number;
  /** Frame the curves start from, and the audio-clock time they start at. */
  startFrame: number;
  t0: number;
  duration: number;
  speed: number;
  /** Audio-clock time the fade-out completes. */
  end: number;
  /** Wall-clock cap, so a suspended context can't hang the poll forever. */
  wallDeadline: number;
}

export function usePinkTrombone(): PinkTromboneHandle {
  const elementRef = useRef<PinkTromboneElement | null>(null);
  const constrictionRef = useRef<PinkTromboneConstriction | null>(null);
  const lipConstrictionRef = useRef<PinkTromboneConstriction | null>(null);
  const masterGainRef = useRef<GainNode | null>(null);
  /** Audio-clock time when the last scheduled utterance ends. */
  const scheduleEndRef = useRef(0);
  /** Bumped by stop()/scrub()/speak() to settle any in-flight speak(). */
  const playTokenRef = useRef(0);
  /** Non-null only while an utterance is scheduled and unfinished. */
  const playbackRef = useRef<Playback | null>(null);
  /** Recording taps, set up in init(): what the synth outputs, and the stacked
   * canvases the tract is drawn on (bottom first). */
  const recordDestRef = useRef<MediaStreamAudioDestinationNode | null>(null);
  const canvasesRef = useRef<HTMLCanvasElement[]>([]);
  const recorderRef = useRef<VideoRecorder | null>(null);

  const init = useCallback(async (element: PinkTromboneElement) => {
    if (elementRef.current) return;
    // Load the vendored bundle outside the bundler's module graph; for module
    // scripts "load" fires after evaluation, i.e. after customElements.define.
    if (!document.querySelector("script[data-pink-trombone]")) {
      await new Promise<void>((resolve, reject) => {
        const script = document.createElement("script");
        script.type = "module";
        script.src = "/pink-trombone/pink-trombone.min.js";
        script.dataset.pinkTrombone = "true";
        script.onload = () => resolve();
        script.onerror = () =>
          reject(new Error("failed to load pink-trombone.min.js"));
        document.head.appendChild(script);
      });
    }
    await customElements.whenDefined("pink-trombone");
    const ctx = new AudioContext();
    await element.setAudioContext(ctx);

    // Match the model's frozen values (JS defaults differ for vibrato) and
    // keep the glottis silent until an utterance is scheduled.
    element.vibrato.gain.value = 0;
    element.vibrato.wobble.value = 0;
    element.intensity.value = 0;

    // Two persistent constriction slots driven by the model trajectories:
    // the tongue-tip constriction (movable index) and the lip constriction
    // (fixed at the last tract index, mirrors _LIP_INDEX in pink_trombone.py).
    // Start fully open (diameter 3 ≈ no constriction).
    const constriction = element.newConstriction(33, 3);
    const lipConstriction = element.newConstriction(43, 3);
    // newConstriction() hands out the first slot it considers free, so a
    // regression there would silently give both trajectories the same
    // AudioParams (the second setValueCurveAtTime then overwrites the first).
    if (!constriction || !lipConstriction)
      throw new Error("Pink Trombone ran out of constriction slots");
    if (constriction.diameter === lipConstriction.diameter)
      throw new Error(
        "Pink Trombone returned the same constriction slot twice — the tongue-tip and lip trajectories would collide",
      );
    constrictionRef.current = constriction;
    lipConstrictionRef.current = lipConstriction;

    // Master gain is the click-free utterance gate (see speak()). The energy
    // envelope itself comes from the model's per-frame intensity trajectory.
    const masterGain = ctx.createGain();
    masterGain.gain.value = 0;
    element.connect(masterGain);
    masterGain.connect(ctx.destination);
    masterGainRef.current = masterGain;
    // Second tap on the same signal, for the video's soundtrack. Always
    // connected — a destination node nobody records from costs nothing.
    recordDestRef.current = ctx.createMediaStreamDestination();
    masterGain.connect(recordDestRef.current);

    element.start();
    element.enableUI();
    element.startUI();
    // enableUI() has now appended TractUI's canvases. They are transparent and
    // stacked by z-index (background 0, tract 1), which is the order the video
    // recorder has to composite them in — DOM order is the other way round.
    canvasesRef.current = Array.from(
      element.querySelectorAll("canvas"),
    ).sort((a, b) => Number(a.style.zIndex || 0) - Number(b.style.zIndex || 0));
    elementRef.current = element;
  }, []);

  const resume = useCallback(async () => {
    const element = elementRef.current;
    if (!element) throw new Error("Pink Trombone not ready");
    await element.audioContext.resume();
  }, []);

  const speak = useCallback(
    async (response: SynthResponse, options: SpeakOptions = {}) => {
      const { speed = 1, startFrac = 0, onProgress } = options;
      const element = elementRef.current;
      const constriction = constrictionRef.current;
      const lipConstriction = lipConstrictionRef.current;
      const masterGain = masterGainRef.current;
      if (!element || !constriction || !lipConstriction || !masterGain)
        throw new Error("Pink Trombone not ready");

      const ctx = element.audioContext;
      await ctx.resume();
      const token = ++playTokenRef.current;

      const { frame_rate: frameRate } = response;
      const values = curveValues(response);
      const nFrames = response.params.voiceness.length;
      // setValueCurveAtTime needs >= 2 values, so start at least 2 frames
      // before the end.
      const startFrame = Math.min(
        Math.max(0, Math.floor(startFrac * nFrames)),
        nFrames - 2,
      );
      const duration = (nFrames - startFrame) / frameRate / speed;
      const now = ctx.currentTime;
      // Never overlap the previous utterance's curves: the UI's notion of
      // "playback finished" runs on the wall clock, which can outpace a
      // throttled audio clock — and an in-progress curve makes any scheduling
      // inside its interval throw NotSupportedError.
      const t0 = Math.max(now + LEAD_S, scheduleEndRef.current + 0.02);

      const params = automatedParams(element, constriction, lipConstriction);
      params.forEach((param, i) => {
        // Clear leftovers from a previous utterance. cancelScheduledValues does
        // NOT remove a curve that is already in progress (start < now), which is
        // why t0 must lie beyond the previous utterance's end (above).
        param.cancelScheduledValues(now);
        scheduleCurve(param, values[i].subarray(startFrame), t0, duration);
      });

      // Utterance gate: element.start()/stop() switch gain instantly (clicks),
      // so fade the master gain instead. This used to ride on element.intensity,
      // which the model now drives per-frame as its energy envelope.
      const gate = masterGain.gain;
      gate.cancelScheduledValues(now);
      gate.setValueAtTime(gate.value, now);
      gate.linearRampToValueAtTime(1, t0);
      gate.setValueAtTime(1, t0 + duration);
      gate.linearRampToValueAtTime(0, t0 + duration + FADE_OUT_S);

      const end = t0 + duration + FADE_OUT_S;
      scheduleEndRef.current = end;

      // Wait for the *audio clock* to pass the end of the utterance — a plain
      // wall-clock timeout can fire while audio is still playing.
      const playback: Playback = {
        values,
        nFrames,
        frameRate,
        startFrame,
        t0,
        duration,
        speed,
        end,
        wallDeadline: performance.now() + (end - now + 5) * 1000,
      };
      playbackRef.current = playback;
      await new Promise<void>((resolve) => {
        const poll = () => {
          if (playTokenRef.current !== token) {
            resolve(); // interrupted by stop()/scrub()/another speak()
            return;
          }
          // Read through the object: setPlaybackSpeed rewrites these mid-flight.
          const t = ctx.currentTime;
          onProgress?.(
            Math.min(
              1,
              (playback.startFrame +
                Math.max(0, t - playback.t0) * playback.speed * frameRate) /
                nFrames,
            ),
          );
          if (
            t >= playback.end + 0.05 ||
            performance.now() > playback.wallDeadline
          ) {
            onProgress?.(1);
            if (playbackRef.current === playback) playbackRef.current = null;
            resolve();
          } else {
            setTimeout(poll, 50);
          }
        };
        poll();
      });
    },
    [],
  );

  const setPlaybackSpeed = useCallback((speed: number) => {
    const playback = playbackRef.current;
    const element = elementRef.current;
    const constriction = constrictionRef.current;
    const lipConstriction = lipConstrictionRef.current;
    const masterGain = masterGainRef.current;
    if (!playback || !element || !constriction || !lipConstriction || !masterGain)
      return;
    if (speed === playback.speed) return;
    // Without cancelAndHoldAtTime the running curve cannot be truncated, and
    // setValueCurveAtTime throws when it overlaps one. Let the new speed take
    // effect at the next play there instead.
    if (!canCancelAndHold(masterGain.gain)) return;
    const { values, nFrames, frameRate } = playback;
    const ctx = element.audioContext;
    const now = ctx.currentTime;

    // Where the old curves have got to; still in the lead-in => the start frame.
    const played = Math.max(0, now - playback.t0) * playback.speed * frameRate;
    const frame = Math.floor(playback.startFrame + played);
    // setValueCurveAtTime needs >= 2 values; that close to the end, let the
    // utterance run out at its current speed rather than re-scheduling.
    if (frame > nFrames - 3) return;

    // Params hold for RESCHEDULE_LEAD_S and then resume from `frame`, so the
    // seam is a freeze of a few tens of ms rather than a jump.
    const t0 = now + RESCHEDULE_LEAD_S;
    const duration = (nFrames - frame) / frameRate / speed;
    const params = automatedParams(element, constriction, lipConstriction);
    params.forEach((param, i) => {
      // Truncate the running curve; cancelScheduledValues cannot (see
      // cancelAndHold), so on Firefox the old curve plays on underneath.
      cancelAndHold(param, now);
      scheduleCurve(param, values[i].subarray(frame), t0, duration);
    });

    const end = t0 + duration + FADE_OUT_S;
    const gate = masterGain.gain;
    cancelAndHold(gate, now);
    // Already open mid-utterance; a ramp only matters if we cut into the fade-in.
    gate.linearRampToValueAtTime(1, t0);
    gate.setValueAtTime(1, t0 + duration);
    gate.linearRampToValueAtTime(0, end);

    playback.startFrame = frame;
    playback.t0 = t0;
    playback.duration = duration;
    playback.speed = speed;
    playback.end = end;
    playback.wallDeadline = performance.now() + (end - now + 5) * 1000;
    scheduleEndRef.current = end;
  }, []);

  const stop = useCallback(() => {
    playTokenRef.current++;
    playbackRef.current = null;
    const element = elementRef.current;
    const constriction = constrictionRef.current;
    const lipConstriction = lipConstrictionRef.current;
    const masterGain = masterGainRef.current;
    if (!element || !constriction || !lipConstriction || !masterGain) return;
    const now = element.audioContext.currentTime;
    for (const param of automatedParams(element, constriction, lipConstriction)) {
      cancelAndHold(param, now);
    }
    cancelAndHold(masterGain.gain, now);
    masterGain.gain.setTargetAtTime(0, now, SCRUB_TAU_S);
    scheduleEndRef.current = now;
  }, []);

  const scrub = useCallback((response: SynthResponse, frac: number) => {
    playTokenRef.current++;
    playbackRef.current = null;
    const element = elementRef.current;
    const constriction = constrictionRef.current;
    const lipConstriction = lipConstrictionRef.current;
    const masterGain = masterGainRef.current;
    if (!element || !constriction || !lipConstriction || !masterGain) return;
    const ctx = element.audioContext;
    void ctx.resume(); // called from a user gesture

    const values = curveValues(response);
    const nFrames = response.params.voiceness.length;
    const frame = Math.min(
      nFrames - 1,
      Math.max(0, Math.round(frac * (nFrames - 1))),
    );
    const now = ctx.currentTime;
    const params = automatedParams(element, constriction, lipConstriction);
    params.forEach((param, i) => {
      cancelAndHold(param, now);
      param.setTargetAtTime(values[i][frame], now, SCRUB_TAU_S);
    });
    // Open the gate; the scrubbed frame's own intensity is set by the loop
    // above, so a silent frame stays silent.
    cancelAndHold(masterGain.gain, now);
    masterGain.gain.setTargetAtTime(1, now, 0.03);
    scheduleEndRef.current = now;
  }, []);

  const endScrub = useCallback(() => {
    const masterGain = masterGainRef.current;
    if (!masterGain) return;
    const now = masterGain.context.currentTime;
    cancelAndHold(masterGain.gain, now);
    masterGain.gain.setTargetAtTime(0, now, FADE_OUT_S);
  }, []);

  const startRecording = useCallback(() => {
    if (recorderRef.current) return;
    recorderRef.current = startVideoRecording(
      canvasesRef.current,
      recordDestRef.current?.stream ?? null,
    );
  }, []);

  const stopRecording = useCallback(async () => {
    const recorder = recorderRef.current;
    recorderRef.current = null;
    return recorder ? recorder.stop() : null;
  }, []);

  const ready = useCallback(() => elementRef.current !== null, []);

  // Stable handle: effects depending on it must not re-run on re-render.
  return useMemo(
    () => ({
      init,
      resume,
      speak,
      setPlaybackSpeed,
      stop,
      scrub,
      endScrub,
      startRecording,
      stopRecording,
      ready,
    }),
    [
      init,
      resume,
      speak,
      setPlaybackSpeed,
      stop,
      scrub,
      endScrub,
      startRecording,
      stopRecording,
      ready,
    ],
  );
}
