import { useCallback, useMemo, useRef } from "react";
import type {
  PinkTromboneConstriction,
  PinkTromboneElement,
} from "@/types/pink-trombone";
import type { SynthResponse } from "@/lib/audio";

/** Headroom between scheduling and curve start; also the voicing fade-in. */
const LEAD_S = 0.15;
const FADE_OUT_S = 0.05;

export interface PinkTromboneHandle {
  /** Load the synth + visualization. Safe to call at page load: the
   * AudioContext starts suspended, but the tract UI still renders. */
  init: (element: PinkTromboneElement) => Promise<void>;
  /** Resume the AudioContext; must be called from a user gesture. */
  resume: () => Promise<void>;
  /** Schedule a model response onto the synth. Resolves when playback ends. */
  speak: (response: SynthResponse) => Promise<void>;
  ready: () => boolean;
}

function scheduleCurve(
  param: AudioParam,
  values: Float32Array,
  now: number,
  t0: number,
  duration: number,
) {
  // setValueCurveAtTime throws if any event overlaps the curve interval, so
  // clear leftovers from a previous utterance first. cancelScheduledValues
  // does NOT remove a curve that is already in progress (start < now), which
  // is why t0 must lie beyond the previous utterance's end (see speak()).
  param.cancelScheduledValues(now);
  param.setValueAtTime(values[0], t0 - 0.01);
  param.setValueCurveAtTime(values, t0, duration);
}

export function usePinkTrombone(): PinkTromboneHandle {
  const elementRef = useRef<PinkTromboneElement | null>(null);
  const constrictionRef = useRef<PinkTromboneConstriction | null>(null);
  const masterGainRef = useRef<GainNode | null>(null);
  /** Audio-clock time when the last scheduled utterance ends. */
  const scheduleEndRef = useRef(0);

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

    // Single persistent constriction slot driven by the model trajectories.
    // Start fully open (diameter 3 ≈ no constriction).
    constrictionRef.current = element.newConstriction(33, 3);

    // Master gain for the per-frame volume-match envelope (training only
    // ever evaluated volume-matched audio, so the raw synth has no
    // meaningful envelope of its own).
    const masterGain = ctx.createGain();
    masterGain.gain.value = 1;
    element.connect(masterGain);
    masterGain.connect(ctx.destination);
    masterGainRef.current = masterGain;

    element.start();
    element.enableUI();
    element.startUI();
    elementRef.current = element;
  }, []);

  const resume = useCallback(async () => {
    const element = elementRef.current;
    if (!element) throw new Error("Pink Trombone not ready");
    await element.audioContext.resume();
  }, []);

  const speak = useCallback(async (response: SynthResponse) => {
    const element = elementRef.current;
    const constriction = constrictionRef.current;
    const masterGain = masterGainRef.current;
    if (!element || !constriction || !masterGain)
      throw new Error("Pink Trombone not ready");

    const ctx = element.audioContext;
    await ctx.resume();

    const { frame_rate: frameRate, params } = response;
    const nFrames = params.voiceness.length;
    const duration = nFrames / frameRate;
    const now = ctx.currentTime;
    // Never overlap the previous utterance's curves: the UI's notion of
    // "playback finished" runs on the wall clock, which can outpace a
    // throttled audio clock — and an in-progress curve makes any scheduling
    // inside its interval throw NotSupportedError.
    const t0 = Math.max(now + LEAD_S, scheduleEndRef.current + 0.02);

    const voiceness = Float32Array.from(params.voiceness);
    // Mirror of the Python synth (pink_trombone.py): tenseness = voiceness,
    // loudness = voiceness ** 0.25.
    const curves: Array<[AudioParam, Float32Array]> = [
      [element.frequency, Float32Array.from(params.frequency)],
      [element.tenseness, voiceness],
      [element.loudness, voiceness.map((v) => Math.pow(v, 0.25))],
      [element.tongue.index, Float32Array.from(params.tongueIndex)],
      [element.tongue.diameter, Float32Array.from(params.tongueDiameter)],
      [constriction.index, Float32Array.from(params.constrictionIndex)],
      [constriction.diameter, Float32Array.from(params.constrictionDiameter)],
      [masterGain.gain, Float32Array.from(response.gain)],
    ];
    for (const [param, values] of curves) {
      scheduleCurve(param, values, now, t0, duration);
    }

    // Voicing gate: element.start()/stop() switch gain instantly (clicks), so
    // fade with the intensity param instead.
    const intensity = element.intensity;
    intensity.cancelScheduledValues(now);
    intensity.setValueAtTime(intensity.value, now);
    intensity.linearRampToValueAtTime(1, t0);
    intensity.setValueAtTime(1, t0 + duration);
    intensity.linearRampToValueAtTime(0, t0 + duration + FADE_OUT_S);

    const end = t0 + duration + FADE_OUT_S;
    scheduleEndRef.current = end;

    // Wait for the *audio clock* to pass the end of the utterance — a plain
    // wall-clock timeout can fire while audio is still playing. Wall-clock
    // cap so a suspended context can't hang us forever.
    const wallDeadline = performance.now() + (end - now + 5) * 1000;
    await new Promise<void>((resolve) => {
      const poll = () => {
        if (ctx.currentTime >= end + 0.05 || performance.now() > wallDeadline) {
          resolve();
        } else {
          setTimeout(poll, 100);
        }
      };
      poll();
    });
  }, []);

  const ready = useCallback(() => elementRef.current !== null, []);

  // Stable handle: effects depending on it must not re-run on re-render.
  return useMemo(
    () => ({ init, resume, speak, ready }),
    [init, resume, speak, ready],
  );
}
