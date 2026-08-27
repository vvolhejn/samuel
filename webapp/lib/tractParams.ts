/** The tract's parameters as the looper thinks of them, and how they reach the
 * synth. One "channel" per thing the model predicts, which is not one-to-one
 * with the element's AudioParams: `voiceness` fans out to both `tenseness` and
 * `loudness` (mirroring the Python synth, see usePinkTrombone.curveValues), and
 * the frozen params are nobody's business here. */

import type {
  PinkTromboneConstriction,
  PinkTromboneElement,
} from "@/types/pink-trombone";

/** Every channel a trajectory carries and MIDI may override, in the order the
 * UI lists them: the two that matter most first. */
export const CHANNELS = [
  "frequency",
  "intensity",
  "voiceness",
  "tongueIndex",
  "tongueDiameter",
  "constrictionIndex",
  "constrictionDiameter",
  "lipDiameter",
] as const;

export type Channel = (typeof CHANNELS)[number];

export type ChannelValues = Record<Channel, number>;

/** How a manual value combines with the trajectory's in `offset` mode. Pitch
 * combines as a ratio so a transposition is the same interval wherever the
 * contour happens to be; `intensity` scales, so overriding it re-shapes the
 * recorded envelope rather than flattening it; the rest add. */
export type Combine = "semitones" | "scale" | "add";

export const CHANNEL_COMBINE: Record<Channel, Combine> = {
  frequency: "semitones",
  intensity: "scale",
  voiceness: "add",
  tongueIndex: "add",
  tongueDiameter: "add",
  constrictionIndex: "add",
  constrictionDiameter: "add",
  lipDiameter: "add",
};

export const CHANNEL_LABELS: Record<Channel, string> = {
  frequency: "pitch",
  intensity: "loudness",
  voiceness: "voicing",
  tongueIndex: "tongue index",
  tongueDiameter: "tongue diameter",
  constrictionIndex: "constriction index",
  constrictionDiameter: "constriction diameter",
  lipDiameter: "lip diameter",
};

/** `[lo, hi]` per channel. The backend reports the loaded checkpoint's own
 * (GET /api/health), which is what a client should use; this is the fallback
 * for when it can't be reached, and matches model._DEFAULT_PARAM_SPEC. */
export const DEFAULT_RANGES: Record<Channel, [number, number]> = {
  frequency: [70, 500],
  intensity: [0, 1],
  voiceness: [0, 1],
  tongueIndex: [10, 35],
  tongueDiameter: [1.5, 3.5],
  constrictionIndex: [22, 39],
  constrictionDiameter: [-2, 3],
  lipDiameter: [0, 3],
};

/** Where the tract sits with nothing driving it: silent, and open enough that
 * the first frame of a trajectory doesn't have to unpinch it. Used before the
 * first loop is recorded and after one is cleared. */
export const REST_VALUES: ChannelValues = {
  frequency: 140,
  intensity: 0,
  voiceness: 0,
  tongueIndex: 20,
  tongueDiameter: 2.4,
  constrictionIndex: 33,
  constrictionDiameter: 3,
  lipDiameter: 3,
};

/** The AudioParams a trajectory drives, plus the gate that fades the whole
 * thing in and out. Handed out by usePinkTrombone once the element is up. */
export interface TractTargets {
  ctx: AudioContext;
  /** Master gain: the click-free on/off, not an energy envelope (that is
   * `intensity`, which the model predicts per frame). */
  gate: AudioParam;
  frequency: AudioParam;
  tenseness: AudioParam;
  loudness: AudioParam;
  intensity: AudioParam;
  tongueIndex: AudioParam;
  tongueDiameter: AudioParam;
  constrictionIndex: AudioParam;
  constrictionDiameter: AudioParam;
  lipDiameter: AudioParam;
}

export function tractTargets(
  element: PinkTromboneElement,
  constriction: PinkTromboneConstriction,
  lipConstriction: PinkTromboneConstriction,
  gate: AudioParam,
): TractTargets {
  return {
    ctx: element.audioContext,
    gate,
    frequency: element.frequency,
    tenseness: element.tenseness,
    loudness: element.loudness,
    intensity: element.intensity,
    tongueIndex: element.tongue.index,
    tongueDiameter: element.tongue.diameter,
    constrictionIndex: constriction.index,
    constrictionDiameter: constriction.diameter,
    lipDiameter: lipConstriction.diameter,
  };
}

/** Every AudioParam in `targets`, for the callers that clear or freeze the lot. */
export function allParams(targets: TractTargets): AudioParam[] {
  return [
    targets.frequency,
    targets.tenseness,
    targets.loudness,
    targets.intensity,
    targets.tongueIndex,
    targets.tongueDiameter,
    targets.constrictionIndex,
    targets.constrictionDiameter,
    targets.lipDiameter,
  ];
}

/** Truncate a param's automation at `t`, holding its current value.
 * cancelScheduledValues cannot cut short a ramp already in progress — it
 * removes the ramp's end event and the value snaps back to where the ramp
 * began. cancelAndHoldAtTime does the right thing but is Chrome/Safari-only;
 * on Firefox the running ramp plays out underneath. */
export function cancelAndHold(param: AudioParam, t: number) {
  const p = param as AudioParam & {
    cancelAndHoldAtTime?: (when: number) => AudioParam;
  };
  if (p.cancelAndHoldAtTime) p.cancelAndHoldAtTime(t);
  else param.cancelScheduledValues(t);
}

/** Pin a param at the value it currently holds, so a later
 * linearRampToValueAtTime has a defined place to ramp *from*. Without this the
 * ramp interpolates from whatever the last scheduled event left behind, which
 * after a setTargetAtTime is not where the param actually is. */
export function pin(param: AudioParam, t: number) {
  cancelAndHold(param, t);
  param.setValueAtTime(param.value, t);
}

/** Write one frame of channel values, each as a linear ramp landing at `t`.
 * Ramping rather than stepping is what the Python synth does between control
 * frames (pink_trombone._upsample_params), so this is the same signal the
 * model was trained through. */
export function writeFrame(
  targets: TractTargets,
  values: ChannelValues,
  t: number,
  skip?: ReadonlySet<Channel>,
) {
  for (const channel of CHANNELS) {
    if (skip?.has(channel)) continue;
    writeChannel(targets, channel, values[channel], t);
  }
}

export function writeChannel(
  targets: TractTargets,
  channel: Channel,
  value: number,
  t: number,
) {
  if (channel === "voiceness") {
    targets.tenseness.linearRampToValueAtTime(value, t);
    targets.loudness.linearRampToValueAtTime(voicenessLoudness(value), t);
    return;
  }
  targets[channel].linearRampToValueAtTime(value, t);
}

/** The element's `loudness`, which the Python synth derives from voiceness
 * rather than taking as its own parameter. */
export function voicenessLoudness(voiceness: number): number {
  return Math.pow(Math.max(voiceness, 1e-6), 0.25);
}

export function clamp(value: number, [lo, hi]: [number, number]): number {
  return Math.min(hi, Math.max(lo, value));
}
