/** Hand-drawn parameter curves for the curve editor (app/curves): breakpoints
 * per parameter, interpolated into the per-frame trajectories a SynthResponse
 * carries, so playback can reuse usePinkTrombone as-is. */

import type { SynthResponse } from "@/lib/audio";

export interface Breakpoint {
  id: number;
  /** Seconds. */
  t: number;
  v: number;
}

export type Interpolation = "linear" | "smooth";

export interface CurveParam {
  key: string;
  label: string;
  lo: number;
  hi: number;
  init: number;
}

/** The model's trainable parameters with its ranges (_DEFAULT_PARAM_SPEC in
 * samuel/model.py), plus frequency, which the model takes from pyin instead.
 * Defaults are a neutral open vowel: both constrictions out of the way. */
export const CURVE_PARAMS: CurveParam[] = [
  { key: "frequency", label: "frequency", lo: 50, hi: 400, init: 140 },
  { key: "voiceness", label: "voiceness", lo: 0, hi: 1, init: 0.6 },
  { key: "intensity", label: "intensity", lo: 0, hi: 1, init: 1 },
  { key: "tongueIndex", label: "tongue index", lo: 10, hi: 35, init: 20 },
  { key: "tongueDiameter", label: "tongue diameter", lo: 1.5, hi: 3.5, init: 2.4 },
  { key: "constrictionIndex", label: "constriction index", lo: 22, hi: 39, init: 33 },
  { key: "constrictionDiameter", label: "constriction diameter", lo: -2, hi: 3, init: 3 },
  { key: "lipDiameter", label: "lip diameter", lo: 0, hi: 3, init: 3 },
];

/** Bounds and default for the editable length, in seconds. */
export const MIN_DURATION_S = 1;
export const MAX_DURATION_S = 5;
export const DEFAULT_DURATION_S = 1;
const FRAME_RATE = 100;

/** The curve's value at `t`. Held flat before the first breakpoint and after
 * the last. "smooth" is monotone cubic (Fritsch–Carlson): it never overshoots
 * its neighbours, so a curve can't leave the range its points are in. */
export function evaluate(
  points: Breakpoint[],
  t: number,
  mode: Interpolation,
): number {
  const n = points.length;
  if (t <= points[0].t) return points[0].v;
  if (t >= points[n - 1].t) return points[n - 1].v;
  let i = 0;
  while (points[i + 1].t < t) i++;
  const a = points[i];
  const b = points[i + 1];
  const h = b.t - a.t;
  if (h <= 0) return b.v;
  const s = (t - a.t) / h;
  if (mode === "linear") return a.v + (b.v - a.v) * s;
  const m0 = tangent(points, i) * h;
  const m1 = tangent(points, i + 1) * h;
  const s2 = s * s;
  const s3 = s2 * s;
  return (
    (2 * s3 - 3 * s2 + 1) * a.v +
    (s3 - 2 * s2 + s) * m0 +
    (-2 * s3 + 3 * s2) * b.v +
    (s3 - s2) * m1
  );
}

/** Fritsch–Carlson tangent at point `i`: zero at a local extremum (and at the
 * ends), otherwise the harmonic mean of the neighbouring slopes. */
function tangent(points: Breakpoint[], i: number): number {
  if (i === 0 || i === points.length - 1) return 0;
  const d0 = slope(points[i - 1], points[i]);
  const d1 = slope(points[i], points[i + 1]);
  if (d0 * d1 <= 0) return 0;
  return (2 * d0 * d1) / (d0 + d1);
}

function slope(a: Breakpoint, b: Breakpoint): number {
  const h = b.t - a.t;
  return h > 0 ? (b.v - a.v) / h : 0;
}

/** Sample every curve into the response shape speak()/scrub() take. */
export function toResponse(
  curves: Record<string, Breakpoint[]>,
  mode: Interpolation,
  duration: number,
): SynthResponse {
  const nFrames = Math.round(duration * FRAME_RATE);
  const params: Record<string, number[]> = {};
  for (const { key } of CURVE_PARAMS) {
    params[key] = Array.from({ length: nFrames }, (_, f) =>
      evaluate(curves[key], f / FRAME_RATE, mode),
    );
  }
  return {
    frame_rate: FRAME_RATE,
    n_frames: nFrames,
    duration_s: duration,
    params,
    voiced: new Array(nFrames).fill(true),
  };
}
