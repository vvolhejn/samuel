/** The precomputed loss landscapes for app/loss-landscape: a loss against a
 * target sound over each pair of tongueIndex, tongueDiameter and lipDiameter,
 * the third held at a reference pose, with -- in some -- the loss's gradient
 * from autograd through the synth (scripts/export_loss_landscape.py). */

import type { SynthResponse } from "@/lib/audio";
import { CURVE_PARAMS } from "@/lib/curves";

export interface LandscapeAxis {
  key: string;
  lo: number;
  hi: number;
  /** Grid points, evenly spaced from lo to hi inclusive. */
  n: number;
}

export interface Landscape {
  /** Every synth param of the reference pose: the target sound's own for the
   * synth target, the published model's fit to it for a recording. */
  reference: Record<string, number>;
  target: TargetKind;
  /** The swept params. */
  params: LandscapeAxis[];
  /** One per pair, x < y indexing params. */
  slices: LandscapeSlice[];
}

/** Grids laid out as rows of y (from lo up), each over x. */
export interface LandscapeSlice {
  x: number;
  y: number;
  loss: number[];
  /** d loss / d param, per unit of the x and y params; when absent, descent
   * takes finite differences of the loss. */
  grad_x?: number[];
  grad_y?: number[];
}

/** The losses: middle-frame MFCC, log-mel spectrogram, multi-scale STFT, and
 * the published model's self-supervised loss (layer-6 wav2vec2 features, L1). */
export type LossKind = "mfcc" | "mel" | "stft" | "ssl";

/** The target sound: the synth at one reference config, or a recorded vowel. */
export type TargetKind = "synth" | "ah" | "e-1" | "e-2";

export async function loadLandscape(
  target: TargetKind,
  loss: LossKind,
): Promise<Landscape> {
  const file = `loss-landscape-${target}-${loss}.json`;
  const res = await fetch(`/${file}`);
  if (!res.ok) throw new Error(`${file}: HTTP ${res.status}`);
  return res.json();
}

/** The lowest and highest loss over all three maps. */
export function lossRange(landscape: Landscape): [number, number] {
  let lo = Infinity;
  let hi = -Infinity;
  for (const { loss } of landscape.slices) {
    for (const v of loss) {
      lo = Math.min(lo, v);
      hi = Math.max(hi, v);
    }
  }
  return [lo, hi];
}

/** Fractional grid index of a value along an axis, clamped to the grid. */
export function gridIndex(axis: LandscapeAxis, v: number): number {
  const f = ((v - axis.lo) / (axis.hi - axis.lo)) * (axis.n - 1);
  return Math.min(axis.n - 1, Math.max(0, f));
}

/** The slice over params `x` < `y`. */
export function pairSlice(
  landscape: Landscape,
  x: number,
  y: number,
): LandscapeSlice {
  const found = landscape.slices.find((s) => s.x === x && s.y === y);
  if (!found) throw new Error(`no slice for params ${x}, ${y}`);
  return found;
}

/** A grid's value at fractional indices, interpolated bilinearly. */
function bilinear(
  data: ArrayLike<number>,
  nx: number,
  ny: number,
  fx: number,
  fy: number,
): number {
  const [ix, iy] = [Math.floor(fx), Math.floor(fy)];
  const [tx, ty] = [fx - ix, fy - iy];
  const at = (i: number, j: number) =>
    data[Math.min(j, ny - 1) * nx + Math.min(i, nx - 1)];
  return (
    (1 - ty) * ((1 - tx) * at(ix, iy) + tx * at(ix + 1, iy)) +
    ty * ((1 - tx) * at(ix, iy + 1) + tx * at(ix + 1, iy + 1))
  );
}

/** The loss in slice `x` × `y` (x < y) at the given values, interpolated
 * bilinearly between grid points. */
export function sampleAt(
  landscape: Landscape,
  x: number,
  y: number,
  values: Record<string, number>,
): number {
  const [ax, ay] = [landscape.params[x], landscape.params[y]];
  return bilinear(
    pairSlice(landscape, x, y).loss,
    ax.n,
    ay.n,
    gridIndex(ax, values[ax.key]),
    gridIndex(ay, values[ay.key]),
  );
}

/** A grid's partial derivatives per unit of each param, by central differences
 * (one-sided at the edges). */
function finiteDifferences(
  data: number[],
  ax: LandscapeAxis,
  ay: LandscapeAxis,
): { grad_x: number[]; grad_y: number[] } {
  const [dx, dy] = [(ax.hi - ax.lo) / (ax.n - 1), (ay.hi - ay.lo) / (ay.n - 1)];
  const grad_x = new Array<number>(data.length);
  const grad_y = new Array<number>(data.length);
  for (let j = 0; j < ay.n; j++) {
    for (let i = 0; i < ax.n; i++) {
      const [i0, i1] = [Math.max(0, i - 1), Math.min(ax.n - 1, i + 1)];
      const [j0, j1] = [Math.max(0, j - 1), Math.min(ay.n - 1, j + 1)];
      grad_x[j * ax.n + i] =
        (data[j * ax.n + i1] - data[j * ax.n + i0]) / ((i1 - i0) * dx);
      grad_y[j * ax.n + i] =
        (data[j1 * ax.n + i] - data[j0 * ax.n + i]) / ((j1 - j0) * dy);
    }
  }
  return { grad_x, grad_y };
}

const fdCache = new WeakMap<
  LandscapeSlice,
  ReturnType<typeof finiteDifferences>
>();

/** A slice's gradient grids: the stored ones, or finite differences. */
function gradients(landscape: Landscape, slice: LandscapeSlice) {
  if (slice.grad_x && slice.grad_y)
    return { grad_x: slice.grad_x, grad_y: slice.grad_y };
  let fd = fdCache.get(slice);
  if (!fd) {
    fd = finiteDifferences(
      slice.loss,
      landscape.params[slice.x],
      landscape.params[slice.y],
    );
    fdCache.set(slice, fd);
  }
  return fd;
}

/** Gradient descent settings. Steps are in fractions of each param's range,
 * and the loss is rescaled to [0, 1] over its maps -- as the colors show it --
 * so one rate suits every loss (MFCC's range is ~0.73, wav2vec2's ~0.2).
 * The rescaled gradient's median is ~1.4 on the MFCC maps. Plain descent zigzags across the narrow valleys at
 * any step size that still gets somewhere in a few dozen steps — the gradient there
 * points at the walls, not along the floor — and smoother interpolation
 * doesn't help. Momentum does: the back-and-forth cancels and the drift along
 * the floor adds up. Simulated over the three maps (20 steps at half this
 * rate), half of the late steps reverse direction without it and under a
 * tenth with it; at this rate it's ~30%, a little lively but quicker. */
const LEARNING_RATE = 0.0087;
const MOMENTUM = 0.8;
const MAX_STEP = 0.03;

/** Gradient descent with momentum on slice `x` × `y` (x < y) from the given values,
 * following the synth's true gradient where stored, else finite differences of
 * the loss grid, interpolated bilinearly between grid points. Steps are taken in fractions of each param's range, so the two axes
 * count equally. Returns the start and each step's values, for the two mapped
 * params only. */
export function descend(
  landscape: Landscape,
  x: number,
  y: number,
  start: Record<string, number>,
  steps: number,
): Record<string, number>[] {
  const [ax, ay] = [landscape.params[x], landscape.params[y]];
  const { grad_x, grad_y } = gradients(landscape, pairSlice(landscape, x, y));
  const [lo, hi] = lossRange(landscape);
  // Position in [0, 1] along each axis.
  let u = gridIndex(ax, start[ax.key]) / (ax.n - 1);
  let v = gridIndex(ay, start[ay.key]) / (ay.n - 1);
  const toValues = () => ({
    [ax.key]: ax.lo + u * (ax.hi - ax.lo),
    [ay.key]: ay.lo + v * (ay.hi - ay.lo),
  });
  const path = [toValues()];
  let [vu, vv] = [0, 0];
  for (let s = 0; s < steps; s++) {
    const [fx, fy] = [u * (ax.n - 1), v * (ay.n - 1)];
    // Per unit of u and v rather than of the params, of the rescaled loss.
    const gu =
      (bilinear(grad_x, ax.n, ay.n, fx, fy) * (ax.hi - ax.lo)) / (hi - lo);
    const gv =
      (bilinear(grad_y, ax.n, ay.n, fx, fy) * (ay.hi - ay.lo)) / (hi - lo);
    vu = MOMENTUM * vu - LEARNING_RATE * gu;
    vv = MOMENTUM * vv - LEARNING_RATE * gv;
    let [du, dv] = [vu, vv];
    const len = Math.hypot(du, dv);
    if (len > MAX_STEP)
      [du, dv] = [(du / len) * MAX_STEP, (dv / len) * MAX_STEP];
    u = Math.min(1, Math.max(0, u + du));
    v = Math.min(1, Math.max(0, v + dv));
    path.push(toValues());
  }
  return path;
}

/** A constant pose as the response shape scrub()/pose() take. */
export function poseResponse(values: Record<string, number>): SynthResponse {
  const nFrames = 2;
  const params = Object.fromEntries(
    CURVE_PARAMS.map(({ key, init }) => [
      key,
      new Array(nFrames).fill(values[key] ?? init),
    ]),
  );
  return {
    frame_rate: 100,
    n_frames: nFrames,
    duration_s: nFrames / 100,
    params,
    voiced: new Array(nFrames).fill(true),
  };
}

/** Loss colormap: dark indigo where the pose matches the target, through
 * magenta to the site's pink and a pale pink where it's furthest off.
 * Lightness rises steadily, so it still reads in greyscale. */
const STOPS = [
  [20, 14, 54],
  [74, 22, 110],
  [150, 35, 125],
  [222, 70, 125],
  [250, 175, 200],
];

export function lossColor(t: number): [number, number, number] {
  const f = Math.min(1, Math.max(0, t)) * (STOPS.length - 1);
  const i = Math.min(STOPS.length - 2, Math.floor(f));
  const s = f - i;
  const [a, b] = [STOPS[i], STOPS[i + 1]];
  return [0, 1, 2].map((c) => Math.round(a[c] + (b[c] - a[c]) * s)) as [
    number,
    number,
    number,
  ];
}
