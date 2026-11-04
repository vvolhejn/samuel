"use client";

import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  useSyncExternalStore,
} from "react";
import { TractStage } from "@/components/TractStage";
import { usePinkTrombone } from "@/lib/usePinkTrombone";
import { insecureContextMessage } from "@/lib/secureContext";
import {
  descend,
  loadLandscape,
  poseResponse,
  pairSlice,
  sampleAt,
  viridis,
  type Landscape,
} from "@/lib/lossLandscape";
import type { PinkTromboneElement } from "@/types/pink-trombone";

const LABELS: Record<string, string> = {
  tongueIndex: "tongue index",
  tongueDiameter: "tongue diameter",
  lipDiameter: "lip diameter",
};
/** Tract index from which a drag on the mouth is the lips (as on /curves). */
const LIP_FROM = 41;
/** Space the axes take around the heatmap. */
const AXIS_L = 44;
const AXIS_B = 36;
/** Gradient descent steps shown from the hovered point. */
const DESCENT_STEPS = 25;

const subscribeNever = () => () => {};

/** Hover the heatmap to move two of the three swept params, and hold it down to
 * hear the pose; it shows the precomputed MFCC loss of every pose in that
 * plane against the target sound, the third param held at the target. */
export default function LossLandscapePage() {
  const insecure = useSyncExternalStore(
    subscribeNever,
    insecureContextMessage,
    () => null,
  );
  const trombone = usePinkTrombone();
  const [landscape, setLandscape] = useState<Landscape | null>(null);
  const [values, setValues] = useState<Record<string, number> | null>(null);
  /** The pair on the heatmap, as an index into landscape.slices. */
  const [pair, setPair] = useState(0);
  /** Sounding: while the heatmap (or space) is held down. */
  const [isPlaying, setIsPlaying] = useState(false);
  const [hovering, setHovering] = useState(false);
  const [ready, setReady] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadLandscape()
      .then((l) => {
        setLandscape(l);
        setValues(l.reference);
      })
      .catch((e) => setError(e instanceof Error ? e.message : String(e)));
  }, []);

  useEffect(() => {
    const element = insecureContextMessage()
      ? null
      : document.querySelector<PinkTromboneElement>("pink-trombone");
    if (!element) return;
    trombone
      .init(element)
      .then(() => setReady(true))
      .catch((e) => setError(e instanceof Error ? e.message : String(e)));
  }, [trombone]);

  // The mouth follows the values: sounding while playing, silently otherwise.
  useEffect(() => {
    if (!ready || !values) return;
    if (isPlaying) trombone.scrub(poseResponse(values), 0);
    else trombone.pose(poseResponse(values), 0);
  }, [ready, isPlaying, values, trombone]);

  /** Set by the first hold; the hint is gone for good after it. */
  const [played, setPlayed] = useState(false);

  const play = useCallback(() => {
    if (!trombone.ready()) return;
    setIsPlaying(true);
    setPlayed(true);
  }, [trombone]);

  // Fading an already silent voice is harmless, so this needn't check.
  const stopPlaying = useCallback(() => {
    trombone.endScrub();
    setIsPlaying(false);
  }, [trombone]);

  // A hold ends wherever the pointer is let go, on the map or not.
  useEffect(() => {
    window.addEventListener("pointerup", stopPlaying);
    window.addEventListener("pointercancel", stopPlaying);
    return () => {
      window.removeEventListener("pointerup", stopPlaying);
      window.removeEventListener("pointercancel", stopPlaying);
    };
  }, [stopPlaying]);

  const slice = landscape?.slices[pair];

  /** Keys of the two params on the map: the only ones that can move. */
  const mapped = useMemo(
    () =>
      landscape && slice
        ? [slice.x, slice.y].map((i) => landscape.params[i].key)
        : [],
    [landscape, slice],
  );

  const set = useCallback(
    (changes: Record<string, number>) => {
      const allowed = Object.fromEntries(
        Object.entries(changes).filter(([key]) => mapped.includes(key)),
      );
      setValues((v) => v && { ...v, ...allowed });
    },
    [mapped],
  );

  // Drags on the mouth move the mapped params rather than the synth directly
  // (the element's own handlers are skipped, as on /curves): the tongue, and
  // the lips. Everything else on the mouth is held at the target.
  useEffect(() => {
    const element =
      document.querySelector<PinkTromboneElement>("pink-trombone");
    if (!element) return;
    const lip = (e: Event) =>
      set({
        lipDiameter: Math.min(3, Math.max(0, (e as CustomEvent).detail.diameter)),
      });
    const handlers: Record<string, (e: Event) => void> = {
      setParameter: (e) => {
        const { parameterName, newValue } = (e as CustomEvent).detail;
        if (parameterName === "tongue.index")
          set({ tongueIndex: Math.min(35, Math.max(10, Number(newValue))) });
        else if (parameterName === "tongue.diameter")
          set({ tongueDiameter: Math.min(3.5, Math.max(1.5, Number(newValue))) });
      },
      newConstriction: (e) => {
        const { index, touchIdentifier } = (e as CustomEvent).detail;
        if (index < LIP_FROM) return;
        lip(e);
        e.target?.dispatchEvent(
          new CustomEvent("didNewConstriction", {
            detail: { touchIdentifier, constrictionIndex: "lip" },
          }),
        );
      },
      setConstriction: (e) => {
        if ((e as CustomEvent).detail.constrictionIndex === "lip") lip(e);
      },
      removeConstriction: () => {},
      voicebox: (e) => e.preventDefault(),
    };
    const wrapped = Object.entries(handlers).map(
      ([type, handler]) =>
        [
          type,
          (e: Event) => {
            if (e.target === element) return; // our own calls, not a drag
            e.stopImmediatePropagation();
            handler(e);
          },
        ] as const,
    );
    for (const [type, h] of wrapped) element.addEventListener(type, h, true);
    return () => {
      for (const [type, h] of wrapped)
        element.removeEventListener(type, h, true);
    };
  }, [set]);

  // Holding space plays too, from anywhere on the page.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== " ") return;
      e.preventDefault();
      if (e.type === "keyup") stopPlaying();
      else if (!e.repeat) play();
    };
    window.addEventListener("keydown", onKey);
    window.addEventListener("keyup", onKey);
    return () => {
      window.removeEventListener("keydown", onKey);
      window.removeEventListener("keyup", onKey);
    };
  }, [play, stopPlaying]);

  /** Switching pairs puts the param that leaves the map back at its target:
   * the map is the slice through the target. */
  const pickPair = (i: number) => {
    if (!landscape) return;
    const next = landscape.slices[i];
    const off = 3 - next.x - next.y;
    const key = landscape.params[off].key;
    setValues((v) => v && { ...v, [key]: landscape.reference[key] });
    setPair(i);
  };

  const loss =
    landscape && values && slice
      ? sampleAt(landscape, slice.x, slice.y, values)
      : null;

  return (
    <main className="flex flex-1 flex-col items-start gap-6 px-4 py-4 md:flex-row md:gap-8 md:p-8">
      <div className="flex w-full max-w-[420px] shrink-0 flex-col gap-4">
        {(insecure ?? error) && (
          <p className="text-sm text-red-600">{insecure ?? error}</p>
        )}

        {insecure === null && (
          <TractStage>
            <pink-trombone className="block h-[600px] w-[600px]" />
          </TractStage>
        )}

        {/* Hidden rather than removed once played, so nothing below moves. */}
        <p
          className={`text-center text-sm text-neutral-500 ${
            played ? "invisible" : ""
          } ${hovering ? "font-semibold" : ""}`}
        >
          Click and hold the map to play audio
        </p>

        {landscape && (
          <div className="flex flex-col gap-2 text-sm">
            {landscape.slices.map((sl, i) => (
              <button
                key={i}
                onClick={() => pickPair(i)}
                aria-pressed={pair === i}
                className={`rounded-full border px-4 py-1.5 text-left ${
                  pair === i
                    ? "border-transparent bg-highlight-600 font-semibold text-white"
                    : "border-neutral-300 bg-white text-neutral-600 hover:bg-neutral-50"
                }`}
              >
                {LABELS[landscape.params[sl.x].key]} ×{" "}
                {LABELS[landscape.params[sl.y].key]}
              </button>
            ))}
          </div>
        )}
      </div>

      {landscape && values && (
        <div className="flex w-full min-w-0 max-w-2xl flex-1 flex-col gap-2 text-xs text-neutral-500 select-none">
          <p className="text-sm text-neutral-600">
            MFCC loss vs. the target:{" "}
            <span className="font-semibold tabular-nums text-neutral-800">
              {loss?.toFixed(3)}
            </span>
          </p>
          {slice && (
            <Heatmap
              landscape={landscape}
              x={slice.x}
              y={slice.y}
              values={values}
              hovering={hovering}
              onHoverChange={setHovering}
              onHover={set}
              onPress={play}
            />
          )}
        </div>
      )}
    </main>
  );
}

function Heatmap({
  landscape,
  x,
  y,
  values,
  hovering,
  onHoverChange,
  onHover,
  onPress,
}: {
  landscape: Landscape;
  x: number;
  y: number;
  values: Record<string, number>;
  /** The pointer is on the map; the descent path is drawn only then. */
  hovering: boolean;
  onHoverChange: (hovering: boolean) => void;
  onHover: (changes: Record<string, number>) => void;
  onPress: () => void;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const ax = landscape.params[x];
  const ay = landscape.params[y];

  // One color scale for all three maps, so a color means the same loss on each.
  const [lo, hi] = useMemo(() => {
    let lo = Infinity;
    let hi = -Infinity;
    for (const { loss } of landscape.slices) {
      for (const v of loss) {
        lo = Math.min(lo, v);
        hi = Math.max(hi, v);
      }
    }
    return [lo, hi];
  }, [landscape]);

  const data = pairSlice(landscape, x, y).loss;

  useEffect(() => {
    const ctx = canvasRef.current?.getContext("2d");
    if (!ctx) return;
    const img = ctx.createImageData(ax.n, ay.n);
    for (let iy = 0; iy < ay.n; iy++) {
      for (let ix = 0; ix < ax.n; ix++) {
        const [r, g, b] = viridis((data[iy * ax.n + ix] - lo) / (hi - lo));
        // Canvas rows run top-down; the y axis runs bottom-up.
        const o = ((ay.n - 1 - iy) * ax.n + ix) * 4;
        img.data.set([r, g, b, 255], o);
      }
    }
    ctx.putImageData(img, 0, 0);
  }, [data, ax.n, ay.n, lo, hi]);

  // Cells are centered on the grid points, so the map reaches half a cell
  // past lo and hi on each side.
  const extent = (a: typeof ax) => {
    const half = (a.hi - a.lo) / (a.n - 1) / 2;
    return [a.lo - half, a.hi + half] as const;
  };
  const [x0, x1] = extent(ax);
  const [y0, y1] = extent(ay);
  const pctX = (v: number) => ((v - x0) / (x1 - x0)) * 100;
  const pctY = (v: number) => (1 - (v - y0) / (y1 - y0)) * 100;

  const onMove = (e: React.PointerEvent<HTMLDivElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const fx = (e.clientX - rect.left) / rect.width;
    const fy = 1 - (e.clientY - rect.top) / rect.height;
    const clamp = (v: number, a: typeof ax) =>
      Math.min(a.hi, Math.max(a.lo, v));
    onHover({
      [ax.key]: clamp(x0 + fx * (x1 - x0), ax),
      [ay.key]: clamp(y0 + fy * (y1 - y0), ay),
    });
  };

  const path = useMemo(
    () =>
      hovering ? descend(landscape, x, y, values, DESCENT_STEPS) : null,
    [hovering, landscape, x, y, values],
  );

  const ticks = (a: typeof ax) =>
    Array.from({ length: 5 }, (_, i) => a.lo + ((a.hi - a.lo) * i) / 4);

  return (
    <div className="flex gap-3">
      <div
        className="relative flex-1"
        style={{ paddingLeft: AXIS_L, paddingBottom: AXIS_B }}
      >
        <div
          className="relative aspect-square w-full cursor-crosshair touch-none"
          onPointerMove={onMove}
          onPointerDown={(e) => {
            onMove(e);
            onPress();
          }}
          onPointerEnter={() => onHoverChange(true)}
          onPointerLeave={() => onHoverChange(false)}
        >
          {/* One pixel per grid point, upscaled with the browser's bilinear
              smoothing: pixel centers sit on the grid points, so it's the
              same interpolation the loss readout and descent use. */}
          <canvas
            ref={canvasRef}
            width={ax.n}
            height={ay.n}
            className="absolute inset-0 h-full w-full"
          />
          <Marker
            left={pctX(landscape.reference[ax.key])}
            top={pctY(landscape.reference[ay.key])}
            kind="target"
          />
          {path && (
            <DescentPath
              points={path.map((p) => [pctX(p[ax.key]), pctY(p[ay.key])])}
            />
          )}
          <Marker
            left={pctX(values[ax.key])}
            top={pctY(values[ay.key])}
            kind="current"
          />
          {ticks(ax).map((v) => (
            <span
              key={v}
              className="absolute top-full mt-1 -translate-x-1/2 tabular-nums"
              style={{ left: `${pctX(v)}%` }}
            >
              {v.toFixed(1)}
            </span>
          ))}
          {ticks(ay).map((v) => (
            <span
              key={v}
              className="absolute right-full mr-1.5 -translate-y-1/2 tabular-nums"
              style={{ top: `${pctY(v)}%` }}
            >
              {v.toFixed(1)}
            </span>
          ))}
          <span className="absolute top-full left-1/2 mt-5 -translate-x-1/2 text-sm text-neutral-600">
            {LABELS[ax.key]}
          </span>
          <span className="absolute top-1/2 right-full mr-8 origin-center translate-x-1/2 -rotate-90 text-sm whitespace-nowrap text-neutral-600">
            {LABELS[ay.key]}
          </span>
        </div>
      </div>
      <Colorbar lo={lo} hi={hi} bottom={AXIS_B} />
    </div>
  );
}

/** Gradient descent from the current point, in % of the map: a line through
 * each step, a dot per step, and a ring where it ends up. */
function DescentPath({ points }: { points: number[][] }) {
  const end = points[points.length - 1];
  return (
    <>
      <svg
        viewBox="0 0 100 100"
        preserveAspectRatio="none"
        className="pointer-events-none absolute inset-0 h-full w-full"
      >
        <polyline
          points={points.map(([px, py]) => `${px},${py}`).join(" ")}
          fill="none"
          strokeWidth={1.5}
          vectorEffect="non-scaling-stroke"
          className="stroke-white"
        />
      </svg>
      {points.slice(1).map(([px, py], i) => (
        <div
          key={i}
          className="pointer-events-none absolute h-1.5 w-1.5 -translate-x-1/2 -translate-y-1/2 rounded-full bg-white"
          style={{ left: `${px}%`, top: `${py}%` }}
        />
      ))}
      <div
        className="pointer-events-none absolute h-3 w-3 -translate-x-1/2 -translate-y-1/2 rounded-full border-2 border-white"
        style={{ left: `${end[0]}%`, top: `${end[1]}%` }}
      />
    </>
  );
}

function Marker({
  left,
  top,
  kind,
}: {
  left: number;
  top: number;
  kind: "target" | "current";
}) {
  return (
    <div
      className="pointer-events-none absolute -translate-x-1/2 -translate-y-1/2"
      style={{ left: `${left}%`, top: `${top}%` }}
    >
      {kind === "target" ? (
        <svg width={14} height={14} className="overflow-visible">
          <path
            d="M2,2L12,12M12,2L2,12"
            strokeWidth={4}
            className="stroke-black/60"
          />
          <path d="M2,2L12,12M12,2L2,12" strokeWidth={2} className="stroke-white" />
        </svg>
      ) : (
        <div className="h-3.5 w-3.5 rounded-full border-2 border-white bg-highlight-600 shadow" />
      )}
    </div>
  );
}

function Colorbar({
  lo,
  hi,
  bottom,
}: {
  lo: number;
  hi: number;
  bottom: number;
}) {
  const stops = Array.from({ length: 9 }, (_, i) => {
    const [r, g, b] = viridis(i / 8);
    return `rgb(${r},${g},${b}) ${(i / 8) * 100}%`;
  });
  return (
    <div className="flex gap-1" style={{ paddingBottom: bottom }}>
      <div
        className="w-3"
        style={{ background: `linear-gradient(to top, ${stops.join(",")})` }}
      />
      <div className="flex flex-col justify-between tabular-nums">
        <span>{hi.toFixed(2)}</span>
        <span>{lo.toFixed(2)}</span>
      </div>
    </div>
  );
}
