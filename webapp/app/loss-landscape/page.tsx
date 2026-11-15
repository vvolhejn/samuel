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
  lossRange,
  poseResponse,
  pairSlice,
  sampleAt,
  lossColor,
  type Landscape,
  type LossKind,
  type TargetKind,
} from "@/lib/lossLandscape";
import type { PinkTromboneElement } from "@/types/pink-trombone";

const LOSS_LABELS: Record<LossKind, string> = {
  mfcc: "MFCC",
  mel: "Mel",
  stft: "STFT",
  ssl: "wav2vec2",
};
/** The targets offered: two of the recordings (the data also has the synth
 * target and the 132 Hz "e"). */
const TARGET_LABELS = {
  ah: "“ah”",
  "e-1": "“e”",
} satisfies Partial<Record<TargetKind, string>>;
type ShownTarget = keyof typeof TARGET_LABELS;
const LABELS: Record<string, string> = {
  tongueIndex: "tongue index",
  tongueDiameter: "tongue height",
};
/** The map: tongueIndex × tongueDiameter, of the three swept params' pairs
 * (the data also has each with lipDiameter). */
const [MAP_X, MAP_Y] = [0, 1];
/** Params shown reversed: a smaller tongue diameter (the gap under the
 * palate) is a higher tongue, so it runs from high values to low, as "tongue
 * height". */
const REVERSED = new Set(["tongueDiameter"]);
/** Space the axes take around the heatmap. */
const AXIS_L = 44;
const AXIS_B = 36;
/** Gradient descent steps shown from the hovered point. */
const DESCENT_STEPS = 25;

const subscribeNever = () => () => {};

/** Hover the heatmap to move the tongue, and click it to toggle hearing the
 * pose; it shows the precomputed loss (as picked) of every tongue pose against
 * a recorded vowel, the other params held at the published model's fit. */
export default function LossLandscapePage() {
  const insecure = useSyncExternalStore(
    subscribeNever,
    insecureContextMessage,
    () => null,
  );
  const trombone = usePinkTrombone();
  const [target, setTarget] = useState<ShownTarget>("ah");
  const [kind, setKind] = useState<LossKind>("mfcc");
  /** Each target and loss's landscape, by `${target}-${kind}`, fetched the
   * first time it's picked. */
  const [landscapes, setLandscapes] = useState<Record<string, Landscape>>({});
  const landscapeKey = `${target}-${kind}`;
  const landscape = landscapes[landscapeKey] ?? null;
  const [values, setValues] = useState<Record<string, number> | null>(null);
  /** Sounding: toggled by clicking the heatmap (or space). */
  const [isPlaying, setIsPlaying] = useState(false);
  const [hovering, setHovering] = useState(false);
  const targetAudio = useRef<HTMLAudioElement | null>(null);
  const [ready, setReady] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (landscapes[landscapeKey]) return;
    loadLandscape(target, kind)
      .then((l) => {
        setLandscapes((ls) => ({ ...ls, [landscapeKey]: l }));
        // A target's losses share its reference; values are reset to null on
        // switching targets, so only the first load sets them.
        setValues((v) => v ?? l.reference);
        setError(null);
      })
      .catch((e) => setError(e instanceof Error ? e.message : String(e)));
  }, [target, kind, landscapeKey, landscapes]);

  /** A new target starts from its reference pose. */
  const pickTarget = (t: ShownTarget) => {
    if (t === target) return;
    const loaded = Object.values(landscapes).find((l) => l.target === t);
    setValues(loaded?.reference ?? null);
    setTarget(t);
  };

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

  // The mouth follows the values: sounding while playing and the pointer is on
  // the map, silently otherwise (so leaving the map pauses the sound without
  // turning it off).
  const sounding = isPlaying && hovering;
  const wasSounding = useRef(false);
  useEffect(() => {
    if (!ready || !values) return;
    if (sounding) trombone.scrub(poseResponse(values), 0);
    else {
      if (wasSounding.current) trombone.endScrub();
      trombone.pose(poseResponse(values), 0);
    }
    wasSounding.current = sounding;
  }, [ready, sounding, values, trombone]);

  /** A target's 2 s, looped while its button is held. */
  const startTarget = (t: TargetKind) => {
    const src = `/loss-landscape-${t}.wav`;
    if (!targetAudio.current?.src.endsWith(src)) {
      targetAudio.current?.pause();
      targetAudio.current = new Audio(src);
      targetAudio.current.loop = true;
    }
    targetAudio.current.currentTime = 0;
    targetAudio.current
      .play()
      .catch((e) => setError(e instanceof Error ? e.message : String(e)));
  };
  const stopTarget = () => targetAudio.current?.pause();

  const togglePlay = useCallback(() => {
    if (!trombone.ready()) return;
    setIsPlaying(!isPlaying);
  }, [isPlaying, trombone]);

  /** Keys of the two params on the map: the only ones that can move. */
  const mapped = useMemo(
    () => (landscape ? [MAP_X, MAP_Y].map((i) => landscape.params[i].key) : []),
    [landscape],
  );

  /** Moves the mapped params, kept within the map. */
  const set = useCallback(
    (changes: Record<string, number>) => {
      if (!landscape) return;
      const allowed = Object.fromEntries(
        landscape.params
          .filter(({ key }) => mapped.includes(key) && key in changes)
          .map(({ key, lo, hi }) => [
            key,
            Math.min(hi, Math.max(lo, changes[key])),
          ]),
      );
      setValues((v) => v && { ...v, ...allowed });
    },
    [landscape, mapped],
  );

  // Drags on the tongue move the mapped params rather than the synth directly
  // (the element's own handlers are skipped, as on /curves). Everything else
  // on the mouth is held at the reference.
  useEffect(() => {
    const element =
      document.querySelector<PinkTromboneElement>("pink-trombone");
    if (!element) return;
    const handlers: Record<string, (e: Event) => void> = {
      setParameter: (e) => {
        const { parameterName, newValue } = (e as CustomEvent).detail;
        if (parameterName === "tongue.index")
          set({ tongueIndex: Number(newValue) });
        else if (parameterName === "tongue.diameter")
          set({ tongueDiameter: Number(newValue) });
      },
      newConstriction: () => {},
      setConstriction: () => {},
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

  // Space toggles too, from anywhere on the page. It's handled on keyup as
  // well, or a focused button would take the same press as a click.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== " ") return;
      e.preventDefault();
      if (e.type === "keydown" && !e.repeat) togglePlay();
    };
    window.addEventListener("keydown", onKey);
    window.addEventListener("keyup", onKey);
    return () => {
      window.removeEventListener("keydown", onKey);
      window.removeEventListener("keyup", onKey);
    };
  }, [togglePlay]);

  const loss =
    landscape && values ? sampleAt(landscape, MAP_X, MAP_Y, values) : null;

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

        <p
          className={`text-center text-sm text-neutral-500 ${
            hovering ? "font-semibold" : ""
          }`}
        >
          Click the map to {isPlaying ? "stop" : "play"} audio
        </p>
      </div>

      {landscape && values && (
        <div className="flex w-full min-w-0 max-w-2xl flex-1 flex-col gap-2 text-xs text-neutral-500 select-none">
          {/* Labels right-aligned left of the center, buttons left-aligned
              right of it. */}
          <div className="grid grid-cols-2 items-center gap-x-3 gap-y-2 text-sm text-neutral-600">
            <p className="justify-self-end">
              Loss{" "}
              <span className="font-semibold tabular-nums text-neutral-800">
                {loss?.toFixed(3)}
              </span>
            </p>
            <div className="justify-self-start">
              <Segmented
                options={Object.keys(LOSS_LABELS) as LossKind[]}
                labels={LOSS_LABELS}
                value={kind}
                onChange={setKind}
              />
            </div>
            <p className="justify-self-end">Target</p>
            <div className="justify-self-start">
              <Segmented
                options={Object.keys(TARGET_LABELS) as ShownTarget[]}
                labels={TARGET_LABELS}
                value={target}
                onChange={pickTarget}
                onHold={(t) => {
                  pickTarget(t);
                  startTarget(t);
                }}
                onRelease={stopTarget}
              />
            </div>
          </div>
          <Heatmap
            landscape={landscape}
            x={MAP_X}
            y={MAP_Y}
            values={values}
            hovering={hovering}
            onHoverChange={setHovering}
            onHover={set}
            onPress={togglePlay}
          />
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
  const [lo, hi] = useMemo(() => lossRange(landscape), [landscape]);

  const data = pairSlice(landscape, x, y).loss;

  useEffect(() => {
    const ctx = canvasRef.current?.getContext("2d");
    if (!ctx) return;
    const img = ctx.createImageData(ax.n, ay.n);
    for (let iy = 0; iy < ay.n; iy++) {
      for (let ix = 0; ix < ax.n; ix++) {
        const [r, g, b] = lossColor((data[iy * ax.n + ix] - lo) / (hi - lo));
        // Canvas rows run top-down; the y axis runs bottom-up, unless reversed.
        const col = REVERSED.has(ax.key) ? ax.n - 1 - ix : ix;
        const row = REVERSED.has(ay.key) ? iy : ay.n - 1 - iy;
        const o = (row * ax.n + col) * 4;
        img.data.set([r, g, b, 255], o);
      }
    }
    ctx.putImageData(img, 0, 0);
  }, [data, ax.n, ay.n, ax.key, ay.key, lo, hi]);

  // Cells are centered on the grid points, so the map reaches half a cell
  // past lo and hi on each side. Fraction of the way along an axis (from the
  // left, or the bottom) of a value, and back.
  const frac = (a: typeof ax, v: number) => {
    const half = (a.hi - a.lo) / (a.n - 1) / 2;
    const f = (v - (a.lo - half)) / (a.hi - a.lo + 2 * half);
    return REVERSED.has(a.key) ? 1 - f : f;
  };
  const value = (a: typeof ax, f: number) => {
    const half = (a.hi - a.lo) / (a.n - 1) / 2;
    const g = REVERSED.has(a.key) ? 1 - f : f;
    const v = a.lo - half + g * (a.hi - a.lo + 2 * half);
    return Math.min(a.hi, Math.max(a.lo, v));
  };
  const pctX = (v: number) => frac(ax, v) * 100;
  const pctY = (v: number) => (1 - frac(ay, v)) * 100;

  const onMove = (e: React.PointerEvent<HTMLDivElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    onHover({
      [ax.key]: value(ax, (e.clientX - rect.left) / rect.width),
      [ay.key]: value(ay, 1 - (e.clientY - rect.top) / rect.height),
    });
  };

  const path = useMemo(
    () => (hovering ? descend(landscape, x, y, values, DESCENT_STEPS) : null),
    [hovering, landscape, x, y, values],
  );

  /** Values and labels; a reversed axis's values run backwards, so it gets
   * words at its ends instead. */
  const ticks = (a: typeof ax): [number, string][] =>
    REVERSED.has(a.key)
      ? [
          [a.hi, "low"],
          [a.lo, "high"],
        ]
      : Array.from({ length: 5 }, (_, i) => {
          const v = a.lo + ((a.hi - a.lo) * i) / 4;
          return [v, v.toFixed(1)];
        });

  return (
    <div className="flex gap-3">
      <div
        className="relative flex-1"
        style={{ paddingLeft: AXIS_L, paddingBottom: AXIS_B }}
      >
        <div
          className="relative aspect-square w-full cursor-crosshair touch-none"
          onPointerMove={onMove}
          onPointerDown={onMove}
          onClick={onPress}
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
            kind={landscape.target === "synth" ? "target" : "fit"}
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
          {ticks(ax).map(([v, label]) => (
            <span
              key={v}
              className="absolute top-full mt-1 -translate-x-1/2 tabular-nums"
              style={{ left: `${pctX(v)}%` }}
            >
              {label}
            </span>
          ))}
          {ticks(ay).map(([v, label]) => (
            <span
              key={v}
              className="absolute right-full mr-1.5 -translate-y-1/2 tabular-nums"
              style={{ top: `${pctY(v)}%` }}
            >
              {label}
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
  /** The synth target's pose, the model's fit to a recording, or the
   * hovered pose. */
  kind: "target" | "fit" | "current";
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
          <path
            d="M2,2L12,12M12,2L2,12"
            strokeWidth={2}
            className="stroke-white"
          />
        </svg>
      ) : kind === "fit" ? (
        <div className="h-4 w-4 rounded-full border-2 border-white shadow" />
      ) : (
        <div className="h-3.5 w-3.5 rounded-full border-2 border-white bg-neutral-900 shadow" />
      )}
    </div>
  );
}

function Segmented<T extends string>({
  options,
  labels,
  value,
  onChange,
  onHold,
  onRelease,
}: {
  options: T[];
  labels: Record<T, string>;
  value: T;
  onChange: (value: T) => void;
  /** Pressing an option down, and letting go of it (or leaving it). */
  onHold?: (value: T) => void;
  onRelease?: () => void;
}) {
  const held = onHold && onRelease;
  return (
    <div className="flex overflow-hidden rounded-full border border-neutral-300 text-sm">
      {options.map((o) => (
        <button
          key={o}
          onClick={() => onChange(o)}
          {...(held && {
            onPointerDown: () => onHold(o),
            onPointerUp: onRelease,
            onPointerLeave: onRelease,
            onPointerCancel: onRelease,
            onContextMenu: (e: React.MouseEvent) => e.preventDefault(),
          })}
          aria-pressed={value === o}
          className={`px-3 py-1 whitespace-nowrap ${held ? "touch-none select-none " : ""}${
            value === o
              ? "bg-highlight-600 font-semibold text-white"
              : "bg-white text-neutral-600 hover:bg-neutral-50"
          }`}
        >
          {labels[o]}
        </button>
      ))}
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
    const [r, g, b] = lossColor(i / 8);
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
