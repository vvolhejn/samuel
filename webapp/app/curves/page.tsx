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
import { MUTED_LINK } from "@/components/ui";
import { usePinkTrombone } from "@/lib/usePinkTrombone";
import { insecureContextMessage } from "@/lib/secureContext";
import {
  CURVE_PARAMS,
  DEFAULT_DURATION_S,
  MAX_DURATION_S,
  MIN_DURATION_S,
  evaluate,
  toResponse,
  type Breakpoint,
  type CurveParam,
  type Interpolation,
} from "@/lib/curves";
import type {
  PinkTromboneElement,
  VoiceboxEventDetail,
} from "@/types/pink-trombone";

const ROW_H = 56;
/** Keeps a point at either end of the range clear of the row's edge. */
const ROW_PAD = 7;
/** Pointer distance (px) at which a breakpoint counts as under it. */
const HIT_PX = 8;
/** A mouth drag edits the breakpoint this close to the playhead, if any. */
const SNAP_S = 0.02;
/** Tract indices for drags on the mouth: from LIP_FROM on (the last three
 * sections) it's the lips; within the tongue tip's range, that. */
const LIP_FROM = 41;
const TIP_LO = 22;
const TIP_HI = 39;

let nextId = 0;

function initialCurves(): Record<string, Breakpoint[]> {
  return Object.fromEntries(
    CURVE_PARAMS.map((p) => [p.key, [{ id: nextId++, t: 0, v: p.init }]]),
  );
}

const STORAGE_KEY = "samuel-curves";

/** What localStorage and the Save/Load files hold. */
interface Saved {
  curves: Record<string, Breakpoint[]>;
  mode: Interpolation;
  duration: number;
}

function clampDuration(d: number): number {
  return Math.min(MAX_DURATION_S, Math.max(MIN_DURATION_S, d));
}

/** Read a saved state back, or null if it isn't one. A missing curve gets its
 * default; ids are handed out afresh, since nextId restarts. */
function parseSaved(saved: unknown): Saved | null {
  if (!saved || typeof saved !== "object" || !("curves" in saved)) return null;
  const { curves: raw, mode, duration } = saved as {
    curves: Record<string, unknown>;
    mode?: unknown;
    duration?: unknown;
  };
  const curves = initialCurves();
  for (const { key } of CURVE_PARAMS) {
    const points = raw?.[key];
    if (!Array.isArray(points)) continue;
    const valid = points
      .filter((p) => Number.isFinite(p?.t) && Number.isFinite(p?.v))
      .map((p: Breakpoint) => ({
        id: nextId++,
        t: Math.min(MAX_DURATION_S, Math.max(0, p.t)),
        v: p.v,
      }))
      .sort((a, b) => a.t - b.t);
    if (valid.length > 0) curves[key] = valid;
  }
  return {
    curves,
    mode: mode === "linear" ? "linear" : "smooth",
    duration:
      typeof duration === "number"
        ? clampDuration(duration)
        : DEFAULT_DURATION_S,
  };
}

/** The last session's state, from localStorage. */
function loadSaved(): Saved | null {
  try {
    return parseSaved(JSON.parse(localStorage.getItem(STORAGE_KEY) ?? "null"));
  } catch {
    return null;
  }
}

/** The state as a file's worth of JSON: ids are only for React. */
function serialize({ curves, mode, duration }: Saved): string {
  const plain = Object.fromEntries(
    Object.entries(curves).map(([key, points]) => [
      key,
      points.map(({ t, v }) => ({ t, v })),
    ]),
  );
  return JSON.stringify({ duration, mode, curves: plain }, null, 2);
}

/** Labelled tick spacing for a ruler of the given length. */
function tickFor(duration: number): number {
  return duration <= 1 ? 0.25 : duration <= 2.5 ? 0.5 : 1;
}

const subscribeNever = () => () => {};

/** Client-only: the saved curves live in localStorage, which the prerendered
 * page doesn't have, so the editor mounts once there is a client to read it. */
export default function Curves() {
  const client = useSyncExternalStore(
    subscribeNever,
    () => true,
    () => false,
  );
  return client ? <Editor /> : null;
}

function Editor() {
  const insecure = useSyncExternalStore(
    subscribeNever,
    insecureContextMessage,
    () => null,
  );
  const trombone = usePinkTrombone();
  const [saved] = useState(loadSaved);
  const [curves, setCurves] = useState(() => saved?.curves ?? initialCurves());
  const [mode, setMode] = useState<Interpolation>(saved?.mode ?? "smooth");
  const [duration, setDuration] = useState(
    saved?.duration ?? DEFAULT_DURATION_S,
  );
  /** The length field while it's being typed in; applied on blur/Enter. */
  const [durationText, setDurationText] = useState(String(duration));
  const fileRef = useRef<HTMLInputElement>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [hold, setHold] = useState(false);
  /** Playback/scrub position in [0, 1]. */
  const [frac, setFrac] = useState(0);
  const fracRef = useRef(0);
  const playIdRef = useRef(0);
  const scrubbingRef = useRef(false);
  const [error, setError] = useState<string | null>(null);

  const response = useMemo(
    () => toResponse(curves, mode, duration),
    [curves, mode, duration],
  );

  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, serialize({ curves, mode, duration }));
    } catch {
      // Storage blocked or full: the editor still works, it just won't persist.
    }
  }, [curves, mode, duration]);

  /** Clamp what was typed into [1, 5] s (to 0.1 s), or put back the old value
   * if it isn't a number. */
  const applyDuration = () => {
    const typed = Number(durationText);
    const next =
      durationText.trim() !== "" && Number.isFinite(typed)
        ? Math.round(clampDuration(typed) * 10) / 10
        : duration;
    setDuration(next);
    setDurationText(String(next));
  };

  const save = () => {
    const blob = new Blob([serialize({ curves, mode, duration })], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "curves.json";
    a.click();
    URL.revokeObjectURL(url);
  };

  const load = async (file: File) => {
    let loaded: Saved | null = null;
    try {
      loaded = parseSaved(JSON.parse(await file.text()));
    } catch {
      // Not JSON: reported below like any other unreadable file.
    }
    if (!loaded) {
      setError(`${file.name} isn't a saved set of curves`);
      return;
    }
    setError(null);
    setCurves(loaded.curves);
    setMode(loaded.mode);
    setDuration(loaded.duration);
    setDurationText(String(loaded.duration));
  };

  useEffect(() => {
    const element = insecureContextMessage()
      ? null
      : document.querySelector<PinkTromboneElement>("pink-trombone");
    if (element) {
      trombone.init(element).catch((e) => {
        setError(e instanceof Error ? e.message : String(e));
      });
    }
  }, [trombone]);

  const seek = useCallback((f: number) => {
    fracRef.current = f;
    setFrac(f);
  }, []);

  // Read by each new lap of the loop, so edits made while playing are heard
  // from the next one.
  const responseRef = useRef(response);
  useEffect(() => {
    responseRef.current = response;
  }, [response]);

  /** Plays in a loop until something else takes over (which bumps playIdRef). */
  const playFrom = useCallback(
    async (from: number) => {
      if (!trombone.ready()) return;
      const id = ++playIdRef.current;
      setIsPlaying(true);
      for (let start = from; playIdRef.current === id; start = 0) {
        await trombone.speak(responseRef.current, {
          startFrac: start,
          onProgress: seek,
        });
      }
    },
    [trombone, seek],
  );

  /** With hold on, playing sounds the pose at the playhead without moving it,
   * and keeps sounding it as the curves are edited (see the effect below). */
  const togglePlay = useCallback(() => {
    if (isPlaying) {
      playIdRef.current++;
      if (hold) trombone.endScrub();
      else trombone.stop();
      setIsPlaying(false);
      return;
    }
    if (hold) setIsPlaying(true);
    else void playFrom(fracRef.current >= 0.995 ? 0 : fracRef.current);
  }, [isPlaying, hold, trombone, playFrom]);

  const toggleHold = useCallback(() => {
    // Orphans a running playback, whose curves the scrub below cuts short.
    playIdRef.current++;
    if (hold && isPlaying) void playFrom(fracRef.current);
    setHold(!hold);
  }, [hold, isPlaying, playFrom]);

  // The mouth always shows the curves at the playhead: sounding, with hold on;
  // silently, when stopped. Playback moves it by itself.
  useEffect(() => {
    if (hold && isPlaying) trombone.scrub(response, frac);
    else if (!isPlaying && !scrubbingRef.current) trombone.pose(response, frac);
  }, [hold, isPlaying, trombone, response, frac]);

  /** Set some parameters at the playhead: move the breakpoint there, or add
   * one if there is none. */
  const editAtPlayhead = useCallback((values: Record<string, number>) => {
    const t = fracRef.current * duration;
    setCurves((c) => {
      const next = { ...c };
      for (const [key, raw] of Object.entries(values)) {
        const { lo, hi } = CURVE_PARAMS.find((p) => p.key === key)!;
        const v = Math.min(hi, Math.max(lo, raw));
        const points = c[key];
        const at = points.find((p) => Math.abs(p.t - t) <= SNAP_S);
        next[key] = at
          ? points.map((p) => (p === at ? { ...p, v } : p))
          : [...points, { id: nextId++, t, v }].sort((a, b) => a.t - b.t);
      }
      return next;
    });
  }, [duration]);

  // Drags on the mouth become breakpoints instead of reaching the synth: the
  // element's own handlers are skipped (capture phase, then stopped), and the
  // pose effect above moves the drawing to match the curves.
  useEffect(() => {
    const element =
      document.querySelector<PinkTromboneElement>("pink-trombone");
    if (!element) return;
    const constriction = (e: Event, target: "lip" | "tip") => {
      const { index, diameter } = (e as CustomEvent).detail;
      if (target === "lip") editAtPlayhead({ lipDiameter: diameter });
      else
        editAtPlayhead({
          constrictionIndex: index,
          constrictionDiameter: diameter,
        });
    };
    const handlers: Record<string, (e: Event) => void> = {
      setParameter: (e) => {
        const { parameterName, newValue } = (e as CustomEvent).detail;
        if (parameterName === "tongue.index")
          editAtPlayhead({ tongueIndex: Number(newValue) });
        else if (parameterName === "tongue.diameter")
          editAtPlayhead({ tongueDiameter: Number(newValue) });
      },
      // Near the lips it's the lip constriction, snapped to its fixed index;
      // in the tongue tip's range, that; anywhere else, nothing.
      newConstriction: (e) => {
        const { index, touchIdentifier } = (e as CustomEvent).detail;
        const target =
          index >= LIP_FROM ? "lip" : index >= TIP_LO && index <= TIP_HI ? "tip" : null;
        if (!target) return;
        constriction(e, target);
        // What TractUI waits for before it reports the drag's moves; the
        // target rides along as the "constriction index", fixing it for the
        // whole drag.
        e.target?.dispatchEvent(
          new CustomEvent("didNewConstriction", {
            detail: { touchIdentifier, constrictionIndex: target },
          }),
        );
      },
      setConstriction: (e) => {
        const target = (e as CustomEvent).detail.constrictionIndex;
        if (target === "lip" || target === "tip") constriction(e, target);
      },
      // Letting go leaves the breakpoint where it is.
      removeConstriction: () => {},
      voicebox: (e) => {
        e.preventDefault();
        const { frequency, tenseness } = (
          e as CustomEvent<VoiceboxEventDetail>
        ).detail;
        if (frequency === undefined || tenseness === undefined) return;
        editAtPlayhead({ frequency, voiceness: tenseness });
      },
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
  }, [editAtPlayhead]);

  // Space plays/pauses and H toggles hold, from anywhere on the page. Space is
  // handled on keyup too, or a focused button would take the same press as a
  // click and undo it.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement) return;
      if (e.key === "h" && e.type === "keydown" && !e.metaKey && !e.ctrlKey) {
        toggleHold();
        return;
      }
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
  }, [togglePlay, toggleHold]);

  /** Holding the timeline sounds the pose under the pointer; letting go plays
   * on from there, as on the main page — unless hold is on, when it just moves
   * the playhead. */
  const onScrub = useCallback(
    (f: number) => {
      if (!trombone.ready()) return;
      if (hold && isPlaying) {
        seek(f);
        return;
      }
      playIdRef.current++;
      setIsPlaying(false);
      scrubbingRef.current = true;
      seek(f);
      trombone.scrub(response, f);
    },
    [trombone, response, seek, hold, isPlaying],
  );

  const onScrubEnd = useCallback(() => {
    if (!scrubbingRef.current) return;
    scrubbingRef.current = false;
    if (!hold && fracRef.current < 0.995) void playFrom(fracRef.current);
    else trombone.endScrub();
  }, [trombone, playFrom, hold]);

  const setPoints = useCallback((key: string, points: Breakpoint[]) => {
    setCurves((c) => ({ ...c, [key]: points }));
  }, []);

  return (
    <main className="flex flex-1 flex-col items-start gap-6 px-4 py-4 md:flex-row md:gap-8 md:p-8">
      {/* Left: the mouth and what plays it. Right: the curves it follows. */}
      <div className="flex w-full max-w-[420px] shrink-0 flex-col gap-4">
        {(insecure ?? error) && (
          <p className="text-sm text-red-600">{insecure ?? error}</p>
        )}

        {insecure === null && (
          <TractStage>
            <pink-trombone className="block h-[600px] w-[600px]" />
          </TractStage>
        )}

        <div className="flex items-center gap-3">
          <button
            onClick={togglePlay}
            className="w-16 rounded-full border border-highlight-300 bg-white py-1.5 text-sm font-medium text-highlight-700 hover:bg-highlight-50"
          >
            {isPlaying ? "Pause" : "Play"}
          </button>
          <button
            onClick={toggleHold}
            aria-pressed={hold}
            className={
              hold
                ? "rounded-full border border-transparent bg-highlight-600 px-4 py-1.5 text-sm font-semibold text-white hover:bg-highlight-700"
                : "rounded-full border border-neutral-300 bg-white px-4 py-1.5 text-sm font-medium text-neutral-600 hover:bg-neutral-50"
            }
          >
            Hold
          </button>
          <div className="flex overflow-hidden rounded-full border border-neutral-300 text-sm">
            {(["linear", "smooth"] as const).map((m) => (
              <button
                key={m}
                onClick={() => setMode(m)}
                className={`px-3 py-1.5 capitalize ${
                  mode === m
                    ? "bg-highlight-600 font-semibold text-white"
                    : "bg-white text-neutral-600 hover:bg-neutral-50"
                }`}
              >
                {m}
              </button>
            ))}
          </div>
        </div>

        <div className="flex items-center gap-3 text-sm text-neutral-600">
          <label className="flex items-center gap-1.5">
            Length
            <input
              type="text"
              inputMode="decimal"
              value={durationText}
              onChange={(e) => setDurationText(e.currentTarget.value)}
              onBlur={applyDuration}
              onKeyDown={(e) => {
                if (e.key === "Enter") e.currentTarget.blur();
              }}
              className="w-12 rounded-md border border-neutral-300 px-2 py-1 text-right tabular-nums"
            />
            s
          </label>
          <button onClick={save} className={`${MUTED_LINK} ml-auto`}>
            Save
          </button>
          <button
            onClick={() => fileRef.current?.click()}
            className={MUTED_LINK}
          >
            Load
          </button>
          <input
            ref={fileRef}
            type="file"
            accept="application/json,.json"
            className="hidden"
            onChange={(e) => {
              const file = e.currentTarget.files?.[0];
              e.currentTarget.value = ""; // so the same file can load again
              if (file) void load(file);
            }}
          />
        </div>
      </div>

      <div className="w-full min-w-0 max-w-4xl flex-1 text-xs text-neutral-500 select-none">
        <div className="flex">
          <div className="w-36 shrink-0" />
          <Timeline
            duration={duration}
            frac={frac}
            onScrub={onScrub}
            onScrubEnd={onScrubEnd}
          />
        </div>
        {CURVE_PARAMS.map((param) => (
          <CurveRow
            key={param.key}
            param={param}
            points={curves[param.key]}
            mode={mode}
            duration={duration}
            frac={frac}
            onChange={(points) => setPoints(param.key, points)}
            onScrub={onScrub}
            onScrubEnd={onScrubEnd}
          />
        ))}
      </div>
    </main>
  );
}

/** Tracks an element's width, for the rows' pixel-space drawing. */
function useWidth() {
  const ref = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(0);
  useEffect(() => {
    const element = ref.current;
    if (!element) return;
    const observer = new ResizeObserver(([entry]) =>
      setWidth(entry.contentRect.width),
    );
    observer.observe(element);
    return () => observer.disconnect();
  }, []);
  return [ref, width] as const;
}

function Playhead({ frac }: { frac: number }) {
  return (
    <div
      className="pointer-events-none absolute inset-y-0 w-px bg-highlight-600"
      style={{ left: `${frac * 100}%` }}
    />
  );
}

/** The seconds ruler; pressing and dragging on it scrubs. */
function Timeline({
  duration,
  frac,
  onScrub,
  onScrubEnd,
}: {
  duration: number;
  frac: number;
  onScrub: (frac: number) => void;
  onScrubEnd: () => void;
}) {
  const tick = tickFor(duration);
  const fracAt = (e: React.PointerEvent<HTMLDivElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    return Math.min(1, Math.max(0, (e.clientX - rect.left) / rect.width));
  };
  return (
    <div
      className="relative h-6 flex-1 cursor-pointer border-b border-neutral-300"
      onPointerDown={(e) => {
        e.currentTarget.setPointerCapture(e.pointerId);
        onScrub(fracAt(e));
      }}
      onPointerMove={(e) => {
        if (e.currentTarget.hasPointerCapture(e.pointerId)) onScrub(fracAt(e));
      }}
      onPointerUp={onScrubEnd}
      onPointerCancel={onScrubEnd}
    >
      {Array.from({ length: Math.floor(duration / tick + 1e-9) + 1 }, (_, i) => (
        <span
          key={i}
          className="absolute bottom-0 h-2 border-l border-neutral-300"
          style={{ left: `${((i * tick) / duration) * 100}%` }}
        >
          <span className="absolute bottom-2 -translate-x-1/2 tabular-nums">
            {i * tick}s
          </span>
        </span>
      ))}
      <Playhead frac={frac} />
    </div>
  );
}

/** One parameter's curve. Double-click (or Option-click) adds a breakpoint, or
 * removes the one under the pointer; drag a breakpoint to move it. Pressing
 * anywhere else scrubs, as on the ruler. */
function CurveRow({
  param,
  points,
  mode,
  duration,
  frac,
  onChange,
  onScrub,
  onScrubEnd,
}: {
  param: CurveParam;
  points: Breakpoint[];
  mode: Interpolation;
  duration: number;
  frac: number;
  onChange: (points: Breakpoint[]) => void;
  onScrub: (frac: number) => void;
  onScrubEnd: () => void;
}) {
  const [ref, width] = useWidth();
  /** The breakpoint being dragged, "scrub" while scrubbing, else null. */
  const dragRef = useRef<number | "scrub" | null>(null);
  const { lo, hi } = param;

  const x = (t: number) => (t / duration) * width;
  const y = (v: number) =>
    ROW_PAD + (1 - (v - lo) / (hi - lo)) * (ROW_H - 2 * ROW_PAD);
  const fromPointer = (e: React.MouseEvent<SVGSVGElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const px = Math.min(width, Math.max(0, e.clientX - rect.left));
    const py = Math.min(
      ROW_H - ROW_PAD,
      Math.max(ROW_PAD, e.clientY - rect.top),
    );
    return {
      px,
      py,
      t: (px / width) * duration,
      v: lo + (1 - (py - ROW_PAD) / (ROW_H - 2 * ROW_PAD)) * (hi - lo),
    };
  };
  // Points past the end (left by shortening the length) still shape the curve
  // but aren't drawn, or they would sit beyond the row.
  const visible = points.filter((p) => p.t <= duration);
  const hit = (px: number, py: number) =>
    visible.find((p) => Math.hypot(x(p.t) - px, y(p.v) - py) <= HIT_PX);

  /** Add a breakpoint at the pointer, or remove the one under it (never the
   * last: a curve needs a value). */
  const toggleAt = (e: React.MouseEvent<SVGSVGElement>) => {
    const { px, py, t, v } = fromPointer(e);
    const under = hit(px, py);
    if (under) {
      if (points.length > 1) onChange(points.filter((p) => p !== under));
      return;
    }
    onChange([...points, { id: nextId++, t, v }].sort((a, b) => a.t - b.t));
  };

  const endDrag = () => {
    if (dragRef.current === "scrub") onScrubEnd();
    dragRef.current = null;
  };

  const path = useMemo(() => {
    if (width === 0) return "";
    const steps = Math.ceil(width / 2);
    return Array.from({ length: steps + 1 }, (_, i) => {
      const t = (i / steps) * duration;
      const v = evaluate(points, t, mode);
      return `${i === 0 ? "M" : "L"}${((i / steps) * width).toFixed(1)},${(
        ROW_PAD +
        (1 - (v - lo) / (hi - lo)) * (ROW_H - 2 * ROW_PAD)
      ).toFixed(1)}`;
    }).join("");
  }, [points, mode, duration, width, lo, hi]);

  const value = evaluate(points, frac * duration, mode);

  return (
    <div className="flex border-b border-neutral-200">
      <div className="flex w-36 shrink-0 flex-col justify-center pr-3">
        <span className="text-neutral-600">{param.label}</span>
        <span className="tabular-nums">{value.toFixed(2)}</span>
      </div>
      <div ref={ref} className="relative flex-1" style={{ height: ROW_H }}>
        <svg
          width={width}
          height={ROW_H}
          className="absolute inset-0 overflow-visible"
          onPointerDown={(e) => {
            if (e.altKey) {
              toggleAt(e);
              return;
            }
            const { px, py } = fromPointer(e);
            const under = hit(px, py);
            dragRef.current = under ? under.id : "scrub";
            e.currentTarget.setPointerCapture(e.pointerId);
            if (!under) onScrub(px / width);
          }}
          onPointerMove={(e) => {
            if (dragRef.current === null) return;
            const { px, t, v } = fromPointer(e);
            if (dragRef.current === "scrub") {
              onScrub(px / width);
              return;
            }
            onChange(
              points
                .map((p) => (p.id === dragRef.current ? { ...p, t, v } : p))
                .sort((a, b) => a.t - b.t),
            );
          }}
          onPointerUp={endDrag}
          onPointerCancel={endDrag}
          onDoubleClick={(e) => {
            if (!e.altKey) toggleAt(e);
          }}
        >
          <path d={path} fill="none" strokeWidth={1.5} className="stroke-highlight-600" />
          {visible.map((p) => (
            <circle
              key={p.id}
              cx={x(p.t)}
              cy={y(p.v)}
              r={4}
              strokeWidth={1.5}
              className="cursor-grab fill-white stroke-highlight-600 hover:fill-highlight-600"
            />
          ))}
        </svg>
        <Playhead frac={frac} />
      </div>
    </div>
  );
}
