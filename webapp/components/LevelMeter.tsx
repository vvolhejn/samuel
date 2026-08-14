import { useSyncExternalStore } from "react";

/** Pipes in the level meter. Unlit ones stay on screen, greyed. */
const METER_SLOTS = 14;

/** Dots typed out after the meter during the redemption window. */
const METER_DOTS = 3;

/** RMS range spanned by the meter, in dBFS: under a quiet room to under a
 * shout, so ordinary speech lands mid-scale. */
const METER_FLOOR_DB = -55;
const METER_CEIL_DB = -18;

/** Long enough that the dips between words don't strobe the meter, short
 * enough not to read as a countdown of its own — the dots do that. */
const METER_FADE_MS = 150;

export function levelToSlots(frame: Float32Array): number {
  let sum = 0;
  for (const sample of frame) sum += sample * sample;
  const rms = Math.sqrt(sum / frame.length);
  if (rms <= 0) return 0;
  const db = 20 * Math.log10(rms);
  const frac = (db - METER_FLOOR_DB) / (METER_CEIL_DB - METER_FLOOR_DB);
  return Math.max(0, Math.min(METER_SLOTS, Math.round(frac * METER_SLOTS)));
}

/** Publishes the mic level outside React, so the ~31 frames a second repaint
 * the meter rather than the whole page. Quantised to a slot count, so most
 * frames don't notify at all. */
export function makeLevelStore() {
  let slots = 0;
  const listeners = new Set<() => void>();
  return {
    set(next: number) {
      if (next === slots) return;
      slots = next;
      for (const listener of listeners) listener();
    },
    subscribe(listener: () => void) {
      listeners.add(listener);
      return () => {
        listeners.delete(listener);
      };
    },
    get: () => slots,
  };
}

export type LevelStore = ReturnType<typeof makeLevelStore>;

/** Mic level as a line of text. Lit pipes are pink while `active` and grey
 * otherwise, unlit ones fainter still, with dots typed out through `pending`. */
export function LevelMeter({
  store,
  active,
  pending,
}: {
  store: LevelStore;
  active: boolean;
  pending: boolean;
}) {
  const slots = useSyncExternalStore(store.subscribe, store.get, () => 0);
  return (
    // Tracked out because `|` has almost no side bearing: packed tight, the
    // antialiased stems bleed into each other and the colours look mixed.
    <p aria-hidden className="tracking-[0.2em] text-neutral-200">
      <span
        className="ease-linear"
        style={{
          color: active
            ? "var(--color-highlight-600)"
            : "var(--color-neutral-400)",
          transitionProperty: "color",
          transitionDuration: `${METER_FADE_MS}ms`,
        }}
      >
        {"|".repeat(slots)}
      </span>
      {"|".repeat(METER_SLOTS - slots)}
      {/* Staggered in CSS, so mounting starts the animation and unmounting
          wipes it. */}
      {pending && (
        <span className="meter-dots text-neutral-400">
          {Array.from({ length: METER_DOTS }, (_, i) => (
            <span key={i}>.</span>
          ))}
        </span>
      )}
    </p>
  );
}
