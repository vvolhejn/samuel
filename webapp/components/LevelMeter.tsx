import { useSyncExternalStore } from "react";
import { METER_SLOTS, LevelStore } from "@/lib/levelStore";

/** Dots typed out after the meter during the redemption window. */
const METER_DOTS = 3;

/** Long enough that the dips between words don't strobe the meter, short
 * enough not to read as a countdown of its own — the dots do that. */
const METER_FADE_MS = 150;

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
