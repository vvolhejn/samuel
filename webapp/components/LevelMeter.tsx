import { useSyncExternalStore } from "react";
import { METER_SLOTS, LevelStore } from "@/lib/levelStore";

/** Long enough that the dips between words don't strobe the meter. */
const METER_FADE_MS = 150;

/** Mic level as a line of text. Lit pipes are pink while `active` — the VAD
 * hears speech — and grey otherwise, unlit ones fainter still. */
export function LevelMeter({
  store,
  active,
}: {
  store: LevelStore;
  active: boolean;
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
    </p>
  );
}
