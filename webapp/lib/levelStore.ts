/** Pipes in the level meter. Unlit ones stay on screen, greyed. */
export const METER_SLOTS = 14;

/** RMS range spanned by the meter, in dBFS: under a quiet room to under a
 * shout, so ordinary speech lands mid-scale. */
const METER_FLOOR_DB = -55;
const METER_CEIL_DB = -18;

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
