/*
    Added for the samuel project (GPL-3.0, see Pink-Trombone/LICENSE):
      2026-08-14  extracted from TractUI.js so the voicebox (GlottisUI) draws
                  from the same palette and greys out with the tract
*/

// Matches the webapp's --font-sans stack (webapp/app/globals.css).
export const FONT_FAMILY = '"Helvetica Neue", Helvetica, Arial, Roboto, "Noto Sans", sans-serif';

// The webapp's accent ramp (--color-highlight-* in webapp/app/globals.css):
// #f92672 and steps at the same oklch hue, as sRGB hex. The active scheme
// replaces the original's pink/palePink/orchid/#C070C6 so the drawing matches
// the page around it; the inactive one greys the whole thing out while there is
// no audio input selected.
//
// The original uses palePink for three things — the tongue-control pad, the
// voicebox bars and the buttons — so `tongueControl` carries all three here,
// and `accent` carries everything the original drew in orchid.
export const COLOR_SCHEMES = {
  active: {
    tongueControl: "#ffe7eb", // ~highlight-100 — the tongue-control pad
    tract: "#ffa7ba", // ~highlight-300 — tract/nose fill
    wall: "#d40c5d", // highlight-700 — tract outline
    accent: "#f92672", // highlight-600 — labels, markers, amplitudes
    innerLabel: "#ffffff", // labels drawn inside the tract
  },
  inactive: {
    tongueControl: "#f5f5f5", // neutral-100
    tract: "#e5e5e5", // neutral-200
    wall: "#a1a1a1", // neutral-400
    accent: "#a1a1a1", // neutral-400
    innerLabel: "#737373", // neutral-500 — white would vanish on the grey fill
  },
};
