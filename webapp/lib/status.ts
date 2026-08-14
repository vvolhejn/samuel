/** What the page is doing, in one word. Shared because the mic and the
 * transport both hand it back and forth. */
export type Status =
  "idle" | "listening" | "recording" | "processing" | "speaking" | "muted";
