/** Who has the mouth. Everything that makes a sound takes it first and hands it
 * back when it stops, so only one thing at a time drives the synth. "model" is
 * the wait for an answer, which nothing may interrupt. */
export type Owner = "none" | "mic" | "model" | "playback" | "scrub" | "manual";
