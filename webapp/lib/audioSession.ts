/**
 * We need this for two bugs
 * - no playback when mute switch is active
 * - playback after microphone was activated would be quiet because it went through the earpiece
 *     (until you switch to "play original" and back)
 * 
 * Claude description:
 * iOS Safari infers an audio session category from what the page does, and
 * neither guess it makes for us is right: a bare AudioContext gets "ambient",
 * which the hardware mute switch silences, and an open getUserMedia stream gets
 * "play-and-record", which routes the output to the earpiece. So state the
 * category instead — "playback" while the page only makes sound, and
 * "play-and-record" around the mic, since "playback" forbids capture.
 *
 * Safari 17 and up. Elsewhere `navigator.audioSession` is absent and this is a
 * no-op, mute switch included.
 */

/** https://www.w3.org/TR/audio-session/ — not in the TypeScript DOM lib yet. */
type AudioSessionType = "playback" | "play-and-record";

/** Tell iOS what the page is about to do with audio. */
export function setAudioSessionType(type: AudioSessionType): void {
  if (typeof navigator === "undefined") return;
  const session = (
    navigator as Navigator & { audioSession?: { type: AudioSessionType } }
  ).audioSession;
  if (session) session.type = type;
}
