import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { PinkTromboneHandle } from "@/lib/usePinkTrombone";

/** Tape-scrub preview of the original: dragging the bar plays a short window of
 * audio from under the thumb. One grain per move, each superseding the last.
 * GRAIN_S is long enough to hear a vowel, and outlives GRAIN_INTERVAL_MS so a
 * steady drag sounds continuous rather than stuttered; the fades are what keep
 * the cut edges from clicking. */
const GRAIN_S = 0.12;
const GRAIN_FADE_S = 0.01;
const GRAIN_INTERVAL_MS = 55;

export interface OriginalAudioHandle {
  /** Is there audio the model heard, i.e. can the original be played? */
  loaded: boolean;
  /** Remember the audio a response was made from, so it can be replayed. Takes
   * ownership of the object URL and revokes the one it replaces. */
  set: (url: string) => void;
  /** Start playing from `startFrac`. Rejects if the browser refuses. */
  play: (startFrac: number) => Promise<void>;
  pause: () => void;
  /** True when nothing is sounding, including before anything is loaded. */
  paused: () => boolean;
  ended: () => boolean;
  /** Move the playhead without starting or stopping playback. */
  seek: (frac: number) => void;
  /** Playback position in [0, 1], or 0 before the metadata has arrived. */
  currentFrac: () => number;
  /** Play a short window from `frac` — the sound of dragging the bar. */
  playGrain: (frac: number) => void;
  /** Fade out the grain in flight. */
  stopGrain: () => void;
}

/** The audio the model heard: one long-lived media element for playback, plus a
 * decoded copy for the scrub preview. */
export function useOriginalAudio(
  trombone: PinkTromboneHandle,
): OriginalAudioHandle {
  const [loaded, setLoaded] = useState(false);
  const urlRef = useRef<string | null>(null);
  /** Plays urlRef. One long-lived element, so its currentTime can drive the
   * scrub bar and be driven by it. */
  const audioRef = useRef<HTMLAudioElement | null>(null);
  /** The same audio decoded, for the scrub preview — grains need random access,
   * which a media element's seek latency can't give. Null until the decode
   * lands (or for good, if it fails: the preview is then simply silent). */
  const bufferRef = useRef<AudioBuffer | null>(null);
  /** Bumped per original, so a slow decode can't install a stale buffer. */
  const tokenRef = useRef(0);
  /** The scrub grain now sounding, kept so the next one can cut it short. */
  const grainRef = useRef<{
    source: AudioBufferSourceNode;
    gain: GainNode;
  } | null>(null);
  const lastGrainAtRef = useRef(0);

  // Made once: a fresh element per play has no stable currentTime for the
  // scrub bar to read or write.
  useEffect(() => {
    const audio = new Audio();
    audio.preload = "metadata";
    audioRef.current = audio;
    return () => {
      audio.pause();
      audioRef.current = null;
    };
  }, []);

  /** Length of the loaded original, or 0 before its metadata has arrived. */
  const duration = useCallback(() => {
    const d = audioRef.current?.duration ?? 0;
    return Number.isFinite(d) && d > 0 ? d : 0;
  }, []);

  const set = useCallback(
    (url: string) => {
      const previous = urlRef.current;
      urlRef.current = url;
      const audio = audioRef.current;
      if (audio) {
        audio.pause();
        audio.src = url;
        audio.load(); // so duration is known before the first play
      }
      // Only after the element has let go of it.
      if (previous) URL.revokeObjectURL(previous);
      setLoaded(true);

      // Decode a second copy for the scrub preview. Straight off the object URL
      // rather than threading the blob through every caller; the token guards
      // against a slow decode landing after the next original has replaced it.
      const token = ++tokenRef.current;
      bufferRef.current = null;
      void (async () => {
        const ctx = trombone.audioContext();
        if (!ctx) return;
        try {
          const bytes = await (await fetch(url)).arrayBuffer();
          const buffer = await ctx.decodeAudioData(bytes);
          if (tokenRef.current === token) bufferRef.current = buffer;
        } catch {
          // Only the scrub preview depends on this; playback goes through the
          // media element either way, so a failure here stays silent.
        }
      })();
    },
    [trombone],
  );

  const seek = useCallback(
    (frac: number) => {
      const audio = audioRef.current;
      const d = duration();
      if (audio && d) audio.currentTime = Math.min(d - 0.01, frac * d);
    },
    [duration],
  );

  const play = useCallback(
    async (startFrac: number) => {
      const audio = audioRef.current;
      if (!audio || !urlRef.current) return;
      seek(startFrac);
      await audio.play();
    },
    [seek],
  );

  const pause = useCallback(() => audioRef.current?.pause(), []);

  const paused = useCallback(() => audioRef.current?.paused ?? true, []);

  const ended = useCallback(() => audioRef.current?.ended ?? false, []);

  const currentFrac = useCallback(() => {
    const audio = audioRef.current;
    const d = duration();
    return audio && d ? Math.min(1, audio.currentTime / d) : 0;
  }, [duration]);

  /** Ramped, not stopped dead: at these lengths a hard cut is audible as a
   * click on every move of the thumb. */
  const stopGrain = useCallback(() => {
    const grain = grainRef.current;
    if (!grain) return;
    grainRef.current = null;
    const t = grain.gain.context.currentTime;
    const gain = grain.gain.gain;
    gain.cancelScheduledValues(t);
    gain.setValueAtTime(gain.value, t);
    gain.linearRampToValueAtTime(0, t + GRAIN_FADE_S);
    grain.source.stop(t + GRAIN_FADE_S);
  }, []);

  /** Rate-limited, since a fast drag fires moves far quicker than a grain lasts
   * and they would otherwise pile up into mush. */
  const playGrain = useCallback(
    (frac: number) => {
      const ctx = trombone.audioContext();
      const buffer = bufferRef.current;
      if (!ctx || !buffer) return; // decode failed or hasn't landed
      const now = performance.now();
      if (now - lastGrainAtRef.current < GRAIN_INTERVAL_MS) return;
      lastGrainAtRef.current = now;
      void ctx.resume(); // we're in a user gesture

      const offset = Math.max(0, Math.min(1, frac)) * buffer.duration;
      const length = Math.min(GRAIN_S, buffer.duration - offset);
      if (length <= 2 * GRAIN_FADE_S) return; // at the very end; nothing to hear
      stopGrain();

      const t = ctx.currentTime;
      const gain = ctx.createGain();
      gain.gain.setValueAtTime(0, t);
      gain.gain.linearRampToValueAtTime(1, t + GRAIN_FADE_S);
      gain.gain.setValueAtTime(1, t + length - GRAIN_FADE_S);
      gain.gain.linearRampToValueAtTime(0, t + length);
      gain.connect(ctx.destination);
      const source = ctx.createBufferSource();
      source.buffer = buffer;
      source.connect(gain);
      source.onended = () => gain.disconnect();
      source.start(t, offset, length);
      grainRef.current = { source, gain };
    },
    [trombone, stopGrain],
  );

  return useMemo(
    () => ({
      loaded,
      set,
      play,
      pause,
      paused,
      ended,
      seek,
      currentFrac,
      playGrain,
      stopGrain,
    }),
    [
      loaded,
      set,
      play,
      pause,
      paused,
      ended,
      seek,
      currentFrac,
      playGrain,
      stopGrain,
    ],
  );
}
