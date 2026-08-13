"use client";

import {
  useCallback,
  useEffect,
  useRef,
  useState,
  useSyncExternalStore,
} from "react";
import type { MicVAD } from "@ricky0123/vad-web";
import {
  fetchDatasetClips,
  fetchHealth,
  fetchPrecomputedIndex,
  precomputedMatches,
  synthesizeDatasetClip,
  synthesizeUtterance,
  wavBlob,
  DatasetClip,
  HealthResponse,
  PrecomputedIndex,
  SynthResponse,
} from "@/lib/audio";
import { insecureContextMessage, micErrorMessage } from "@/lib/secureContext";
import { downloadBlob, makeZip } from "@/lib/zip";
import { usePinkTrombone } from "@/lib/usePinkTrombone";
import type { PinkTromboneElement } from "@/types/pink-trombone";

type Status =
  "idle" | "listening" | "recording" | "processing" | "speaking" | "muted";

/** The three stages of the page, in the order you use them. Exactly one is
 * highlighted at a time, which is how the eye gets handed along. */
type Section = "input" | "playback" | "tract";

/** Faint box around a stage; the active one drops the outline and fills with the
 * accent wash instead, so only one thing at a time reads as raised. The border
 * stays declared but transparent when lit, or the box would shift a pixel.
 * Padding is left to the call site. */
function sectionBox(active: boolean): string {
  return `rounded-xl border transition-[color,background-color,border-color,filter] ${
    active
      ? "border-transparent bg-highlight-50/60"
      : "border-neutral-200 bg-white"
  }`;
}

/** Which of the two audio sources the transport plays: the model's imitation,
 * or the audio it was made from. */
type Source = "samuel" | "original";

/** Floor under both faces of the input box, so toggling the mic doesn't resize
 * it and shove the page around. */
const INPUT_MIN_H = "min-h-[5.5rem]";

/** Silence the VAD waits through before it calls an utterance finished. The
 * meter greys out over exactly this window, so the wait reads as a countdown. */
const REDEMPTION_MS = 800;

/** vad-web's hysteresis, restated so the meter watches the same edges it does:
 * speech starts above the first, ends below the second. */
const POSITIVE_SPEECH_THRESHOLD = 0.3;
const NEGATIVE_SPEECH_THRESHOLD = 0.25;

/** Heading in the recording panel. */
function micHeading(status: Status): string {
  if (status === "processing") return "Thinking…";
  if (status === "speaking") return "Answering…";
  return "Recording";
}

/** Pipes in the level meter. Unlit ones stay on screen, greyed. */
const METER_SLOTS = 14;

/** Dots typed out after the meter during the redemption window. */
const METER_DOTS = 3;

/** RMS range spanned by the meter, in dBFS: under a quiet room to under a
 * shout, so ordinary speech lands mid-scale. */
const METER_FLOOR_DB = -55;
const METER_CEIL_DB = -18;

function levelToSlots(frame: Float32Array): number {
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
function makeLevelStore() {
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

type LevelStore = ReturnType<typeof makeLevelStore>;

/** Mic level as a line of text: pink while `active`, draining to grey over the
 * redemption window, with dots typed out through `pending`. */
function LevelMeter({
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
    <p aria-hidden className="text-neutral-300">
      <span
        className="ease-linear"
        style={{
          color: active ? "var(--color-highlight-600)" : "inherit",
          transitionProperty: "color",
          transitionDuration: active ? "60ms" : `${REDEMPTION_MS}ms`,
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

/** Tape-scrub preview of the original: dragging the bar plays a short window of
 * audio from under the thumb. One grain per move, each superseding the last.
 * GRAIN_S is long enough to hear a vowel, and outlives GRAIN_INTERVAL_MS so a
 * steady drag sounds continuous rather than stuttered; the fades are what keep
 * the cut edges from clicking. */
const GRAIN_S = 0.12;
const GRAIN_FADE_S = 0.01;
const GRAIN_INTERVAL_MS = 55;

/** Nothing ever invalidates the secure-context snapshot. */
const subscribeNever = () => () => {};

/** One round trip: what the model heard, and what it said back. Held in memory
 * for the whole session, so cap it — a few seconds of 16 kHz float WAV is
 * ~250 kB per side, and nothing else evicts them. */
const MAX_HISTORY = 50;

interface Recording {
  kind: "mic" | "dataset";
  /** Audio the model was given: a WAV from the mic, an MP3 for a clip. */
  input: Blob;
  /** WAV of the model's output, rendered by the Python synth. Null for a
   * precomputed clip: those responses drop the reference audio. */
  output: Blob | null;
}

/** How long a precomputed clip pretends to think, in ms. The answer is on disk
 * and comes back in no time, which reads as a button that didn't work — and
 * makes the six clips feel unlike the mic, which really does take a moment. */
const FAKE_THINKING_MS = [500, 1000] as const;

const sleep = (ms: number) =>
  new Promise<void>((resolve) => setTimeout(resolve, ms));

function fakeThinking(): Promise<void> {
  const [lo, hi] = FAKE_THINKING_MS;
  return sleep(lo + Math.random() * (hi - lo));
}

/** Browser mic processing, toggleable so we can A/B it against training audio,
 * which has none of it. vad-web hardcodes all three on; we pass our own
 * getStream/resumeStream instead (see startMic). */
type MicProcessing = {
  echoCancellation: boolean;
  autoGainControl: boolean;
  noiseSuppression: boolean;
};

const MIC_PROCESSING_DEFAULTS: MicProcessing = {
  echoCancellation: true,
  autoGainControl: true,
  noiseSuppression: true,
};

const MIC_PROCESSING_LABELS: Array<{
  key: keyof MicProcessing;
  label: string;
}> = [
  { key: "echoCancellation", label: "echo cancellation" },
  { key: "autoGainControl", label: "auto gain control" },
  { key: "noiseSuppression", label: "noise suppression" },
];

const PANEL_PARAMS: Array<{ key: string; label: string; digits: number }> = [
  { key: "frequency", label: "frequency (Hz)", digits: 1 },
  { key: "voiceness", label: "voiceness", digits: 3 },
  { key: "intensity", label: "intensity", digits: 3 },
  { key: "tongueIndex", label: "tongue index", digits: 2 },
  { key: "tongueDiameter", label: "tongue diameter", digits: 2 },
  { key: "constrictionIndex", label: "constriction index", digits: 2 },
  { key: "constrictionDiameter", label: "constriction diameter", digits: 2 },
  { key: "lipDiameter", label: "lip diameter", digits: 2 },
];

/** Exact model-output values at the current playback/scrub position. */
function ModelOutput({
  response,
  frac,
}: {
  response: SynthResponse | null;
  frac: number;
}) {
  const nFrames = response?.n_frames ?? 0;
  const frame = response
    ? Math.min(nFrames - 1, Math.max(0, Math.round(frac * (nFrames - 1))))
    : 0;
  const row = (label: string, value: string) => (
    <div key={label} className="flex items-baseline justify-between gap-3">
      <dt className="text-neutral-500">{label}</dt>
      <dd className="font-mono tabular-nums text-neutral-800">{value}</dd>
    </div>
  );
  return (
    <section>
      <div className="mb-2 flex items-baseline justify-between">
        <SectionTitle>model output</SectionTitle>
        <span className="tabular-nums text-neutral-400">
          {response ? `frame ${frame + 1}/${nFrames}` : "no clip"}
        </span>
      </div>
      <dl className="space-y-1.5">
        {PANEL_PARAMS.map(({ key, label, digits }) =>
          // Older checkpoints predate some params (e.g. lipDiameter).
          row(
            label,
            response?.params[key]
              ? response.params[key][frame].toFixed(digits)
              : "–",
          ),
        )}
        {row(
          "voiced (pyin)",
          response ? (response.voiced[frame] ? "yes" : "no") : "–",
        )}
      </dl>
    </section>
  );
}

/** ".", "..", "..." on a loop. The dots are always laid out and only fade in and
 * out, so the label after them never shifts. */
function Ellipsis() {
  return (
    <span aria-hidden className="ellipsis">
      <span>.</span>
      <span>.</span>
      <span>.</span>
    </span>
  );
}

/** A link in the prose. Anything off-origin opens in a new tab — which, as it
 * stands, is every link here.
 *
 * `muted` is for asides nobody needs to follow (the QWOP joke, the credit for
 * the original): the pink underline still says "link", but the text stays the
 * colour of the sentence around it so it doesn't ask for the click. */
function TextLink({
  href,
  muted,
  children,
}: {
  href: string;
  muted?: boolean;
  children: React.ReactNode;
}) {
  const external = /^(https?:)?\/\//.test(href);
  return (
    <a
      href={href}
      target={external ? "_blank" : undefined}
      rel={external ? "noreferrer" : undefined}
      className={
        muted
          ? "text-inherit underline decoration-highlight-600 decoration-dotted underline-offset-2 hover:text-highlight-600"
          : "text-highlight-600 underline decoration-dotted hover:text-highlight-700"
      }
    >
      {children}
    </a>
  );
}

function SectionTitle({ children }: { children: React.ReactNode }) {
  return (
    <span className="font-semibold tracking-wide text-neutral-600 uppercase">
      {children}
    </span>
  );
}

/** Everything that is useful while developing but noise while using the thing:
 * which checkpoint is loaded, the live parameter trajectories, and the mic
 * capture settings. Collapsed to a tab on the right by default. */
function DebugPanel({
  open,
  onToggle,
  health,
  precomputed,
  response,
  frac,
  micProcessing,
  onToggleMicProcessing,
  historyCount,
  onDownloadHistory,
}: {
  open: boolean;
  onToggle: () => void;
  health: HealthResponse | null;
  precomputed: PrecomputedIndex | null;
  response: SynthResponse | null;
  frac: number;
  micProcessing: MicProcessing;
  onToggleMicProcessing: (key: keyof MicProcessing) => void;
  historyCount: number;
  onDownloadHistory: () => void;
}) {
  if (!open) {
    return (
      <button
        onClick={onToggle}
        title="Show the debug panel"
        className="rounded-full border border-neutral-300 px-4 py-1.5 text-sm font-medium text-neutral-600 hover:bg-neutral-50"
      >
        Debug
      </button>
    );
  }
  return (
    <aside className="w-full max-w-md space-y-4 rounded-lg border border-neutral-200 bg-neutral-50 p-3 text-xs">
      <div className="flex items-baseline justify-between">
        <SectionTitle>debug</SectionTitle>
        <button
          onClick={onToggle}
          title="Hide the debug panel"
          className="text-neutral-400 hover:text-highlight-600"
        >
          hide
        </button>
      </div>

      <section>
        <div className="mb-1">
          <SectionTitle>checkpoint</SectionTitle>
        </div>
        {health ? (
          <p className="font-mono break-all text-neutral-500">
            {health.checkpoint.startsWith("https://") ? (
              <a
                href={health.checkpoint}
                target="_blank"
                rel="noreferrer"
                className="underline decoration-dotted hover:text-highlight-600"
              >
                {health.checkpoint}
              </a>
            ) : (
              health.checkpoint
            )}
          </p>
        ) : (
          <p className="text-neutral-400">backend unreachable</p>
        )}
      </section>

      <section>
        <div className="mb-1">
          <SectionTitle>precomputed clips</SectionTitle>
        </div>
        {!precomputed ? (
          <p className="text-neutral-500">
            none committed — every clip goes to the backend
          </p>
        ) : precomputedMatches(precomputed, health?.model_fingerprint) ? (
          <p className="font-mono text-neutral-500">
            {precomputed.clips.length} clip(s), {precomputed.model_fingerprint}
          </p>
        ) : (
          <p className="text-red-600">
            stale: made by {precomputed.model_fingerprint}, backend serves{" "}
            {health?.model_fingerprint}. Falling back to the backend — re-run{" "}
            <code>scripts/precompute_clip_responses.py</code>.
          </p>
        )}
      </section>

      <ModelOutput response={response} frac={frac} />

      <section title="Browser mic processing. Training audio has none of it; the server RMS-normalises either way, so these change the shape of the input, not its level.">
        <div className="mb-1">
          <SectionTitle>mic processing</SectionTitle>
        </div>
        <div className="space-y-1 text-neutral-600">
          {MIC_PROCESSING_LABELS.map(({ key, label }) => (
            <label key={key} className="flex items-center gap-1.5">
              <input
                type="checkbox"
                checked={micProcessing[key]}
                onChange={() => onToggleMicProcessing(key)}
                className="accent-highlight-600"
              />
              {label}
            </label>
          ))}
        </div>
      </section>

      <section>
        <div className="mb-1">
          <SectionTitle>session audio</SectionTitle>
        </div>
        <button
          onClick={onDownloadHistory}
          disabled={historyCount === 0}
          title="Zip of every utterance this session: what the model heard and what it said back"
          className="rounded-full border border-neutral-300 px-3 py-1 text-xs font-medium text-neutral-600 hover:bg-neutral-100 disabled:opacity-40 disabled:hover:bg-transparent"
        >
          {historyCount === 0
            ? "nothing recorded yet"
            : `download ${historyCount} utterance${historyCount === 1 ? "" : "s"}`}
        </button>
      </section>
    </aside>
  );
}

export default function Home() {
  const [status, setStatus] = useState<Status>("idle");
  const [error, setError] = useState<string | null>(null);
  /** null until /api/health answers (or if the backend is down). */
  const [health, setHealth] = useState<HealthResponse | null>(null);
  /** Render-side mirror of lastResponse (refs must not be read in render). */
  const [viewResponse, setViewResponse] = useState<SynthResponse | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  /** Which source the transport is pointed at. */
  const [source, setSource] = useState<Source>("samuel");
  /** Playback/scrub position within the current source, in [0, 1]. */
  const [scrubFrac, setScrubFrac] = useState(0);
  const [micProcessing, setMicProcessing] = useState<MicProcessing>(
    MIC_PROCESSING_DEFAULTS,
  );
  /** Is there audio the model heard, i.e. can "Original" play anything? */
  const [hasOriginal, setHasOriginal] = useState(false);
  /** User-intended mic state — the Record/Stop toggle. Mirrors micOnRef; the
   * status alone can't stand in for it (a dataset clip is "speaking" too). */
  const [micOn, setMicOn] = useState(false);
  /** Is the VAD hearing speech *this frame*? Drives the ring around the
   * recording panel, and nothing else — unlike status "recording" it flips tens
   * of times a second, so it must never gate a control. */
  const [heard, setHeard] = useState(false);
  /** Pre-recorded clips shipped with the app, and the last one played (which
   * is the button left filled in). */
  const [clips, setClips] = useState<DatasetClip[]>([]);
  const [playedClip, setPlayedClip] = useState<string>("");
  /** The committed clip answers, if any were generated for this checkpoint. */
  const [precomputed, setPrecomputed] = useState<PrecomputedIndex | null>(null);
  const [debugOpen, setDebugOpen] = useState(false);
  /** Why this origin can't run the app at all, or null. Read through
   * useSyncExternalStore because it is a client-only fact: the exported HTML is
   * built without an origin. */
  const insecure = useSyncExternalStore(
    subscribeNever,
    insecureContextMessage,
    () => null,
  );

  const trombone = usePinkTrombone();
  const vadRef = useRef<MicVAD | null>(null);
  const lastResponse = useRef<SynthResponse | null>(null);
  const originalUrlRef = useRef<string | null>(null); // audio the model heard
  /** Every utterance this session, for the debug panel's download. */
  const historyRef = useRef<Recording[]>([]);
  const [historyCount, setHistoryCount] = useState(0);
  /** Plays originalUrlRef. One long-lived element, so its currentTime can drive
   * the scrub bar and be driven by it. */
  const originalAudioRef = useRef<HTMLAudioElement | null>(null);
  /** The same audio decoded, for the scrub preview — grains need random access,
   * which a media element's seek latency can't give. Null until the decode
   * lands (or for good, if it fails: the preview is then simply silent). */
  const originalBufferRef = useRef<AudioBuffer | null>(null);
  /** Bumped per original, so a slow decode can't install a stale buffer. */
  const originalTokenRef = useRef(0);
  /** The scrub grain now sounding, kept so the next one can cut it short. */
  const grainRef = useRef<{
    source: AudioBufferSourceNode;
    gain: GainNode;
  } | null>(null);
  const lastGrainAtRef = useRef(0);
  const busyRef = useRef(false); // ignore VAD events while processing/speaking
  /** Waiting on the model — the one state nothing can interrupt. */
  const processingRef = useRef(false);
  const micOnRef = useRef(false); // user-intended mic state (start/mute toggle)
  const scrubbingRef = useRef(false); // pointer is down on the scrub bar
  /** Bumped whenever a playback starts or is cut short, so the completion of a
   * superseded one can't clear the new one's state out from under it. */
  const playIdRef = useRef(0);
  /** Mirror of scrubFrac for the callbacks that must not re-bind on every
   * progress tick. */
  const scrubFracRef = useRef(0);
  const sourceRef = useRef<Source>("samuel");
  // Read by getStream/resumeStream, which vad-web calls on every start().
  const micProcessingRef = useRef<MicProcessing>(MIC_PROCESSING_DEFAULTS);
  /** Mirror of `heard`, so the per-frame callback only re-renders on edges. */
  const heardRef = useRef(false);
  /** Mirror of the VAD's own `speaking` flag, which unlike heardRef stays up
   * through the redemption window. */
  const speakingRef = useRef(false);
  const [levelStore] = useState(makeLevelStore);

  // Bring up the synth + tract visualization immediately; the AudioContext
  // stays suspended until the first user gesture (Start).
  useEffect(() => {
    // On an insecure origin nothing here works, and starting the synth anyway
    // is what makes the visualization glitch — see insecureContextMessage.
    const element = insecureContextMessage()
      ? null
      : document.querySelector<PinkTromboneElement>("pink-trombone");
    if (element) {
      trombone.init(element).catch((e) => {
        setError(e instanceof Error ? e.message : String(e));
      });
    }
    return () => {
      vadRef.current?.destroy();
      vadRef.current = null;
    };
  }, [trombone]);

  // Which checkpoint the backend is serving (shown under the title).
  useEffect(() => {
    void fetchHealth().then(setHealth);
  }, []);

  // The pre-recorded clips, one button each, and the answers committed for them.
  useEffect(() => {
    void fetchDatasetClips().then(setClips);
    void fetchPrecomputedIndex().then(setPrecomputed);
  }, []);

  // The element that plays the original. Made once: a fresh element per play
  // has no stable currentTime for the scrub bar to read or write.
  useEffect(() => {
    const audio = new Audio();
    audio.preload = "metadata";
    originalAudioRef.current = audio;
    return () => {
      audio.pause();
      originalAudioRef.current = null;
    };
  }, []);

  /** Edge-triggered setter for the meter's colour. */
  const showHeard = useCallback((value: boolean) => {
    if (heardRef.current === value) return;
    heardRef.current = value;
    setHeard(value);
  }, []);

  /** Resume VAD only if the user hasn't muted the mic. */
  const restoreMic = useCallback(async () => {
    if (scrubbingRef.current) return; // the scrub owns the synth until pointer-up
    speakingRef.current = false; // any pause() left the frame processor reset
    if (micOnRef.current) {
      await vadRef.current?.start();
      setStatus("listening");
    } else {
      setStatus(vadRef.current ? "muted" : "idle");
    }
  }, []);

  const updateFrac = useCallback((frac: number) => {
    scrubFracRef.current = frac;
    setScrubFrac(frac);
  }, []);

  /** Length of the loaded original, or 0 before its metadata has arrived. */
  const originalDuration = useCallback(() => {
    const d = originalAudioRef.current?.duration ?? 0;
    return Number.isFinite(d) && d > 0 ? d : 0;
  }, []);

  /** Fade out the grain in flight. Ramped, not stopped dead: at these lengths a
   * hard cut is audible as a click on every move of the thumb. */
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

  /** Play a short window of the original from `frac` — the sound of dragging
   * the bar. Rate-limited, since a fast drag fires moves far quicker than a
   * grain lasts and they would otherwise pile up into mush. */
  const playGrain = useCallback(
    (frac: number) => {
      const ctx = trombone.audioContext();
      const buffer = originalBufferRef.current;
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

  /** Silence whatever is playing and orphan its completion handler, leaving the
   * scrub position where it stands. Every new playback starts here, which is
   * what makes a second clip interrupt the first instead of being locked out. */
  const stopPlayback = useCallback(() => {
    playIdRef.current++;
    trombone.stop();
    originalAudioRef.current?.pause();
    stopGrain();
    setIsPlaying(false);
  }, [trombone, stopGrain]);

  /** End of a playback that wasn't superseded: hand the mic back. */
  const finishPlayback = useCallback(
    async (id: number) => {
      if (playIdRef.current !== id) return;
      setIsPlaying(false);
      busyRef.current = false;
      await restoreMic();
    },
    [restoreMic],
  );

  const playSamuel = useCallback(
    async (response: SynthResponse, startFrac = 0) => {
      stopPlayback();
      const id = playIdRef.current;
      busyRef.current = true;
      setStatus("speaking");
      setIsPlaying(true);
      await vadRef.current?.pause(); // don't let the synth retrigger the mic
      try {
        await trombone.speak(response, {
          startFrac,
          onProgress: (frac) => {
            if (playIdRef.current === id) updateFrac(frac);
          },
        });
      } finally {
        await finishPlayback(id);
      }
    },
    [trombone, stopPlayback, finishPlayback, updateFrac],
  );

  /** Play the audio the model heard — your trimmed recording, or the dataset
   * clip — at its recorded level; the RMS normalisation happens server-side.
   * Drives the same scrub bar as the imitation, off the element's clock. */
  const playOriginal = useCallback(
    async (startFrac = 0) => {
      const audio = originalAudioRef.current;
      if (!audio || !originalUrlRef.current) return;
      stopPlayback();
      const id = playIdRef.current;
      busyRef.current = true;
      setStatus("speaking");
      setIsPlaying(true);
      await vadRef.current?.pause();
      const duration = originalDuration();
      if (duration)
        audio.currentTime = Math.min(duration - 0.01, startFrac * duration);
      try {
        await audio.play();
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
        await finishPlayback(id);
        return;
      }
      // Poll rather than lean on timeupdate, which fires a few times a second
      // and leaves the bar visibly stepping.
      const tick = () => {
        if (playIdRef.current !== id) return;
        const d = originalDuration();
        if (d) updateFrac(Math.min(1, audio.currentTime / d));
        if (audio.ended) {
          updateFrac(1);
          void finishPlayback(id);
        } else if (audio.paused) {
          void finishPlayback(id);
        } else {
          requestAnimationFrame(tick);
        }
      };
      requestAnimationFrame(tick);
    },
    [stopPlayback, finishPlayback, originalDuration, updateFrac],
  );

  /** Remember the audio a response was made from, so it can be replayed. */
  const setOriginal = useCallback(
    (url: string) => {
      const previous = originalUrlRef.current;
      originalUrlRef.current = url;
      const audio = originalAudioRef.current;
      if (audio) {
        audio.pause();
        audio.src = url;
        audio.load(); // so duration is known before the first play
      }
      // Only after the element has let go of it.
      if (previous) URL.revokeObjectURL(previous);
      setHasOriginal(true);

      // Decode a second copy for the scrub preview. Straight off the object URL
      // rather than threading the blob through every caller; the token guards
      // against a slow decode landing after the next original has replaced it.
      const token = ++originalTokenRef.current;
      originalBufferRef.current = null;
      void (async () => {
        const ctx = trombone.audioContext();
        if (!ctx) return;
        try {
          const bytes = await (await fetch(url)).arrayBuffer();
          const buffer = await ctx.decodeAudioData(bytes);
          if (originalTokenRef.current === token)
            originalBufferRef.current = buffer;
        } catch {
          // Only the scrub preview depends on this; playback goes through the
          // media element either way, so a failure here stays silent.
        }
      })();
    },
    [trombone],
  );

  const remember = useCallback((recording: Recording) => {
    const history = historyRef.current;
    history.push(recording);
    if (history.length > MAX_HISTORY)
      history.splice(0, history.length - MAX_HISTORY);
    setHistoryCount(history.length);
  }, []);

  /** One zip of every input/output pair this session. */
  const downloadHistory = useCallback(async () => {
    const entries = await Promise.all(
      historyRef.current.flatMap((recording, i) => {
        const stem = `${String(i + 1).padStart(2, "0")}-${recording.kind}`;
        const inputExt = recording.input.type === "audio/mpeg" ? "mp3" : "wav";
        const files: Array<[string, Blob]> = [
          [`${stem}-input.${inputExt}`, recording.input],
        ];
        // Absent for a precomputed clip — see Recording.output.
        if (recording.output)
          files.push([`${stem}-output.wav`, recording.output]);
        return files.map(([name, blob]) =>
          blob.arrayBuffer().then((b) => ({ name, bytes: new Uint8Array(b) })),
        );
      }),
    );
    downloadBlob(makeZip(entries), "samuel-session.zip");
  }, []);

  const onUtterance = useCallback(
    async (audio: Float32Array) => {
      if (busyRef.current) return;
      busyRef.current = true;
      processingRef.current = true;
      setStatus("processing");
      try {
        const { response, inputUrl, inputBlob } =
          await synthesizeUtterance(audio);
        lastResponse.current = response;
        setOriginal(inputUrl);
        remember({
          kind: "mic",
          input: inputBlob,
          output: response.synth_audio_b64
            ? wavBlob(response.synth_audio_b64)
            : null,
        });
        setViewResponse(response);
        processingRef.current = false;
        // A fresh imitation is what you want to hear, whatever the toggle was
        // left on.
        sourceRef.current = "samuel";
        setSource("samuel");
        await playSamuel(response);
      } catch (e) {
        busyRef.current = false;
        processingRef.current = false;
        setError(e instanceof Error ? e.message : String(e));
        await restoreMic();
      }
    },
    [playSamuel, restoreMic, setOriginal, remember],
  );

  const startMic = useCallback(async () => {
    setError(null);
    try {
      const current = await fetchHealth();
      if (!current) {
        throw new Error(
          "Model backend unreachable — run: uv run --extra server uvicorn samuel.server:app --port 8000",
        );
      }
      setHealth(current); // also picks up a checkpoint swap since page load
      await trombone.resume(); // we're in a user gesture

      if (!vadRef.current) {
        const { MicVAD } = await import("@ricky0123/vad-web");
        // vad-web's default getStream/resumeStream hardcode all three
        // processing flags on; ours re-read the ref, and since pause() stops
        // the tracks and start() re-acquires, a toggle lands on the next
        // listening cycle without rebuilding the VAD.
        const getStream = () =>
          navigator.mediaDevices.getUserMedia({
            audio: { channelCount: 1, ...micProcessingRef.current },
          });
        vadRef.current = await MicVAD.new({
          model: "v5",
          baseAssetPath: "/vad/",
          onnxWASMBasePath: "/vad/",
          getStream,
          resumeStream: getStream,
          redemptionMs: REDEMPTION_MS,
          preSpeechPadMs: 150, // default 800ms puts noticeable silence before speech
          positiveSpeechThreshold: POSITIVE_SPEECH_THRESHOLD,
          negativeSpeechThreshold: NEGATIVE_SPEECH_THRESHOLD,
          onSpeechStart: () => {
            speakingRef.current = true;
            if (!busyRef.current && micOnRef.current) setStatus("recording");
          },
          onVADMisfire: () => {
            speakingRef.current = false;
            levelStore.set(0);
            if (!busyRef.current && micOnRef.current) setStatus("listening");
          },
          onSpeechEnd: (audio) => {
            speakingRef.current = false;
            showHeard(false);
            levelStore.set(0);
            void onUtterance(audio);
          },
          onFrameProcessed: ({ isSpeech }, frame) => {
            if (busyRef.current || !micOnRef.current) return;
            if (isSpeech >= POSITIVE_SPEECH_THRESHOLD) showHeard(true);
            else if (isSpeech < NEGATIVE_SPEECH_THRESHOLD) showHeard(false);
            // Hold the level through the redemption window: left live it would
            // collapse to nothing in the silence, leaving nothing to drain.
            const redeeming = speakingRef.current && !heardRef.current;
            if (!redeeming) levelStore.set(levelToSlots(frame));
          },
        });
      }
      micOnRef.current = true;
      setMicOn(true);
      await vadRef.current.start();
      setStatus("listening");
    } catch (e) {
      setError(micErrorMessage(e));
      micOnRef.current = false;
      setMicOn(false);
      setStatus(vadRef.current ? "muted" : "idle");
    }
  }, [trombone, onUtterance, showHeard, levelStore]);

  /** Turn the mic off. `submit` sends a half-spoken utterance rather than
   * binning it — true from the button, since people press it meaning "I'm
   * done"; false when the tab is left, where an unasked-for answer is worse. */
  const stopMic = useCallback(
    async (submit = false) => {
      micOnRef.current = false;
      setMicOn(false);
      showHeard(false);
      speakingRef.current = false;
      levelStore.set(0);
      const vad = vadRef.current;
      // Scoped to this one pause: the others (playback starting, a
      // mic-processing toggle) must keep discarding, or the response would feed
      // itself back in as a new utterance.
      if (submit) vad?.setOptions({ submitUserSpeechOnPause: true });
      await vad?.pause(); // fires onSpeechEnd synchronously if it had one
      if (submit) vad?.setOptions({ submitUserSpeechOnPause: false });
      // onUtterance has already claimed busyRef by now if something was sent.
      if (!busyRef.current) setStatus("muted");
    },
    [showHeard, levelStore],
  );

  // Leaving shouldn't leave a live mic behind: mute when the tab is hidden or
  // the window loses focus (another window on top still hears you), and stay
  // muted on return so coming back is an explicit user gesture.
  useEffect(() => {
    const leave = () => {
      if (micOnRef.current) void stopMic();
    };
    const onVisibility = () => {
      if (document.hidden) leave();
    };
    document.addEventListener("visibilitychange", onVisibility);
    window.addEventListener("blur", leave);
    return () => {
      document.removeEventListener("visibilitychange", onVisibility);
      window.removeEventListener("blur", leave);
    };
  }, [stopMic]);

  /** Flip one mic-processing flag. If we're listening right now, cycle the
   * stream so it takes effect immediately rather than after the next
   * utterance. */
  const toggleMicProcessing = useCallback(async (key: keyof MicProcessing) => {
    const next = {
      ...micProcessingRef.current,
      [key]: !micProcessingRef.current[key],
    };
    micProcessingRef.current = next;
    setMicProcessing(next);

    const vad = vadRef.current;
    if (!vad || busyRef.current || !micOnRef.current || !vad.listening) return;
    try {
      await vad.pause();
      await vad.start();
    } catch (e) {
      setError(micErrorMessage(e));
    }
  }, []);

  /** Mimic one of the pre-recorded clips. Interrupts anything playing — only a
   * request already in flight to the model holds it off. */
  const playClip = useCallback(
    async (name: string) => {
      const clip = clips.find((c) => c.name === name);
      if (!clip || processingRef.current) return;
      stopPlayback();
      busyRef.current = true;
      processingRef.current = true;
      setError(null);
      setPlayedClip(name);
      setStatus("processing");
      await vadRef.current?.pause();
      try {
        await trombone.resume(); // we're in a user gesture
        console.log(`${clip.name}: ${clip.source} @${clip.offset_s}s`);
        const { response, inputUrl, inputBlob, precomputed } =
          await synthesizeDatasetClip(clip, health?.model_fingerprint);
        // A committed answer is there instantly; hold it for a beat so the
        // clips behave like the mic rather than firing on mousedown.
        if (precomputed) await fakeThinking();
        lastResponse.current = response;
        setOriginal(inputUrl);
        remember({
          kind: "dataset",
          input: inputBlob,
          output: response.synth_audio_b64
            ? wavBlob(response.synth_audio_b64)
            : null,
        });
        setViewResponse(response);
        processingRef.current = false;
        // A new clip is a new thing to listen to: back to the imitation, from
        // the top.
        sourceRef.current = "samuel";
        setSource("samuel");
        await playSamuel(response);
      } catch (e) {
        busyRef.current = false;
        processingRef.current = false;
        setError(e instanceof Error ? e.message : String(e));
        await restoreMic();
      }
    },
    [
      clips,
      health,
      trombone,
      stopPlayback,
      playSamuel,
      restoreMic,
      setOriginal,
      remember,
    ],
  );

  /** Start whichever source is selected, from `frac`. */
  const playFrom = useCallback(
    async (which: Source, frac: number) => {
      setError(null);
      if (which === "original") {
        await playOriginal(frac);
      } else if (lastResponse.current) {
        await playSamuel(lastResponse.current, frac);
      }
    },
    [playOriginal, playSamuel],
  );

  /** Play from the scrub position (or the start, if at the end); pause if
   * already playing. */
  const togglePlay = useCallback(async () => {
    if (isPlaying) {
      stopPlayback();
      busyRef.current = false;
      await restoreMic();
      return;
    }
    if (processingRef.current) return;
    const from = scrubFracRef.current >= 0.995 ? 0 : scrubFracRef.current;
    await playFrom(sourceRef.current, from);
  }, [isPlaying, stopPlayback, restoreMic, playFrom]);

  /** Flip between the imitation and the original. Playback follows: the two are
   * the same utterance, so the position carries over. */
  const toggleSource = useCallback(() => {
    const next: Source = sourceRef.current === "samuel" ? "original" : "samuel";
    if (next === "original" && !originalUrlRef.current) return;
    sourceRef.current = next;
    setSource(next);
    if (isPlaying) void playFrom(next, scrubFracRef.current);
  }, [isPlaying, playFrom]);

  const onScrub = useCallback(
    (frac: number) => {
      updateFrac(frac);
      if (!scrubbingRef.current) {
        scrubbingRef.current = true;
        void vadRef.current?.pause(); // scrubbing makes sound; don't feed it back
        setStatus("speaking");
      }
      if (sourceRef.current === "original") {
        const audio = originalAudioRef.current;
        const duration = originalDuration();
        if (audio && duration)
          audio.currentTime = Math.min(duration - 0.01, frac * duration);
        // Mid-playback the element is already the sound, and dragging just
        // seeks it. Stopped, the grains are the sound.
        if (!audio || audio.paused) playGrain(frac);
        return;
      }
      const response = lastResponse.current;
      if (!response) return;
      trombone.scrub(response, frac);
    },
    [trombone, updateFrac, originalDuration, playGrain],
  );

  const onScrubEnd = useCallback(async () => {
    if (!scrubbingRef.current) return;
    scrubbingRef.current = false;
    trombone.endScrub();
    stopGrain();
    if (!busyRef.current) await restoreMic();
  }, [trombone, stopGrain, restoreMic]);

  // "recording" only means the VAD currently hears *something* — a cough or a
  // door must not disable the buttons, or every other click gets swallowed
  // while it flaps. busyRef is the real guard, and starting a playback pauses
  // the VAD, which discards the in-flight segment. Playback is *not* a reason
  // to disable anything: every control below interrupts it cleanly.
  const notBusy = insecure === null && status !== "processing";
  const hasSource = source === "original" ? hasOriginal : viewResponse !== null;
  // Is there anything at all to play, on either source? Not `hasSource`:
  // flipping the switch to a side that happens to be empty shouldn't drain the
  // colour out of the switch you'd flip back with.
  const transportReady = hasOriginal || viewResponse !== null;
  // A live mic and the transport are mutually exclusive: scrubbing sustains the
  // synth into your own microphone. Turning the mic off hands the page over.
  const canPlay = hasSource && notBusy && !micOn;
  const canScrub = hasSource && status !== "processing" && !micOn;
  // Mid-utterance but hearing nothing, i.e. the redemption window: "recording"
  // is set on speech start and cleared only when the segment ends or misfires.
  const pending = status === "recording" && !heard;
  // The tract is drawn grey until an audio input is picked: the mic is on, or a
  // pre-recorded clip has come back from the model.
  const tractActive = micOn || viewResponse !== null;
  // At most one box is lit, and it's wherever the interesting thing is: pick an
  // input, watch the tract while it thinks and answers, then the transport
  // takes over — unless the mic is still on, in which case it never left the
  // input box. The original doesn't move the tract, so it stays on the
  // transport. "tract" lights nothing: the tract has no box, it just moves.
  const section: Section =
    status === "processing" || (isPlaying && source === "samuel")
      ? "tract"
      : micOn
        ? "input"
        : tractActive
          ? "playback"
          : "input";

  return (
    <main className="flex flex-1 flex-wrap items-start gap-8 p-8">
      {/* Left: everything you operate. Right: the thing you look at. */}
      <div className="flex min-w-md max-w-md flex-1 flex-col items-start gap-4">
        <header>
          <h1 className="text-5xl font-bold text-highlight-600">Samuel</h1>
        </header>
        {/* TODO: write the real intro. */}
        <p className="text-neutral-600">
          Samuel is a model that learns to control this silly mouth on the right
          to mimic speech. Say something and it will parrot after you, or if
          you&apos;re shy, try one of the pre-made clips.
        </p>
        <p className="text-neutral-600">
          The mouth itself is Pink Trombone, a project originally by{" "}
          <TextLink href="https://dood.al/pinktrombone/" muted>
            Neil Thapen
          </TextLink>
          , described by him as &quot;bare-handed speech synthesis&quot;.
          It&apos;s the{" "}
          <TextLink href="https://www.foddy.net/Athletics.html" muted>
            QWOP
          </TextLink>{" "}
          of text-to-speech.
        </p>
        <p className="text-neutral-600">
          Made by{" "}
          <TextLink href="https://vvolhejn.com">Václav Volhejn</TextLink>. Code{" "}
          <TextLink href="https://github.com/vvolhejn/samuel">
            on GitHub
          </TextLink>
          .
        </p>
        <div
          className={`flex w-full flex-col gap-2 p-3 ${sectionBox(section === "input")}`}
        >
          {micOn ? (
            <div
              className={`flex flex-col justify-between gap-2 ${INPUT_MIN_H}`}
            >
              <div className="flex min-w-0 flex-col gap-1">
                <span className="flex items-center gap-2 text-sm font-semibold text-highlight-700">
                  <span
                    aria-hidden
                    className="h-2.5 w-2.5 shrink-0 rounded-full bg-highlight-600"
                  />
                  {micHeading(status)}
                </span>
                <span className="text-neutral-500">
                  Samuel answers after you pause
                </span>
              </div>

              <div className="flex items-end justify-between gap-4">
                <LevelMeter
                  store={levelStore}
                  active={heard}
                  pending={pending}
                />

                <button
                  onClick={() => void stopMic(true)}
                  title="Stop listening and hand the page back to the playback controls"
                  className="rounded-full border border-highlight-300 px-4 py-1.5 text-sm font-medium text-highlight-700 hover:bg-highlight-50"
                >
                  Turn off
                </button>
              </div>
            </div>
          ) : (
            /* Two columns: talk to it on the left, or pick a canned clip on the
               right. Wraps to one column when there isn't room for both. */
            <div
              className={`flex flex-wrap items-center justify-around gap-6 ${INPUT_MIN_H}`}
            >
              <div className="flex min-w-0 flex-col items-center gap-1.5">
                <button
                  onClick={() => void startMic()}
                  disabled={insecure !== null}
                  title={
                    insecure
                      ? "Unavailable on an insecure origin"
                      : "Listen continuously and mimic every utterance"
                  }
                  className="rounded-full bg-highlight-600 px-4 py-1.5 text-sm font-semibold text-white hover:bg-highlight-700 disabled:opacity-40 disabled:hover:bg-highlight-600"
                >
                  Microphone
                </button>

                {/* Reassurance, not an action: recordings go to the server only
                    to be mimicked back, and nothing is written to disk there.
                    Only the "self-host" escape hatch is clickable. */}
                <span
                  className="text-center text-xs text-neutral-500"
                  title="Recordings are sent to the server only to be mimicked back; they are never written to disk."
                >
                  Your audio is not stored.
                  <br />
                  <TextLink href="https://github.com/vvolhejn/samuel" muted>
                    Self-host
                  </TextLink>{" "}
                  if you don&apos;t trust me
                </span>
              </div>

              <div className="flex shrink-0 flex-col gap-1.5">
                <span className="text-sm text-neutral-500">
                  or use pre-recorded audio
                </span>

                {/* One button per committed clip, numbered rather than named:
                    which recording is which only matters once you've heard
                    them. */}
                <div className="grid grid-cols-3 gap-1.5">
                  {clips.map((clip, i) => (
                    <button
                      key={clip.name}
                      onClick={() => void playClip(clip.name)}
                      disabled={!notBusy}
                      title={
                        insecure
                          ? "Unavailable on an insecure origin"
                          : `Mimic ${clip.duration_s.toFixed(0)}s of held-out speech`
                      }
                      className={
                        clip.name === playedClip
                          ? "rounded-md bg-highlight-600 px-3 py-1 text-sm font-semibold text-white disabled:opacity-40"
                          : /* White like the Play pill, not transparent: these
                             keep their own ground when the box behind them
                             lights up. */
                            "rounded-md border border-highlight-300 bg-white px-3 py-1 text-sm font-medium text-highlight-700 hover:bg-highlight-50 disabled:opacity-40 disabled:hover:bg-white"
                      }
                    >
                      {i + 1}
                    </button>
                  ))}
                </div>
              </div>
            </div>
          )}
        </div>
        {/* Same grey-until-there's-something rule as the tract, done in one
            place: with nothing to play, every accent inside the transport —
            the Play pill, the source switch, the scrubber — desaturates
            together rather than each control needing its own dead colour. */}
        <div
          className={`flex w-full flex-col gap-3 p-3 ${sectionBox(section === "playback")} ${
            transportReady ? "" : "grayscale"
          }`}
        >
          <div className="flex items-center gap-4">
            <button
              onClick={() => void togglePlay()}
              disabled={!canPlay && !isPlaying}
              title={
                source === "original"
                  ? "Play the audio the model heard, at its recorded level"
                  : "Play the model's imitation from the scrub position"
              }
              /* White, not accent-filled: the solid pill is the microphone's,
                 and two of them made the transport shout for a click it doesn't
                 need. Explicitly white rather than transparent so it keeps its
                 own ground when the box around it lights up. */
              className="w-24 shrink-0 rounded-full border border-highlight-300 bg-white py-1.5 text-sm font-medium text-highlight-700 hover:bg-highlight-50 disabled:opacity-40 disabled:hover:bg-white"
            >
              {isPlaying ? "Pause" : "Play"}
            </button>

            {/* Which of the two takes of the same utterance you're listening to.
                Flipping it mid-playback carries the position across. */}
            <div className="flex items-center gap-2">
              <button
                role="switch"
                aria-checked={source === "original"}
                aria-label="Play the original instead of the imitation"
                onClick={toggleSource}
                disabled={!hasOriginal}
                title="Switch between the model's imitation and the audio it heard"
                className={`relative h-6 w-11 shrink-0 rounded-full transition-colors disabled:opacity-40 ${
                  source === "original" ? "bg-neutral-500" : "bg-highlight-600"
                }`}
              >
                <span
                  aria-hidden
                  className={`absolute top-0.5 left-0.5 h-5 w-5 rounded-full bg-white shadow transition-transform ${
                    source === "original" ? "translate-x-5" : "translate-x-0"
                  }`}
                />
              </button>
              <span className="text-sm text-neutral-600">
                {source === "original" ? "Original" : "Imitated"}
              </span>
            </div>
          </div>

          <input
            type="range"
            min={0}
            max={1000}
            value={Math.round(scrubFrac * 1000)}
            disabled={!canScrub}
            aria-label="Scrub through the current audio"
            onPointerDown={() => onScrub(scrubFrac)}
            onChange={(e) => onScrub(Number(e.currentTarget.value) / 1000)}
            onPointerUp={() => void onScrubEnd()}
            onKeyUp={() => void onScrubEnd()}
            onBlur={() => void onScrubEnd()}
            className="w-full accent-highlight-600 disabled:opacity-40"
          />
        </div>
        {insecure && (
          <p className="rounded-lg border border-amber-300 bg-amber-50 p-3 text-sm text-amber-900">
            {insecure}
          </p>
        )}
        {error && <p className="text-sm text-red-600">{error}</p>}
        {/* Dev-only: inlined at build time, so the deployed bundle has no
            debug UI at all. */}
        {process.env.NODE_ENV === "development" && (
          <DebugPanel
            open={debugOpen}
            onToggle={() => setDebugOpen((v) => !v)}
            health={health}
            precomputed={precomputed}
            response={viewResponse}
            frac={scrubFrac}
            micProcessing={micProcessing}
            onToggleMicProcessing={(key) => void toggleMicProcessing(key)}
            historyCount={historyCount}
            onDownloadHistory={() => void downloadHistory()}
          />
        )}
      </div>

      {/* The element's canvases are a fixed 600×500 anchored top-left, so
          size the host to match rather than letting it stretch. Left out
          entirely on an insecure origin, where it would stay blank: the synth
          it draws is never started there. No box of its own either: a tract
          that's moving already announces itself, and the greyed-out state
          covers the rest. */}
      {insecure === null && (
        <div className="relative shrink-0 p-2">
          <pink-trombone
            className="block h-[500px] w-[600px]"
            inactive={tractActive ? undefined : "true"}
          />
          {/* Over the tract rather than beside the buttons: the wait is the one
              moment nothing else on the page moves, and tucked next to the
              controls it was easy to miss. */}
          {status === "processing" && (
            <div className="absolute inset-0 z-10 flex items-center justify-center rounded-xl bg-white/70">
              {/* Solid strip behind the words: the tract is all thin dark lines,
                  and the wash alone doesn't hide enough of them to read over. */}
              <span className="w-full bg-white py-2 text-center text-2xl text-neutral-600">
                Thinking
                <Ellipsis />
              </span>
            </div>
          )}
        </div>
      )}
    </main>
  );
}
