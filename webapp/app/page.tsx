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
  synthesizeDatasetClip,
  synthesizeUtterance,
  wavBlob,
  DatasetClip,
  HealthResponse,
  SynthResponse,
} from "@/lib/audio";
import { insecureContextMessage, micErrorMessage } from "@/lib/secureContext";
import { downloadBlob, makeZip } from "@/lib/zip";
import { usePinkTrombone } from "@/lib/usePinkTrombone";
import type { PinkTromboneElement } from "@/types/pink-trombone";

type Status =
  | "idle"
  | "listening"
  | "recording"
  | "processing"
  | "speaking"
  | "muted";

/** Shown next to the transport controls, each followed by an animated ellipsis
 * (so no trailing "…" here). `null` states say nothing: the Record button
 * already reads as "the mic is off". */
const STATUS_LABEL: Record<Status, string | null> = {
  idle: null,
  muted: null,
  listening: "Listening",
  recording: "Hearing you",
  processing: "Thinking",
  speaking: "Speaking back",
};

const SPEEDS = [0.25, 0.5, 1] as const;

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
  /** WAV of the model's output, rendered by the Python synth. */
  output: Blob;
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
 * stands, is every link here. */
function TextLink({
  href,
  children,
}: {
  href: string;
  children: React.ReactNode;
}) {
  const external = /^(https?:)?\/\//.test(href);
  return (
    <a
      href={href}
      target={external ? "_blank" : undefined}
      rel={external ? "noreferrer" : undefined}
      className="text-highlight-600 underline decoration-dotted hover:text-highlight-700"
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
  const [speed, setSpeed] = useState<number>(1);
  /** Playback/scrub position within the last response, in [0, 1]. */
  const [scrubFrac, setScrubFrac] = useState(0);
  const [micProcessing, setMicProcessing] = useState<MicProcessing>(
    MIC_PROCESSING_DEFAULTS,
  );
  /** Is there audio the model heard, i.e. can "Original" play anything? */
  const [hasOriginal, setHasOriginal] = useState(false);
  /** The original is playing, so its segment of the pill offers Pause. */
  const [playingOriginal, setPlayingOriginal] = useState(false);
  /** User-intended mic state — the Record/Stop toggle. Mirrors micOnRef; the
   * status alone can't stand in for it (a dataset clip is "speaking" too). */
  const [micOn, setMicOn] = useState(false);
  /** Pre-recorded clips shipped with the app, and the last one played (which
   * is the button left filled in). */
  const [clips, setClips] = useState<DatasetClip[]>([]);
  const [playedClip, setPlayedClip] = useState<string>("");
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
  const debugAudioRef = useRef<HTMLAudioElement | null>(null);
  const busyRef = useRef(false); // ignore VAD events while processing/speaking
  const micOnRef = useRef(false); // user-intended mic state (start/mute toggle)
  const speedRef = useRef(1); // playResponse is a stable callback; read via ref
  const scrubbingRef = useRef(false); // pointer is down on the scrub bar
  // Read by getStream/resumeStream, which vad-web calls on every start().
  const micProcessingRef = useRef<MicProcessing>(MIC_PROCESSING_DEFAULTS);

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

  // The pre-recorded clips, one button each.
  useEffect(() => {
    void fetchDatasetClips().then(setClips);
  }, []);

  /** Resume VAD only if the user hasn't muted the mic. */
  const restoreMic = useCallback(async () => {
    if (scrubbingRef.current) return; // the scrub owns the synth until pointer-up
    if (micOnRef.current) {
      await vadRef.current?.start();
      setStatus("listening");
    } else {
      setStatus(vadRef.current ? "muted" : "idle");
    }
  }, []);

  const playResponse = useCallback(
    async (response: SynthResponse, startFrac = 0) => {
      setStatus("speaking");
      setIsPlaying(true);
      await vadRef.current?.pause(); // don't let the synth retrigger the mic
      try {
        await trombone.speak(response, {
          speed: speedRef.current,
          startFrac,
          onProgress: setScrubFrac,
        });
      } finally {
        setIsPlaying(false);
        busyRef.current = false;
        await restoreMic();
      }
    },
    [trombone, restoreMic],
  );

  /** Remember the audio a response was made from, so it can be replayed. */
  const setOriginal = useCallback((url: string) => {
    if (originalUrlRef.current) URL.revokeObjectURL(originalUrlRef.current);
    originalUrlRef.current = url;
    setHasOriginal(true);
  }, []);

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
          [`${stem}-output.wav`, recording.output],
        ];
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
      setStatus("processing");
      try {
        const { response, inputUrl, inputBlob } =
          await synthesizeUtterance(audio);
        lastResponse.current = response;
        setOriginal(inputUrl);
        remember({
          kind: "mic",
          input: inputBlob,
          output: wavBlob(response.synth_audio_b64),
        });
        setViewResponse(response);
        await playResponse(response);
      } catch (e) {
        busyRef.current = false;
        setError(e instanceof Error ? e.message : String(e));
        await restoreMic();
      }
    },
    [playResponse, restoreMic, setOriginal, remember],
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
          redemptionMs: 800,
          preSpeechPadMs: 150, // default 800ms puts noticeable silence before speech
          onSpeechStart: () => {
            if (!busyRef.current && micOnRef.current) setStatus("recording");
          },
          onVADMisfire: () => {
            if (!busyRef.current && micOnRef.current) setStatus("listening");
          },
          onSpeechEnd: (audio) => void onUtterance(audio),
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
  }, [trombone, onUtterance]);

  const stopMic = useCallback(async () => {
    micOnRef.current = false;
    setMicOn(false);
    await vadRef.current?.pause();
    if (!busyRef.current) setStatus("muted");
  }, []);

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

  /** Play a WAV object URL to completion through a throwaway <audio> element.
   *
   * A fresh element per call, and listeners attached before play(): with one
   * shared element, a second playback overwrote the first's ended/error
   * handlers, so the first promise never settled and busyRef stayed true —
   * every button dead until reload. Resolving on `pause` covers both the user
   * hitting Pause and a playback cut short by the device switching under us
   * (Chrome does that when a mic stream with echo cancellation comes and
   * goes) — no pause event can reach us before play() is called. */
  const playUrl = useCallback(async (url: string) => {
    debugAudioRef.current?.pause(); // supersede anything still running
    const audio = new Audio();
    debugAudioRef.current = audio;
    await new Promise<void>((resolve, reject) => {
      let done = false;
      const finish = () => {
        if (!done) {
          done = true;
          resolve();
        }
      };
      audio.addEventListener("ended", finish, { once: true });
      audio.addEventListener("error", finish, { once: true });
      audio.addEventListener("pause", finish, { once: true }); // truncated; don't hang
      audio.src = url;
      audio.play().catch((e: DOMException) => {
        if (e.name === "AbortError")
          finish(); // superseded by another play
        else reject(e);
      });
    });
  }, []);

  /** Mimic one of the pre-recorded clips. */
  const playClip = useCallback(
    async (name: string) => {
      const clip = clips.find((c) => c.name === name);
      if (!clip || busyRef.current) return;
      busyRef.current = true;
      setError(null);
      setPlayedClip(name);
      setStatus("processing");
      await vadRef.current?.pause();
      try {
        await trombone.resume(); // we're in a user gesture
        console.log(`${clip.name}: ${clip.source} @${clip.offset_s}s`);
        const { response, inputUrl, inputBlob } =
          await synthesizeDatasetClip(clip);
        lastResponse.current = response;
        setOriginal(inputUrl);
        remember({
          kind: "dataset",
          input: inputBlob,
          output: wavBlob(response.synth_audio_b64),
        });
        setViewResponse(response);
        await playResponse(response);
      } catch (e) {
        busyRef.current = false;
        setError(e instanceof Error ? e.message : String(e));
        await restoreMic();
      }
    },
    [clips, trombone, playResponse, restoreMic, setOriginal, remember],
  );

  /** Play from the scrub position (or the start, if at the end); pause if
   * already playing. */
  const togglePlay = useCallback(async () => {
    if (isPlaying) {
      trombone.stop(); // settles the in-flight speak(), which restores the mic
      return;
    }
    if (!lastResponse.current || busyRef.current) return;
    busyRef.current = true;
    setError(null);
    const from = scrubFrac >= 0.995 ? 0 : scrubFrac;
    try {
      await playResponse(lastResponse.current, from);
    } catch (e) {
      busyRef.current = false;
      setError(e instanceof Error ? e.message : String(e));
      await restoreMic();
    }
  }, [isPlaying, scrubFrac, trombone, playResponse, restoreMic]);

  /** Speed applies to the next play, and to one already in flight. */
  const changeSpeed = useCallback(
    (s: number) => {
      speedRef.current = s;
      setSpeed(s);
      trombone.setPlaybackSpeed(s);
    },
    [trombone],
  );

  const onScrub = useCallback(
    (frac: number) => {
      const response = lastResponse.current;
      if (!response) return;
      if (!scrubbingRef.current) {
        scrubbingRef.current = true;
        void vadRef.current?.pause(); // the sustained synth would trigger it
        setStatus("speaking");
      }
      setScrubFrac(frac);
      trombone.scrub(response, frac);
    },
    [trombone],
  );

  const onScrubEnd = useCallback(async () => {
    if (!scrubbingRef.current) return;
    scrubbingRef.current = false;
    trombone.endScrub();
    if (!busyRef.current) await restoreMic();
  }, [trombone, restoreMic]);

  /** Play the audio the model heard — your trimmed recording, or the dataset
   * clip — holding the mic and busy flag like a real response. Raw level: the
   * RMS normalisation happens server-side. Pause stops it for good; the next
   * click starts over, since there is nothing to scrub here. */
  const toggleOriginal = useCallback(async () => {
    if (playingOriginal) {
      debugAudioRef.current?.pause(); // settles the in-flight playUrl()
      return;
    }
    const url = originalUrlRef.current;
    if (!url || busyRef.current) return;
    busyRef.current = true;
    setError(null);
    setStatus("speaking");
    setPlayingOriginal(true);
    await vadRef.current?.pause();
    try {
      await playUrl(url);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setPlayingOriginal(false);
      busyRef.current = false;
      await restoreMic();
    }
  }, [playingOriginal, playUrl, restoreMic]);

  // "recording" only means the VAD currently hears *something* — a cough or a
  // door must not disable the buttons, or every other click gets swallowed
  // while it flaps. busyRef is the real guard, and starting a playback pauses
  // the VAD, which discards the in-flight segment.
  const notBusy =
    insecure === null && status !== "processing" && status !== "speaking";
  const canReplay = viewResponse !== null && notBusy;
  const canPlayOriginal = hasOriginal && notBusy;
  const canScrub = viewResponse !== null && status !== "processing";
  // The tract is drawn grey until an audio input is picked: the mic is on, or a
  // pre-recorded clip has come back from the model.
  const tractActive = micOn || viewResponse !== null;

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
          <TextLink href="https://dood.al/pinktrombone/">Neil Thapen</TextLink>,
          described by
          him as &quot;bare-handed speech synthesis&quot;. It&apos;s the QWOP of
          text-to-speech.
        </p>
        <p className="text-neutral-600">
          Made by <TextLink href="https://vvolhejn.com">Václav Volhejn</TextLink>
          .
        </p>
        <div className="flex w-full flex-col gap-2">
          <div className="font-bold text-neutral-500">Audio input</div>

          <div className="flex flex-wrap items-center gap-3">
            <button
              onClick={() => void (micOn ? stopMic() : startMic())}
              disabled={insecure !== null}
              title={
                insecure
                  ? "Unavailable on an insecure origin"
                  : micOn
                    ? "Stop listening"
                    : "Listen continuously and mimic every utterance"
              }
              className={
                micOn
                  ? "rounded-full border border-highlight-300 px-4 py-1.5 text-sm font-medium text-highlight-700 hover:bg-highlight-50 disabled:opacity-40"
                  : "rounded-full bg-highlight-600 px-4 py-1.5 text-sm font-semibold text-white hover:bg-highlight-700 disabled:opacity-40 disabled:hover:bg-highlight-600"
              }
            >
              {micOn ? "Stop" : "Microphone"}
            </button>

            <span className="text-sm text-neutral-500">
              or use pre-recorded audio
            </span>

            {/* One button per committed clip, numbered rather than named: which
                recording is which only matters once you've heard them. */}
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
                      ? "rounded-md bg-sky-600 px-3 py-1 text-sm font-semibold text-white disabled:opacity-40"
                      : "rounded-md border border-sky-300 px-3 py-1 text-sm font-medium text-sky-700 hover:bg-sky-50 disabled:opacity-40 disabled:hover:bg-transparent"
                  }
                >
                  {i + 1}
                </button>
              ))}
            </div>

            {STATUS_LABEL[status] && (
              <span className="text-sm text-neutral-500">
                {STATUS_LABEL[status]}
                <Ellipsis />
              </span>
            )}
          </div>
        </div>
        <div className="flex w-full flex-col gap-3">
          <div className="flex items-center gap-3">
            {/* Own chevron: the native one is glued to the border box, so padding
                can't give it any room inside the pill. */}
            <div className="relative shrink-0">
              <select
                value={speed}
                onChange={(e) => changeSpeed(Number(e.currentTarget.value))}
                title="Playback speed"
                className="appearance-none rounded-full border border-neutral-300 py-1 pr-7 pl-2.5 text-xs font-medium text-neutral-600 hover:border-highlight-300 hover:text-highlight-700"
              >
                {SPEEDS.map((s) => (
                  <option key={s} value={s}>
                    {s}× speed
                  </option>
                ))}
              </select>
              <span
                aria-hidden
                className="pointer-events-none absolute inset-y-0 right-2.5 flex items-center text-[0.5rem] text-neutral-500"
              >
                ▼
              </span>
            </div>

            {/* One pill, two sources: the model's imitation and the audio it was
                made from. Whichever is playing offers Pause; the other is out of
                reach until it stops, since they'd fight over the mic and busy
                flag. */}
            <div className="flex shrink-0 overflow-hidden rounded-full border border-highlight-300 text-sm">
              <button
                onClick={togglePlay}
                disabled={!canReplay && !isPlaying}
                title="Play the model's imitation from the scrub position"
                className="w-24 bg-highlight-600 py-1.5 font-semibold text-white hover:bg-highlight-700 disabled:opacity-40 disabled:hover:bg-highlight-600"
              >
                {isPlaying ? "Pause" : "Play"}
              </button>
              <button
                onClick={() => void toggleOriginal()}
                disabled={!canPlayOriginal && !playingOriginal}
                title="Play the audio the model heard, at its recorded level"
                className="w-24 border-l border-highlight-300 py-1.5 font-medium text-highlight-700 hover:bg-highlight-50 disabled:opacity-40 disabled:hover:bg-transparent"
              >
                {playingOriginal ? "Pause" : "Original"}
              </button>
            </div>
          </div>

          <div className="flex items-center gap-3">
            <input
              type="range"
              min={0}
              max={1000}
              value={Math.round(scrubFrac * 1000)}
              disabled={!canScrub}
              aria-label="Scrub through the last response"
              onPointerDown={() => onScrub(scrubFrac)}
              onChange={(e) => onScrub(Number(e.currentTarget.value) / 1000)}
              onPointerUp={() => void onScrubEnd()}
              onKeyUp={() => void onScrubEnd()}
              onBlur={() => void onScrubEnd()}
              className="flex-1 accent-highlight-600 disabled:opacity-40"
            />

            <span className="w-12 text-right text-xs tabular-nums text-neutral-500">
              {viewResponse
                ? `${(scrubFrac * viewResponse.duration_s).toFixed(1)}s`
                : "–"}
            </span>
          </div>
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
          it draws is never started there. */}
      {insecure === null && (
        <pink-trombone
          className="block h-[500px] w-[600px] shrink-0"
          inactive={tractActive ? undefined : "true"}
        />
      )}
    </main>
  );
}
