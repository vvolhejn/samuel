"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import type { MicVAD } from "@ricky0123/vad-web";
import {
  backendHealthy,
  fetchDatasetClip,
  synthesizeUtterance,
  synthAudioUrl,
  SynthResponse,
} from "@/lib/audio";
import { usePinkTrombone } from "@/lib/usePinkTrombone";
import type { PinkTromboneElement } from "@/types/pink-trombone";

type Status =
  | "idle"
  | "listening"
  | "recording"
  | "processing"
  | "speaking"
  | "muted";

const STATUS_LABEL: Record<Status, string> = {
  idle: "Mic off — click start and allow the microphone",
  listening: "Listening — say something",
  recording: "Hearing you…",
  processing: "Thinking…",
  speaking: "Speaking back",
  muted: "Mic muted",
};

const STATUS_DOT: Record<Status, string> = {
  idle: "bg-neutral-400",
  listening: "bg-emerald-500",
  recording: "bg-emerald-500 animate-pulse",
  processing: "bg-amber-500 animate-pulse",
  speaking: "bg-fuchsia-400 animate-pulse",
  muted: "bg-neutral-400",
};

const SPEEDS = [0.25, 0.5, 1] as const;

const PANEL_PARAMS: Array<{ key: string; label: string; digits: number }> = [
  { key: "frequency", label: "frequency (Hz)", digits: 1 },
  { key: "voiceness", label: "voiceness", digits: 3 },
  { key: "tongueIndex", label: "tongue index", digits: 2 },
  { key: "tongueDiameter", label: "tongue diameter", digits: 2 },
  { key: "constrictionIndex", label: "constriction index", digits: 2 },
  { key: "constrictionDiameter", label: "constriction diameter", digits: 2 },
];

/** Exact model-output values at the current playback/scrub position. */
function ParamPanel({
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
    <aside className="w-56 shrink-0 self-start rounded-lg border border-neutral-200 bg-neutral-50 p-3 text-xs">
      <div className="mb-2 flex items-baseline justify-between">
        <span className="font-semibold text-neutral-700">Model output</span>
        <span className="tabular-nums text-neutral-400">
          {response ? `frame ${frame + 1}/${nFrames}` : "no clip"}
        </span>
      </div>
      <dl className="space-y-1.5">
        {PANEL_PARAMS.map(({ key, label, digits }) =>
          row(label, response ? response.params[key][frame].toFixed(digits) : "–"),
        )}
        {row("gain", response ? response.gain[frame].toFixed(2) : "–")}
        {row(
          "voiced (pyin)",
          response ? (response.voiced[frame] ? "yes" : "no") : "–",
        )}
      </dl>
    </aside>
  );
}

export default function Home() {
  const [status, setStatus] = useState<Status>("idle");
  const [error, setError] = useState<string | null>(null);
  /** Render-side mirror of lastResponse (refs must not be read in render). */
  const [viewResponse, setViewResponse] = useState<SynthResponse | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [speed, setSpeed] = useState<number>(1);
  /** Playback/scrub position within the last response, in [0, 1]. */
  const [scrubFrac, setScrubFrac] = useState(0);

  const trombone = usePinkTrombone();
  const vadRef = useRef<MicVAD | null>(null);
  const lastResponse = useRef<SynthResponse | null>(null);
  const synthUrlRef = useRef<string | null>(null);
  const debugAudioRef = useRef<HTMLAudioElement | null>(null);
  const busyRef = useRef(false); // ignore VAD events while processing/speaking
  const micOnRef = useRef(false); // user-intended mic state (start/mute toggle)
  const speedRef = useRef(1); // playResponse is a stable callback; read via ref
  const scrubbingRef = useRef(false); // pointer is down on the scrub bar

  // Bring up the synth + tract visualization immediately; the AudioContext
  // stays suspended until the first user gesture (Start).
  useEffect(() => {
    const element =
      document.querySelector<PinkTromboneElement>("pink-trombone");
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

  const onUtterance = useCallback(
    async (audio: Float32Array) => {
      if (busyRef.current) return;
      busyRef.current = true;
      setStatus("processing");
      try {
        const response = await synthesizeUtterance(audio);
        lastResponse.current = response;
        if (synthUrlRef.current) URL.revokeObjectURL(synthUrlRef.current);
        synthUrlRef.current = synthAudioUrl(response);
        setViewResponse(response);
        await playResponse(response);
      } catch (e) {
        busyRef.current = false;
        setError(e instanceof Error ? e.message : String(e));
        await restoreMic();
      }
    },
    [playResponse, restoreMic],
  );

  const startMic = useCallback(async () => {
    setError(null);
    try {
      if (!(await backendHealthy())) {
        throw new Error(
          "Model backend unreachable — run: uv run --extra server uvicorn samuel.server:app --port 8000",
        );
      }
      await trombone.resume(); // we're in a user gesture

      if (!vadRef.current) {
        const { MicVAD } = await import("@ricky0123/vad-web");
        vadRef.current = await MicVAD.new({
          model: "v5",
          baseAssetPath: "/vad/",
          onnxWASMBasePath: "/vad/",
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
      await vadRef.current.start();
      setStatus("listening");
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setStatus(vadRef.current ? "muted" : "idle");
    }
  }, [trombone, onUtterance]);

  const muteMic = useCallback(async () => {
    micOnRef.current = false;
    await vadRef.current?.pause();
    if (!busyRef.current) setStatus("muted");
  }, []);

  /** Play a WAV object URL through the hidden debug <audio> element. */
  const playUrl = useCallback(async (url: string) => {
    const audio = (debugAudioRef.current ??= new Audio());
    audio.src = url;
    await audio.play();
    await new Promise<void>((resolve) => {
      audio.onended = () => resolve();
      audio.onerror = () => resolve();
    });
  }, []);

  const playDatasetClip = useCallback(async () => {
    if (busyRef.current) return;
    busyRef.current = true;
    setError(null);
    setStatus("processing");
    await vadRef.current?.pause();
    try {
      await trombone.resume(); // we're in a user gesture
      const response = await fetchDatasetClip();
      lastResponse.current = response;
      if (synthUrlRef.current) URL.revokeObjectURL(synthUrlRef.current);
      synthUrlRef.current = synthAudioUrl(response);
      setViewResponse(response);
      await playResponse(response);
    } catch (e) {
      busyRef.current = false;
      setError(e instanceof Error ? e.message : String(e));
      await restoreMic();
    }
  }, [trombone, playResponse, restoreMic]);

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

  const playPythonSynth = useCallback(async () => {
    if (!synthUrlRef.current || busyRef.current) return;
    busyRef.current = true;
    setError(null);
    setStatus("speaking");
    await vadRef.current?.pause();
    try {
      await playUrl(synthUrlRef.current);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      busyRef.current = false;
      await restoreMic();
    }
  }, [playUrl, restoreMic]);

  const micActive = status !== "idle";
  const notBusy =
    status === "idle" || status === "listening" || status === "muted";
  const canReplay = viewResponse !== null && notBusy;
  const canScrub = viewResponse !== null && status !== "processing";

  return (
    <main className="flex flex-1 flex-col items-center gap-6 p-8">
      <header className="text-center">
        <h1 className="font-display text-4xl tracking-wide text-fuchsia-600">
          Samuel
        </h1>
        <p className="mt-1 text-sm text-neutral-500">
          Speak — the vocal tract model mimics you.
        </p>
      </header>

      <div className="flex flex-wrap items-center justify-center gap-3">
        <span className="inline-flex items-center gap-2 rounded-full border border-neutral-200 bg-neutral-50 px-3 py-1 text-sm text-neutral-700">
          <span className={`h-2 w-2 rounded-full ${STATUS_DOT[status]}`} />
          {STATUS_LABEL[status]}
        </span>

        {!micActive || status === "muted" ? (
          <button
            onClick={startMic}
            className="rounded-full bg-fuchsia-600 px-4 py-1.5 text-sm font-semibold text-white hover:bg-fuchsia-700"
          >
            {micActive ? "Mic on" : "Start"}
          </button>
        ) : (
          <button
            onClick={muteMic}
            className="rounded-full border border-fuchsia-300 px-4 py-1.5 text-sm font-medium text-fuchsia-700 hover:bg-fuchsia-50"
          >
            Mic off
          </button>
        )}

        <button
          onClick={playDatasetClip}
          disabled={!notBusy}
          title="Mimic a random 10s clip from the training dataset"
          className="rounded-full border border-sky-300 px-4 py-1.5 text-sm font-medium text-sky-700 hover:bg-sky-50 disabled:opacity-40 disabled:hover:bg-transparent"
        >
          Dataset clip
        </button>

        <button
          onClick={playPythonSynth}
          disabled={!canReplay}
          title="Reference resynthesis from the Python synth (debug)"
          className="rounded-full border border-dashed border-neutral-300 px-4 py-1.5 text-sm font-medium text-neutral-700 hover:border-fuchsia-300 hover:bg-fuchsia-50 hover:text-fuchsia-700 disabled:opacity-40 disabled:hover:border-neutral-300 disabled:hover:bg-transparent disabled:hover:text-neutral-700"
        >
          Python synth
        </button>
      </div>

      <div className="flex w-full max-w-3xl items-center gap-3">
        <button
          onClick={togglePlay}
          disabled={!canReplay && !isPlaying}
          title="Play the last response from the scrub position"
          className="w-20 rounded-full bg-fuchsia-600 px-4 py-1.5 text-sm font-semibold text-white hover:bg-fuchsia-700 disabled:opacity-40 disabled:hover:bg-fuchsia-600"
        >
          {isPlaying ? "Pause" : "Play"}
        </button>

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
          className="flex-1 accent-fuchsia-600 disabled:opacity-40"
        />

        <span className="w-12 text-right text-xs tabular-nums text-neutral-500">
          {viewResponse
            ? `${(scrubFrac * viewResponse.duration_s).toFixed(1)}s`
            : "–"}
        </span>

        <div className="flex overflow-hidden rounded-full border border-neutral-300 text-xs">
          {SPEEDS.map((s) => (
            <button
              key={s}
              onClick={() => {
                speedRef.current = s;
                setSpeed(s);
              }}
              title="Playback speed (applies to the next play)"
              className={`px-2.5 py-1 font-medium ${
                s === speed
                  ? "bg-fuchsia-600 text-white"
                  : "text-neutral-600 hover:bg-fuchsia-50 hover:text-fuchsia-700"
              }`}
            >
              {s}×
            </button>
          ))}
        </div>
      </div>

      {error && (
        <p className="max-w-xl text-center text-sm text-red-600">{error}</p>
      )}

      <div className="flex w-full max-w-5xl items-stretch gap-4">
        <ParamPanel response={viewResponse} frac={scrubFrac} />
        <pink-trombone className="block h-[60vh] min-w-0 flex-1" />
      </div>
    </main>
  );
}
