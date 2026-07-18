"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import type { MicVAD } from "@ricky0123/vad-web";
import {
  backendHealthy,
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

export default function Home() {
  const [status, setStatus] = useState<Status>("idle");
  const [error, setError] = useState<string | null>(null);
  const [hasLastUtterance, setHasLastUtterance] = useState(false);

  const trombone = usePinkTrombone();
  const vadRef = useRef<MicVAD | null>(null);
  const lastResponse = useRef<SynthResponse | null>(null);
  const synthUrlRef = useRef<string | null>(null);
  const debugAudioRef = useRef<HTMLAudioElement | null>(null);
  const busyRef = useRef(false); // ignore VAD events while processing/speaking
  const micOnRef = useRef(false); // user-intended mic state (start/mute toggle)

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
    if (micOnRef.current) {
      await vadRef.current?.start();
      setStatus("listening");
    } else {
      setStatus(vadRef.current ? "muted" : "idle");
    }
  }, []);

  const playResponse = useCallback(
    async (response: SynthResponse) => {
      setStatus("speaking");
      await vadRef.current?.pause(); // don't let the synth retrigger the mic
      try {
        await trombone.speak(response);
      } finally {
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
        setHasLastUtterance(true);
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

  const replay = useCallback(async () => {
    if (!lastResponse.current || busyRef.current) return;
    busyRef.current = true;
    setError(null);
    try {
      await playResponse(lastResponse.current);
    } catch (e) {
      busyRef.current = false;
      setError(e instanceof Error ? e.message : String(e));
      await restoreMic();
    }
  }, [playResponse, restoreMic]);

  const playPythonSynth = useCallback(async () => {
    if (!synthUrlRef.current || busyRef.current) return;
    busyRef.current = true;
    setError(null);
    setStatus("speaking");
    await vadRef.current?.pause();
    try {
      const audio = (debugAudioRef.current ??= new Audio());
      audio.src = synthUrlRef.current;
      await audio.play();
      await new Promise<void>((resolve) => {
        audio.onended = () => resolve();
        audio.onerror = () => resolve();
      });
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      busyRef.current = false;
      await restoreMic();
    }
  }, [restoreMic]);

  const micActive = status !== "idle";
  const canReplay =
    hasLastUtterance && (status === "listening" || status === "muted");

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
          onClick={replay}
          disabled={!canReplay}
          className="rounded-full border border-neutral-300 px-4 py-1.5 text-sm font-medium text-neutral-700 hover:border-fuchsia-300 hover:bg-fuchsia-50 hover:text-fuchsia-700 disabled:opacity-40 disabled:hover:border-neutral-300 disabled:hover:bg-transparent disabled:hover:text-neutral-700"
        >
          Replay
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

      {error && (
        <p className="max-w-xl text-center text-sm text-red-600">{error}</p>
      )}

      <pink-trombone className="block h-[60vh] w-full max-w-3xl" />
    </main>
  );
}
