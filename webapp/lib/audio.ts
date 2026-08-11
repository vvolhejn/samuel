import { utils } from "@ricky0123/vad-web";

/**
 * Origin of the Python backend, without a trailing slash.
 *
 * Empty by default, i.e. same-origin: `samuel.server` serves this build itself,
 * and `pnpm dev` proxies `/api/*` (see next.config.ts). Set
 * `NEXT_PUBLIC_API_BASE` at build time to host the frontend separately from the
 * backend — the value is inlined into the bundle by `next build`, and the
 * backend must then allow this origin via `SAMUEL_ALLOW_ORIGINS`.
 *
 * Only `/api/*` uses it. Static assets (`/clips`, `/vad`, `/pink-trombone`)
 * ship with the frontend and stay relative.
 */
const API_BASE = (process.env.NEXT_PUBLIC_API_BASE ?? "").replace(/\/+$/, "");

/** Response of POST /api/synthesize (samuel.server). Trajectories are in
 * Pink Trombone's native units at `frame_rate` frames per second. */
export interface SynthResponse {
  frame_rate: number;
  n_frames: number;
  duration_s: number;
  params: Record<string, number[]>;
  voiced: boolean[];
  /** Python-synth reference audio (WAV) for A/B debugging. Not volume-matched:
   * the model produces its own level via the `intensity` trajectory. */
  synth_audio_b64: string;
}

/** Decode one of the base64 WAV fields into a blob. */
export function wavBlob(b64: string): Blob {
  const bytes = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
  return new Blob([bytes], { type: "audio/wav" });
}

/** One pre-recorded clip under public/clips, as listed in sources.json. That
 * file is also the recipe the WAVs are cut from — see
 * scripts/build_webapp_clips.py — hence the dataset fields, which the UI does
 * not use beyond the console log. */
export interface DatasetClip {
  name: string;
  source: string;
  offset_s: number;
  duration_s: number;
}

let clipIndex: Promise<DatasetClip[]> | null = null;

/** The committed clip list, fetched once per page load. */
export function fetchDatasetClips(): Promise<DatasetClip[]> {
  clipIndex ??= fetch("/clips/sources.json").then(async (res) => {
    if (!res.ok) throw new Error(`clip index: ${res.status}`);
    return res.json() as Promise<DatasetClip[]>;
  });
  return clipIndex;
}

/**
 * Trim trailing near-silence from a VAD utterance.
 *
 * `FrameProcessor.endSegment` in @ricky0123/vad-web appends the *entire*
 * redemption window to the emitted audio (see redemptionMs in page.tsx) —
 * up to ~800ms of post-speech silence gets sent to the model every time.
 * The controller was never trained on trailing silence and audibly loses
 * the plot during it, so cut back to the last frame louder than
 * `silenceThresholdDb` below the utterance peak, plus a small release pad.
 */
export function trimTrailingSilence(
  audio: Float32Array,
  sampleRate = 16000,
  {
    frameMs = 30,
    silenceThresholdDb = 30,
    releasePadMs = 100,
  }: { frameMs?: number; silenceThresholdDb?: number; releasePadMs?: number } = {},
): Float32Array {
  const frameLen = Math.max(1, Math.round((sampleRate * frameMs) / 1000));
  const frameCount = Math.ceil(audio.length / frameLen);
  const rms = new Float32Array(frameCount);
  let peak = 0;
  for (let i = 0; i < frameCount; i++) {
    const start = i * frameLen;
    const end = Math.min(audio.length, start + frameLen);
    let sumSq = 0;
    for (let j = start; j < end; j++) sumSq += audio[j] * audio[j];
    rms[i] = Math.sqrt(sumSq / (end - start));
    peak = Math.max(peak, rms[i]);
  }
  if (peak === 0) return audio;

  const threshold = peak * 10 ** (-silenceThresholdDb / 20);
  let lastVoiced = frameCount - 1;
  while (lastVoiced >= 0 && rms[lastVoiced] < threshold) lastVoiced--;
  if (lastVoiced < 0) return audio; // never exceeded threshold; leave as-is

  const releasePadSamples = Math.round((sampleRate * releasePadMs) / 1000);
  const cut = Math.min(
    audio.length,
    (lastVoiced + 1) * frameLen + releasePadSamples,
  );
  return audio.subarray(0, cut);
}

export interface UtteranceResult {
  response: SynthResponse;
  /** Object URL for the WAV we actually sent — the trimmed recording, i.e.
   * exactly what the model heard. The caller owns it (revoke when replacing). */
  inputUrl: string;
  /** The same WAV as a blob, for the session download. */
  inputBlob: Blob;
}

/** Send one audio file to the model backend, which sniffs the format itself
 * (WAV from the mic, MP3 for the pre-recorded clips). */
async function synthesizeBlob(blob: Blob): Promise<UtteranceResult> {
  const res = await fetch(`${API_BASE}/api/synthesize`, {
    method: "POST",
    headers: { "Content-Type": blob.type || "application/octet-stream" },
    body: blob,
  });
  if (!res.ok) {
    throw new Error(`backend error ${res.status}: ${await res.text()}`);
  }
  // URL only on success, so a failed request leaves nothing to revoke.
  return {
    response: (await res.json()) as SynthResponse,
    inputUrl: URL.createObjectURL(blob),
    inputBlob: blob,
  };
}

/** Send one VAD utterance (Float32Array at 16 kHz) to the model backend. */
export function synthesizeUtterance(
  audio: Float32Array,
): Promise<UtteranceResult> {
  // defaults: 32-bit float WAV, 16 kHz mono — matches what MicVAD emits
  const wav = utils.encodeWAV(trimTrailingSilence(audio));
  return synthesizeBlob(new Blob([wav], { type: "audio/wav" }));
}

/** Run one pre-recorded clip through the model. */
export async function synthesizeDatasetClip(
  clip: DatasetClip,
): Promise<UtteranceResult> {
  const res = await fetch(`/clips/${clip.name}`);
  if (!res.ok) throw new Error(`clip ${clip.name}: ${res.status}`);
  return synthesizeBlob(await res.blob());
}

/** Response of GET /api/health. */
export interface HealthResponse {
  status: string;
  frame_rate: number;
  device: string;
  /** Wandb run URL, or the resolved local .pt path. */
  checkpoint: string;
}

/** null when the backend is unreachable or not serving a model. */
export async function fetchHealth(): Promise<HealthResponse | null> {
  try {
    const res = await fetch(`${API_BASE}/api/health`);
    return res.ok ? await res.json() : null;
  } catch {
    return null;
  }
}
