import { utils } from "@ricky0123/vad-web";

/** Response of POST /api/synthesize (samuel.server). Trajectories are in
 * Pink Trombone's native units at `frame_rate` frames per second. */
export interface SynthResponse {
  frame_rate: number;
  n_frames: number;
  duration_s: number;
  params: Record<string, number[]>;
  voiced: boolean[];
  /** Python-synth reference audio (WAV) for A/B debugging. Not volume-matched:
   * the model produces its own level via the `intensity` trajectory. Absent
   * from the precomputed clip responses, which drop it for size. */
  synth_audio_b64?: string;
}

/** Decode one of the base64 WAV fields into a blob. */
export function wavBlob(b64: string): Blob {
  const bytes = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
  return new Blob([bytes], { type: "audio/wav" });
}

/** Who to credit for a clip: the LibriVox recording the dataset took it from.
 * `url` is the book's LibriVox page, `file_url` the section itself. */
export interface LibrivoxCredit {
  title: string;
  author: string;
  reader: string;
  url: string;
  file_url: string;
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
  librivox?: LibrivoxCredit;
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
 * Trim near-silence from both ends of a recording.
 *
 * The VAD trim in lib/useMicVad.ts works in whole 32ms frames and pads either
 * end, so some silence always survives it. The controller was never trained on
 * silence and audibly loses the plot during it, so cut back to the outermost
 * frames louder than `silenceThresholdDb` below the recording's peak, plus a
 * small pad so the first and last words keep their attack and tail.
 */
export function trimSilence(
  audio: Float32Array,
  sampleRate = 16000,
  {
    frameMs = 30,
    silenceThresholdDb = 30,
    padMs = 100,
  }: { frameMs?: number; silenceThresholdDb?: number; padMs?: number } = {},
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
  let firstVoiced = 0;
  while (rms[firstVoiced] < threshold) firstVoiced++;

  const pad = Math.round((sampleRate * padMs) / 1000);
  return audio.subarray(
    Math.max(0, firstVoiced * frameLen - pad),
    Math.min(audio.length, (lastVoiced + 1) * frameLen + pad),
  );
}

export interface UtteranceResult {
  response: SynthResponse;
  /** Object URL for the WAV we actually sent — the trimmed recording, i.e.
   * exactly what the model heard. The caller owns it (revoke when replacing). */
  inputUrl: string;
  /** The same WAV as a blob, for the session download. */
  inputBlob: Blob;
  /** Came from the committed precomputed responses rather than the backend, so
   * it arrived instantly and the caller should fake the thinking pause. */
  precomputed: boolean;
}

/** What a backend that is still waking up is busy with: "boot" (the container
 * itself, reported by the Worker), then samuel.server's "model", "warm-up" and
 * "tts". The container sleeps when nobody uses it, so the first request after
 * a quiet spell finds it in one of these. */
export type WakeStage = "boot" | "model" | "warm-up" | "tts";
/** Called whenever a request finds the backend still waking up. */
export type OnWaking = (stage: WakeStage) => void;

/** The stages each endpoint waits out: text-to-speech loads last, and the
 * imitation endpoint doesn't wait for it. */
const SYNTHESIZE_WAITS: readonly WakeStage[] = ["boot", "model", "warm-up"];
const TTS_WAITS: readonly WakeStage[] = [...SYNTHESIZE_WAITS, "tts"];

const WAKE_POLL_MS = 1000;
/** Generous: booting and loading should take well under this. */
const WAKE_TIMEOUT_MS = 3 * 60_000;
/** Times a request is resent after its wait. One should do; the slack is for
 * a container that goes back to sleep in between. */
const WAKE_ATTEMPTS = 3;

/** A model request that failed on the backend's side. `code` is the HTTP
 * status, or "wake-timeout" if the backend never woke up; it is what
 * analytics reports, since the message can carry the server's own text. */
export class BackendError extends Error {
  constructor(
    message: string,
    readonly code: string,
  ) {
    super(message);
  }
}

const sleep = (ms: number) =>
  new Promise<void>((resolve) => setTimeout(resolve, ms));

/** The stage of a 503 "still loading" answer, or null for any other answer. */
async function wakeStage(res: Response): Promise<WakeStage | null> {
  if (res.status !== 503) return null;
  try {
    const body = await res.clone().json();
    return body?.detail?.status === "loading" ? body.detail.stage : null;
  } catch {
    return null;
  }
}

/** Poll /api/health until the backend is past every stage in `waits`. */
async function waitUntilAwake(
  waits: readonly WakeStage[],
  onWaking: OnWaking,
): Promise<void> {
  const deadline = Date.now() + WAKE_TIMEOUT_MS;
  while (Date.now() < deadline) {
    await sleep(WAKE_POLL_MS);
    let res: Response;
    try {
      res = await fetch("/api/health");
    } catch {
      continue;
    }
    const stage = await wakeStage(res);
    // Any other answer, an error included, is for the real request to report.
    if (stage === null || !waits.includes(stage)) return;
    onWaking(stage);
  }
  throw new BackendError(
    "The model didn't wake up. Try again in a minute.",
    "wake-timeout",
  );
}

/** fetch() for a model endpoint: if the backend is still waking up, report it,
 * wait until it is up, and send the request again. */
async function fetchAwake(
  path: string,
  init: RequestInit,
  waits: readonly WakeStage[],
  onWaking: OnWaking,
): Promise<Response> {
  for (let attempt = 1; ; attempt++) {
    const res = await fetch(path, init);
    const stage = await wakeStage(res);
    if (stage === null || attempt === WAKE_ATTEMPTS) return res;
    onWaking(stage);
    await waitUntilAwake(waits, onWaking);
  }
}

/** Send one audio file to the model backend, which sniffs the format itself
 * (WAV from the mic, MP3 for the pre-recorded clips). */
async function synthesizeBlob(
  blob: Blob,
  onWaking: OnWaking,
): Promise<UtteranceResult> {
  const res = await fetchAwake(
    "/api/synthesize",
    {
      method: "POST",
      headers: { "Content-Type": blob.type || "application/octet-stream" },
      body: blob,
    },
    SYNTHESIZE_WAITS,
    onWaking,
  );
  if (!res.ok) {
    throw new BackendError(
      `backend error ${res.status}: ${await res.text()}`,
      String(res.status),
    );
  }
  // URL only on success, so a failed request leaves nothing to revoke.
  return {
    response: (await res.json()) as SynthResponse,
    inputUrl: URL.createObjectURL(blob),
    inputBlob: blob,
    precomputed: false,
  };
}

/** Send one VAD utterance (Float32Array at 16 kHz) to the model backend. */
export function synthesizeUtterance(
  audio: Float32Array,
  onWaking: OnWaking,
): Promise<UtteranceResult> {
  // defaults: 32-bit float WAV, 16 kHz mono — matches what MicVAD emits
  const wav = utils.encodeWAV(trimSilence(audio));
  return synthesizeBlob(new Blob([wav], { type: "audio/wav" }), onWaking);
}

/** Have the text-to-speech model say `text`. Same answer as for audio, minus
 * an original: there is no recording behind it. */
export async function synthesizeText(
  text: string,
  onWaking: OnWaking,
): Promise<SynthResponse> {
  const res = await fetchAwake(
    "/api/tts",
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    },
    TTS_WAITS,
    onWaking,
  );
  if (!res.ok) {
    throw new BackendError(
      `backend error ${res.status}: ${await res.text()}`,
      String(res.status),
    );
  }
  return (await res.json()) as SynthResponse;
}

/** What the model answers with for each committed clip, computed ahead of time
 * by scripts/precompute_clip_responses.py. `clips` are sources.json names;
 * `model_fingerprint` is the checkpoint these were computed from. */
export interface PrecomputedIndex {
  model_fingerprint: string;
  clips: string[];
}

let precomputedIndex: Promise<PrecomputedIndex | null> | null = null;

/** The precomputed-response index, fetched once per page load, or null if it
 * isn't there (nobody has run the script yet). */
export function fetchPrecomputedIndex(): Promise<PrecomputedIndex | null> {
  precomputedIndex ??= fetch("/clips/precomputed/index.json")
    .then((res) => (res.ok ? (res.json() as Promise<PrecomputedIndex>) : null))
    .catch(() => null);
  return precomputedIndex;
}

/** Whether the precomputed responses were made by the checkpoint the backend is
 * currently serving. A null fingerprint (backend unreachable) is not a mismatch:
 * offline, the stored answers are the only ones there are. */
export function precomputedMatches(
  index: PrecomputedIndex | null,
  modelFingerprint: string | null | undefined,
): boolean {
  if (!index) return false;
  return !modelFingerprint || index.model_fingerprint === modelFingerprint;
}

/** The committed answer for one clip, or null if there isn't a usable one. */
async function fetchPrecomputedClip(
  clip: DatasetClip,
  modelFingerprint: string | null | undefined,
): Promise<SynthResponse | null> {
  const index = await fetchPrecomputedIndex();
  if (!index?.clips.includes(clip.name)) return null;
  if (!precomputedMatches(index, modelFingerprint)) {
    console.warn(
      `precomputed clip responses are from checkpoint ${index.model_fingerprint}, ` +
        `backend serves ${modelFingerprint} — asking the backend instead. ` +
        "Re-run scripts/precompute_clip_responses.py.",
    );
    return null;
  }
  const stem = clip.name.replace(/\.[^.]+$/, "");
  const res = await fetch(`/clips/precomputed/${stem}.json`);
  return res.ok ? ((await res.json()) as SynthResponse) : null;
}

/** Run one pre-recorded clip through the model — or, since the clips never
 * change, read the answer straight off disk when it was precomputed for the
 * checkpoint the backend is serving. The clip itself is fetched either way: the
 * Original button plays it back. */
export async function synthesizeDatasetClip(
  clip: DatasetClip,
  modelFingerprint: string | null | undefined,
  onWaking: OnWaking,
): Promise<UtteranceResult> {
  const res = await fetch(`/clips/${clip.name}`);
  if (!res.ok) throw new Error(`clip ${clip.name}: ${res.status}`);
  const blob = await res.blob();
  const response = await fetchPrecomputedClip(clip, modelFingerprint);
  if (response) {
    return {
      response,
      inputUrl: URL.createObjectURL(blob),
      inputBlob: blob,
      precomputed: true,
    };
  }
  return synthesizeBlob(blob, onWaking);
}

/** Response of GET /api/health. */
export interface HealthResponse {
  status: "ok";
  frame_rate: number;
  device: string;
  /** Wandb run URL, or the resolved local .pt path. */
  checkpoint: string;
  /** Content hash of the loaded weights — the same string wherever they are
   * loaded from, so the committed clip responses can be checked against it. */
  model_fingerprint: string;
  /** Whether POST /api/tts is served. */
  tts: boolean;
}

/** /api/health of a backend that is still waking up. */
export interface WakingResponse {
  status: "loading";
  stage: WakeStage;
}

/** null when the backend is unreachable or not serving a model. */
export async function fetchHealth(): Promise<
  HealthResponse | WakingResponse | null
> {
  try {
    const res = await fetch("/api/health");
    const stage = await wakeStage(res);
    if (stage !== null) return { status: "loading", stage };
    return res.ok ? await res.json() : null;
  } catch {
    return null;
  }
}

/** The health of a backend that is waking up, once it has: null if it never
 * does, or comes up broken. */
export async function waitForHealth(): Promise<HealthResponse | null> {
  try {
    await waitUntilAwake(TTS_WAITS, () => {});
  } catch {
    return null;
  }
  const health = await fetchHealth();
  return health?.status === "ok" ? health : null;
}
