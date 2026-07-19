import { utils } from "@ricky0123/vad-web";

/** Response of POST /api/synthesize (samuel.server). Trajectories are in
 * Pink Trombone's native units at `frame_rate` frames per second. */
export interface SynthResponse {
  frame_rate: number;
  n_frames: number;
  duration_s: number;
  params: Record<string, number[]>;
  voiced: boolean[];
  /** Per-frame RMS volume-match gain (training's _volume_match). */
  gain: number[];
  /** Python-synth reference audio (volume-matched WAV), for A/B debugging. */
  synth_audio_b64: string;
}

/** Response of POST /api/dataset_clip: a synthesize payload plus the source
 * dataset clip so the browser can play it before the imitation. */
export interface DatasetClipResponse extends SynthResponse {
  clip_path: string;
  clip_offset_s: number;
  clip_audio_b64: string;
}

function wavBlobUrl(b64: string): string {
  const bytes = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
  return URL.createObjectURL(new Blob([bytes], { type: "audio/wav" }));
}

/** Object URL for the backend's Python-synth reference audio. */
export function synthAudioUrl(response: SynthResponse): string {
  return wavBlobUrl(response.synth_audio_b64);
}

/** Object URL for the original dataset clip. */
export function clipAudioUrl(response: DatasetClipResponse): string {
  return wavBlobUrl(response.clip_audio_b64);
}

/** Ask the backend for a random dataset clip run through the model. */
export async function fetchDatasetClip(): Promise<DatasetClipResponse> {
  const res = await fetch("/api/dataset_clip", { method: "POST" });
  if (!res.ok) {
    throw new Error(`backend error ${res.status}: ${await res.text()}`);
  }
  return res.json();
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

/** Send one VAD utterance (Float32Array at 16 kHz) to the model backend. */
export async function synthesizeUtterance(
  audio: Float32Array,
): Promise<SynthResponse> {
  // defaults: 32-bit float WAV, 16 kHz mono — matches what MicVAD emits
  const wav = utils.encodeWAV(trimTrailingSilence(audio));
  const res = await fetch("/api/synthesize", {
    method: "POST",
    headers: { "Content-Type": "audio/wav" },
    body: new Blob([wav], { type: "audio/wav" }),
  });
  if (!res.ok) {
    throw new Error(`backend error ${res.status}: ${await res.text()}`);
  }
  return res.json();
}

export async function backendHealthy(): Promise<boolean> {
  try {
    const res = await fetch("/api/health");
    return res.ok;
  } catch {
    return false;
  }
}
