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

/** Object URL for the backend's Python-synth reference audio. */
export function synthAudioUrl(response: SynthResponse): string {
  const bytes = Uint8Array.from(atob(response.synth_audio_b64), (c) =>
    c.charCodeAt(0),
  );
  return URL.createObjectURL(new Blob([bytes], { type: "audio/wav" }));
}

/** Send one VAD utterance (Float32Array at 16 kHz) to the model backend. */
export async function synthesizeUtterance(
  audio: Float32Array,
): Promise<SynthResponse> {
  // defaults: 32-bit float WAV, 16 kHz mono — matches what MicVAD emits
  const wav = utils.encodeWAV(audio);
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
