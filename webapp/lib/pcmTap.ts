/**
 * Records what the browser synth outputs, sample for sample, as a WAV.
 *
 * MediaRecorder would only give us a compressed webm/mp4 soundtrack, so this
 * taps the audio graph directly instead: an AudioWorklet that, while
 * recording, batches up the raw samples and posts them back to the page.
 */
import { utils } from "@ricky0123/vad-web";

const PROCESSOR = "samuel-pcm-tap";

/** Samples per message posted back to the page: ~85 ms at 48 kHz, rather than
 * one message per 128-sample render quantum. */
const BATCH = 4096;

/** Inlined as a blob URL so there is no extra file to serve. Records the first
 * channel only — the synth is mono. Its own output stays silent. */
const PROCESSOR_SOURCE = `
class PcmTap extends AudioWorkletProcessor {
  constructor() {
    super();
    this.recording = false;
    this.buffer = new Float32Array(${BATCH});
    this.filled = 0;
    this.port.onmessage = ({ data }) => {
      if (data === "start") {
        this.recording = true;
        this.filled = 0;
      } else if (data === "stop") {
        this.recording = false;
        this.port.postMessage(this.buffer.slice(0, this.filled));
        this.port.postMessage("done");
      }
    };
  }
  process(inputs) {
    const channel = inputs[0][0];
    if (!this.recording || !channel) return true;
    for (let i = 0; i < channel.length; i++) {
      this.buffer[this.filled++] = channel[i];
      if (this.filled === ${BATCH}) {
        this.port.postMessage(this.buffer);
        this.buffer = new Float32Array(${BATCH});
        this.filled = 0;
      }
    }
    return true;
  }
}
registerProcessor("${PROCESSOR}", PcmTap);
`;

export interface PcmTap {
  /** Start collecting samples, dropping any from a previous recording. */
  start: () => void;
  /** Stop, and resolve with a 16-bit WAV of everything since start(); null
   * if nothing was recorded. */
  stop: () => Promise<Blob | null>;
}

/** Hang a tap off `source`. Null if the browser has no AudioWorklet — the
 * caller just gets no audio file. */
export async function createPcmTap(source: AudioNode): Promise<PcmTap | null> {
  const ctx = source.context as AudioContext;
  if (!ctx.audioWorklet) return null;
  const url = URL.createObjectURL(
    new Blob([PROCESSOR_SOURCE], { type: "text/javascript" }),
  );
  try {
    await ctx.audioWorklet.addModule(url);
  } finally {
    URL.revokeObjectURL(url);
  }
  const node = new AudioWorkletNode(ctx, PROCESSOR, { numberOfOutputs: 1 });
  source.connect(node);
  // Connected to the speakers so every browser keeps pulling it; it only ever
  // outputs silence.
  node.connect(ctx.destination);

  let chunks: Float32Array[] = [];
  let onDone: (() => void) | null = null;
  node.port.onmessage = ({ data }) => {
    if (data === "done") onDone?.();
    else chunks.push(data as Float32Array);
  };

  return {
    start: () => {
      chunks = [];
      node.port.postMessage("start");
    },
    stop: () =>
      new Promise<Blob | null>((resolve) => {
        onDone = () => {
          onDone = null;
          const length = chunks.reduce((n, c) => n + c.length, 0);
          if (length === 0) {
            resolve(null);
            return;
          }
          const samples = new Float32Array(length);
          let offset = 0;
          for (const chunk of chunks) {
            samples.set(chunk, offset);
            offset += chunk.length;
          }
          chunks = [];
          // Format 1, 16 bit: plain PCM, which everything opens.
          const wav = utils.encodeWAV(samples, 1, ctx.sampleRate, 1, 16);
          resolve(new Blob([wav], { type: "audio/wav" }));
        };
        node.port.postMessage("stop");
      }),
  };
}
