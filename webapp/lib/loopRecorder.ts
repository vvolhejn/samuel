/** The microphone, kept in a rolling buffer that can be cut at an exact sample.
 *
 * The looper does not start and stop the mic around a take: it listens
 * continuously and, when a loop boundary goes by, reaches back into the buffer
 * for the window that just finished. That is what makes a take begin on the
 * downbeat rather than a block-and-a-bit after it, and it is also why arming
 * the recorder can be instant — the audio is already there.
 *
 * Input latency is the one thing this cannot see. The samples that arrive
 * tagged with frame F were spoken some milliseconds earlier, and no browser
 * API reports how many, so a take always sits slightly late against the grid.
 * a take is therefore recorded with a pad either side of the bar, and the
 * alignment is fixed afterwards by sliding the trajectory. See
 * lib/loopTrajectory.ts. */

import { levelToSlots, makeLevelStore, LevelStore } from "@/lib/levelStore";
import { MicProcessing, MIC_PROCESSING_DEFAULTS } from "@/lib/micProcessing";

interface Block {
  startFrame: number;
  samples: Float32Array;
}

/** How much audio to keep until a take says otherwise. A loop can be far
 * longer than this — eight slow bars is well over a minute — so `retain()`
 * raises it when one is armed; this is only what to hold on to meanwhile. */
const DEFAULT_RETAIN_SECONDS = 12;

export class LoopRecorder {
  readonly levelStore: LevelStore = makeLevelStore();
  processing: MicProcessing = { ...MIC_PROCESSING_DEFAULTS };

  private ctx: AudioContext | null = null;
  private stream: MediaStream | null = null;
  private source: MediaStreamAudioSourceNode | null = null;
  private node: AudioWorkletNode | null = null;
  private sink: GainNode | null = null;
  private blocks: Block[] = [];
  private retainFrames = 0;

  get listening(): boolean {
    return this.node !== null;
  }

  get sampleRate(): number {
    return this.ctx?.sampleRate ?? 48000;
  }

  /** Open the mic and start filling the buffer. Must be called from a user
   * gesture, like anything else that touches getUserMedia. */
  async start(ctx: AudioContext, processing?: MicProcessing) {
    if (this.node) return;
    if (processing) this.processing = { ...processing };
    this.ctx = ctx;
    this.retainFrames = Math.ceil(DEFAULT_RETAIN_SECONDS * ctx.sampleRate);

    // Auto gain control fights the level the model was trained to read (see
    // data.target_rms) and, worse, moves under a loop that is already down.
    // Echo cancellation earns its keep here: without headphones the synth is
    // playing into the same room the take is recorded in.
    this.stream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, ...this.processing },
    });

    await ctx.audioWorklet.addModule("/looper/recorder-worklet.js");
    this.source = ctx.createMediaStreamSource(this.stream);
    this.node = new AudioWorkletNode(ctx, "loop-recorder", {
      numberOfInputs: 1,
      numberOfOutputs: 1,
      channelCount: 1,
    });
    this.node.port.onmessage = (event) => this.receive(event.data as Block);
    // A worklet is only pulled if its output reaches the destination, so it
    // gets there through a gain of zero.
    this.sink = ctx.createGain();
    this.sink.gain.value = 0;
    this.source.connect(this.node);
    this.node.connect(this.sink);
    this.sink.connect(ctx.destination);
  }

  stop() {
    this.node?.port.close();
    this.node?.disconnect();
    this.source?.disconnect();
    this.sink?.disconnect();
    for (const track of this.stream?.getTracks() ?? []) track.stop();
    this.node = null;
    this.source = null;
    this.sink = null;
    this.stream = null;
    this.blocks = [];
    this.levelStore.set(0);
  }

  /** Re-open the mic with different processing flags. */
  async setProcessing(processing: MicProcessing) {
    this.processing = { ...processing };
    if (!this.node || !this.ctx) return;
    const ctx = this.ctx;
    this.stop();
    await this.start(ctx, processing);
  }

  /** Keep at least `seconds` of audio. Called when a take is armed: the window
   * it will ask for is a whole loop long, and a buffer shorter than that can
   * only ever answer with nothing. Never shrinks below the default. */
  retain(seconds: number) {
    this.retainFrames = Math.ceil(
      Math.max(DEFAULT_RETAIN_SECONDS, seconds) * this.sampleRate,
    );
  }

  /** The most recent frame index the buffer holds, or null if nothing has
   * arrived — how the caller knows a window has finished coming in. */
  bufferedThrough(): number | null {
    const last = this.blocks[this.blocks.length - 1];
    return last ? last.startFrame + last.samples.length : null;
  }

  /** Pull `[fromTime, toTime)` on the AudioContext clock out of the buffer.
   *
   * Returns null if the window is not entirely in the buffer, so a caller that
   * asks too early gets an honest no rather than a take with a hole in it. */
  extract(fromTime: number, toTime: number): Float32Array | null {
    const rate = this.sampleRate;
    const from = Math.round(fromTime * rate);
    const to = Math.round(toTime * rate);
    const length = to - from;
    if (length <= 0) return null;

    const first = this.blocks[0];
    const through = this.bufferedThrough();
    if (!first || through === null) return null;
    if (from < first.startFrame || to > through) return null;

    const out = new Float32Array(length);
    for (const block of this.blocks) {
      const blockEnd = block.startFrame + block.samples.length;
      if (blockEnd <= from) continue;
      if (block.startFrame >= to) break;
      const start = Math.max(from, block.startFrame);
      const end = Math.min(to, blockEnd);
      out.set(
        block.samples.subarray(start - block.startFrame, end - block.startFrame),
        start - from,
      );
    }
    return out;
  }

  private receive(block: Block) {
    this.blocks.push(block);
    this.levelStore.set(levelToSlots(block.samples));
    const cutoff = block.startFrame + block.samples.length - this.retainFrames;
    // Blocks arrive in order, so everything to drop is at the front.
    let drop = 0;
    while (
      drop < this.blocks.length &&
      this.blocks[drop].startFrame + this.blocks[drop].samples.length < cutoff
    )
      drop++;
    if (drop > 0) this.blocks.splice(0, drop);
  }
}
