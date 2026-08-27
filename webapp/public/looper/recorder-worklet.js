/** Microphone capture for the looper, at the AudioContext's own sample rate.
 *
 * Every block is tagged with the frame index it starts at. `currentFrame` runs
 * on the same timeline as `currentTime * sampleRate`, which is the timeline the
 * loop clock schedules on, so a bar boundary can be turned into an exact sample
 * offset instead of "whichever block arrived around then". MediaRecorder and
 * the VAD's frame callback both lose that: they hand over audio with no
 * position on the graph's clock, and a loop cut on their timing drifts.
 *
 * Loaded by URL rather than bundled — addModule takes a script, not a module
 * from the graph, same as the vendored Pink Trombone worklet. */
class LoopRecorderProcessor extends AudioWorkletProcessor {
  process(inputs) {
    const channel = inputs[0] && inputs[0][0];
    // No input connected yet: stay alive, there is nothing to send.
    if (!channel) return true;
    // The render quantum's buffer is reused between calls, so this has to be a
    // copy; transferring it then costs nothing on the way out.
    const samples = new Float32Array(channel);
    this.port.postMessage({ startFrame: currentFrame, samples }, [
      samples.buffer,
    ]);
    return true;
  }
}

registerProcessor("loop-recorder", LoopRecorderProcessor);
