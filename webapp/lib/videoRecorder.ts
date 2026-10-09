/**
 * Records the vocal-tract visualization as a video with the synth's audio.
 *
 * There is no single canvas to capture: TractUI stacks a background + tract
 * pair, and GlottisUI stacks another pair for the voicebox strip below it —
 * each canvas transparent, each anchored to its own row of the pink-trombone
 * grid. So we composite them onto an offscreen canvas every animation frame,
 * placing each one at its layout offset within the host element — over the
 * page's white, since a transparent webm renders black in most players — and
 * capture that instead.
 */

/** First one the browser admits to supporting wins. Chrome/Firefox land on
 * webm; Safari only records mp4. */
const MIME_CANDIDATES = [
  "video/webm;codecs=vp9,opus",
  "video/webm;codecs=vp8,opus",
  "video/webm",
  "video/mp4",
];

const FPS = 30;
const PAGE_BACKGROUND = "#ffffff";

export interface VideoRecording {
  blob: Blob;
  /** Container extension for the negotiated mime type ("webm" or "mp4"). */
  extension: string;
}

export interface VideoRecorder {
  /** Resolves with the finished video, or null if nothing was captured. */
  stop: () => Promise<VideoRecording | null>;
}

/**
 * Start compositing `canvases` (bottom first) inside `host` and recording them
 * together with `audio`. Composite dimensions come from the host so the frame
 * matches its design bitmap; each canvas is drawn at its layout offset within
 * the host, so the tract's 600×500 pair and the voicebox's 600×125 pair land
 * where they visually sit rather than all at (0, 0). Returns null when the
 * browser has no MediaRecorder or canvas capture — the caller just gets no
 * video, everything else still works.
 */
export function startVideoRecording(
  host: HTMLElement,
  canvases: HTMLCanvasElement[],
  audio: MediaStream | null,
): VideoRecorder | null {
  const [first] = canvases;
  if (typeof MediaRecorder === "undefined") return null;
  if (!first || typeof first.captureStream !== "function") return null;

  const composite = document.createElement("canvas");
  // offsetWidth/Height ignore CSS transforms, so they give the host's design
  // bitmap (600×600 for pink-trombone) even when TractStage has scaled it.
  composite.width = host.offsetWidth;
  composite.height = host.offsetHeight;
  const context = composite.getContext("2d");
  if (!context) return null;

  // Not in the DOM: captureStream reads whatever we draw here regardless.
  let frame = requestAnimationFrame(function draw() {
    frame = requestAnimationFrame(draw);
    context.fillStyle = PAGE_BACKGROUND;
    context.fillRect(0, 0, composite.width, composite.height);
    // getBoundingClientRect returns transform-scaled coords, so both sides of
    // the ratio scale together and the offsets come out in composite pixels.
    const hostRect = host.getBoundingClientRect();
    const scaleX = hostRect.width ? composite.width / hostRect.width : 1;
    const scaleY = hostRect.height ? composite.height / hostRect.height : 1;
    for (const canvas of canvases) {
      const rect = canvas.getBoundingClientRect();
      const x = (rect.left - hostRect.left) * scaleX;
      const y = (rect.top - hostRect.top) * scaleY;
      context.drawImage(canvas, x, y);
    }
  });

  const stream = composite.captureStream(FPS);
  // The audio track belongs to a destination node that outlives every
  // recording, so it is only borrowed — stop() must hand it back unstopped.
  for (const track of audio?.getAudioTracks() ?? []) stream.addTrack(track);

  const mimeType = MIME_CANDIDATES.find((type) =>
    MediaRecorder.isTypeSupported(type),
  );
  let recorder: MediaRecorder;
  try {
    recorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);
  } catch {
    cancelAnimationFrame(frame);
    return null;
  }

  const chunks: Blob[] = [];
  recorder.ondataavailable = (event) => {
    if (event.data.size > 0) chunks.push(event.data);
  };
  // Timeslice keeps MediaRecorder from buffering the whole recording internally.
  // Without it, long recordings sometimes truncate their tail on stop — one
  // dataavailable event has to flush everything, and Chrome will drop clusters
  // rather than delay the stop. A 1 s slice gives dozens of small chunks with
  // no user-visible cost.
  recorder.start(1000);

  return {
    stop: () =>
      new Promise<VideoRecording | null>((resolve) => {
        cancelAnimationFrame(frame);
        const finish = () => {
          for (const track of stream.getVideoTracks()) track.stop();
          for (const track of stream.getAudioTracks()) stream.removeTrack(track);
          if (chunks.length === 0) {
            resolve(null);
            return;
          }
          const type = recorder.mimeType || mimeType || "video/webm";
          resolve({
            blob: new Blob(chunks, { type }),
            extension: type.includes("mp4") ? "mp4" : "webm",
          });
        };
        if (recorder.state === "inactive") finish();
        else {
          recorder.onstop = finish;
          recorder.stop();
        }
      }),
  };
}
