/**
 * Records the vocal-tract visualization as a video with the synth's audio.
 *
 * There is no single canvas to capture: TractUI stacks two transparent ones
 * (the static background labels below, the animated tract above). So we
 * composite them onto an offscreen canvas every animation frame — over the
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
 * Start compositing `canvases` (bottom first) and recording them together with
 * `audio`. Returns null when the browser has no MediaRecorder or canvas
 * capture — the caller just gets no video, everything else still works.
 */
export function startVideoRecording(
  canvases: HTMLCanvasElement[],
  audio: MediaStream | null,
): VideoRecorder | null {
  const [first] = canvases;
  if (typeof MediaRecorder === "undefined") return null;
  if (!first || typeof first.captureStream !== "function") return null;

  const composite = document.createElement("canvas");
  composite.width = first.width;
  composite.height = first.height;
  const context = composite.getContext("2d");
  if (!context) return null;

  // Not in the DOM: captureStream reads whatever we draw here regardless.
  let frame = requestAnimationFrame(function draw() {
    frame = requestAnimationFrame(draw);
    context.fillStyle = PAGE_BACKGROUND;
    context.fillRect(0, 0, composite.width, composite.height);
    for (const canvas of canvases) context.drawImage(canvas, 0, 0);
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
  // No timeslice: one dataavailable event, delivered on stop().
  recorder.start();

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
