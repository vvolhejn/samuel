/**
 * Browsers gate both things this app needs — `navigator.mediaDevices` for the
 * mic, and `AudioWorklet` for the synth — behind a secure context: HTTPS, or
 * http on localhost. Served over plain http from anywhere else, the mic throws
 * "Cannot read properties of undefined (reading 'getUserMedia')" and Pink
 * Trombone quietly falls back to its legacy ScriptProcessorNode path, whose
 * tract simulation is fed the parameter channel instead of the noise input:
 * the amplitudes blow up to ~1e6 and the visualization smears across the canvas
 * for a second. So refuse to start there, with an explanation.
 */
export function insecureContextMessage(): string | null {
  if (typeof window === "undefined") return null;
  // Cached: an origin can't change within a page load, and useSyncExternalStore
  // requires a snapshot that stays identical between renders.
  if (cached === undefined) cached = computeInsecureContextMessage();
  return cached;
}

let cached: string | null | undefined;

function computeInsecureContextMessage(): string | null {
  if (window.isSecureContext) return null;
  const { hostname, port } = window.location;
  const forward = `ssh -L ${port || 80}:localhost:${port || 80} ${hostname}`;
  return (
    `This page is served over plain HTTP (${window.location.origin}), which the browser ` +
    `treats as an insecure origin: it blocks microphone access and the AudioWorklet the ` +
    `synthesizer runs in, so nothing here can work. Serve it over HTTPS, or reach it ` +
    `through localhost — e.g. \`${forward}\`, then open http://localhost:${port || 80}.`
  );
}

/** Turn a failure to open the mic into something actionable. */
export function micErrorMessage(error: unknown): string {
  if (!navigator.mediaDevices) {
    return (
      insecureContextMessage() ??
      "This browser exposes no microphone API (navigator.mediaDevices is undefined)."
    );
  }
  if (!(error instanceof Error)) return String(error);
  switch (error.name) {
    case "NotAllowedError":
      return "Microphone access was denied. Allow it for this site in the browser's address bar, then try again.";
    case "NotFoundError":
      return "No microphone found. Connect one and try again.";
    case "NotReadableError":
      return "The microphone is busy — another application is holding it. Close that one and try again.";
    default:
      return error.message;
  }
}
