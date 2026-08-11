import { Container, getContainer } from "@cloudflare/containers";

/** The Python backend (samuel.server), as a container instance. */
export class SamuelBackend extends Container {
  defaultPort = 8080;
  // Memory and disk are billed for as long as the instance is awake, so the
  // idle window is a direct cost knob. Waking costs a container boot plus
  // `import torch` and the checkpoint load, so don't set this too low either.
  sleepAfter = "10m";
}

interface Env {
  BACKEND: DurableObjectNamespace<SamuelBackend>;
}

// `run_worker_first` in wrangler.jsonc sends only /api/* here; every other path
// is served straight from the static export without waking the container.
export default {
  fetch(request: Request, env: Env): Promise<Response> {
    return getContainer(env.BACKEND).fetch(request);
  },
} satisfies ExportedHandler<Env>;
