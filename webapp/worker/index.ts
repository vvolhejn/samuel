import { Container, getContainer } from "@cloudflare/containers";

/** The Python backend (samuel.server), as a container instance. */
export class SamuelBackend extends Container {
  defaultPort = 8080;
  // Memory and disk are billed for as long as the instance is awake, so the
  // idle window is a direct cost knob. Waking costs a container boot plus
  // `import torch` and the checkpoint load, so don't set this too low either.
  sleepAfter = "10m";

  /** The boot in progress, shared by every request that arrives during it. */
  private booting: Promise<void> | null = null;
  /** Why the last boot failed, handed to the next request and then forgotten,
   * so the one after that tries again. */
  private bootError: string | null = null;

  /** A cold container gets an immediate 503 in the same shape samuel.server
   * uses while it loads, and boots in the background. Left to the library,
   * the request would hang through the boot and give up after 20 s with a
   * 500, which is about as long as the boot takes; this way the page can say
   * it is waking up and poll /api/health until it has. */
  override async fetch(request: Request): Promise<Response> {
    const { status } = await this.getState();
    if (this.ctx.container?.running && status === "healthy") {
      return super.fetch(request);
    }
    if (this.bootError !== null) {
      const message = this.bootError;
      this.bootError = null;
      return new Response(`Failed to start container: ${message}`, {
        status: 500,
      });
    }
    this.booting ??= this.startAndWaitForPorts({
      cancellationOptions: {
        instanceGetTimeoutMS: 60_000,
        portReadyTimeoutMS: 120_000,
      },
    })
      .catch((e: unknown) => {
        this.bootError = e instanceof Error ? e.message : String(e);
        console.error(`container boot failed: ${this.bootError}`);
      })
      .finally(() => {
        this.booting = null;
      });
    this.ctx.waitUntil(this.booting);
    return Response.json(
      { detail: { status: "loading", stage: "boot" } },
      { status: 503 },
    );
  }
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
