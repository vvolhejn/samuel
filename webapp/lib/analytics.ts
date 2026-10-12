/** Send a custom event to Plausible. A no-op wherever the Analytics component
 * didn't set up `window.plausible`, i.e. anywhere but the canonical host.
 * Before the script has loaded, the stub there queues the event. Each event
 * also needs a matching goal in the Plausible dashboard to show up there. */
export function trackEvent(name: string, props: Record<string, string>) {
  window.plausible?.(name, { props });
}

/** Rough bucket for a wait, so the dashboard's breakdown stays readable. */
export function waitBucket(ms: number): string {
  const s = ms / 1000;
  if (s < 10) return "<10s";
  if (s < 30) return "10-30s";
  if (s < 60) return "30-60s";
  return "60s+";
}
