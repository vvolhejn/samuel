"use client";

import { useEffect } from "react";

/** Privacy-friendly analytics by Plausible, loaded only on the canonical
 * deployment. Anyone self-hosting this repo serves it from another hostname, so
 * the third-party script never loads for them and their traffic never lands in
 * someone else's dashboard. */
const ANALYTICS_HOST = "samuel.vvolhejn.com";
const SCRIPT_SRC = "https://plausible.io/js/pa-cikIUzmvJJCid8KHUc0V0.js";

type Plausible = {
  (...args: unknown[]): void;
  q?: unknown[];
  o?: unknown;
  init?: (options?: unknown) => void;
};

declare global {
  interface Window {
    plausible?: Plausible;
  }
}

export function Analytics() {
  useEffect(() => {
    if (window.location.hostname !== ANALYTICS_HOST) return;

    // Queue the pageview until the real script loads and drains the queue.
    const queued: Plausible =
      window.plausible ??
      ((...args: unknown[]) => {
        (queued.q ??= []).push(args);
      });
    window.plausible = queued;
    queued.init ??= (options?: unknown) => {
      queued.o = options ?? {};
    };
    queued.init();

    const script = document.createElement("script");
    script.src = SCRIPT_SRC;
    script.async = true;
    document.head.appendChild(script);
  }, []);

  return null;
}
