import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Emit a static site into `out/` on `next build`. The Python backend
  // (samuel.server) serves that directory and handles `/api/*` on the same
  // origin, so no proxy is needed in production.
  output: "export",
  // The rewrite below only takes effect under `next dev` (it is ignored by
  // the static export, which prints a harmless warning at build time). It
  // keeps the standalone dev workflow — `pnpm dev` on :3000 proxying to the
  // uvicorn backend on :8000 — working unchanged.
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: "http://localhost:8471/api/:path*",
      },
    ];
  },
};

export default nextConfig;
