#!/usr/bin/env bash
# Vendors browser assets into public/:
#   - Pink Trombone ESM bundle + AudioWorklet processor: built from
#     ../Pink-Trombone/script/, then copied here with the hardcoded worklet URL
#     patched (it resolves relative to the *page* URL, so the unpatched bundle
#     only works when the page serves the whole script/ tree at the root).
#   - Silero VAD model/worklet (@ricky0123/vad-web) and the onnxruntime-web
#     wasm runtime it needs, so nothing is fetched from a CDN.
# Runs automatically as package.json's "prebuild", so an edit under
# Pink-Trombone/script/ can never ship stale: neither the bundles nor
# public/pink-trombone are committed, and both are regenerated here.
# Run it by hand only to refresh assets without a build.
set -euo pipefail
cd "$(dirname "$0")/.."

PT_SRC=../Pink-Trombone
PT_OUT=public/pink-trombone
VAD_OUT=public/vad

# Both dirs are gitignored and fully regenerated here, so wipe them rather than
# copying over the top: a file that stops being vendored must stop shipping.
rm -rf "$PT_OUT" "$VAD_OUT"
mkdir -p "$PT_OUT" "$VAD_OUT"

# Separate pnpm project (its only dependency is rollup), not a workspace member.
pnpm --dir "$PT_SRC" install --frozen-lockfile
pnpm --dir "$PT_SRC" run build

cp "$PT_SRC/pink-trombone.min.js" "$PT_OUT/"
cp "$PT_SRC/pink-trombone-worklet-processor.min.js" "$PT_OUT/"
sed 's|"./script/audio/nodes/pinkTrombone/processors/WorkletProcessor.js"|"/pink-trombone/pink-trombone-worklet-processor.min.js"|' \
  "$PT_OUT/pink-trombone.min.js" > "$PT_OUT/pink-trombone.min.js.tmp"
mv "$PT_OUT/pink-trombone.min.js.tmp" "$PT_OUT/pink-trombone.min.js"
if ! grep -q '/pink-trombone/pink-trombone-worklet-processor.min.js' "$PT_OUT/pink-trombone.min.js"; then
  echo "ERROR: worklet URL patch did not apply — did the Pink-Trombone build change?" >&2
  exit 1
fi

VAD_DIST=$(node -e "console.log(require('path').dirname(require.resolve('@ricky0123/vad-web/package.json')))")/dist
# onnxruntime-web is not hoisted by pnpm; resolve it via vad-web's own tree
# so the copied wasm always matches the version vad-web was built against.
ORT_DIST=$(node -e "console.log(require('path').dirname(require.resolve('onnxruntime-web', {paths: ['$VAD_DIST']})))")

cp "$VAD_DIST/silero_vad_v5.onnx" "$VAD_DIST/silero_vad_legacy.onnx" \
   "$VAD_DIST/vad.worklet.bundle.min.js" "$VAD_OUT/"
# Only the plain wasm build: MicVAD imports `onnxruntime-web/wasm`, so the
# jsep/jspi/asyncify variants are never fetched (66 MB of the 79 MB). The jsep
# one is also 25.6 MB, over Cloudflare's 25 MiB per-asset cap.
cp "$ORT_DIST"/ort-wasm-simd-threaded.wasm "$ORT_DIST"/ort-wasm-simd-threaded.mjs "$VAD_OUT/"

echo "Vendored:"
ls -la "$PT_OUT" "$VAD_OUT"
