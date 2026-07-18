#!/usr/bin/env bash
# Vendors browser assets into public/:
#   - Pink Trombone ESM bundle + AudioWorklet processor (from ../Pink-Trombone),
#     patching the hardcoded worklet URL (it resolves relative to the *page*
#     URL, so the unpatched bundle only works when the page serves the whole
#     script/ tree at the root).
#   - Silero VAD model/worklet (@ricky0123/vad-web) and the onnxruntime-web
#     wasm runtime it needs, so nothing is fetched from a CDN.
# Re-run after rebuilding Pink-Trombone or bumping @ricky0123/vad-web.
set -euo pipefail
cd "$(dirname "$0")/.."

PT_SRC=../Pink-Trombone
PT_OUT=public/pink-trombone
VAD_OUT=public/vad

mkdir -p "$PT_OUT" "$VAD_OUT"

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
cp "$ORT_DIST"/ort-wasm-simd-threaded*.wasm "$ORT_DIST"/ort-wasm-simd-threaded*.mjs "$VAD_OUT/"

echo "Vendored:"
ls -la "$PT_OUT" "$VAD_OUT"
