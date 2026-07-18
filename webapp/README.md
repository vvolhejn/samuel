# Samuel webapp — speak, and the vocal tract mimics you

Records your voice in the browser, detects end-of-utterance with Silero VAD,
sends the audio to a Python backend running the trained controller
(`onset-off_20260527-193518` / wandb `i30dfe0t`), and plays the predicted
parameter trajectories through the vendored Pink Trombone synth — the tract
visualization animates along.

## Run

1. Vendor the browser assets (once, and after rebuilding `../Pink-Trombone`
   or bumping `@ricky0123/vad-web`):

   ```bash
   pnpm install
   ./scripts/vendor-pink-trombone.sh
   ```

2. Start the model backend (from the repo root):

   ```bash
   uv run --extra server uvicorn samuel.server:app --port 8000
   ```

   Env overrides: `SAMUEL_CHECKPOINT` (local `.pt` or wandb artifact ref),
   `SAMUEL_RUN_CONFIG` (run `config.json`; only its `model` block is used).

3. Start the frontend:

   ```bash
   pnpm dev
   ```

   Open http://localhost:3000 (`/api/*` is proxied to the backend), click
   Start, allow the microphone, speak, pause — it speaks back.

## How it works

- `app/page.tsx` — state machine `idle → listening → recording → processing →
  speaking`; mic + Silero VAD via `@ricky0123/vad-web` (assets self-hosted
  under `public/vad/`).
- `lib/audio.ts` — encodes the 16 kHz VAD segment as WAV, POSTs to
  `/api/synthesize`.
- `src/samuel/server.py` (repo root) — resamples to 44.1 kHz, extracts pyin
  f0 (the model's external `frequency` input), runs the checkpoint, returns
  all 11 Pink Trombone parameter trajectories at ≈86.13 fps in native units.
- `lib/usePinkTrombone.ts` — schedules each trajectory onto the synth's
  AudioParams with one `setValueCurveAtTime` per parameter (mirroring the
  Python synth: `tenseness = voiceness`, `loudness = voiceness^0.25`), gating
  voicing with short `intensity` ramps. The tract UI polls the audio worklet
  each frame, so it animates automatically. The synth + visualization load at
  page open (suspended AudioContext, resumed on Start).
- **Volume envelope**: training only ever evaluated per-frame RMS
  volume-matched audio (`train._volume_match`), so the raw model output has
  no meaningful envelope — the backend returns the per-frame `gain` curve
  (computed against its own Python resynthesis) and the frontend applies it
  via a master GainNode. Without it, silence hums.
- **Debug**: the "Python synth" button plays the backend's own
  (volume-matched) resynthesis of the last utterance, for A/B comparison with
  the browser synth. "Mic off/on" toggles the VAD without tearing it down.
- `public/pink-trombone/` — vendored build of `../Pink-Trombone` with the
  worklet URL patched to an absolute path (see
  `scripts/vendor-pink-trombone.sh`).

Expect rough, babble-adjacent speech — the checkpoint's eval WER is ~0.9;
the webapp faithfully reproduces what the model predicts.
