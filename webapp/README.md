# Samuel webapp — speak, and the vocal tract mimics you

Records your voice in the browser, detects end-of-utterance with Silero VAD,
sends the audio to a Python backend running the trained controller
(`onset-off_20260527-193518` / wandb `i30dfe0t`), and plays the predicted
parameter trajectories through the vendored Pink Trombone synth — the tract
visualization animates along.

## Run

Vendor the browser assets first (once, and after rebuilding `../Pink-Trombone`
or bumping `@ricky0123/vad-web`):

```bash
pnpm install
./scripts/vendor-pink-trombone.sh
```

### One process (recommended)

The backend serves both the UI and the API. On startup it builds the Next.js
static export (`out/`, via `pnpm build`) and mounts it at `/`:

```bash
uv run --extra server python -m samuel.server
```

Open http://127.0.0.1:8471, click Start, allow the microphone, speak, pause —
it speaks back. Env overrides:

- `SAMUEL_CHECKPOINT` — local `.pt` or wandb artifact ref. The run's
  `config.json` is found next to it (`<run_dir>/config.json`) automatically
- `SAMUEL_RUN_CONFIG` — override the auto-found config (only its `model` block
  is used); required only for a wandb-artifact checkpoint
- `SAMUEL_PORT` / `SAMUEL_HOST` — default `127.0.0.1:8471`
- `SAMUEL_FRONTEND_SKIP_BUILD=1` — serve an existing `out/` without rebuilding
- `SAMUEL_SERVE_FRONTEND=0` — API only (use the dev-mode workflow below)

### Two processes (frontend dev mode)

For live-reloading frontend work, run the backend API-only and Next's dev
server separately:

```bash
SAMUEL_SERVE_FRONTEND=0 uv run --extra server uvicorn samuel.server:app --port 8471
pnpm dev   # in another shell
```

Open http://localhost:3000 — `/api/*` is proxied to the backend on :8471.

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
