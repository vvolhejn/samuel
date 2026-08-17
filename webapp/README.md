# Samuel webapp — speak, and the vocal tract mimics you

**Note to humans: slop below, refer to main README**

Records your voice in the browser (press Microphone, press Stop; Silero VAD
only trims the recording to the speech in it), sends the audio to a Python
backend running the trained controller
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

Open http://127.0.0.1:8471. See `--help` for more flags.

### Two processes (frontend dev mode)

For live-reloading frontend work, run the backend API-only and Next's dev
server separately:

```bash
SAMUEL_SERVE_FRONTEND=0 uv run --extra server uvicorn samuel.server:app --port 8471
pnpm dev   # in another shell
```

Open http://localhost:3000 — `/api/*` is proxied to the backend on :8471.

### Deployment

`deploy/` puts both halves on Cloudflare as a single Worker: the static export
from `out/`, and the backend as a container on the same origin. See
`deploy/README.md`.

## How it works

- `app/page.tsx` — owns the synth: who has it (`lib/owner.ts`) and what the
  page shows for it.
- `lib/useMicVad.ts` — the mic: recording runs from Microphone to Stop, and
  Silero VAD via `@ricky0123/vad-web` (assets self-hosted under `public/vad/`)
  is used only to trim the recording to the frames it heard speech in.
- `lib/audio.ts` — encodes the 16 kHz recording as WAV, POSTs to
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
- **Precomputed clips**: the six numbered buttons feed the model committed MP3s
  from `public/clips/`, so their answers never change — they are generated once
  by `scripts/precompute_clip_responses.py` into `public/clips/precomputed/`
  and played from there, with a 0.5–1 s fake pause so the button still reads as
  thinking. Regenerate them after a checkpoint swap: `index.json` records a
  fingerprint of the weights they came from, and when it disagrees with
  `/api/health` the frontend ignores them (console warning, and a red line in
  the debug panel) and asks the backend as before.
- **Debug**: the "Python synth" button plays the backend's own
  (volume-matched) resynthesis of the last utterance, for A/B comparison with
  the browser synth. "Mic off/on" toggles the VAD without tearing it down.
- `public/pink-trombone/` — vendored build of `../Pink-Trombone` with the
  worklet URL patched to an absolute path (see
  `scripts/vendor-pink-trombone.sh`).

Expect rough, babble-adjacent speech — the checkpoint's eval WER is ~0.9;
the webapp faithfully reproduces what the model predicts.
