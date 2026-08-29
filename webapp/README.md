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

## Looper (`/looper`)

A second page, and a different way of using the same model: record a
bar-aligned take, loop the parameter trajectory it comes back as, and play
over it from a MIDI keyboard.

Nothing here streams. A take is a finite recording sent to the same
`/api/synthesize` as the main page, which is what makes the whole thing cheap:
no streaming encoder, no streaming pitch tracker, no causal input
normalisation, and a full loop's worth of wall clock to compute the next
loop in. It also means the checkpoint does not have to be causal.

- **Recording is cut from a rolling buffer, not started and stopped.**
  `lib/loopRecorder.ts` keeps the mic in memory via an AudioWorklet
  (`public/looper/recorder-worklet.js`) that tags each block with its
  `currentFrame`. A take is then an exact sample range on the same clock the
  loop is scheduled against — the VAD's frame callback and MediaRecorder both
  lose that, and a loop cut on their timing drifts. What no API reports is
  microphone *input* latency, so a take lands slightly late against the grid.
  A take is therefore cut with a pad of extra audio either side of the bar, and
  the "record offset" slider slides the *predicted parameters* over that pad
  rather than re-cutting the audio: the alignment is fixed after the fact, to
  one model frame (~12 ms), on the loop you are already listening to.
- **The trajectory is indexed by loop phase, not by frame**
  (`lib/loopTrajectory.ts`), so the loop follows tempo for free. At half the
  tempo the articulation reads out half as fast and nothing transposes, because
  pitch is a separate parameter rather than a property of a resampled waveform.
  The last few frames are bent to meet the first so the loop point doesn't jump.
  The loop is a window over the padded take, copied out whenever the record
  offset moves, which is what makes that control retroactive.
- **`lib/loopClock.ts` answers one question** — what phase is it at time `t` —
  from either its own tempo or MIDI clock. External sync averages the tempo
  continuously but re-anchors the phase only at loop boundaries: USB clock
  jitter is around a millisecond, and correcting mid-loop is audible as a lurch.
- **`lib/loopScheduler.ts` writes the tract** on the usual two-clock
  arrangement: a coarse timer wakes often and schedules every parameter frame
  inside a 30 ms lookahead, each as a ramp landing at an exact time. Per
  channel it keeps a weight: 0 is purely the recording, 1 is purely your hands,
  and the value written is the crossfade. Nothing smooths the output itself —
  at weight 0 the recorded articulation reaches the synth exactly as the model
  predicted it.
- **`lib/metronome.ts` clicks on every beat**, scheduled the same way and off
  the same clock, so the click and the loop cannot disagree. Beat times are
  re-derived from the clock at every wake-up rather than accumulated, which is
  what keeps them on the loop when the tempo moves or an external clock
  re-anchors the downbeat. The click defaults to sounding for a take only, and
  the beat it is on is marked on the phase bar as well.
- **The loop is muted for the take itself**, not for the count-in: the take is
  of you, and what is already looping would come back through the mic and be
  re-synthesised on top of itself. Muting rides the same output gate as
  start/stop, so the trajectory keeps turning against the clock and comes back
  where it has got to. Turn it off to overdub against what is playing.

### Saved takes

Every take is written to IndexedDB as the model predicted it — the parameter
frames, not the audio — so a refresh does not cost you the loop
(`lib/takeStore.ts`). The frames are small, they load with no backend and no
wait, and they are exactly what the trajectory is built from. The pads are
stored with them, so the record offset still slides on a loaded take. The
newest 20 are kept and older ones drop off on their own. Loading a take
restores the metre it was recorded in, but not the tempo: the loop follows
whatever tempo is running.

Storage is allowed to fail. A private window or a browser with site data turned
off refuses, and the looper works as before without it.

### Handover

Each channel is `recorded`, `transpose`/`scale`/`offset`, or `replace`. The
middle one is usually what you want: it keeps the recorded contour and moves
it, and that contour is most of what makes the loop sound like speech rather
than a held vowel.

Pitch is worth understanding separately, for two reasons.

The model never sees pitch. `frequency` is not an encoder input — it is
scattered into the output parameter vector (`model.py`), having come from pyin.
So substituting a MIDI note for it is exact: there is no distribution to be
out of, and nothing to retrain.

And pitch in `replace` mode is written the instant the note arrives rather than
at the next scheduled frame, with the scheduler standing off that parameter
until the note lifts. That takes the lookahead window out of note-on latency.
`transpose` cannot do this — it has to keep tracking the contour it is
transposing — so it costs one window of delay.

`intensity` is the loudness channel; `voiceness` is breathy-to-pressed timbre,
not level (it drives the synth's `tenseness` *and* `loudness`). Everything
except pitch is driven by a controller, assigned by clicking "learn CC" and
reaching for a knob. The computer keyboard plays notes too (`a w s e d f…`,
`z`/`x` for octave) so none of this needs hardware to try.

Mono only — one tract, one glottis.

### Caveats

- Web MIDI is Chromium and Firefox (behind a prompt); Safari has none. The
  computer keyboard still works there.
- Keep echo cancellation on unless you are wearing headphones: the synth is
  playing into the room the take is recorded in. Auto gain control is off by
  default here — it fights the level the model reads as loudness, and moves
  under a loop that is already down.
- The route is served explicitly by `samuel.server` (`/looper`): Next's static
  export writes it as `looper.html`, which the catch-all static mount would not
  find.

Expect rough, babble-adjacent speech — the checkpoint's eval WER is ~0.9;
the webapp faithfully reproduces what the model predicts.
