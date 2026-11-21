# CHIMERA port: status and what's left

A differentiable Torch port of the CHIMERA vocal synth's human voice
(github.com/jamesondh/chimera-vocal-synth, Rust engine in `crates/chimera/`), sample by sample
(the full time-domain simulation, not impulse responses). Speed comes later: chunked overlap-add
with ~25 ms warm-up per chunk was measured to be indistinguishable from the full render
(Rust experiment `chimera chunked`, branch `experiment/chunked-ola`).

## Where things are

- **This branch** (`chimera-synth` in samuel, worktree `/Users/vaclav/prog/samuel-chimera`):
  `src/samuel/chimera/`, tests in `tests/test_chimera.py` (5 tests, ~28 s).
- **Rust reference tools**: vvolhejn/chimera-vocal-synth branch `experiment/torch-port-dump`
  (worktree `/Users/vaclav/prog/chimera-torch-dump`, built with `cargo build --release -p chimera-cli`):
  - `chimera dump <out.json> <fs_out> <oversample> "Text."`: a full trace of an utterance. Per
    control block: network targets, noise sources, glottis params, lung pressure, articulators,
    resampled tract. Per internal sample: glottal area, network output, glottal flow; plus the final
    output. `CHIMERA_DUMP_NONOISE=1` turns noise off. The limiter is off.
  - `chimera dump-geometry <trace.json> <out.json> [every_nth] [jitter_copies]`: the geometry
    alone for each articulation in a trace, plus jittered copies (`CHIMERA_GEO_JITTER`, a fraction
    of each articulator's range, default 0.1).
  - Large references live in `artifacts/chimera_ref/` (gitignored): `peter.json`,
    `peter_nonoise.json`, `peter_geometry.json`, `peter_geometry_wild.json`. Regenerate with the
    commands above ("Peter picked.", 44100, 2).
- Run tests with the main checkout's venv: `PYTHONPATH=src /Users/vaclav/prog/samuel/.venv/bin/python -m pytest tests/test_chimera.py`.

## Done, and exact against Rust

| Module | Ports | Check |
|---|---|---|
| `network.py` | `AirwayNetwork::step` (FDTD airway, walls, radiation, kinetic losses, squeeze spill, sealed cells, Re²-gated noise) | trace replay: 6e-13 relative on the whole 1.09 s phrase, with and without noise |
| `glottis.py`, `rng.py` | `ParametricGlottis::next`, mulberry32 `Rng`, `Drift` | same replay |
| `airway.py` | `HumanAirway::new` (topology, index for index) | equal to the dump's |
| `voice.py` | `HumanVoice::render` minus the limiter; FIR decimator, DC blocker. `render(blocks=list or fn(i, net) -> Block, noise="rust"/"independent")` | same replay |
| `geometry.py` | `HumanVocalTract::compute` + `resample` (articulators → areas, minima, lengths, teeth/lip position, contacts) | 1232 articulations incl. ±60 % range jitter: ≤1.4e-13; gradients reach ≥12 articulators |
| `control.py` | `HumanAirway::commit` (slew limits, tube targets with the stretch, velum port), `update_noise` (three slots, the junction skip), `apply_source` + the score's abduction/closure; `replay_articulated` drives the voice from a trace's `artic` and source settings alone | whole phrase, with and without noise: every block's targets, noise sources and glottis parameters ≤3e-13, network output 9e-13, output 5e-8 (the f32 reference) |

## Next steps

1. ~~End-to-end parity from articulators~~: done (`control.replay_articulated`, test on the
   200 ms fixture). The dump records the glottis port's target after the block (the last glottal
   area), so comparisons leave that port out. Not checked whether this phrase exercises them:
   blocks where `enforce_contacts` moved the tongue (the Rust sets the noise list twice there) and
   the junction skip in `_noise` (written to match the Rust's `to_branch` → −1).
2. **The samuel-style API** (`synth.py`), following `pink_trombone.py` / `model.py`:
   `chimera(params [B, T, P], seed=None, control_rate=...) -> [B, T·spf]` at 44.1 kHz (airway at
   2×), `PARAM_NAMES`, and a `(lo, hi, init)` spec per parameter like `_DEFAULT_PARAM_SPEC`.
   - Parameters: the 17 articulators (`geometry.ARTIC_NAMES`, ranges `ARTIC_RANGES`, init
     `NEUTRAL`), plus source `f0` (Hz), `pressure` (0..~15 cmH₂O, 7.2 typical), `abduction`
     (0..0.5; the planner uses 0.1–0.36), `closure` (0..1), `breathiness`, `tension`,
     `brightness` (0..1). Freeze the rarely used ones (dorsum, ttCurl, lateral, epilarynx,
     ventricular, labiodental?) by default.
   - Upsample like `_upsample_params`: frame t starts at sample t·spf and ramps to frame t+1;
     sample the ramp at each CHIMERA control block start (every 64 output samples).
   - No articulator dynamics (the controller sets positions directly), no f0 glide, no
     `enforce_contacts` (the tongue can sink into the palate; the contact model still seals).
     Say so in the docstring.
   - `noise="independent"` with a `torch.Generator` from `seed`; all three noise slots always
     present with gain faded to 0 when off (already in `control.py`'s `exact=False`).
   - Start from the first frame's geometry, snapped (no transient from neutral).
3. **Gradient sanity**: backprop a loss on a short render to every parameter; look for NaNs at
   closures (`sqrt` at 0 in `close_r` slew, `pow(d, 1.4)` at 0, `acos` near 1, divisions guarded by
   `torch.where`; prefer `safe_div` patterns already in `geometry.py`).
4. **Then, separately**: float32 and GPU, speed (torch.compile the step, chunked OLA with ~25 ms
   warm-up), the noise-rate scaling fix (the Rust at this commit normalises noise by √(96 kHz/fs);
   a fix to √(fs/96 kHz) was prepared in another session; `network._noise` follows the old one),
   and the limiter if needed. Not in scope: training code.

## Not ported (on purpose, for now)

Output limiter, fry's second pulse, chaos, the female/child bodies (geometry and airway take a
`Body`, untested), creature voices, articulator dynamics, gestural score and planner.
