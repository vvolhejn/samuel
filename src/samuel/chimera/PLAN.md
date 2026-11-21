# CHIMERA port: status and what's left

A differentiable Torch port of the CHIMERA vocal synth's human voice
(github.com/jamesondh/chimera-vocal-synth, Rust engine in `crates/chimera/`), sample by sample
(the full time-domain simulation, not impulse responses). Speed comes later: chunked overlap-add
with ~25 ms warm-up per chunk was measured to be indistinguishable from the full render
(Rust experiment `chimera chunked`, branch `experiment/chunked-ola`).

## Where things are

- **This branch** (`chimera-synth` in samuel, worktree `/Users/vaclav/prog/samuel-chimera`):
  `src/samuel/chimera/`, tests in `tests/test_chimera.py` (9 tests, ~65 s).
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
| `synth.py` | the samuel API: `chimera(params [B, T, 25], seed, control_rate) -> [B, T·spf]`, `PARAM_NAMES`, `DEFAULT_PARAM_SPEC` / `DEFAULT_FROZEN_VALUES`, `default_params`; the voice's constants and decimator built without a trace | its block renderer from the trace's articulators: constants equal, output 5e-8 on the 200 ms fixture. Through `chimera()` at the block rate (first frame's pose, independent noise): 1.1 % RMS from the Rust |

## Next steps

1. ~~End-to-end parity from articulators~~: done (`control.replay_articulated`, test on the
   200 ms fixture). The dump records the glottis port's target after the block (the last glottal
   area), so comparisons leave that port out. Not checked whether this phrase exercises them:
   blocks where `enforce_contacts` moved the tongue (the Rust sets the noise list twice there) and
   the junction skip in `_noise` (written to match the Rust's `to_branch` → −1).
2. ~~The samuel-style API~~: done (`synth.py`). 17 articulators + `f0`, `pressure`, `abduction`,
   `closure`, `breathiness`, `tension`, `brightness`, `roughness`; frozen by default: ttCurl,
   labiodental, epilarynx, ventricular, lateral, dorsum, roughness. Simulates in float64 and casts
   back to the params' dtype. ~6 s per 100 ms of audio per batch of 1–2 on CPU. The noise draws
   depend on the batch size (one `rand(L, B, 3)` per block), unlike Pink Trombone's per-item seeds.
   Not wired into `model.py` / training.
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
