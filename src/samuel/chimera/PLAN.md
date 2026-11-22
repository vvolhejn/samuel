# CHIMERA port: status and what's left

A differentiable Torch port of the CHIMERA vocal synth's human voice
(github.com/jamesondh/chimera-vocal-synth, Rust engine in `crates/chimera/`), sample by sample
(the full time-domain simulation, not impulse responses). For speed, `chimera(chunk_ms=...)`
renders chunks side by side in the batch with ~25 ms warm-up each and overlap-adds them (as the
Rust experiment `chimera chunked`, branch `experiment/chunked-ola`, did).

## Where things are

- **This branch** (`chimera-synth` in samuel, worktree `/Users/vaclav/prog/samuel-chimera`):
  `src/samuel/chimera/`, tests in `tests/test_chimera.py` (14 tests, ~160 s).
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
3. ~~Gradient sanity~~: done. One NaN source, fixed: `sqrt(area)` in `Network.step` at a sealed
   node (lips shut in /p/) sent inf·0 back to every articulator; now a double `where` (exact
   forward). Checked: finite gradients on the 200 ms /p/ clip (every trainable parameter reached),
   on geometry at 665 jittered + 4000 random/corner articulations, and on 12 random full-range
   renders; autograd = finite differences to ~1e-8 per frame from a moving pose. Known, by design:
   - the velum gets no gradient while shut (port area = max(velum, 0); init −0.15). If training
     needs to discover nasals, start it at ≥ 0 or put a surrogate slope on the clamp;
   - at a perfectly static pose (`default_params`) the network's branch conditions (e.g. the
     radiation's "did the lip area grow") sit on their switch points, so gradients are one-sided
     there (per-frame up to ~20 % off a central difference; their sum over frames is exact);
   - backward costs ~2.5× the forward (200 ms: 19 s forward, 50 s backward, CPU).
4. ~~float32, GPU, speed~~: done as far as this Mac goes.
   - **Speed** is per-op overhead, not arithmetic: ~300 small ops per internal sample, 88 200
     samples per second of audio. CPU float64, B=1: ~60 s per audio second; B=64 costs only ~5×
     B=1. `compile=True` (`torch.compile` of one sample's lungs + glottis + airway, `voice._sample`)
     is ~2.3× forward and ~5× forward+backward at B=1, nothing at B=16 with grad (compute-bound),
     equal to eager to 1e-14. Compiles once per batch size, dtype, device and grad mode (~10 s).
   - **Chunked** (`chunk_ms`, `warmup_ms=25`, `crossfade_ms=10`; `synth.render_chunked`): chunk 0
     is the sequential render's start, later chunks start `warmup + crossfade/2` early with the
     airway at rest; the glottis's cycle phase/jitter/shimmer (`glottis.cycle_state_at`, a
     cycle-level pass equal to the sample loop to 1e-11) and the pitch modulation follow the
     sequential timeline, so pulses line up. Whole phrase, CPU B=1: 72 s → 12–17 s; against the
     sequential render it differs as much as another noise seed does (SNR 11 dB, log-mel
     1.7–2.0 dB vs 10.8 / 1.69 for a seed change; on the quieter 200 ms clip 36–40 dB vs 43).
     Gradients flow through it. On a GPU, where a step costs the same at any batch size, this
     divides the sequential length outright.
   - **float32** on the 200 ms clip: 70 dB SNR against the Rust (float64: 153), finite; compiled
     B=1 float32 runs at ~12 s per audio second. Not checked: float32 gradients, stability at
     extreme poses.
   - **MPS** works eager in float32 but is launch-bound: ~760 s per audio second, flat in batch
     size (≈3 s per item-second at B=256). `compile=True` fails there: Inductor fuses more than
     Metal's 31 buffer arguments into one kernel. Not tried: CUDA (the realistic target; expect
     launch-bound eager like MPS, so compile + chunking matter most there).
5. **Next**: a CUDA run (compile, chunked, float32, gradients), then wiring into `model.py` /
   training (out of this plan's scope until asked). Upstream has moved since the dump branch:
   jamesondh/chimera-vocal-synth#143 (merged) keeps the frication source for a constriction just
   past the velum, i.e. it removes the junction skip that `control._noise` copies; regenerate the
   references from a newer Rust before following it. The noise-rate normalisation (√(96 kHz/fs))
   stays as it is.

## Not ported (on purpose, for now)

Output limiter, fry's second pulse, chaos, the female/child bodies (geometry and airway take a
`Body`, untested), creature voices, articulator dynamics, gestural score and planner.
