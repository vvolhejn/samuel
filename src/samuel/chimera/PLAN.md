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

## Written but not yet run: `control.py`

`AirwayControl` (slew limits = `HumanAirway::commit`, `set_tube_targets` with the tube stretch,
velum port target, `update_noise` with the three noise slots) and `glottis_params`
(`apply_source` plus the score's abduction/closure). Imports, nothing else checked.

## Next steps

1. **End-to-end parity from articulators.** Build a replay that takes the trace's per-block
   `artic` and source settings instead of its targets, runs `VocalTract.compute` for all blocks
   at once, then per block `AirwayControl.block(...)` inside `render` (block function form, so the
   teeth source sees `net.u` at the block's start). Compare against the trace's targets and audio.
   - Initial state: the Rust starts from the neutral posture with the tongue tip at rest
     (`ArticulatorBank::new`: `NEUTRAL` with `tt_x, tt_y = rest_tip(jaw, tb_x, tb_y)`, see
     `rest_tip` in `anatomy/human.rs`), geometry snapped (`AirwayControl.start` + `init_state`).
   - Source from the trace: `f0` = glottis f0; `pressure` = `p_lung / CM_H2O`;
     `abduction` = `(glottis.abduction − 0.1·0.15²)` (male, k = 1); `closure`;
     breathiness 0.15, tension 0.5, brightness 0.25, roughness 0.1, drift 5, vibrato rate 6,
     `vibrato_depth` from the glottis, fry 0 (the `DEFAULT_SOURCE`).
   - Expected small mismatches: blocks where `enforce_contacts` corrected the tongue (the Rust
     sets the noise list twice, so filter-state carry can differ), and `p_lung == 0` tiny-flushes.
   - Check `AirwayControl.block` against Rust where it's subtle: the port-target line that keeps
     the glottis port's current area; `lips_from` truncation; the order (slew, then mins capped by
     the slewed areas, then clamped at 0).
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
