# Pink Trombone — Agent Guide

A modularized, programmable version of Neil Thapen's [Pink Trombone](https://dood.al/pinktrombone/) speech synthesizer. It physically models the human vocal tract using Web Audio API to generate realistic speech sounds, exposed as a `<pink-trombone>` custom HTML element.

## Build

```bash
npm install
npm run build
```

Produces two Rollup bundles:
- `pink-trombone.min.js` — main module (entry: `script/component.js`)
- `pink-trombone-worklet-processor.min.js` — AudioWorklet processor

## Architecture

```
script/
├── PinkTrombone.js              # Public API wrapper
├── component.js                 # Custom HTML element (<pink-trombone>)
├── audio/nodes/pinkTrombone/
│   ├── AudioNode.js             # Web Audio node wrapper
│   └── processors/
│       ├── WorkletProcessor.js  # AudioWorklet entry point
│       ├── Processor.js         # Sample-level pipeline (coordinates Glottis + Tract)
│       ├── Glottis.js           # Glottal source: pitch, tenseness, vibrato
│       ├── Tract.js             # Digital waveguide mesh (vocal tract filter)
│       ├── Nose.js              # Nasal cavity
│       ├── Transient.js         # Click/transient sounds
│       ├── SimplexNoise.js      # Noise for natural jitter
│       └── ParameterDescriptors.js
└── graphics/                    # Canvas-based visualization UI
```

## Key Constraints

**Real-time audio DSP.** `Tract.js` (517 lines) runs per-sample in an AudioWorklet. It implements a digital waveguide mesh — changes require understanding of acoustic modeling. Performance is critical; the tract processes each sample twice for accuracy with a 0.125 amplitude compensation.

**Cross-thread communication.** The AudioWorklet processor runs on the audio thread; constrictions and configuration are sent via `MessagePort`. Parameter automation uses `AudioParam` scheduling, not direct assignment.

**Browser compatibility.** The code maintains a `ScriptProcessor` fallback for browsers without AudioWorklet support. Do not break this path.

**No test suite.** Validate changes by loading `index.html` in a browser and exercising the audio output.

## Public API

```javascript
const el = document.querySelector('pink-trombone');
el.setAudioContext(ctx);
el.connect(destination);
el.start();

// AudioParam controls (supports .value, .setValueAtTime, etc.)
el.frequency        // Hz
el.intensity        // 0–1
el.loudness         // 0–1
el.tenseness        // 0–1
el.tongue.index     // tract position
el.tongue.diameter  // tongue shape

el.vibrato.frequency
el.vibrato.gain
el.vibrato.wobble

// Constrictions (for consonants — up to 4)
const c = el.newConstriction(index, diameter);
el.removeConstriction(c);
```

## Synthesis Model Mathematics

The synthesizer is a **source-filter** model: a glottal source signal is filtered by a vocal tract model. Every audio sample runs through both stages.

### 1. Glottal Source — LF Model (`Glottis.js`)

The glottal waveform uses the **Liljencrants-Fant (LF) model**, a standard acoustic phonetics model for the glottal flow derivative. Each pitch period is divided into two phases:

**Open phase** (0 ≤ t ≤ Tₑ):
```
voice(t) = E₀ · exp(α·t) · sin(ω·t)
```
An exponentially-enveloped sinusoid that rises and falls as the vocal folds open.

**Return/closing phase** (Tₑ < t ≤ 1):
```
voice(t) = (−exp(−ε·(t − Tₑ)) + shift) / Δ
```
An exponential decay back to baseline as the folds snap shut.

The shape coefficients (α, ε, ω, E₀, Tₑ) are recomputed each pitch period from `tenseness` via intermediate parameters Rd, Ra, Rk, Rg (defined in the LF literature). High tenseness → pressed phonation; low → breathy/slack.

**Vibrato and noise** are added to the instantaneous frequency each sample:
```
f_eff = f · (1 + vibratoGain·sin(2π·vibratoFreq·t)
              + 0.02·simplex(4.07t) + 0.04·simplex(2.15t)
              + wobble·[0.2·simplex(0.98t) + 0.4·simplex(0.5t)])
```
Two octaves of Simplex noise add organic jitter; `wobble` adds slower drift.

**Fricative noise** is shaped by the glottal opening: a half-wave rectified sine (`max(0, sin(2π·t)) · 0.2 + 0.1`) gates broadband noise so turbulence peaks when the glottis is most open.

---

### 2. Vocal Tract Filter — Kelly-Lochbaum Waveguide (`Tract.js`)

The tract is modelled as a chain of **N = 88 cylindrical tube sections** (default 44, scaled 2×). Each section has cross-sectional area:
```
A[i] = diameter[i]²
```

**Reflection coefficients** at each junction follow the acoustic transmission-line formula:
```
r[i] = (A[i−1] − A[i]) / (A[i−1] + A[i])
```
This is exact for lossless plane-wave propagation in adjacent tubes of different areas.

**Wave propagation** per sample: right-traveling (`R`) and left-traveling (`L`) waves scatter at each junction:
```
R_junction[i] = R[i−1] − r[i]·(R[i−1] + L[i])
L_junction[i] = L[i]   + r[i]·(R[i−1] + L[i])
```

**Boundary conditions:**
- Glottis end (i = 0): `R_junction[0] = L[0]·0.75 + glottis_sample` (partial reflection + source injection)
- Lip end (i = N):     `L_junction[N] = R[N−1]·(−0.85)` (strong negative reflection — open end)
- A damping factor of 0.999 per step models small viscous/thermal losses.

**Three-way junction at the velum** (nose opening): the nose branches off at a fixed tract position. The scattering equations distribute energy among the oral tract, nasal cavity, and the returning wave using area-weighted reflection coefficients. Velum target diameter (`velum.target`) is normally 0.01 (closed); nasal consonants open it to 0.4.

**Double-pass oversampling** (`Processor.js`): each audio sample runs through `tract.process()` twice (at integer and half-integer sample offsets), and the outputs are summed and scaled by 0.125. This improves numerical accuracy without full 2× upsampling of the source.

---

### 3. Tongue and Constrictions — Diameter Shaping

**Tongue** sets the resting diameter profile from blade to lip using a raised cosine:
```
angle  = 1.1π · (tongueIndex − tractIndex) / (tipStart − bladeStart)
curve  = (1.5 − diameter + 1.7) · cos(angle)
rest[i] = 1.5 − curve
```
This approximates how a real tongue body raises and lowers the mid-tract.

**Constrictions** (tongue tip, lips, teeth for consonants) are applied as a smooth taper over the surrounding tract sections using a raised cosine kernel:
```
scalar = 0.5 · (1 − cos(π · relpos / range))
diameter[i] = newDiameter + (diameter[i] − newDiameter) · scalar
```
`range` depends on tract position (wider in the back, narrower at the tip), approximating how different articulators affect different spans of the tract. A completely closed constriction (diameter ≤ 0) schedules a **transient** (click) when it reopens.

---

### Signal Flow Summary

```
tenseness, frequency
       │
       ▼
  [LF Glottis] ──→ glottal sample + fricative noise
                              │
                              ▼
               [Kelly-Lochbaum Tract × 2 passes]
                  ← shaped by diameter[0..N]
                  ← diameter set by tongue + constrictions
                              │
                              ├──→ lip output (oral)
                              └──→ nose output (nasal)
                              │
                              ▼
                         audio output
```

## Phoneme Reference

Vowels are shaped by `tongue.index` / `tongue.diameter`. Consonants use `newConstriction()`. See the README for specific parameter values per phoneme.
