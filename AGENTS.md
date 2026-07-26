Under Pink-Trombone/: A modularized, programmable version of Neil Thapen's [Pink Trombone](https://dood.al/pinktrombone/) speech synthesizer. It physically models the human vocal tract using Web Audio API to generate realistic speech sounds, exposed as a `<pink-trombone>` custom HTML element. See Pink-Trombone/AGENTS.md

Top level and under samuel/: Python implementation of the same model, we want to train a model to control the synthesis model.

For plotting, use Plotly rather than Matplotlib.

This is a uv project: run Python via `uv run python ...` (the shell's default `python` is an unrelated mamba env) and manage dependencies with `uv add` / `uv pip install`, never plain `pip`.