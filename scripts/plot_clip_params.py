"""Plot the Pink Trombone parameters that the model predicts for one clip.

Runs the served checkpoint over a slice of an audio file and writes one PNG
with a separate line plot for each parameter. Parameters that never move over
the slice are left out and named on stdout.

    uv run --extra server python scripts/plot_clip_params.py
    uv run --extra server python scripts/plot_clip_params.py --clip path/to.wav --start 0 --end 3
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import librosa
import numpy as np
import plotly.graph_objects as go
import soundfile as sf
import torch
from plotly.subplots import make_subplots

from samuel import server
from samuel.pink_trombone import PARAM_NAMES, SAMPLE_RATE

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CLIP = REPO_ROOT / "webapp" / "public" / "clips" / "clip-21b.mp3"
DEFAULT_OUT = REPO_ROOT / "iclc" / "iclc-2027-paper" / "images" / "clip-21b-params.png"
# Read row by row: each index/diameter pair sits side by side in one row.
PLOT_ORDER = [
    "frequency",
    "voiceness",
    "intensity",
    "lipDiameter",
    "tongueIndex",
    "tongueDiameter",
    "constrictionIndex",
    "constrictionDiameter",
]
COLUMNS = 4
ROW_HEIGHT = 210
ROW_GAP = 0.035
FONT_SIZE = 15
LINE_COLOR = "#4269d0"
# frequency comes from the pitch tracker, not from the model.
LINE_COLORS = {"frequency": "#9498a0"}


def load_audio(path: Path, start: float, end: float) -> np.ndarray:
    """Read ``path`` as mono at the synthesizer's sample rate, cut to a slice."""
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
    lo = int(start * SAMPLE_RATE)
    hi = len(audio) if end <= 0 else int(end * SAMPLE_RATE)
    audio = audio[lo:hi]
    if len(audio) == 0:
        raise SystemExit(f"{path} has nothing between {start} s and {end} s")
    return audio


def predict(audio: np.ndarray) -> tuple[np.ndarray, float]:
    """Predict parameters for ``audio``. Returns the frames and the frame rate."""
    model = server._load_model()
    print("checkpoint:", server._checkpoint, "fingerprint:", server._fingerprint)

    t_ctrl = model.t_ctrl_for(len(audio))
    f0, _voiced = server._pitch_track(audio, model.samples_per_frame, t_ctrl)
    audio_in = server._rms_normalize(audio, server._target_rms())

    device = next(model.parameters()).device
    with torch.no_grad():
        params = model(
            torch.from_numpy(audio_in).to(device)[None, None, :],
            torch.from_numpy(f0).to(device)[None, :],
        )[0]
    return params.cpu().numpy(), model.config.frame_rate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip", type=Path, default=DEFAULT_CLIP)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--start", type=float, default=7.5, help="slice start, seconds")
    parser.add_argument(
        "--end", type=float, default=8.8, help="slice end in seconds; 0 is end of clip"
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-6,
        help="a parameter counts as constant if its range over the slice is below this",
    )
    args = parser.parse_args()

    audio = load_audio(args.clip, args.start, args.end)
    params, frame_rate = predict(audio)

    spread = params.max(axis=0) - params.min(axis=0)
    order = PLOT_ORDER + [n for n in PARAM_NAMES if n not in PLOT_ORDER]
    moving = [n for n in order if spread[PARAM_NAMES.index(n)] > args.tol]
    constant = [n for n in order if spread[PARAM_NAMES.index(n)] <= args.tol]
    if not moving:
        raise SystemExit("every parameter is constant over this slice, nothing to plot")
    print("constant, not plotted:", ", ".join(constant) or "none")

    t = np.arange(params.shape[0]) / frame_rate
    rows = math.ceil(len(moving) / COLUMNS)

    fig = make_subplots(
        rows=rows,
        cols=COLUMNS,
        shared_xaxes=True,
        vertical_spacing=ROW_GAP,
        horizontal_spacing=0.055,
        subplot_titles=moving,
    )
    for cell, name in enumerate(moving):
        row, col = cell // COLUMNS + 1, cell % COLUMNS + 1
        fig.add_trace(
            go.Scatter(
                x=t,
                y=params[:, PARAM_NAMES.index(name)],
                mode="lines",
                name=name,
                line=dict(color=LINE_COLORS.get(name, LINE_COLOR), width=1.6),
                showlegend=False,
            ),
            row=row,
            col=col,
        )

    fig.update_layout(
        template="plotly_white",
        font=dict(size=FONT_SIZE),
        width=350 * COLUMNS,
        height=int((60 + ROW_HEIGHT * rows) / (1 - ROW_GAP * (rows - 1))),
        hovermode="x unified",
        margin=dict(t=30, l=60, r=20, b=50),
    )
    for annotation in fig.layout.annotations:
        annotation.update(font=dict(size=FONT_SIZE))
    fig.update_xaxes(showgrid=True, gridcolor="#eceff3")
    fig.update_yaxes(showgrid=True, gridcolor="#eceff3")
    for col in range(1, COLUMNS + 1):
        fig.update_xaxes(title_text="time (s)", row=rows, col=col)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(args.out, scale=2)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
