"""Animate an impulse travelling through the vocal tract.

The tract is held at the shape of one control frame of a webapp clip. The
tract starts from silence, the glottis emits a single impulse, and there is no
turbulence. The left panel shows the wave amplitude in every tube section, the
right panel shows the impulse response coming out of the lips.

    uv run python scripts/animate_waveguide_waves.py

Writes an mp4 and a folder of the individual frames beside it. The waveguide
runs at 88200 steps per second and every step is drawn.

The waveguide carries two independent copies of the wave, interleaved section
by section. The animation draws one of them, so the wave moves one tube
section per frame.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import torch
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).resolve().parent))

from plot_waveguide_waves import (  # noqa: E402
    LEFT_COLOR,
    RIGHT_COLOR,
    load_params,
    pick_frame,
)
from samuel import pink_trombone as pt  # noqa: E402

LIPS_COLOR = "#444444"
STEP_RATE = 2 * pt.SAMPLE_RATE
STRIDE = 1
WIDTH = 1600
HEIGHT = 520


def record_impulse_response(
    params: torch.Tensor, frame: int, steps: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Send one impulse into a silent tract held at the shape of ``frame``.

    Returns the right-going waves, the left-going waves (both [steps, N]), the
    output at the lips [steps] and the tube diameter profile [N].
    """
    samples = (steps + 1) // 2
    p = {name: params[0, frame, i] for i, name in enumerate(pt.PARAM_NAMES)}
    const = {
        key: p[key].view(1, 1).expand(1, samples)
        for key in (
            "tongueIndex",
            "tongueDiameter",
            "constrictionIndex",
            "constrictionDiameter",
            "lipDiameter",
        )
    }
    glottis_out = torch.zeros(1, samples)
    glottis_out[0, 0] = 1.0

    recorded: list[tuple[np.ndarray, np.ndarray, float]] = []
    original_step = pt._waveguide_step

    def recording_step(*args, **kwargs):
        state = original_step(*args, **kwargs)
        right, left, out = state[0], state[1], state[4]
        recorded.append(
            (
                right[0].detach().numpy().copy(),
                left[0].detach().numpy().copy(),
                float(out[0]),
            )
        )
        return state

    pt._waveguide_step = recording_step
    try:
        with torch.no_grad():
            pt._tract(
                glottis_out=glottis_out,
                noise_mod=torch.zeros(1, samples),
                tongue_index=const["tongueIndex"],
                tongue_diameter=const["tongueDiameter"],
                constriction_index=const["constrictionIndex"],
                constriction_diameter=const["constrictionDiameter"],
                lip_diameter=const["lipDiameter"],
            )
    finally:
        pt._waveguide_step = original_step

    head = recorded[:steps]
    right = np.stack([r for r, _, _ in head])
    left = np.stack([left_ for _, left_, _ in head])
    lips = np.array([o for _, _, o in head])
    diameter = pt._compute_diameter_profile(
        const["tongueIndex"][:, :1],
        const["tongueDiameter"][:, :1],
        const["constrictionIndex"][:, :1],
        const["constrictionDiameter"][:, :1],
        const["lipDiameter"][:, :1],
        pt._TRACT_N,
    )
    return right, left, lips, diameter[0, 0].numpy()


def frame_figure(
    right: np.ndarray,
    left: np.ndarray,
    lips: np.ndarray,
    diameter: np.ndarray,
    step: int,
    limit: float,
    lips_limit: float,
    with_lips: bool,
) -> go.Figure:
    n = right.shape[0]
    x = np.arange(n)
    # The waveguide holds two interleaved copies of the wave. Draw one: its
    # sections shift by one per step, so the mask follows the step.
    field = ((x + step) % 2 == 0).astype(float)

    fig = make_subplots(
        rows=1,
        cols=2 if with_lips else 1,
        column_widths=[0.7, 0.3] if with_lips else None,
        horizontal_spacing=0.05,
    )

    # The tract, one rectangle per tube section. The bars are narrow enough
    # that the gap between two sections survives the pixel grid.
    shape = diameter / diameter.max() * limit
    fig.add_trace(
        go.Bar(
            x=x,
            y=2 * shape,
            base=-shape,
            width=0.85,
            marker=dict(color="rgba(136,136,136,0.18)", line_width=0),
            hoverinfo="skip",
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=x - 0.18,
            y=left,
            width=0.32,
            marker=dict(color=LEFT_COLOR, line_width=0, opacity=1.0 - field),
            name="left-going  ←",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=x + 0.18,
            y=right,
            width=0.32,
            marker=dict(color=RIGHT_COLOR, line_width=0, opacity=field),
            name="right-going  →",
        ),
        row=1,
        col=1,
    )
    fig.update_yaxes(range=[-limit, limit], title_text="wave amplitude", row=1, col=1)
    fig.update_xaxes(
        range=[-0.7, n - 0.3],
        title_text="tube section",
        dtick=1,
        tickfont=dict(size=9),
        row=1,
        col=1,
    )

    if with_lips:
        history = np.full(len(lips), np.nan)
        history[: step + 1] = lips[: step + 1]
        fig.add_trace(
            go.Scatter(
                x=np.arange(len(lips)),
                y=history,
                mode="lines",
                line=dict(color=LIPS_COLOR, width=1.5),
                showlegend=False,
            ),
            row=1,
            col=2,
        )
        fig.add_trace(
            go.Scatter(
                x=[step],
                y=[lips[step]],
                mode="markers",
                marker=dict(color=RIGHT_COLOR, size=8),
                showlegend=False,
            ),
            row=1,
            col=2,
        )
        fig.update_yaxes(
            range=[-lips_limit, lips_limit], title_text="output at the lips", row=1, col=2
        )
        fig.update_xaxes(range=[0, len(lips) - 1], title_text="step", row=1, col=2)

    fig.update_layout(
        width=WIDTH,
        height=HEIGHT,
        template="plotly_white",
        bargap=0,
        barmode="overlay",
        margin=dict(t=30, b=60),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=0.99,
            x=0.01,
            bgcolor="rgba(255,255,255,0.7)",
        ),
    )
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip", default="clip-21b")
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--frame", type=int, default=None, help="control frame index")
    parser.add_argument("--fps", type=float, default=12.5)
    parser.add_argument(
        "--lips",
        action="store_true",
        help="add a panel with the impulse response coming out of the lips",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("/tmp/waveguide-impulse.mp4"),
        help="mp4 path; the frames go to a folder of the same name",
    )
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame_dir = args.out.with_suffix("") / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    for old in frame_dir.glob("*.png"):
        old.unlink()

    params, frame_rate, voiced = load_params(args.clip)
    intensity = params[0, :, pt.PARAM_NAMES.index("intensity")].numpy()
    frame = args.frame if args.frame is not None else pick_frame(voiced, intensity)

    right, left, lips, diameter = record_impulse_response(params, frame, args.steps)
    limit = float(np.abs(np.concatenate([right, left])).max()) * 1.1
    lips_limit = float(np.abs(lips).max()) * 1.1
    shown = list(range(0, args.steps, STRIDE))

    figures = [
        frame_figure(
            right[i], left[i], lips, diameter, i, limit, lips_limit, args.lips
        )
        for i in shown
    ]
    # write_images ignores the figure size, so pass it explicitly.
    pio.write_images(
        figures,
        [frame_dir / f"{j:04d}.png" for j in range(len(shown))],
        scale=1,
        width=WIDTH,
        height=HEIGHT,
    )

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-framerate",
            f"{args.fps:g}",
            "-i",
            str(frame_dir / "%04d.png"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(args.out),
        ],
        check=True,
    )
    slowdown = STEP_RATE / args.fps * STRIDE
    print(
        f"frame {frame} ({frame / frame_rate:.2f} s), {args.steps} steps, "
        f"{len(shown)} rendered, {slowdown:.0f}x slow motion"
    )
    print(f"wrote {args.out} and {frame_dir}")


if __name__ == "__main__":
    main()
