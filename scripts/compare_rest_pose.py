"""Measure rest-pose distance for checkpoints that predate ``loss.rest``.

The baseline run logged no rest metrics -- the term did not exist -- so the
only way to say what the prior actually bought is to score its checkpoint
offline on the same eval clips and the same silent-frame split. Runs are
scored one after another through the identical path, so the numbers are
directly comparable to each other and to the ``eval/rest_*`` series.
"""

import argparse
import json
from pathlib import Path

import torch
from omegaconf import OmegaConf

from samuel.config import TrainConfig
from samuel.model import PinkTromboneController
from samuel.train import (
    _eval_setup,
    _normalized_trainable_diffs,
    _rest_pose_distance,
    _rms_normalize,
    _run_eval_batched,
    _silent_frame_rest_metrics,
)

REST_TARGETS = {
    "lipDiameter": 0.3,
    "constrictionDiameter": 3.0,
    "tongueDiameter": 3.5,
}


def score(run_dir: Path, ckpt: str, n_clips: int, device: torch.device) -> dict:
    cfg = TrainConfig.from_hydra(
        OmegaConf.create(json.loads((run_dir / "config.json").read_text()))
    )
    cfg.log.asr_whisper_size = ""
    cfg.log.n_eval_clips = n_clips
    # Score every checkpoint against the same posture, whatever the run trained
    # with (the baseline trained with none).
    cfg.loss.rest_targets = dict(REST_TARGETS)

    model = PinkTromboneController(cfg.model).to(device)
    state = torch.load(run_dir / "checkpoints" / ckpt, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()

    setup = _eval_setup(cfg, cfg.model.samples_per_frame, device)
    target = _rms_normalize(setup.val_wavs.to(device), cfg.data.target_rms)
    f0 = setup.val_f0.to(device)
    params, _ = _run_eval_batched(model, target, f0, cfg, cfg.model.frame_rate)

    dist = _rest_pose_distance(params, model, REST_TARGETS)
    out = {"step": state.get("step"), "run": run_dir.name}
    for name, v in zip(model.trainable_names_, dist.tolist()):
        if name in REST_TARGETS:
            out[f"all/{name}"] = v
    out.update(
        {
            k.replace("eval/rest_dist_silent/", "silent/").replace("eval/", ""): v
            for k, v in _silent_frame_rest_metrics(params, model, target, cfg).items()
        }
    )
    var = _normalized_trainable_diffs(params, model).mean(dim=(0, 1))
    for name, v in zip(model.trainable_names_, var.tolist()):
        if name in REST_TARGETS:
            out[f"var/{name}"] = v
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True, help="run_dir[:ckpt]")
    ap.add_argument("--n-clips", type=int, default=200)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rows = []
    for spec in args.run:
        run_dir, _, ckpt = spec.partition(":")
        rows.append(score(Path(run_dir), ckpt or "last.pt", args.n_clips, device))

    keys = [k for k in rows[0] if k not in ("run", "step")]
    print(f"\n{'metric':38s}" + "".join(f"{r['run'][:22]:>24s}" for r in rows))
    print(f"{'step':38s}" + "".join(f"{r['step']:>24}" for r in rows))
    for k in keys:
        print(f"{k:38s}" + "".join(f"{r.get(k, float('nan')):24.4f}" for r in rows))


if __name__ == "__main__":
    main()
