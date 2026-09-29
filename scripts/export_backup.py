"""Stage a portable backup of code, W&B data, checkpoints and eval audio.

Run with: uv run --with pandas --with pyarrow python scripts/export_backup.py
"""

import argparse
import concurrent.futures
import gzip
import json
import random
import re
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path

import pandas as pd
import wandb
from tqdm import tqdm

REPO = Path(__file__).resolve().parent.parent
WORKTREES = REPO / ".claude" / "worktrees"
EXCLUDED_WORKTREES = {"pocket-tts-samuel"}
MEDIA_RE = re.compile(r"^(.*?)_(\d+)_[0-9a-f]+\.[\w.]+$")


def run_roots() -> list[Path]:
    roots = [REPO / "runs"]
    for wt in sorted(WORKTREES.iterdir()):
        if wt.name not in EXCLUDED_WORKTREES and (wt / "runs").is_dir():
            roots.append(wt / "runs")
    return roots


def run_dirs() -> list[Path]:
    return [d for root in run_roots() for d in sorted(root.iterdir()) if d.is_dir()]


def rel(p: Path) -> str:
    return str(p.relative_to(REPO))


def tar(out: Path, files: list[str], compress: bool) -> None:
    """Tar repo-relative paths, renaming .claude/worktrees/ to worktrees/."""
    with tempfile.NamedTemporaryFile("w", suffix=".txt") as f:
        f.write("\n".join(files) + "\n")
        f.flush()
        cmd = [
            "tar",
            "-C",
            str(REPO),
            "-cf",
            str(out),
            "--transform",
            r"s|^\.claude/worktrees/|worktrees/|",
            "-T",
            f.name,
        ]
        if compress:
            cmd[3:3] = ["-I", "zstd -T0 -10"]
        subprocess.run(cmd, check=True)


def export_code(out: Path) -> None:
    code = out / "code"
    code.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(REPO),
            "bundle",
            "create",
            str(code / "samuel.bundle"),
            "--all",
        ],
        check=True,
    )
    diffs = code / "worktree-uncommitted"
    diffs.mkdir(exist_ok=True)
    for wt in [REPO, *sorted(WORKTREES.iterdir())]:
        if wt.name in EXCLUDED_WORKTREES or not (wt / ".git").exists():
            continue
        name = "main" if wt == REPO else wt.name
        diff = subprocess.run(
            ["git", "-C", str(wt), "diff", "HEAD", "--binary"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        untracked = subprocess.run(
            ["git", "-C", str(wt), "ls-files", "--others", "--exclude-standard"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.split()
        untracked = [u for u in untracked if not u.startswith("pocket-tts/")]
        if diff:
            head = subprocess.run(
                ["git", "-C", str(wt), "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout
            (diffs / f"{name}.patch").write_text(f"# base: {head}{diff}")
        if untracked:
            subprocess.run(
                [
                    "tar",
                    "-C",
                    str(wt),
                    "-cf",
                    str(diffs / f"{name}.untracked.tar"),
                    *untracked,
                ],
                check=True,
            )


def export_wandb_run(run, dest: Path) -> dict:
    d = dest / run.id
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(json.dumps(run.config, indent=1, default=str))
    (d / "summary.json").write_text(
        json.dumps(dict(run.summary._json_dict), indent=1, default=str)
    )
    (d / "metadata.json").write_text(
        json.dumps(run.metadata or {}, indent=1, default=str)
    )
    rows = list(run.scan_history())
    with gzip.open(d / "history.jsonl.gz", "wt") as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")
    scalars = pd.DataFrame(
        [
            {
                k: v
                for k, v in r.items()
                if isinstance(v, (int, float, bool)) or v is None
            }
            for r in rows
        ]
    )
    scalars.to_parquet(d / "history.parquet")
    system = run.history(stream="system", pandas=True, samples=100_000)
    if len(system):
        system.to_parquet(d / "system.parquet")
    return {
        "id": run.id,
        "name": run.name,
        "state": run.state,
        "created_at": run.created_at,
        "tags": ",".join(run.tags),
        "group": run.group,
        "url": run.url,
        "history_rows": len(rows),
    }


def export_wandb(out: Path, entity: str, project: str, workers: int) -> None:
    dest = out / "wandb_export"
    runs = list(wandb.Api(timeout=120).runs(f"{entity}/{project}"))
    index = []
    with concurrent.futures.ThreadPoolExecutor(workers) as pool:
        futures = {pool.submit(export_wandb_run, r, dest / "runs"): r for r in runs}
        for fut in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc="wandb runs",
        ):
            index.append(fut.result())
    local = {}
    for f in (
        d for root in run_roots() for d in root.glob("*/wandb/run-*/run-*.wandb")
    ):
        local[f.stem.removeprefix("run-")] = rel(f.parent.parent.parent)
    df = pd.DataFrame(index).sort_values("created_at")
    df["local_run_dir"] = df["id"].map(local)
    df.to_csv(dest / "runs.csv", index=False)


def wandb_local_files() -> list[str]:
    """Everything in local wandb run dirs except the media folder."""
    files = []
    for rd in run_dirs():
        for p in (rd / "wandb").glob("run-*/**/*"):
            if p.is_file() and not p.is_symlink() and "/files/media/" not in str(p):
                files.append(rel(p))
    return files


def checkpoint_files() -> list[str]:
    files = []
    for rd in run_dirs():
        if (rd / "config.json").exists():
            files.append(rel(rd / "config.json"))
        ckpts = sorted((rd / "checkpoints").glob("[0-9]*.pt"))
        if ckpts:
            files.append(rel(ckpts[-1]))
    return files


def audio_files(seed: int) -> list[str]:
    """All audio at each key's last step, plus one random clip per earlier step."""
    rng = random.Random(seed)
    groups: dict[tuple, dict[int, list[Path]]] = defaultdict(lambda: defaultdict(list))
    for rd in run_dirs():
        for media in (rd / "wandb").glob("run-*/files/media/audio"):
            for p in sorted(media.rglob("*")):
                m = MEDIA_RE.match(p.name)
                if p.is_file() and m:
                    groups[(p.parent, m.group(1))][int(m.group(2))].append(p)
    files = []
    for steps in groups.values():
        last = max(steps)
        for step, ps in sorted(steps.items()):
            files.extend(rel(p) for p in (ps if step == last else [rng.choice(ps)]))
    return files


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path.home() / "samuel-export")
    ap.add_argument("--entity", default="moboehle-kyutai")
    ap.add_argument("--project", default="samuel")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--steps", nargs="+", default=["code", "wandb", "checkpoints", "audio", "misc"]
    )
    args = ap.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    if "code" in args.steps:
        export_code(out)
    if "wandb" in args.steps:
        export_wandb(out, args.entity, args.project, args.workers)
        tar(out / "wandb_local.tar.zst", wandb_local_files(), compress=True)
    if "checkpoints" in args.steps:
        tar(out / "checkpoints.tar", checkpoint_files(), compress=False)
    if "audio" in args.steps:
        tar(out / "audio.tar", audio_files(args.seed), compress=False)
    if "misc" in args.steps:
        misc = [
            m
            for m in ["manifests", "artifacts", "logs", "notebooks", "train.log"]
            if (REPO / m).exists()
        ]
        tar(out / "misc.tar.zst", misc, compress=True)


if __name__ == "__main__":
    main()
