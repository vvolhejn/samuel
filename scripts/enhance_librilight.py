"""Enhance LibriLight audio with resemble-enhance and store it as a parallel tree.

Reads a manifest (default: manifests/librilight_1000h.jsonl), processes the files
in a deterministically shuffled order (fixed --seed), and writes enhanced audio
to --out-root, mirroring the directory structure of the source dataset.

Design notes:
- The train/val split in samuel.data is positional (last val_fraction of the
  manifest), so processing in manifest order would convert all-train-then-val.
  The seeded shuffle here makes any partially converted prefix cover train and
  val proportionally.
- The shuffled order is a stable permutation of the manifest: rerunning with a
  larger --target-hours (or no limit) converts a superset of what an earlier,
  smaller run converted. Pointing at a bigger manifest later also works, since
  files whose output already exists are skipped.
- Crash-robust: outputs are written to a temp file and os.replace()d into
  place, so an interrupted run never leaves a truncated output file; rerunning
  resumes where it left off. Per-file errors are logged to
  <out-root>/failures.jsonl and the run continues (failed files are retried on
  the next run because their output doesn't exist).
- Multi-GPU / multi-job: run N copies with --num-shards N --shard-idx i.
  Shards are disjoint interleaved slices of the shuffled order.

Environment: resemble-enhance pins torch==2.1.1, which is incompatible with the
project venv (py3.12 / torch 2.8), so it lives in a dedicated venv. To recreate:

    uv venv .venv-resemble --python 3.11
    uv pip install --python .venv-resemble torch==2.1.1 torchaudio==2.1.1
    uv pip install --python .venv-resemble 'setuptools<81' wheel packaging
    uv pip install --python .venv-resemble --no-build-isolation resemble-enhance soundfile tqdm

(deepspeed builds from source, hence --no-build-isolation with torch
preinstalled; setuptools<81 because its build needs pkg_resources.)
Model weights auto-download on first run (needs internet once).

Usage (with the dedicated venv):
    .venv-resemble/bin/python scripts/enhance_librilight.py --target-hours 100  # first chunk
    .venv-resemble/bin/python scripts/enhance_librilight.py --target-hours 300  # extends it
    .venv-resemble/bin/python scripts/enhance_librilight.py                     # everything

    # Optional, several GPUs on one node:
    for i in 0 1 2 3; CUDA_VISIBLE_DEVICES=$i .venv-resemble/bin/python \
        scripts/enhance_librilight.py --num-shards 4 --shard-idx $i &

    # After (partial) conversion, write a manifest usable by training
    # (preserves original manifest order, so the train/val split carries over):
    .venv-resemble/bin/python scripts/enhance_librilight.py \
        --write-manifest manifests/librilight_1000h_enhanced.jsonl
"""

import argparse
import json
import os
import random
from pathlib import Path

import soundfile as sf
import torch
import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "manifests" / "librilight_1000h.jsonl"
DEFAULT_OUT_ROOT = Path(
    "/lustre/scwpod02/client/kyutai/vaclav/datasets/librilight-enhanced"
)
DEFAULT_SOURCE_ROOT = Path(
    "/lustre/scwpod02/client/kyutai/datasets/librilight_segmented"
)


def load_manifest(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def output_path(entry: dict, source_root: Path, out_root: Path, ext: str) -> Path:
    src = Path(entry["path"])
    try:
        rel = src.relative_to(source_root)
    except ValueError:
        # Fall back to splitting on the source root's directory name.
        parts = src.parts
        anchor = source_root.name
        if anchor not in parts:
            raise ValueError(
                f"{src} is not under {source_root} and has no '{anchor}' component"
            )
        rel = Path(*parts[parts.index(anchor) + 1 :])
    return out_root / rel.with_suffix(f".{ext}")


def select_files(
    entries: list[dict], seed: int, target_hours: float | None
) -> list[dict]:
    """Deterministically shuffled selection; a larger target_hours yields a superset."""
    entries = list(entries)
    random.Random(seed).shuffle(entries)
    if target_hours is None:
        return entries
    selected, total_s = [], 0.0
    for entry in entries:
        if total_s >= target_hours * 3600:
            break
        selected.append(entry)
        total_s += entry["duration"]
    return selected


def save_audio(path: Path, wav: torch.Tensor, sr: int, ext: str) -> None:
    fmt = {"ogg": ("OGG", "VORBIS"), "mp3": ("MP3", "MPEG_LAYER_III")}[ext]
    tmp = path.with_name(f"{path.name}.tmp{os.getpid()}")
    try:
        sf.write(str(tmp), wav.numpy(), sr, format=fmt[0], subtype=fmt[1])
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def write_manifest(
    entries: list[dict], source_root: Path, out_root: Path, ext: str, manifest_out: Path
) -> None:
    """Manifest of already-converted files, in original manifest order (keeps the
    positional train/val split semantics)."""
    n = 0
    with open(manifest_out, "w") as f:
        for entry in tqdm.tqdm(entries, desc="Writing manifest"):
            out = output_path(entry, source_root, out_root, ext)
            if not out.exists():
                continue
            f.write(
                json.dumps(
                    {
                        "path": str(out),
                        "duration": entry["duration"],
                        "sample_rate": 44100,  # resemble-enhance output rate
                        "size_bytes": out.stat().st_size,
                    }
                )
                + "\n"
            )
            n += 1
    print(f"Wrote {n}/{len(entries)} converted files to {manifest_out}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--ext", choices=["ogg", "mp3"], default="ogg")
    parser.add_argument(
        "--target-hours",
        type=float,
        default=None,
        help="Convert only the first N hours of the shuffled order (default: all). "
        "Increasing this later extends the previous selection.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Shuffle seed. Don't change it between runs, or the "
        "prefix-extension property of --target-hours breaks.",
    )
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-idx", type=int, default=0)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    # resemble-enhance knobs (defaults match its CLI)
    parser.add_argument("--nfe", type=int, default=32)
    parser.add_argument(
        "--solver", choices=["midpoint", "rk4", "euler"], default="midpoint"
    )
    parser.add_argument(
        "--lambd", type=float, default=0.9, help="Denoise strength before enhancement"
    )
    parser.add_argument("--tau", type=float, default=0.5, help="CFM prior temperature")
    parser.add_argument(
        "--dry-run", action="store_true", help="Only print selection stats"
    )
    parser.add_argument(
        "--write-manifest",
        type=Path,
        default=None,
        metavar="PATH",
        help="Don't convert; write a jsonl manifest of already-converted files to PATH",
    )
    args = parser.parse_args()
    assert 0 <= args.shard_idx < args.num_shards

    entries = load_manifest(args.manifest)

    if args.write_manifest is not None:
        write_manifest(
            entries, args.source_root, args.out_root, args.ext, args.write_manifest
        )
        return

    selected = select_files(entries, args.seed, args.target_hours)
    shard = selected[args.shard_idx :: args.num_shards]
    shard_hours = sum(e["duration"] for e in shard) / 3600
    print(
        f"Selected {len(selected)}/{len(entries)} files "
        f"({sum(e['duration'] for e in selected) / 3600:.1f}h); "
        f"shard {args.shard_idx}/{args.num_shards} has {len(shard)} files ({shard_hours:.1f}h)"
    )
    if args.dry_run:
        return

    from resemble_enhance.enhancer.inference import enhance  # slow import; do it late

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    failures_log = args.out_root / "failures.jsonl"
    args.out_root.mkdir(parents=True, exist_ok=True)

    done_s = skipped = failed = 0
    pbar = tqdm.tqdm(shard, desc="Enhancing", unit="file")
    for entry in pbar:
        out = output_path(entry, args.source_root, args.out_root, args.ext)
        if out.exists():
            skipped += 1
            done_s += entry["duration"]
        else:
            try:
                wav, sr = sf.read(entry["path"], dtype="float32", always_2d=True)
                dwav = torch.from_numpy(wav).mean(dim=1)
                hwav, new_sr = enhance(
                    dwav,
                    sr,
                    args.device,
                    nfe=args.nfe,
                    solver=args.solver,
                    lambd=args.lambd,
                    tau=args.tau,
                )
                out.parent.mkdir(parents=True, exist_ok=True)
                save_audio(out, hwav.cpu(), new_sr, args.ext)
                done_s += entry["duration"]
            except KeyboardInterrupt:
                raise
            except Exception as e:
                failed += 1
                if isinstance(e, torch.cuda.OutOfMemoryError):
                    torch.cuda.empty_cache()
                with open(failures_log, "a") as f:
                    f.write(
                        json.dumps({"path": entry["path"], "error": repr(e)}) + "\n"
                    )
        pbar.set_postfix(
            hours=f"{done_s / 3600:.1f}/{shard_hours:.1f}",
            skipped=skipped,
            failed=failed,
        )


if __name__ == "__main__":
    main()
