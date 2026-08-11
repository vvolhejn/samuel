"""Cut webapp/public/clips/*.mp3 from the recipe in sources.json.

``sources.json`` is both the webapp's clip index and the recipe for the clips
beside it: each entry names a dataset file, an offset and a duration. The audio
is committed, so this script is only needed to reproduce it — after editing the
recipe by hand, or to check the committed clips against the dataset.

    uv run python scripts/build_webapp_clips.py           # (re)cut every clip
    uv run python scripts/build_webapp_clips.py --check   # compare, write nothing

Picking clips in the first place is scripts/make_webapp_clips.py, which samples
the dataset and writes the recipe entries.
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = REPO_ROOT / "webapp" / "public" / "clips"
SAMPLE_RATE = 16000
# 0.0 is libsndfile's best-quality end: ~64 kbps at 16 kHz mono, a quarter the
# size of the PCM the clips used to be, ~32 dB SNR against it.
MP3_COMPRESSION = 0.0
# A re-cut clip must decode this close to the committed one. MP3 is lossy and
# encoders differ between machines, so --check can't compare bytes.
MIN_CHECK_SNR_DB = 25.0


def write_clip(path: Path | io.BytesIO, audio: np.ndarray) -> None:
    """Encode ``audio`` as the clips' committed format."""
    sf.write(
        path,
        audio,
        SAMPLE_RATE,
        format="MP3",
        subtype="MPEG_LAYER_III",
        compression_level=MP3_COMPRESSION,
    )


def _recode(audio: np.ndarray) -> np.ndarray:
    """``audio`` through the same encode/decode a committed clip has been."""
    buffer = io.BytesIO()
    write_clip(buffer, audio)
    buffer.seek(0)
    return sf.read(buffer, dtype="float32")[0]


def _snr_db(reference: np.ndarray, other: np.ndarray) -> float:
    """Signal-to-error ratio of ``other`` against ``reference``, or -inf if
    they aren't even the same length."""
    if len(reference) != len(other):
        return float("-inf")
    error = float(np.sqrt(((other - reference) ** 2).mean()))
    if error == 0:
        return float("inf")
    return float(20 * np.log10(np.sqrt((reference**2).mean()) / error))


def _cut(entry: dict) -> np.ndarray:
    """The entry's window of its source file, mono at SAMPLE_RATE."""
    info = sf.info(entry["source"])
    start = round(entry["offset_s"] * info.samplerate)
    frames = round(entry["duration_s"] * info.samplerate)
    audio, sr = sf.read(
        entry["source"], start=start, frames=frames, dtype="float32", always_2d=False
    )
    if len(audio) < frames - 1:
        raise SystemExit(
            f"{entry['name']}: source has {len(audio)} frames from {start}, "
            f"recipe wants {frames} — has {entry['source']} changed?"
        )
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
    return audio


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--check",
        action="store_true",
        help="report whether the committed clips match the recipe, write nothing",
    )
    args = parser.parse_args()

    index = json.loads((args.out_dir / "sources.json").read_text())
    stale = []
    for entry in index:
        path = args.out_dir / entry["name"]
        audio = _cut(entry)
        if not args.check:
            write_clip(path, audio)
            size_kb = path.stat().st_size / 1024
            print(
                f"{entry['name']}  {len(audio) / SAMPLE_RATE:.2f}s  "
                f"{size_kb:.0f} kB  {entry['source']}"
            )
            continue
        # Compare audio, not bytes: two MP3 encoders agree on what you hear,
        # not on how they spell it. Both sides go through the codec, so what
        # is left is the difference between the cuts, not the coding loss.
        on_disk = sf.read(path, dtype="float32")[0] if path.exists() else np.zeros(0)
        snr = _snr_db(_recode(audio), on_disk)
        ok = snr >= MIN_CHECK_SNR_DB
        print(f"{entry['name']}: {'ok' if ok else 'DIFFERS'} ({snr:.1f} dB)")
        if not ok:
            stale.append(entry["name"])

    if stale:
        raise SystemExit(f"{len(stale)} clip(s) do not match the recipe: {stale}")


if __name__ == "__main__":
    main()
