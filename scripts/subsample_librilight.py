import argparse
import json
import multiprocessing
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tqdm


def get_file_duration_and_sample_rate(path: Path):
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(path),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True,
    )

    import json

    data = json.loads(result.stdout)

    duration = float(data["format"]["duration"])
    # Find the first audio stream
    audio_stream = next(
        (s for s in data["streams"] if s["codec_type"] == "audio"), None
    )
    sample_rate = int(audio_stream["sample_rate"]) if audio_stream else None

    return duration, sample_rate


@dataclass
class DatasetFile:
    path: Path
    duration: float | None = None
    sample_rate: int | None = None
    size_bytes: int | None = None


def add_file_info(dataset_file: DatasetFile):
    if dataset_file.duration is not None and dataset_file.sample_rate is not None:
        return dataset_file

    duration, sample_rate = get_file_duration_and_sample_rate(dataset_file.path)
    dataset_file.duration = duration
    dataset_file.sample_rate = sample_rate
    dataset_file.size_bytes = dataset_file.path.stat().st_size
    return dataset_file


def get_metadata(dataset_files: list[DatasetFile], n_processes: int = 4):
    with multiprocessing.Pool(n_processes) as pool:
        new_dataset_files = pool.map(
            add_file_info,
            tqdm.tqdm(
                dataset_files,
                disable=len(dataset_files) < 1000,
                leave=False,
                desc="Getting metadata",
            ),
            chunksize=256,
        )
    return new_dataset_files


def main(librilight_dir: Path, target_hours: float):
    paths = list(tqdm.tqdm(librilight_dir.rglob("*.flac"), desc="Listing audio files"))

    rng = np.random.default_rng(37)
    rng.shuffle(paths)

    dataset_files = [DatasetFile(p) for p in paths]

    print("Computing audio file durations")
    n_files_scanned = 1
    while n_files_scanned < len(paths):
        n_files_scanned = min(n_files_scanned * 2, len(paths))

        t1 = time.time()
        new_dataset_files = get_metadata(dataset_files[:n_files_scanned])
        for i in range(n_files_scanned):
            dataset_files[i] = new_dataset_files[i]
        t2 = time.time()

        total_duration_s = sum([f.duration for f in dataset_files[:n_files_scanned]])
        total_duration_h = total_duration_s / 60 / 60
        print(
            f"First {n_files_scanned:>7} files: {total_duration_h:>7.1f}h. "
            f"Took {t2 - t1:>7.0f}s to process"
        )

        if total_duration_h >= target_hours:
            break

    total_duration_h = 0
    n_files_to_take = 0
    for n_files_to_take in range(len(paths)):
        total_duration_h += dataset_files[n_files_to_take].duration / 60 / 60
        if total_duration_h >= target_hours:
            break

    print(f"Actual total duration: {total_duration_h}h")

    output_filename = f"librilight_{int(target_hours)}h.jsonl"
    output_path = Path(__file__).resolve().parents[1] / "manifests" / output_filename
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, "w") as f:
        for file in dataset_files[:n_files_to_take]:
            line = json.dumps(
                {
                    "path": str(file.path.resolve()),
                    "duration": file.duration,
                    "sample_rate": file.sample_rate,
                    "size_bytes": file.size_bytes,
                },
            )
            f.write(line + "\n")
    print(f"Wrote a list of {n_files_to_take} files to {output_path}")
    total_size_bytes = sum(file.size_bytes for file in dataset_files[:n_files_to_take])
    print(f"The included files have about {total_size_bytes / 1024**2:.1f} MB in total")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("librilight_dir", type=Path)
    parser.add_argument("--target-hours", type=float, required=True)
    args = parser.parse_args()

    main(args.librilight_dir, args.target_hours)
