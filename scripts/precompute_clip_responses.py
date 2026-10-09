"""Precompute the model's answer for each webapp clip, so the demo is instant.

The six numbered buttons in the webapp always feed the model the same six
committed MP3s, so the answer never changes as long as the checkpoint doesn't.
This runs each clip through a running backend once and commits the result under
``webapp/public/clips/precomputed/``; the frontend plays that instead of calling
``/api/synthesize`` (see ``webapp/lib/audio.ts``).

Start a backend first (any of the usual ways, e.g. ``uv run --extra server
python -m samuel.server``), then:

    uv run python scripts/precompute_clip_responses.py           # (re)generate
    uv run python scripts/precompute_clip_responses.py --check   # verify, write nothing

**Re-run this whenever the served checkpoint changes** — otherwise the buttons
demo the old model. Two things make that hard to forget: ``--check`` fails
loudly, and the frontend compares ``index.json``'s fingerprint against
``/api/health`` at runtime, ignoring the precomputed files (and warning in the
debug panel) when they disagree. The fingerprint is a content hash of the
weights, so it is the same on every machine — see ``_model_fingerprint`` in
``samuel/server.py``.

The reference audio (``synth_audio_b64``) is dropped: only the debug panel's
session download uses it, and keeping it would multiply these files tenfold.
"""

from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CLIPS_DIR = REPO_ROOT / "webapp" / "public" / "clips"
DEFAULT_OUT_DIR = CLIPS_DIR / "precomputed"
DEFAULT_SERVER = "http://127.0.0.1:8471"

# Trajectories are in Pink Trombone's native units — tract indices in [0, 44],
# diameters in [-2, 3.5], f0 in Hz. Four decimals is far below what any of them
# can be heard at, and roughly halves the file.
DECIMALS = 4


def _get(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=30) as response:
        return json.load(response)


def _post_audio(url: str, path: Path) -> dict:
    request = urllib.request.Request(
        url,
        data=path.read_bytes(),
        headers={"Content-Type": "audio/mpeg"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        return json.load(response)


def _server_fingerprint(server: str) -> str:
    try:
        health = _get(f"{server}/api/health")
    except (urllib.error.URLError, TimeoutError) as e:
        raise SystemExit(
            f"no backend at {server} ({e}) — start one with "
            "`uv run --extra server python -m samuel.server`, or pass --server"
        )
    fingerprint = health.get("model_fingerprint")
    if not fingerprint:
        raise SystemExit(
            f"{server}/api/health has no model_fingerprint — it is running an "
            "older samuel.server than this script"
        )
    print(f"backend {server}: {health['checkpoint']} (fingerprint {fingerprint})")
    return fingerprint


def _slim(response: dict) -> dict:
    """The response as committed: no reference audio, fewer decimals."""
    return {
        **{k: v for k, v in response.items() if k != "synth_audio_b64"},
        "params": {
            name: [round(v, DECIMALS) for v in values]
            for name, values in response["params"].items()
        },
    }


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", default=DEFAULT_SERVER)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--check",
        action="store_true",
        help="report whether the committed responses match the served model, "
        "write nothing",
    )
    args = parser.parse_args()

    clips = json.loads((CLIPS_DIR / "sources.json").read_text())
    fingerprint = _server_fingerprint(args.server)
    index_path = args.out_dir / "index.json"

    if args.check:
        if not index_path.exists():
            raise SystemExit(f"{index_path} does not exist — run without --check")
        index = json.loads(index_path.read_text())
        missing = [c["name"] for c in clips if c["name"] not in index["clips"]]
        stale = index["model_fingerprint"] != fingerprint
        if stale:
            print(f"fingerprint: {index['model_fingerprint']} (committed) != served")
        if missing:
            print(f"no precomputed response for: {missing}")
        if stale or missing:
            raise SystemExit(
                "precomputed clip responses are out of date — re-run "
                "`uv run python scripts/precompute_clip_responses.py`"
            )
        print(f"ok: {len(index['clips'])} clip(s) match the served model")
        return

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for clip in clips:
        response = _slim(
            _post_audio(f"{args.server}/api/synthesize", CLIPS_DIR / clip["name"])
        )
        path = args.out_dir / f"{Path(clip['name']).stem}.json"
        _write_json(path, response)
        print(
            f"{clip['name']}  {response['n_frames']} frames  "
            f"{path.stat().st_size / 1024:.0f} kB  -> {path.relative_to(REPO_ROOT)}"
        )

    # Written last, so a run interrupted halfway leaves the old (consistent)
    # index rather than one promising responses that were never written.
    _write_json(
        index_path,
        {"model_fingerprint": fingerprint, "clips": [c["name"] for c in clips]},
    )
    print(f"wrote {index_path.relative_to(REPO_ROOT)} (fingerprint {fingerprint})")


if __name__ == "__main__":
    main()
