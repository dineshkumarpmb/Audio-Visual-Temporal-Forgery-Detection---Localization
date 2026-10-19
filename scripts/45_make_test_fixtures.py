"""P15-4 / P15-6 / P15-2: build the committed test fixtures under `tests/fixtures/`.

    python scripts/45_make_test_fixtures.py            # everything
    python scripts/45_make_test_fixtures.py --expected # only re-derive expected_*.json

* `tiny_real.mkv`, `tiny_fake.mkv` -- 2 s synthetic clips (`tests/regression.py` says why
  synthetic, and why `.mkv`). Regenerating them changes their bytes with the ffmpeg build,
  so they are made once and committed.
* `expected_tiny_*.json` -- what `VideoAnalyzer` returns for each, on **CPU**, with the
  platform-independent seeded bundle. The regression test compares against these.
* `lavdf_labels_sample.json` -- real LAV-DF labels for the GT-alignment test, so it runs
  without the dataset: 300 random entries (seed 0, all splits) plus up to 150 whose span
  boundaries sit within float error of a frame edge -- the exact class of label P10-1 found
  being shifted by a frame (`src/localization/targets.py`, `FRAME_SNAP`). Labels only; no
  media, no video ids beyond the public metadata's own.

Re-run `--expected` only when a deliberate pipeline change moves the numbers, and say so in
the commit -- that is the point of a regression fixture.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tests.regression import (  # noqa: E402
    EXPECTED,
    LABELS_SAMPLE,
    REGRESSION_POSTPROCESS,
    TINY_FAKE,
    TINY_REAL,
    make_tiny_videos,
    result_summary,
    seeded_bundle,
)

N_RANDOM = 300
N_EDGE = 150


def labels_sample(manifest_path: Path) -> dict:
    from src.data.manifest import read_manifest
    from src.localization.targets import normalise_periods

    m = read_manifest(manifest_path)
    rows = []
    for r in m.itertuples(index=False):
        spans = normalise_periods(r.fake_periods)
        rows.append((r.video_id, r.split, int(r.label), int(r.n_frames), float(r.fps), spans))

    def on_float_edge(spans: list) -> bool:
        for s, e in spans:
            for x in (s * 25.0, e * 25.0):
                k = round(x)
                if x != k and abs(x - k) < 1e-6:
                    return True
        return False

    rng = np.random.default_rng(0)
    picked = set(rng.choice(len(rows), size=N_RANDOM, replace=False).tolist())
    edge = [i for i, row in enumerate(rows) if on_float_edge(row[5])]
    picked |= set(edge[:N_EDGE])
    entries = [
        {
            "video_id": rows[i][0],
            "split": rows[i][1],
            "label": rows[i][2],
            "n_frames": rows[i][3],
            "fps": rows[i][4],
            "fake_periods": [[s, e] for s, e in rows[i][5]],
        }
        for i in sorted(picked)
    ]
    return {
        "source": manifest_path.as_posix(),
        "n_manifest": len(rows),
        "n_float_edge_in_manifest": len(edge),
        "n_float_edge_in_sample": min(len(edge), N_EDGE),
        "entries": entries,
    }


def expected_outputs() -> None:
    import av
    import torch

    from src.inference.pipeline import VideoAnalyzer

    with VideoAnalyzer(seeded_bundle(), device="cpu") as analyzer:
        for clip, out in EXPECTED.items():
            res = analyzer.analyze(clip, details=True, keyframes=True)
            body = {
                "clip": clip.name,
                "generated_with": {
                    "device": "cpu",
                    "torch": torch.__version__,
                    "numpy": np.__version__,
                    "pyav": av.__version__,
                    "platform": platform.platform(),
                },
                "postprocess": REGRESSION_POSTPROCESS,
                **result_summary(res),
            }
            out.write_text(json.dumps(body, indent=1) + "\n", encoding="utf-8")
            scores = np.asarray(body["frame_scores"])
            print(
                f"  {clip.name}: {body['n_frames']} frames, clip {body['clip_score']:.4f}, "
                f"frame scores {scores.min():.4f}-{scores.max():.4f}, "
                f"{len(body['segments'])} segments -> {out.name}"
            )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--expected", action="store_true", help="only re-derive expected_*.json")
    ap.add_argument("--manifest", default="data/manifests/manifest_v1.parquet")
    args = ap.parse_args()

    if not args.expected:
        real, fake = make_tiny_videos()
        print(
            f"  wrote {real.name} ({real.stat().st_size} B), {fake.name} ({fake.stat().st_size} B)"
        )
        sample = labels_sample(Path(args.manifest))
        LABELS_SAMPLE.write_text(json.dumps(sample, indent=0) + "\n", encoding="utf-8")
        print(
            f"  wrote {LABELS_SAMPLE.name}: {len(sample['entries'])} entries "
            f"({sample['n_float_edge_in_sample']} of {sample['n_float_edge_in_manifest']} "
            "float-edge labels)"
        )
    assert TINY_REAL.is_file() and TINY_FAKE.is_file(), "fixtures missing; run without --expected"
    expected_outputs()
    return 0


if __name__ == "__main__":
    sys.exit(main())
