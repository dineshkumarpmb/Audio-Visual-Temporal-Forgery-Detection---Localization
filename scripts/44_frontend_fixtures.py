"""P14-10 support: real API responses for the frontend's tests, with ground truth beside them.

    python scripts/44_frontend_fixtures.py

Runs a fixed set of LAV-DF **dev** clips through the real Phase 13 app (`api.main.create_app`
on the shipped model, in-process) and saves each `GET /api/v1/jobs/{id}` body to
`frontend/src/test/fixtures/<video_id>.json`, plus `ground_truth.json` with the labelled
`fake_periods`. The frontend's tests render these exact responses, so "the timeline matches
the returned segments" is checked against what the API really returns for a known partially
fake video -- not a hand-written mock.

The set covers each case P14-10 names that LAV-DF has: a real clip; the most-forged fake
on disk (single span); two-span fakes, one of them `visual_only` (the weak class on test,
P11-8); and an `audio_only` fake. **LAV-DF has no fully fake clip** -- the most-forged local
dev clip is 21.7% manipulated -- so "fully fake" is covered only by the highest-coverage clip,
and recorded as such. Dev only; the test split is spent.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from fastapi.testclient import TestClient  # noqa: E402

from api.main import create_app  # noqa: E402
from api.settings import ApiSettings  # noqa: E402
from src.data.manifest import read_manifest  # noqa: E402
from src.utils.console import init_console  # noqa: E402

CLIPS = {
    "dev_004337": "real",
    "dev_034103": "most-forged clip on disk (21.7%, both modalities), single span",
    "dev_004623": "two spans, both modalities",
    "dev_005675": "two spans, visual_only (the weak class on test)",
    "dev_005417": "single span, audio_only",
}


def iou_1d(a: tuple[float, float], b: tuple[float, float]) -> float:
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


def main() -> int:
    init_console()
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--clips-dir", default="data/raw/LAV-DF/dev-2k")
    ap.add_argument("--manifest", default="data/manifests/manifest_v1.parquet")
    ap.add_argument("--out", default="frontend/src/test/fixtures")
    args = ap.parse_args()

    manifest = read_manifest(args.manifest).set_index("video_id")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    settings = ApiSettings(_env_file=None, upload_dir=Path("data/interim/api_fixture_uploads"))
    truth = {}
    with TestClient(create_app(settings)) as client:
        for vid, note in CLIPS.items():
            row = manifest.loc[vid]
            assert row["split"] == "dev", f"{vid} is not dev -- the test split is spent"
            path = Path(args.clips_dir) / f"{vid}.mp4"
            with path.open("rb") as f:
                resp = client.post("/api/v1/analyze", files={"file": (path.name, f, "video/mp4")})
            resp.raise_for_status()
            job_id = resp.json()["job_id"]
            while (body := client.get(f"/api/v1/jobs/{job_id}").json())["status"] not in (
                "succeeded",
                "failed",
            ):
                time.sleep(0.2)
            (out / f"{vid}.json").write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
            gt = [[float(s), float(e)] for s, e in (row["fake_periods"] if row["label"] else [])]
            res = body.get("result") or {}
            pred = [(s["start_s"], s["end_s"]) for s in res.get("segments", [])]
            best = [max((iou_1d(tuple(g), p) for p in pred), default=0.0) for g in gt]
            truth[vid] = {
                "note": note,
                "label": int(row["label"]),
                "class_name": row["class_name"],
                "fake_periods": gt,
                "best_iou_per_true_span": [round(b, 3) for b in best],
            }
            print(
                f"  {vid}  {row['class_name']:<12} {body['status']}  verdict {res.get('verdict')}  "
                f"pred {pred}  truth {gt}  IoU {truth[vid]['best_iou_per_true_span']}"
            )
    (out / "ground_truth.json").write_text(json.dumps(truth, indent=2) + "\n", encoding="utf-8")
    print(f"  wrote {len(CLIPS)} fixtures + ground_truth.json to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
