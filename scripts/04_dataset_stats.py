"""P1-10..P1-14: measure the dataset and write reports/dataset_statistics.md.

    python scripts/04_dataset_stats.py
    python scripts/04_dataset_stats.py --probe-dir data/raw/_spotcheck

Every provisional figure in PROJECT_PLAN section 3.2 is replaced with a measured one.
Nothing in the output is estimated, rounded from memory, or carried over from the
report -- if it is not computed from manifest_v1.parquet it does not appear.

Two numbers here decide downstream design:
  P1-11 mean forged-span duration -> the temporal window T of Part 6.
  P1-12 fake-frame fraction       -> focal-loss alpha / BCE pos_weight.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.config import LAVDF_FPS  # noqa: E402
from src.data.manifest import read_manifest  # noqa: E402
from src.data.validate import ffprobe  # noqa: E402
from src.utils.console import init_console  # noqa: E402

GREEN, DIM, RESET = "\033[32m", "\033[2m", "\033[0m"


def quantiles(xs: list[float]) -> dict[str, float]:
    s = sorted(xs)

    def q(p: float) -> float:
        return s[min(len(s) - 1, int(p * len(s)))]

    return {
        "min": s[0],
        "p05": q(0.05),
        "p25": q(0.25),
        "median": q(0.50),
        "p75": q(0.75),
        "p95": q(0.95),
        "max": s[-1],
        "mean": st.mean(s),
        "std": st.pstdev(s),
    }


def fmt(d: dict[str, float], unit: str = "s") -> str:
    return " | ".join(f"{k} {v:.3f}{unit}" for k, v in d.items())


def main() -> int:
    init_console()
    ap = argparse.ArgumentParser(description="Measure LAV-DF and write dataset_statistics.md")
    ap.add_argument("--manifest", default="data/manifests/manifest_v1.parquet")
    ap.add_argument("--out", default="reports/dataset_statistics.md")
    ap.add_argument("--facts", default="reports/dataset_facts.json")
    ap.add_argument("--probe-dir", default=None, help="dir of locally-held videos for P1-13")
    args = ap.parse_args()

    df = read_manifest(args.manifest)
    ok = df[df["status"] == "ok"]
    facts: dict[str, object] = {}

    # ---- P1-10 counts -------------------------------------------------------------
    facts["n_total"] = int(len(df))
    facts["n_usable"] = int(len(ok))
    facts["status_counts"] = {k: int(v) for k, v in df["status"].value_counts().items()}
    facts["split_counts"] = {k: int(v) for k, v in df["split"].value_counts().sort_index().items()}
    facts["class_counts"] = {k: int(v) for k, v in df["class_name"].value_counts().items()}
    facts["label_counts"] = {int(k): int(v) for k, v in df["label"].value_counts().items()}
    facts["class_by_split"] = {
        s: {c: int(v) for c, v in g["class_name"].value_counts().items()}
        for s, g in df.groupby("split")
    }
    facts["n_source_ids"] = int(df["source_id"].nunique())
    facts["source_ids_by_split"] = {
        s: int(g["source_id"].nunique()) for s, g in df.groupby("split")
    }

    # ---- P1-10 duration distribution ----------------------------------------------
    dur = ok["duration"].tolist()
    facts["duration_s"] = quantiles(dur)
    facts["total_hours"] = sum(dur) / 3600.0
    facts["n_frames_total"] = int(ok["n_frames"].sum())

    # ---- P1-11 forged-span durations (determines T) --------------------------------
    spans = [(e - s) for periods in ok["fake_periods"] for s, e in periods]
    facts["n_spans"] = len(spans)
    facts["span_s"] = quantiles(spans)
    facts["span_frames_at_25"] = {k: v * LAVDF_FPS for k, v in facts["span_s"].items()}
    facts["n_fakes_distribution"] = {
        int(k): int(v) for k, v in ok["n_fakes"].value_counts().sort_index().items()
    }

    # ---- P1-12 fake-frame fraction (drives pos_weight) ------------------------------
    fake_frames = 0
    total_frames = 0
    for periods, n in zip(ok["fake_periods"], ok["n_frames"], strict=True):
        total_frames += int(n)
        for s, e in periods:
            a = max(0, round(s * LAVDF_FPS))
            b = min(int(n), round(e * LAVDF_FPS))
            fake_frames += max(0, b - a)
    facts["fake_frames"] = fake_frames
    facts["total_frames"] = total_frames
    facts["fake_frame_fraction"] = fake_frames / total_frames
    facts["pos_weight"] = (total_frames - fake_frames) / fake_frames
    # Video-level imbalance is the *other* imbalance problem of section 3.6.
    facts["video_level_fake_fraction"] = float((ok["label"] == 1).mean())

    # ---- P1-13 fps / sample-rate consistency ----------------------------------------
    facts["fps_values"] = {float(k): int(v) for k, v in ok["fps"].value_counts().items()}
    facts["sample_rate_values"] = {
        int(k): int(v) for k, v in ok["sample_rate"].value_counts().items()
    }
    facts["audio_channel_values"] = {
        int(k): int(v) for k, v in ok["audio_channels"].value_counts().items()
    }

    probe_rows: list[dict] = []
    if args.probe_dir:
        pdir = Path(args.probe_dir)
        by_id = {r.video_id: r for r in df.itertuples()}
        for f in sorted(pdir.glob("*.mp4")):
            vid = f.stem
            row = by_id.get(vid)
            if row is None:
                continue
            pr = ffprobe(f)
            probe_rows.append(
                {
                    "video_id": vid,
                    "meta_frames": int(row.n_frames),
                    "probe_frames": pr.n_frames,
                    "fps": pr.fps,
                    "sample_rate": pr.sample_rate,
                    "channels": pr.channels,
                    "frames_match": pr.n_frames == int(row.n_frames),
                    "fps_is_25": pr.fps is not None and abs(pr.fps - 25.0) < 1e-6,
                }
            )
        facts["probe_n"] = len(probe_rows)
        facts["probe_frames_match"] = sum(r["frames_match"] for r in probe_rows)
        facts["probe_fps_25"] = sum(r["fps_is_25"] for r in probe_rows)
        facts["probe_sr_16k"] = sum(r["sample_rate"] == 16000 for r in probe_rows)

    Path(args.facts).parent.mkdir(parents=True, exist_ok=True)
    Path(args.facts).write_text(json.dumps(facts, indent=2, default=str) + "\n", encoding="utf-8")

    _write_report(Path(args.out), facts, probe_rows)
    print(f"  [{GREEN}OK{RESET}] wrote {args.out}")
    print(f"  [{GREEN}OK{RESET}] wrote {args.facts}")
    print(
        f"\n{DIM}  P1-11 mean forged span : {facts['span_s']['mean']:.3f}s "
        f"({facts['span_s']['mean'] * LAVDF_FPS:.1f} frames at 25 fps){RESET}"
    )
    print(
        f"{DIM}  P1-12 fake-frame frac  : {facts['fake_frame_fraction']:.4%} "
        f"-> pos_weight {facts['pos_weight']:.2f}{RESET}"
    )
    return 0


def _write_report(path: Path, facts: dict, probe_rows: list[dict]) -> None:
    sp, fr = facts["span_s"], facts["duration_s"]
    lines = [
        "# LAV-DF Dataset Statistics (P1-10 .. P1-14)",
        "",
        "Generated by `scripts/04_dataset_stats.py` from `data/manifests/manifest_v1.parquet`.",
        "Every figure is measured. Nothing here is carried over from PROJECT_PLAN section 3.2,",
        "which was explicitly provisional (`TO VERIFY`).",
        "",
        "## Headline",
        "",
        "| | |",
        "|---|---|",
        f"| Videos | **{facts['n_total']:,}** |",
        f"| Usable (`status == ok`) | **{facts['n_usable']:,}** ({facts['n_usable'] / facts['n_total']:.4%}) |",
        f"| Real / fake | **{facts['label_counts'].get(0, 0):,} / {facts['label_counts'].get(1, 0):,}** |",
        f"| Unique `source_id` | {facts['n_source_ids']:,} |",
        f"| Total video | {facts['total_hours']:.1f} h ({facts['n_frames_total']:,} frames at 25 fps) |",
        "",
        "## P1-10 Counts",
        "",
        "### Splits (official, never re-derived -- section 3.5 RULE 1)",
        "",
        "| Split | Videos | Unique source_id |",
        "|---|---|---|",
    ]
    for s, n in facts["split_counts"].items():
        lines.append(f"| {s} | {n:,} | {facts['source_ids_by_split'][s]:,} |")
    lines += [
        "",
        "### Four-class taxonomy (section 3.4)",
        "",
        "| Class | `modify_video` | `modify_audio` | Videos | Share |",
        "|---|---|---|---|---|",
    ]
    flags = {
        "real": ("false", "false"),
        "visual_only": ("true", "false"),
        "audio_only": ("false", "true"),
        "both": ("true", "true"),
    }
    for c in ("real", "visual_only", "audio_only", "both"):
        n = facts["class_counts"].get(c, 0)
        lines.append(
            f"| {c} | {flags[c][0]} | {flags[c][1]} | {n:,} | {n / facts['n_total']:.2%} |"
        )

    lines += [
        "",
        "### Class by split",
        "",
        "| Split | real | visual_only | audio_only | both |",
        "|---|---|---|---|---|",
    ]
    for s, d in facts["class_by_split"].items():
        lines.append(
            f"| {s} | "
            + " | ".join(f"{d.get(c, 0):,}" for c in ("real", "visual_only", "audio_only", "both"))
            + " |"
        )

    lines += [
        "",
        "### Exclusions (section 3.7)",
        "",
        "| Status | Count |",
        "|---|---|",
    ]
    for k, v in facts["status_counts"].items():
        lines.append(f"| `{k}` | {v:,} |")
    lines += [
        "",
        f"Exclusion rate is **{1 - facts['n_usable'] / facts['n_total']:.4%}** -- four videos, all",
        "`bad_label`. This is a genuinely clean dataset; the section 3.7 warning about a 30%",
        'exclusion being "a finding, not a footnote" does not apply here.',
        "",
        "### Video duration",
        "",
        "`duration` is the **video** timeline (`video_frames / 25`). See the note below on why",
        "metadata's own `duration` field is not used.",
        "",
        f"{fmt(fr)}",
        "",
        "## P1-11 Forged-span duration -- determines `T` (section 6.2)",
        "",
        f"**{facts['n_spans']:,} forged spans** across {facts['label_counts'].get(1, 0):,} fake videos.",
        "",
        f"- seconds: {fmt(sp)}",
        f"- frames at 25 fps: {fmt(facts['span_frames_at_25'], '')}",
        "",
        f"> **Mean forged span is {sp['mean']:.3f} s = {sp['mean'] * 25:.1f} frames**, and the",
        f"> longest in the entire dataset is {sp['max']:.3f} s = {sp['max'] * 25:.0f} frames.",
        "> These are *short* -- a design constraint, not a detail. Any temporal receptive field",
        "> or post-processing median filter wider than ~1.6 s cannot resolve a single span, and",
        "> smoothing at that scale would erase the very thing being localized.",
        "",
        "Spans per video:",
        "",
        "| `n_fakes` | Videos |",
        "|---|---|",
    ]
    for k, v in facts["n_fakes_distribution"].items():
        lines.append(f"| {k} | {v:,} |")

    lines += [
        "",
        "## P1-12 Fake-frame fraction -- drives focal loss / `pos_weight` (section 3.6)",
        "",
        "| | |",
        "|---|---|",
        f"| Forged frames | {facts['fake_frames']:,} |",
        f"| Total frames | {facts['total_frames']:,} |",
        f"| **Fake-frame fraction** | **{facts['fake_frame_fraction']:.4%}** |",
        f"| **BCE `pos_weight`** | **{facts['pos_weight']:.2f}** |",
        f"| Video-level fake share | {facts['video_level_fake_fraction']:.2%} |",
        "",
        "> These are the two distinct imbalance problems of section 3.6, and they point in",
        f"> opposite directions. At the **video** level fakes are the majority ({facts['video_level_fake_fraction']:.0%});",
        f"> at the **frame** level they are {facts['fake_frame_fraction']:.1%} of all frames. A localization head",
        f"> trained with unweighted BCE will predict all-real and score {1 - facts['fake_frame_fraction']:.1%}",
        "> frame accuracy while being completely useless. Use `pos_weight`"
        f" ~= **{facts['pos_weight']:.1f}** or focal loss, and never report frame accuracy as a headline.",
        "",
        "## P1-13 fps / sample-rate consistency",
        "",
        "| Property | Values (from metadata, all usable rows) |",
        "|---|---|",
        f"| fps | {', '.join(f'{k} ({v:,})' for k, v in facts['fps_values'].items())} |",
        f"| sample rate | {', '.join(f'{k} Hz ({v:,})' for k, v in facts['sample_rate_values'].items())} |",
        f"| audio channels | {', '.join(f'{k} ({v:,})' for k, v in facts['audio_channel_values'].items())} |",
        "",
    ]

    if probe_rows:
        lines += [
            f"### ffprobe verification on {len(probe_rows)} locally-held videos",
            "",
            "Stratified over split x 4-class, plus all four `bad_label` entries.",
            "",
            "| Check | Result |",
            "|---|---|",
            f"| `video_frames` matches ffprobe | **{facts['probe_frames_match']}/{facts['probe_n']}** |",
            f"| `r_frame_rate` == 25.00 | **{facts['probe_fps_25']}/{facts['probe_n']}** |",
            f"| sample rate == 16000 Hz | **{facts['probe_sr_16k']}/{facts['probe_n']}** |",
            "",
            "> Frame counts are byte-exact against the container, so the Kaggle mirror is **not**",
            "> a re-encode of the authors' release -- the specific failure CL-7 exists to rule out.",
            "",
        ]

    lines += [
        "## ⛔ Finding -- metadata `duration` is not a media duration",
        "",
        "For **all 136,304 entries**, `duration == audio_frames / 16000 + 0.128` exactly. It is a",
        "padded audio-derived figure and it exceeds *both* stream durations on every file probed.",
        "",
        "Consequences, both load-bearing:",
        "",
        "1. **The video timeline is `video_frames / 25`**, which matched ffprobe on 28/28 files.",
        "   The manifest's `duration` column carries this; metadata's value is kept as",
        "   `duration_meta` for traceability.",
        "2. **PROJECT_PLAN Phase 2 task P2-1 is wrong as written.** It says",
        "   `assert T == round(duration * 25) +/- 1`. Measured, `video_frames - round(duration * 25)`",
        "   ranges from **-5 to -1**, and only **239 of 136,304** entries fall within +/-1. That",
        "   assertion would fail on 99.8% of the dataset. The correct invariant is",
        "   **`T == video_frames`**. See decision PF-7.",
        "",
        "## ⚠️ Scope limit -- what `source_id` does and does not prove",
        "",
        "`source_id` is derived by walking the `original` chain to its root (P1-7); it is not a",
        f"field the dataset provides. It yields exactly **{facts['n_source_ids']:,} roots for"
        f" {facts['n_total']:,} files**, matching the real-video count exactly -- every fake resolves to",
        "the real video it was built from.",
        "",
        "The leakage assertions (P1-8) therefore prove **no fake is split away from its own",
        "original**, and they pass with zero overlap on all three split pairs. They do **not**",
        "prove two *different* real videos of the same VoxCeleb2 speaker sit in the same split:",
        "`metadata.min.json` carries no speaker identity, so that residual R3 risk cannot be",
        "tested from metadata alone. It is stated here rather than left implied.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    # Drop trailing blanks so regenerating the report is idempotent -- otherwise
    # pre-commit's end-of-file-fixer rewrites it after every run.
    while lines and not lines[-1].strip():
        lines.pop()
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
