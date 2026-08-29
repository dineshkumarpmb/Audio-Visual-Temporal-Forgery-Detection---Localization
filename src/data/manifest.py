"""P1-6: build `manifest_v1.parquet` -- the contract every downstream stage reads.

Schema is PROJECT_PLAN section 2A verbatim:

    video_id str | path str | split str | label int | modify_video bool |
    modify_audio bool | fake_periods list[[float,float]] | duration float |
    fps float | n_frames int | sample_rate int | source_id str | status str

Two deviations from the plan's field *semantics*, both measured in P1-5/P1-13 and
recorded in reports/dataset_statistics.md:

- `duration` is the **video** timeline (`video_frames / 25`), not metadata's
  `duration` field. The latter is `audio_frames/16000 + 0.128` -- audio-derived and
  padded -- and exceeds both stream durations on every file probed. Downstream code
  rasterises localization targets against this column, so it must be the timeline the
  frames actually live on. Metadata's value is preserved as `duration_meta` rather
  than discarded.
- `fps` and `sample_rate` are constants of the dataset (25.0 / 16000), verified by
  ffprobe rather than assumed. With `--probe` they carry the measured per-file value.

The manifest is versioned and never mutated in place (section 3.8 rule 3).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import LAVDF_AUDIO_CHANNELS, LAVDF_FPS, LAVDF_SAMPLE_RATE
from src.data.leakage import check_no_leakage, source_ids_by_split
from src.data.metadata import MetadataEntry, derive_source_id, load_metadata
from src.data.validate import classify, ffprobe, summarise

MANIFEST_COLUMNS = [
    "video_id",
    "path",
    "split",
    "label",
    "class_name",
    "modify_video",
    "modify_audio",
    "fake_periods",
    "duration",
    "duration_meta",
    "fps",
    "n_frames",
    "sample_rate",
    "audio_channels",
    "n_fakes",
    "original",
    "source_id",
    "status",
]


def build_manifest(
    entries: list[MetadataEntry],
    data_root: str | Path,
    *,
    probe: bool = False,
    min_duration_s: float = 1.0,
    max_duration_s: float = 60.0,
    progress_every: int = 5000,
) -> pd.DataFrame:
    """Join metadata with derived identity and (optionally) ffprobe facts.

    `probe=False` is the default because decision PF-6 keeps the 25 GB of video on
    Kaggle: on this machine there is nothing to probe. Set it True where the data is
    mounted -- the Kaggle notebook (CL-3) does exactly that.
    """
    root = Path(data_root)
    source_ids = derive_source_id(entries)

    rows: list[dict] = []
    for i, m in enumerate(entries):
        path = root / m.file
        result = ffprobe(path) if probe else None

        if result is not None and result.readable:
            fps = result.fps if result.fps else LAVDF_FPS
            n_frames = result.n_frames if result.n_frames else m.video_frames
            sample_rate = result.sample_rate or LAVDF_SAMPLE_RATE
            channels = result.channels or LAVDF_AUDIO_CHANNELS
        else:
            fps, n_frames = LAVDF_FPS, m.video_frames
            sample_rate, channels = LAVDF_SAMPLE_RATE, m.audio_channels

        rows.append(
            {
                "video_id": m.video_id,
                "path": str(path),
                "split": m.split,
                "label": m.label,
                "class_name": m.class_name,
                "modify_video": m.modify_video,
                "modify_audio": m.modify_audio,
                "fake_periods": m.fake_periods,
                "duration": m.video_duration_s,
                "duration_meta": m.duration,
                "fps": fps,
                "n_frames": n_frames,
                "sample_rate": sample_rate,
                "audio_channels": channels,
                "n_fakes": m.n_fakes,
                "original": m.original,
                "source_id": source_ids[m.file],
                "status": classify(
                    m, result, min_duration_s=min_duration_s, max_duration_s=max_duration_s
                ),
            }
        )
        if probe and progress_every and (i + 1) % progress_every == 0:
            print(f"  probed {i + 1:,}/{len(entries):,}", flush=True)

    return pd.DataFrame(rows, columns=MANIFEST_COLUMNS)


def validate_manifest(df: pd.DataFrame) -> dict:
    """Post-build invariants. Raises on leakage; returns a summary for the report."""
    missing = [c for c in MANIFEST_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"manifest is missing columns: {missing}")

    if df["video_id"].duplicated().any():
        dupes = df.loc[df["video_id"].duplicated(), "video_id"].head(5).tolist()
        raise ValueError(f"duplicate video_id in manifest, e.g. {dupes}")

    # P1-8 / R3 -- raises LeakageError on any overlap.
    by_split = source_ids_by_split(zip(df["split"], df["source_id"], strict=True))
    leakage = check_no_leakage(by_split)

    ok = df[df["status"] == "ok"]
    return {
        "n_total": int(len(df)),
        "n_ok": int(len(ok)),
        "n_excluded": int(len(df) - len(ok)),
        "exclusion_rate": float((len(df) - len(ok)) / len(df)) if len(df) else 0.0,
        "status_counts": summarise(df["status"].tolist()),
        "split_counts": df["split"].value_counts().to_dict(),
        "class_counts": df["class_name"].value_counts().to_dict(),
        "leakage": leakage,
    }


def write_manifest(df: pd.DataFrame, path: str | Path) -> Path:
    """Write parquet. Refuses to overwrite -- manifests are versioned, never mutated."""
    out = Path(path)
    if out.exists():
        raise FileExistsError(
            f"{out} exists. Manifests are versioned, never mutated in place "
            "(section 3.8 rule 3) -- bump the version instead."
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    return out


def read_manifest(path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def build_from_metadata_file(
    metadata_path: str | Path, data_root: str | Path, **kwargs
) -> tuple[pd.DataFrame, dict]:
    """Convenience: parse metadata, build the manifest, validate it."""
    entries = load_metadata(metadata_path)
    df = build_manifest(entries, data_root, **kwargs)
    return df, validate_manifest(df)
