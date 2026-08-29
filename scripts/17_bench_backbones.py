"""P4-3: isolated backbone throughput and VRAM, for Decision D-1.

    python scripts/17_bench_backbones.py

**Why this is separate from `13_extract_visual.py`.** That script's wall-clock is dominated
by video decode and MediaPipe alignment, not by the backbone -- measured end to end,
MobileNetV2 came out *slower* than ResNet-18 (0.80x), which says nothing about the
backbones and everything about the shared decode path in front of them. D-1's tie-break
turns on "extraction is >=2x faster", and answering it needs the backbone measured on its
own, on identical input, with the data-loading removed.

Reports both the pure-backbone figure and, for honesty about what it buys in practice, the
share of end-to-end extraction the backbone actually accounts for.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.models.backbones.visual import BACKBONES, build_backbone  # noqa: E402
from src.seed import seed_everything  # noqa: E402
from src.utils.console import init_console  # noqa: E402

GREEN, DIM, RESET = "\033[32m", "\033[2m", "\033[0m"


@torch.no_grad()
def bench(name: str, device: str, batch: int, size: int, iters: int, warmup: int) -> dict:
    model = build_backbone(name).to(device)
    info = model.info()
    x = torch.randint(0, 255, (batch, 3, size, size), dtype=torch.uint8, device=device)

    for _ in range(warmup):
        model(x)
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    for _ in range(iters):
        model(x)
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    peak = torch.cuda.max_memory_allocated() / 1024**2 if device == "cuda" else 0.0
    del model, x
    if device == "cuda":
        torch.cuda.empty_cache()

    return {
        "backbone": name,
        "feature_dim": info.feature_dim,
        "n_params": info.n_params,
        "n_trainable": info.n_trainable,
        "images_per_second": round(batch * iters / elapsed, 1),
        "ms_per_batch": round(1000 * elapsed / iters, 3),
        "peak_vram_mb": round(peak, 1),
    }


def main() -> int:
    init_console()
    ap = argparse.ArgumentParser(description="Isolated backbone benchmark for D-1")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--size", type=int, default=112)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--out", default="reports/backbone_bench.json")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    seed_everything(1337)
    print(f"{DIM}{'=' * 72}{RESET}\n  P4-3  Isolated backbone benchmark\n{DIM}{'=' * 72}{RESET}")
    print(
        f"  device {args.device}"
        + (f"  {DIM}{torch.cuda.get_device_name(0)}{RESET}" if args.device == "cuda" else "")
    )
    print(
        f"  batch {args.batch} x {args.size}x{args.size}, {args.iters} iters "
        f"after {args.warmup} warmup\n"
    )
    print(f"  {'backbone':<16}{'dim':>6}{'params':>12}{'img/s':>10}{'ms/batch':>10}{'VRAM MB':>10}")

    results = {}
    for name in BACKBONES:
        r = bench(name, args.device, args.batch, args.size, args.iters, args.warmup)
        results[name] = r
        print(
            f"  {name:<16}{r['feature_dim']:>6}{r['n_params']:>12,}"
            f"{r['images_per_second']:>10.1f}{r['ms_per_batch']:>10.2f}{r['peak_vram_mb']:>10.1f}"
        )

    a, b = results["resnet18"], results["mobilenet_v2"]
    speedup = b["images_per_second"] / a["images_per_second"]
    print(f"\n  MobileNetV2 vs ResNet-18: {GREEN}{speedup:.2f}x{RESET} on the backbone alone")

    # How much of end-to-end extraction is actually the backbone? If it is a small share,
    # a backbone speedup cannot move the wall clock much -- which is the honest framing
    # for D-1's "faster extraction buys more experiments" argument.
    share = None
    e2e = Path("reports/visual_features_resnet18.json")
    if e2e.exists():
        facts = json.loads(e2e.read_text(encoding="utf-8"))
        fps_e2e = facts.get("frames_per_second")
        if fps_e2e:
            share = fps_e2e / a["images_per_second"]
            print(
                f"  {DIM}End-to-end extraction runs at {fps_e2e:.1f} frames/s vs the backbone's "
                f"{a['images_per_second']:.0f} img/s{RESET}"
            )
            print(
                f"  {DIM}-> the backbone is ~{share:.1%} of the pipeline; decode + alignment "
                f"dominate{RESET}"
            )

    payload = {
        "device": args.device,
        "gpu": torch.cuda.get_device_name(0) if args.device == "cuda" else None,
        "batch": args.batch,
        "input_size": args.size,
        "iters": args.iters,
        "backbones": results,
        "mobilenet_speedup_vs_resnet": round(speedup, 4),
        "backbone_share_of_pipeline": round(share, 4) if share else None,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\n  wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
