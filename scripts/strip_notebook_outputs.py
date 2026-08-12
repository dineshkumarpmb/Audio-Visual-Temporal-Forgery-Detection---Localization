"""Strip outputs and execution counts from notebooks before they are committed.

Notebook outputs bloat diffs, can embed absolute paths from this machine, and
occasionally carry data samples we are not licensed to redistribute (LAV-DF is
CC BY-NC 4.0). Committed notebooks should carry code and prose only.

Used as a pre-commit hook; also runnable directly:

    python scripts/strip_notebook_outputs.py notebooks/01_dataset_exploration.ipynb
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def strip(path: Path) -> bool:
    """Return True if the file was modified."""
    try:
        nb = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        print(f"{path}: could not parse ({e}) — skipped", file=sys.stderr)
        return False

    changed = False
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        if cell.get("outputs"):
            cell["outputs"] = []
            changed = True
        if cell.get("execution_count") is not None:
            cell["execution_count"] = None
            changed = True

    if changed:
        path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return changed


def main(argv: list[str]) -> int:
    modified = [p for p in (Path(a) for a in argv) if p.exists() and strip(p)]
    for p in modified:
        print(f"stripped outputs: {p}")
    # Non-zero tells pre-commit the files were rewritten, so the commit is
    # retried against the cleaned versions.
    return 1 if modified else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
