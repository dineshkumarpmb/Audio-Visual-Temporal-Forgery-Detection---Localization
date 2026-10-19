"""⛔ P16-11 (the checkable half): every number in the docs traces to a logged run.

    python scripts/46_check_docs.py

The documents quote measured numbers. Each one carries an invisible source tag right after it:

    test ROC-AUC **0.984**<!--n:reports/test_once.json#summary/auc/mean|3-->

`file#path|fmt` names the JSON an experiment script wrote, the path inside it (`/`-separated;
list indices are integers), and how the number is printed:

    3       round to 3 decimals            0.984
    pct1    ×100, 1 decimal (no % sign)    95.2
    int     nearest integer                97
    sign3   3 decimals with an explicit +  +0.087
    mega2   ÷1e6, 2 decimals               5.85  (as in "5.85 M parameters")

The checker recomputes every tagged number from its source and fails on any mismatch, any
missing file or path, and any document below its minimum tag count -- so a number cannot be
edited by hand without the check noticing. It also checks that every `make <target>` a
reader is told to run exists in the Makefile and every relative link resolves to a file git
will actually ship (an ignored file works locally and is dead on GitHub), because "a
stranger can reproduce it from the README" fails first on a dead link or a renamed target.

What it cannot check is prose: an untagged number, or a wrong sentence around a right one.
Untagged decimals in each document are listed so a reviewer can see what was not traced.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from functools import cache
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.utils.console import init_console  # noqa: E402

GREEN, RED, DIM, RESET = "\033[32m", "\033[31m", "\033[2m", "\033[0m"

DOCS = {  # document -> minimum number of traced numbers it must carry
    "README.md": 12,
    "reports/results.md": 30,
    "reports/final_report.md": 25,
    "reports/portfolio.md": 10,
    "reports/interview_prep.md": 10,
}

NUM = r"[+\-−]?\d[\d,]*(?:\.\d+)?"
TAG = re.compile(
    rf"(?P<num>{NUM})(?P<unit>\s?(?:%|ms|s|MB|GB|×|x|pts|points|M))?(?:\*\*|`)?"
    r"<!--n:(?P<src>[^#>]+)#(?P<path>[^|>]+)\|(?P<fmt>[a-z0-9]+)-->"
)
UNTAGGED = re.compile(r"(?<![\w.#/@-])(\d+\.\d{2,})(?![\w.]*-->)(?!\s*<!--n:)")
MAKE = re.compile(r"`make ([a-z0-9][a-z0-9-]*)")
LINK = re.compile(r"\]\((?!https?://|#|mailto:)([^)#\s]+)(?:#[^)]*)?\)")


@cache
def load(src: str):
    return json.loads((REPO_ROOT / src).read_text(encoding="utf-8"))


def resolve(src: str, path: str):
    node = load(src)
    for part in path.split("/"):
        node = node[int(part)] if isinstance(node, list) else node[part]
    return node


def render(value: float, fmt: str) -> str:
    if fmt == "int":
        return str(int(round(value)))
    if fmt.startswith("pct"):
        return f"{value * 100:.{int(fmt[3:])}f}"
    if fmt.startswith("mega"):
        return f"{value / 1e6:.{int(fmt[4:])}f}"
    if fmt.startswith("sign"):
        return f"{value:+.{int(fmt[4:])}f}"
    return f"{value:.{int(fmt)}f}"


def normalise(text: str, fmt: str) -> str:
    text = text.replace("−", "-").replace(",", "")
    if not fmt.startswith("sign"):
        text = text.lstrip("+")
    return text


def git_ignored(path: Path) -> bool:
    """True if git would not commit `path` -- a link to it is dead for everyone else."""
    try:
        rel = path.resolve().relative_to(REPO_ROOT)
    except ValueError:
        return False
    r = subprocess.run(
        ["git", "check-ignore", "-q", rel.as_posix()], cwd=REPO_ROOT, capture_output=True
    )
    return r.returncode == 0


def make_targets() -> set[str]:
    makefile = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")
    return set(re.findall(r"^([a-z0-9][a-z0-9-]*):", makefile, flags=re.M))


def check(doc: str, minimum: int) -> dict:
    text = (REPO_ROOT / doc).read_text(encoding="utf-8")
    bad, ok = [], 0
    for m in TAG.finditer(text):
        src, path, fmt = m["src"].strip(), m["path"].strip(), m["fmt"]
        try:
            want = render(float(resolve(src, path)), fmt)
        except FileNotFoundError:
            bad.append(f"{m['num']}: source file missing: {src}")
            continue
        except (KeyError, IndexError, ValueError, TypeError) as exc:
            bad.append(f"{m['num']}: path {src}#{path} unresolved ({type(exc).__name__}: {exc})")
            continue
        got = normalise(m["num"], fmt)
        if got != want:
            bad.append(f"{m['num']} != {want}  ({src}#{path}|{fmt})")
        else:
            ok += 1
    stripped = TAG.sub("", text)
    stripped = re.sub(r"<!--.*?-->", "", stripped, flags=re.S)
    untagged = sorted(set(UNTAGGED.findall(stripped)))
    targets = make_targets()
    missing_targets = sorted({t for t in MAKE.findall(text) if t not in targets})
    dead_links = []
    for link in sorted(set(LINK.findall(text))):
        target = REPO_ROOT / Path(doc).parent / link
        if not target.exists():
            target = REPO_ROOT / link
        if not target.exists():
            dead_links.append(link)
        elif git_ignored(target):
            # Exists here, but git will never ship it: dead on GitHub and in every clone.
            dead_links.append(f"{link} (git-ignored)")
    return {
        "doc": doc,
        "traced": ok,
        "minimum": minimum,
        "mismatches": bad,
        "untagged_decimals": untagged,
        "missing_make_targets": missing_targets,
        "dead_links": dead_links,
        "passed": not bad and ok >= minimum and not missing_targets and not dead_links,
    }


def main() -> int:
    init_console()
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="reports/docs_check.json")
    args = ap.parse_args()

    print(
        f"{DIM}{'=' * 72}{RESET}\n  ⛔ P16-11  every number traces to a logged run\n{DIM}{'=' * 72}{RESET}"
    )
    results = []
    for doc, minimum in DOCS.items():
        if not (REPO_ROOT / doc).is_file():
            results.append({"doc": doc, "passed": False, "error": "missing"})
            print(f"  [{RED}NO{RESET}] {doc}: missing")
            continue
        r = check(doc, minimum)
        results.append(r)
        mark = GREEN + "ok" if r["passed"] else RED + "NO"
        print(f"  [{mark}{RESET}] {doc}: {r['traced']} traced (min {minimum})")
        for line in r["mismatches"]:
            print(f"        {RED}mismatch{RESET} {line}")
        for t in r["missing_make_targets"]:
            print(f"        {RED}no such make target{RESET} {t}")
        for link in r["dead_links"]:
            print(f"        {RED}dead link{RESET} {link}")
        if r["untagged_decimals"]:
            print(
                f"        {DIM}untagged decimals (review): {', '.join(r['untagged_decimals'][:12])}{RESET}"
            )
    passed = all(r["passed"] for r in results)
    Path(args.out).write_text(
        json.dumps({"passed": passed, "docs": results}, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"\n{GREEN if passed else RED}{'DOCS CHECK PASSED' if passed else 'DOCS CHECK FAILED'}{RESET}"
    )
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
