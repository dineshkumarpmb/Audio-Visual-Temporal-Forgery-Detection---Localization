"""⛔ P16-11 in CI: every traced number in the docs still equals its source.

`scripts/46_check_docs.py` recomputes each tagged number from the JSON an experiment wrote. Running
it here means a re-generated report that moves a number, or a hand edit to a document, fails CI
instead of leaving the README quietly wrong. Also pins the checker's own formatting rules and the
demo launcher's preflight, the two Phase 16 scripts a stranger relies on.
"""

from __future__ import annotations

import importlib.util
import socket
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def load(name: str):
    spec = importlib.util.spec_from_file_location(
        name.replace("-", "_"), ROOT / "scripts" / f"{name}.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def docs():
    return load("46_check_docs")


@pytest.mark.parametrize("doc", list(load("46_check_docs").DOCS))
def test_every_traced_number_matches_its_source(docs, doc):
    r = docs.check(doc, docs.DOCS[doc])
    assert not r["mismatches"], r["mismatches"]
    assert r["traced"] >= r["minimum"], f"{doc}: only {r['traced']} traced numbers"
    assert not r["dead_links"], r["dead_links"]
    assert not r["missing_make_targets"], r["missing_make_targets"]


@pytest.mark.parametrize(
    ("value", "fmt", "want"),
    [
        (0.98420, "3", "0.984"),
        (0.95238, "pct1", "95.2"),
        (96.69, "int", "97"),
        (0.08703, "sign3", "+0.087"),
        (-0.08001, "sign3", "-0.080"),
        (5845581, "mega2", "5.85"),
    ],
)
def test_render_formats(docs, value, fmt, want):
    assert docs.render(value, fmt) == want


def test_a_wrong_number_is_caught(docs, tmp_path, monkeypatch):
    bad = tmp_path / "bad.md"
    bad.write_text(
        "AUC 0.999<!--n:reports/test_once.json#summary/auc/mean|3-->\n", encoding="utf-8"
    )
    monkeypatch.setattr(docs, "REPO_ROOT", ROOT)
    rel = Path("tests") / "unit" / "_tmp_bad_doc.md"
    (ROOT / rel).write_text(bad.read_text(encoding="utf-8"), encoding="utf-8")
    try:
        r = docs.check(rel.as_posix(), 1)
    finally:
        (ROOT / rel).unlink()
    assert r["mismatches"] and not r["passed"]


def test_demo_preflight_names_each_problem(monkeypatch, tmp_path):
    demo = load("47_demo")
    monkeypatch.setattr(demo, "MODEL", tmp_path / "missing.pt")
    monkeypatch.setattr(demo, "VITE", tmp_path / "missing.js")
    with socket.socket() as busy:
        busy.bind(("127.0.0.1", 0))
        busy.listen(1)
        port = busy.getsockname()[1]
        problems = demo.preflight(port, port)
    assert any("model file missing" in p for p in problems)
    assert any("make frontend-install" in p for p in problems)
    assert any(f"port {port}" in p for p in problems)
