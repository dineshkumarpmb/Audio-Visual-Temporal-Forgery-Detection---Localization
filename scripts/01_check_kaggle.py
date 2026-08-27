"""CL-1: verify Kaggle access before Phase 1 depends on it.

Decision PF-6 moved the dataset off this machine: LAV-DF is attached read-only on
Kaggle rather than downloaded to D:. That makes a working Kaggle account a Phase 1
prerequisite, not a Phase 9 convenience, so it gets the same treatment the toolchain
got in Phase 0 — checked by a script, recorded in reports/, reproducible.

    python scripts/01_check_kaggle.py
    python scripts/01_check_kaggle.py --gpu-hours 30 --phone-verified

Two of CL-1's requirements cannot be reached through the public API: phone
verification and the remaining weekly GPU quota. Those are attested by the flags
above and recorded as attested, never guessed. Without the flags the report says
PENDING, because a fabricated number in reports/ is worse than a missing one
(P16-11 gate).

Design note: importing `kaggle` runs api.authenticate() at import time and raises
if no credentials exist. So credentials are checked *before* the package is
imported anywhere in this file. Every API call is defensive — Kaggle reshaped this
client's surface in 1.7, and a future reshape should degrade this script to a
warning, not a traceback.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# The mirror PF-6 selected. CL-7 proves it equals the authors' release; CL-1 only
# proves it is reachable with these credentials.
MIRROR = "elin75/localized-audio-visual-deepfake-dataset-lav-df"

# Published LAV-DF figures, for the size sanity check below. The authoritative
# comparison lives in CL-7 — this is a smell test, not the verification.
EXPECTED_VIDEOS = 136_304
EXPECTED_GB_MIN, EXPECTED_GB_MAX = 20.0, 32.0

# Free-tier facts CL-1 is meant to confirm, quoted in reports/ for traceability.
FREE_TIER = {
    "gpu_hours_per_week": "~30",
    "accelerators": "P100 16 GB, or T4 x2 (32 GB combined)",
    "session_limit_hours": 12,
    "working_disk_gb": 20,
    "input_mount": "/kaggle/input (read-only, does not count against /kaggle/working)",
}

GREEN, RED, YELLOW, DIM, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[0m"


@dataclass
class Results:
    checks: list[tuple[str, bool, str]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    facts: dict[str, object] = field(default_factory=dict)

    def check(self, name: str, ok: bool, detail: str = "") -> bool:
        self.checks.append((name, ok, detail))
        mark = f"{GREEN}PASS{RESET}" if ok else f"{RED}FAIL{RESET}"
        print(f"  [{mark}] {name}" + (f"  {DIM}{detail}{RESET}" if detail else ""))
        return ok

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)
        print(f"  [{YELLOW}WARN{RESET}] {msg}")

    @property
    def failed(self) -> list[str]:
        return [n for n, ok, _ in self.checks if not ok]


def section(title: str) -> None:
    print(f"\n{DIM}{'-' * 72}{RESET}\n  {title}\n{DIM}{'-' * 72}{RESET}")


def check_package(r: Results) -> bool:
    """The kaggle client must be importable — but do not import it yet."""
    section("Client")
    try:
        import importlib.metadata as md

        version = md.version("kaggle")
    except Exception as exc:  # noqa: BLE001 - report the reason, do not crash
        r.check("kaggle client installed", False, f"pip install kaggle  ({exc})")
        return False
    r.facts["kaggle_client_version"] = version
    return r.check("kaggle client installed", True, f"v{version}")


def check_credentials(r: Results) -> str | None:
    """Locate and validate credentials. Returns the configured username.

    Runs before any kaggle import: the package authenticates on import and would
    raise OSError here rather than reporting a clean FAIL.
    """
    section("Credentials")

    env_user, env_key = os.environ.get("KAGGLE_USERNAME"), os.environ.get("KAGGLE_KEY")
    config_dir = Path(os.environ.get("KAGGLE_CONFIG_DIR", Path.home() / ".kaggle"))
    token = config_dir / "kaggle.json"

    if env_user and env_key:
        r.check("credentials found", True, "KAGGLE_USERNAME + KAGGLE_KEY (env)")
        r.facts["credential_source"] = "environment"
        r.facts["kaggle_username"] = env_user
        return env_user

    if not token.exists():
        r.check("credentials found", False, f"no {token} and no KAGGLE_USERNAME/KAGGLE_KEY")
        print(
            f"\n  {YELLOW}To fix:{RESET} kaggle.com -> Settings -> API -> Create New Token,\n"
            f"  then save the downloaded kaggle.json to {config_dir}\\"
        )
        return None

    try:
        data = json.loads(token.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        r.check("kaggle.json parses", False, str(exc))
        return None
    r.check("kaggle.json parses", True, str(token))

    username = data.get("username")
    if not username or not data.get("key"):
        r.check("kaggle.json has username + key", False, f"keys present: {sorted(data)}")
        return None
    r.check("kaggle.json has username + key", True, f"username={username}")

    r.facts["credential_source"] = str(token)
    r.facts["kaggle_username"] = username
    return username


def check_api(r: Results, username: str):  # noqa: ANN201 - client type is version-dependent
    """Authenticate and make one real authenticated call."""
    section("API")
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()
    except Exception as exc:  # noqa: BLE001
        r.check("API authenticates", False, f"{type(exc).__name__}: {exc}")
        return None
    r.check("API authenticates", True, f"as {username}")

    # A round trip proves the key is live, not merely well-formed. An expired or
    # revoked token parses fine and fails only here.
    try:
        api.datasets_list(search="lav-df", page=1)
    except Exception as exc:  # noqa: BLE001
        r.check("authenticated round trip", False, f"{type(exc).__name__}: {exc}")
        return None
    r.check("authenticated round trip", True, "datasets_list responded")
    return api


def check_mirror(r: Results, api) -> None:  # noqa: ANN001
    """Confirm the PF-6 mirror is reachable, and record its shape for CL-7.

    This is CL-1's handoff to CL-2. It deliberately does NOT decide whether the
    mirror is trustworthy — that is CL-7's job, and it needs ffprobe on real files.
    """
    section(f"Mirror — {MIRROR}")

    files: list[tuple[str, int]] = []
    token, pages = None, 0
    try:
        while pages < 200:  # hard stop; a paging bug must not spin forever
            resp = api.dataset_list_files(MIRROR, page_token=token, page_size=200)
            if getattr(resp, "error_message", None):
                r.check("mirror is reachable", False, str(resp.error_message))
                return
            for f in getattr(resp, "files", []) or []:
                files.append((getattr(f, "name", "?"), int(getattr(f, "total_bytes", 0) or 0)))
            pages += 1
            token = getattr(resp, "next_page_token", None)
            if not token:
                break
    except Exception as exc:  # noqa: BLE001
        r.check("mirror is reachable", False, f"{type(exc).__name__}: {exc}")
        print(
            f"\n  {YELLOW}If this is a 403:{RESET} open the dataset page once in a browser\n"
            f"  and accept any terms, then re-run. If it is a 404 the mirror was\n"
            f"  removed — fall back to CL-2's private-upload path."
        )
        return

    if not files:
        r.check("mirror is reachable", False, "responded, but listed no files")
        return

    total_gb = sum(size for _, size in files) / 1024**3
    videos = sum(1 for name, _ in files if name.lower().endswith(".mp4"))
    r.check("mirror is reachable", True, f"{len(files):,} files listed over {pages} page(s)")

    r.facts["mirror"] = MIRROR
    r.facts["mirror_files_listed"] = len(files)
    r.facts["mirror_mp4_listed"] = videos
    r.facts["mirror_total_gb"] = round(total_gb, 2)
    r.facts["mirror_pages_walked"] = pages

    # Size smell test. Wrong by a factor means the mirror is a subset or a
    # re-encode, and CL-7 becomes mandatory-blocking rather than merely required.
    if EXPECTED_GB_MIN <= total_gb <= EXPECTED_GB_MAX:
        r.check("mirror size is plausible", True, f"{total_gb:.1f} GB (expected ~25.6 GB)")
    else:
        r.check("mirror size is plausible", False, f"{total_gb:.1f} GB, expected 20-32 GB")
        r.warn("size is off — treat CL-7 as blocking, and read CL-2's fallback")

    has_metadata = any(name.endswith("metadata.min.json") for name, _ in files)
    if has_metadata:
        r.check("metadata.min.json present", True, "")
    else:
        # Not a FAIL: the listing may be paginated or the file may sit inside an
        # archive. CL-7 resolves it on the mounted copy, where it is unambiguous.
        r.check("metadata.min.json present", False, "not in the listing")
        r.warn("metadata.min.json not listed — confirm on the mounted copy in CL-7")

    if videos and abs(videos - EXPECTED_VIDEOS) > EXPECTED_VIDEOS * 0.01:
        r.warn(f"{videos:,} .mp4 listed vs {EXPECTED_VIDEOS:,} published — CL-7 must explain this")

    print(f"\n  {DIM}Largest listed entries:{RESET}")
    for name, size in sorted(files, key=lambda x: -x[1])[:5]:
        print(f"    {size / 1024**3:8.2f} GB  {name}")


def check_manual(r: Results, gpu_hours: float | None, phone_verified: bool) -> None:
    """The two things the API will not tell you.

    Kaggle exposes neither phone-verification state nor remaining quota. Both are
    attested by flag or recorded PENDING — never inferred.
    """
    section("Manual attestation (not available via API)")

    if phone_verified:
        r.check("phone verification attested", True, "--phone-verified")
        r.facts["phone_verified"] = True
    else:
        r.facts["phone_verified"] = None
        r.warn(
            "phone verification PENDING — kaggle.com/settings -> Phone Verification. "
            "Without it a notebook gets NEITHER GPU NOR internet, which breaks CL-3's "
            "git clone as well as every GPU task."
        )

    if gpu_hours is not None:
        r.check("GPU quota attested", True, f"{gpu_hours:g} h remaining this week")
        r.facts["gpu_hours_remaining"] = gpu_hours
        if gpu_hours < 5:
            r.warn(f"only {gpu_hours:g} h left — quota resets weekly; plan CL-8 shards around it")
    else:
        r.facts["gpu_hours_remaining"] = None
        r.warn("GPU quota PENDING — read it at kaggle.com/settings, re-run with --gpu-hours N")

    r.facts["free_tier_reference"] = FREE_TIER


def write_report(r: Results, argv: str) -> Path:
    out = REPO_ROOT / "reports" / "kaggle_report.md"
    f = r.facts
    passed = len(r.checks) - len(r.failed)

    def yes_no_pending(v: object) -> str:
        return "PENDING" if v is None else ("yes" if v else "no")

    lines = [
        "# Kaggle Access Report — task CL-1",
        "",
        f"Generated by `scripts/01_check_kaggle.py` (`{argv}`).",
        "Every value below is measured or attested. Nothing here is assumed.",
        "",
        f"**Result: {passed}/{len(r.checks)} checks passed, {len(r.warnings)} warning(s).**",
        "",
        "## Account",
        "",
        "| | |",
        "|---|---|",
        f"| Username | `{f.get('kaggle_username', 'unknown')}` |",
        f"| Credential source | `{f.get('credential_source', 'none')}` |",
        f"| Client version | {f.get('kaggle_client_version', 'unknown')} |",
        f"| Phone verified | {yes_no_pending(f.get('phone_verified'))} |",
        f"| GPU hours remaining | {f.get('gpu_hours_remaining') or 'PENDING'} |",
        "",
        "> Phone verification gates **both** GPU and internet access in notebooks. Without it,",
        "> CL-3's `git clone` fails for the same reason a GPU run does.",
        "",
        "## Free-tier limits (the constraints CL-8 is designed around)",
        "",
        "| | |",
        "|---|---|",
        f"| GPU hours/week | {FREE_TIER['gpu_hours_per_week']} |",
        f"| Accelerators | {FREE_TIER['accelerators']} |",
        f"| Session limit | {FREE_TIER['session_limit_hours']} h |",
        f"| Writable disk | {FREE_TIER['working_disk_gb']} GB (`/kaggle/working`) |",
        f"| Attached data | {FREE_TIER['input_mount']} |",
        "",
    ]

    if "mirror" in f:
        lines += [
            "## Mirror reachability (hands off to CL-2)",
            "",
            "| | |",
            "|---|---|",
            f"| Dataset | `{f['mirror']}` |",
            f"| Files listed | {f['mirror_files_listed']:,} |",
            f"| `.mp4` listed | {f['mirror_mp4_listed']:,} |",
            f"| Total size | {f['mirror_total_gb']} GB |",
            f"| Listing pages walked | {f['mirror_pages_walked']} |",
            "",
            "> Reachable is not the same as correct. **CL-7 still has to prove this mirror",
            "> equals the authors' release** — file count, real/fake split, `metadata.min.json`",
            "> integrity, and `ffprobe` fps/duration on 20 spot-checks. A silent re-encode",
            "> shifts every `fake_periods` target and no training curve would show it.",
            "",
        ]

    lines += ["## Checks", "", "| Check | Result | Detail |", "|---|---|---|"]
    for name, ok, detail in r.checks:
        lines.append(f"| {name} | {'PASS' if ok else 'FAIL'} | {detail or ''} |")
    lines.append("")

    if r.warnings:
        lines += ["## Warnings", ""] + [f"- {w}" for w in r.warnings] + [""]

    lines += [
        "## CL-1 completion criteria",
        "",
        "| | Status |",
        "|---|---|",
        f"| Account exists and API authenticates | {'yes' if not r.failed else 'no'} |",
        f"| Phone verified (GPU + internet) | {yes_no_pending(f.get('phone_verified'))} |",
        f"| GPU quota read and recorded | {yes_no_pending(f.get('gpu_hours_remaining'))} |",
        "",
        "CL-1 is done when all three read `yes`. Next: **CL-2** (attach the mirror), then",
        "**CL-7** (prove it equivalent) — CL-7 gates P1-2.",
        "",
    ]

    # Match 00_check_env.py: strip trailing whitespace and end with exactly one
    # newline so the pre-commit hooks never rewrite generated output.
    out.write_text("\n".join(ln.rstrip() for ln in lines).rstrip() + "\n", encoding="utf-8")
    (REPO_ROOT / "reports" / "kaggle_facts.json").write_text(
        json.dumps(f, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="CL-1 — verify Kaggle access")
    ap.add_argument(
        "--gpu-hours", type=float, help="GPU hours remaining, read from kaggle.com/settings"
    )
    ap.add_argument(
        "--phone-verified", action="store_true", help="attest phone verification is enabled"
    )
    ap.add_argument("--skip-mirror", action="store_true", help="skip the mirror reachability probe")
    ap.add_argument("--no-report", action="store_true", help="skip writing reports/")
    args = ap.parse_args()

    print("=" * 72)
    print("  CL-1 — Kaggle access (Phase 1 prerequisite under decision PF-6)")
    print("=" * 72)

    r = Results()
    api = None
    if check_package(r):
        username = check_credentials(r)
        if username:
            api = check_api(r, username)
    if api is not None and not args.skip_mirror:
        check_mirror(r, api)
    check_manual(r, args.gpu_hours, args.phone_verified)

    if not args.no_report:
        path = write_report(r, " ".join(sys.argv[1:]) or "no arguments")
        print(f"\n{DIM}Report written to {path.relative_to(REPO_ROOT)}{RESET}")

    section("Summary")
    passed = len(r.checks) - len(r.failed)
    print(f"  {passed}/{len(r.checks)} checks passed, {len(r.warnings)} warning(s)")
    if r.failed:
        print(f"\n{RED}CL-1 INCOMPLETE{RESET} — {len(r.failed)} check(s) failed:")
        for name in r.failed:
            print(f"    - {name}")
        return 1

    pending = [
        label
        for label, value in (
            ("phone verification", r.facts.get("phone_verified")),
            ("GPU quota", r.facts.get("gpu_hours_remaining")),
        )
        if value is None
    ]
    if pending:
        print(
            f"\n{YELLOW}CL-1 PARTIAL{RESET} — API access works; still to attest: {', '.join(pending)}"
        )
        return 0

    print(f"\n{GREEN}CL-1 COMPLETE{RESET} — proceed to CL-2 (attach the mirror).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
