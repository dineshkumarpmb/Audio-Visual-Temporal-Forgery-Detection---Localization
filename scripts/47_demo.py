"""P16-10: `make demo` -- the API and the front end together, from one command.

    python scripts/47_demo.py                 # API on :8000, front end on :5173
    python scripts/47_demo.py --open          # ... and open the browser when ready

A launcher in Python rather than shell, because `make` on Windows runs recipes in cmd.exe from
PowerShell but in sh from Git Bash, and no one-liner backgrounds a process in both. It:

1. checks what the demo needs before starting anything -- the model file, the front end's
   `node_modules`, and that both ports are free (a server left running from an earlier
   session is the usual culprit, and it would serve old code);
2. starts the API and waits for `/health` (model load + warm-up, ~25 s);
3. starts Vite, pointed at that API, and prints the URL;
4. on Ctrl+C -- or if either process dies -- stops both.

Memory: the API holds the model, ~1-3 GB on this machine (PF-30). Close other heavy apps first.
"""

from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
import time
import urllib.request
import webbrowser
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.utils.console import init_console  # noqa: E402

GREEN, RED, DIM, RESET = "\033[32m", "\033[31m", "\033[2m", "\033[0m"
MODEL = REPO_ROOT / "models" / "avtfd_g_seed0.pt"
FRONTEND = REPO_ROOT / "frontend"
VITE = FRONTEND / "node_modules" / "vite" / "bin" / "vite.js"
STARTUP_TIMEOUT_S = 240


def port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) != 0


def preflight(api_port: int, web_port: int) -> list[str]:
    problems = []
    if not MODEL.is_file():
        shown = MODEL.relative_to(REPO_ROOT) if MODEL.is_relative_to(REPO_ROOT) else MODEL
        problems.append(f"model file missing: {shown} -- run `make export-model`")
    if not VITE.is_file():
        problems.append("front end not installed -- run `make frontend-install` once")
    for name, port in (("API", api_port), ("front end", web_port)):
        if not port_free(port):
            problems.append(
                f"port {port} ({name}) is already in use -- an earlier server is probably still "
                "running (and would serve old code). Stop it with Ctrl+C in its terminal, or pass "
                f"--{'api' if name == 'API' else 'web'}-port"
            )
    return problems


def wait_healthy(url: str, proc: subprocess.Popen, timeout: float) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if proc.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:  # noqa: S310 - localhost only
                if resp.status == 200:
                    return True
        except OSError:
            pass
        time.sleep(1.0)
    return False


def stop(*procs: subprocess.Popen | None) -> None:
    for p in procs:
        if p is not None and p.poll() is None:
            p.terminate()
    for p in procs:
        if p is not None:
            try:
                p.wait(timeout=15)
            except subprocess.TimeoutExpired:
                p.kill()


def main() -> int:
    init_console()
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--api-port", type=int, default=8000)
    ap.add_argument("--web-port", type=int, default=5173)
    ap.add_argument("--open", action="store_true", help="open the browser when ready")
    ap.add_argument("--check", action="store_true", help="run the preflight checks only")
    args = ap.parse_args()

    problems = preflight(args.api_port, args.web_port)
    if problems:
        print(f"{RED}Cannot start the demo:{RESET}")
        for p in problems:
            print(f"  - {p}")
        return 1
    if args.check:
        print(
            f"{GREEN}Ready: model present, front end installed, ports {args.api_port} and {args.web_port} free.{RESET}"
        )
        return 0

    api = web = None
    try:
        print(f"  starting the API on :{args.api_port} (loading the model, ~25 s) ...", flush=True)
        api = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "api.main:app", "--host", "127.0.0.1",
             "--port", str(args.api_port), "--workers", "1"],
            cwd=REPO_ROOT,
        )  # fmt: skip
        if not wait_healthy(f"http://127.0.0.1:{args.api_port}/health", api, STARTUP_TIMEOUT_S):
            print(f"{RED}The API did not become healthy; see its output above.{RESET}")
            return 1
        print(
            f"  {GREEN}API ready{RESET} -- docs at http://127.0.0.1:{args.api_port}/docs",
            flush=True,
        )

        env = {**os.environ, "AVTFD_API_URL": f"http://127.0.0.1:{args.api_port}"}
        # Through node, not npm's .bin shim: the shim breaks on the & in this repo path (PF-31).
        web = subprocess.Popen(
            ["node", str(VITE), "--port", str(args.web_port), "--strictPort"], cwd=FRONTEND, env=env
        )
        url = f"http://127.0.0.1:{args.web_port}/"
        if not wait_healthy(url, web, 60):
            print(f"{RED}The front end did not start; see its output above.{RESET}")
            return 1
        print(
            f"\n  {GREEN}Demo ready:{RESET} {url}\n  {DIM}Ctrl+C stops both.{RESET}\n", flush=True
        )
        if args.open:
            webbrowser.open(url)
        while api.poll() is None and web.poll() is None:
            time.sleep(1.0)
        print(f"{RED}A server exited; stopping the other.{RESET}")
        return 1
    except KeyboardInterrupt:
        print("\n  stopping ...")
        return 0
    finally:
        stop(web, api)


if __name__ == "__main__":
    sys.exit(main())
