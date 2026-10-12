"""Run manifest utilities for recording reproducible run metadata.

Each stage writes a per-stage manifest file `<timestamp>_stage<N>.txt` into its
own output subdirectory, and appends a one-line summary to a shared
`output/runs/runs.log`.
"""
import os
import sys
import json
import time
import datetime
import socket
import subprocess

# Resolved relative to the repo root so it works regardless of CWD.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _git_head() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=_REPO_ROOT,
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


def build_manifest(stage: str, args, started_at: float) -> dict:
    """Collect run metadata common to every stage."""
    return {
        "stage": stage,
        "started_at": datetime.datetime.fromtimestamp(started_at).isoformat(),
        "finished_at": datetime.datetime.now().isoformat(),
        "elapsed_seconds": round(time.time() - started_at, 2),
        "hostname": socket.gethostname(),
        "git_head": _git_head(),
        "argv": sys.argv,
        "args": vars(args),
    }


def _as_kv(d: dict, indent: int = 0) -> str:
    lines = []
    pad = " " * indent
    for k, v in d.items():
        if isinstance(v, dict):
            lines.append(f"{pad}{k}:")
            lines.append(_as_kv(v, indent + 2))
        else:
            lines.append(f"{pad}{k}: {v}")
    return "\n".join(lines)


def write_manifest(output_dir: str, stage: str, manifest: dict) -> str:
    """Write per-stage manifest into output_dir; return the file path.

    Also appends a one-line summary to <repo>/output/runs/runs.log.
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filepath = os.path.join(output_dir, f"{timestamp}_stage{stage}.txt")

    body = _as_kv(manifest)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(body + "\n")

    # Shared summary log under output/runs/.
    runs_dir = os.path.join(_REPO_ROOT, "output", "runs")
    os.makedirs(runs_dir, exist_ok=True)
    summary = {
        "time": manifest["finished_at"],
        "stage": stage,
        "hostname": manifest["hostname"],
        "git_head": manifest["git_head"],
        "output_dir": output_dir,
        "file": filepath,
    }
    with open(os.path.join(runs_dir, "runs.log"), "a", encoding="utf-8") as f:
        f.write(json.dumps(summary, ensure_ascii=False) + "\n")

    return filepath