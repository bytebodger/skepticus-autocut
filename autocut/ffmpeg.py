"""Thin wrapper around ffmpeg / ffprobe.

Responsibilities: locate the binaries, log the exact command, surface stderr on
failure, and support a global dry-run mode (print commands, run nothing). This is
the *only* module that shells out to FFmpeg. Nothing else in the pipeline — and
certainly not Claude Code — invokes ffmpeg directly.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Sequence

log = logging.getLogger("autocut.ffmpeg")


class FFmpegError(RuntimeError):
    def __init__(self, cmd: Sequence[str], returncode: int, stderr: str):
        self.cmd = list(cmd)
        self.returncode = returncode
        self.stderr = stderr
        tail = "\n".join(stderr.strip().splitlines()[-25:])
        super().__init__(
            f"{cmd[0]} exited {returncode}\n"
            f"  cmd: {' '.join(cmd)}\n"
            f"  stderr (tail):\n{tail}"
        )


class FFmpegNotFound(RuntimeError):
    pass


# Dry-run is process-global and toggled by the CLI's --dry-run flag. In dry-run
# no encode runs; commands are logged so you can eyeball the graph.
_DRY_RUN = False


def set_dry_run(value: bool) -> None:
    global _DRY_RUN
    _DRY_RUN = value


def is_dry_run() -> bool:
    return _DRY_RUN


def _binary(name: str) -> str:
    # Allow explicit override; otherwise fall back to PATH. Pinning the build
    # (spec section 15) is done by setting these env vars to an absolute path.
    override = os.environ.get(f"AUTOCUT_{name.upper()}")
    resolved = override or shutil.which(name)
    if not resolved:
        # In dry-run we only print commands, so the binary need not exist yet.
        if _DRY_RUN:
            return name
        raise FFmpegNotFound(
            f"Could not find '{name}'. Install FFmpeg and put it on PATH, or set "
            f"AUTOCUT_{name.upper()} to its absolute path."
        )
    return resolved


def ffmpeg_path() -> str:
    return _binary("ffmpeg")


def ffprobe_path() -> str:
    return _binary("ffprobe")


def _run(cmd: list[str], *, capture: bool, cwd: str | None = None) -> subprocess.CompletedProcess[str]:
    log.info("run: %s%s", " ".join(cmd), f"  (cwd={cwd})" if cwd else "")
    if _DRY_RUN:
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=cwd,
    )
    if proc.returncode != 0:
        raise FFmpegError(cmd, proc.returncode, proc.stderr or "")
    return proc


def run_ffmpeg(args: Sequence[str], *, overwrite: bool = True, cwd: str | None = None) -> str:
    """Run ffmpeg with the given args (excluding the binary itself).

    ``-y``/``-n`` and ``-hide_banner`` are prepended. Returns stderr, which is
    where ffmpeg writes progress and filter diagnostics. ``cwd`` lets callers run
    from the repo root so filter-graph subtitle paths can stay relative — the
    only reliable way to escape them on Windows.
    """
    cmd = [ffmpeg_path(), "-hide_banner", "-y" if overwrite else "-n", *map(str, args)]
    proc = _run(cmd, capture=False, cwd=cwd)
    return proc.stderr or ""


def run_ffmpeg_capture_stderr(args: Sequence[str]) -> str:
    """Run ffmpeg where the useful output is on stderr (e.g. silencedetect).

    Uses ``-nostats`` to keep the progress spam out. Note: silencedetect writes
    to a null muxer, so ``-y`` is harmless.
    """
    cmd = [ffmpeg_path(), "-hide_banner", "-nostats", *map(str, args)]
    proc = _run(cmd, capture=True)
    # silencedetect logs land on stderr regardless of level.
    return proc.stderr or ""


def ffprobe_json(args: Sequence[str]) -> dict:
    """Run ffprobe with ``-of json`` and parse stdout."""
    cmd = [ffprobe_path(), "-hide_banner", "-of", "json", *map(str, args)]
    if _DRY_RUN:
        log.info("run: %s", " ".join(cmd))
        return {}
    proc = _run(cmd, capture=True)
    return json.loads(proc.stdout or "{}")


def ffmpeg_version() -> str:
    """First line of ``ffmpeg -version`` — recorded in the QC report for
    determinism auditing."""
    try:
        cmd = [ffmpeg_path(), "-version"]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, text=True, errors="replace")
        return (proc.stdout or "").splitlines()[0] if proc.stdout else "unknown"
    except (FFmpegNotFound, OSError):
        return "unknown"
