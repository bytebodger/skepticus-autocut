"""Stage 6 — HyperFrames overlays.

Each overlay in the EDL names a composition under ``compositions/`` and supplies
props. HyperFrames renders a PNG sequence (PNGs carry alpha); we encode that to
an alpha-capable VP9 WebM so the card composites *over* the footage rather than
replacing it.

Rendering headless-Chrome frames is slow, so cache aggressively: the key is a
hash of the composition source plus the props plus timing. Most episodes reuse
the same cards with different text.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from pathlib import Path

from . import cache, edl, ffmpeg
from .paths import Episode

log = logging.getLogger("autocut.overlays")

FPS = 30


def _hash_tree(path: Path) -> str:
    """Stable hash of every file under a composition directory."""
    if not path.exists():
        return "missing"
    parts = {}
    for f in sorted(path.rglob("*")):
        if f.is_file():
            parts[f.relative_to(path).as_posix()] = cache.hash_file(f)
    return cache.hash_inputs(parts)


def _run_npx(args: list[str]) -> None:
    npx = shutil.which("npx") or ("npx.cmd" if os.name == "nt" else "npx")
    cmd = [npx, *args]
    log.info("run: %s", " ".join(cmd))
    if ffmpeg.is_dry_run():
        return
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, errors="replace")
    if proc.returncode != 0:
        raise RuntimeError(f"hyperframes failed (exit {proc.returncode}):\n{proc.stdout}")


def _render_frames(ep: Episode, ov: dict, comp_dir: Path, frames_dir: Path) -> None:
    frames_dir.mkdir(parents=True, exist_ok=True)
    props_file = frames_dir.parent / "props.json"
    props_file.write_text(json.dumps(ov.get("props", {}), indent=2), encoding="utf-8")
    # HyperFrames captures PNG frames (with alpha) from the HTML composition.
    _run_npx([
        "hyperframes", "render", str(comp_dir),
        "--props", str(props_file),
        "--fps", str(FPS),
        "--duration", str(ov.get("duration", 4.0)),
        "--out", str(frames_dir),
    ])


def _encode_webm(frames_dir: Path, out_webm: Path) -> None:
    # yuva420p is the alpha-carrying pixel format; VP9-in-WebM is the reliable
    # alpha-capable container FFmpeg handles well.
    ffmpeg.run_ffmpeg([
        "-framerate", FPS,
        "-i", str(frames_dir / "frame_%05d.png"),
        "-c:v", "libvpx-vp9", "-pix_fmt", "yuva420p", "-lossless", "1",
        out_webm,
    ])


def run(ep: Episode, *, force: bool = False) -> list[dict]:
    """Execute stage 6 for every overlay in the EDL. Returns overlay records
    augmented with the rendered webm path."""
    edl_doc = edl.load(ep.edl_json)
    overlays = edl_doc.get("overlays", []) or []
    results = []

    for ov in overlays:
        comp_name = ov["composition"]
        comp_dir = ep.compositions_dir / comp_name
        ov_dir = ep.overlays_dir / ov["id"]
        frames_dir = ov_dir / "frames"
        out_webm = ep.overlays_dir / f"{ov['id']}.webm"

        input_hash = cache.hash_inputs({
            "composition": _hash_tree(comp_dir),
            "props": ov.get("props", {}),
            "duration": ov.get("duration", 4.0),
            "fps": FPS,
        })
        if not force and cache.is_current(ov_dir, input_hash) and out_webm.exists():
            log.info("overlays: %s cache hit (%s)", ov["id"], comp_name)
            results.append({**ov, "webm": str(out_webm)})
            continue

        if not comp_dir.exists() and not ffmpeg.is_dry_run():
            raise FileNotFoundError(
                f"Composition {comp_name!r} not found at {comp_dir}. "
                f"Create it with: npx hyperframes init {comp_dir}"
            )

        log.info("overlays: rendering %s (%s)", ov["id"], comp_name)
        _render_frames(ep, ov, comp_dir, frames_dir)
        _encode_webm(frames_dir, out_webm)
        cache.mark_done(ov_dir, input_hash, extra={"stage": "overlays", "overlay": ov["id"]})
        results.append({**ov, "webm": str(out_webm)})

    return results
