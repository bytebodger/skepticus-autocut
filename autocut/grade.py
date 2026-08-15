"""Stage 5 — grade.

Apply a single 3D LUT, not a stack of eq parameters. The LUT is built once in
DaVinci Resolve from a representative still and versioned by filename, so every
episode gets the same look with zero per-episode tuning.
"""

from __future__ import annotations

import logging

from . import cache, edl, ffmpeg
from .paths import Episode

log = logging.getLogger("autocut.grade")

CRF = 18


def _lut_path(ep: Episode, edl_doc: dict) -> str:
    lut = edl_doc.get("grade")
    if not lut:
        return ""
    # Resolve relative to repo root; store forward-slashed for the filter.
    p = (ep.root / lut)
    return p.as_posix()


def run(ep: Episode, *, force: bool = False) -> None:
    """Execute stage 5. Cached on cut hash + LUT contents."""
    if not ep.cut.exists():
        raise FileNotFoundError(f"No cut at {ep.cut}. Run stage 4 first.")

    edl_doc = edl.load(ep.edl_json)
    lut_rel = edl_doc.get("grade")

    if not lut_rel:
        log.info("grade: no LUT specified; nothing to do")
        return

    lut_file = ep.root / lut_rel
    if not lut_file.exists() and not ffmpeg.is_dry_run():
        raise FileNotFoundError(
            f"LUT not found: {lut_file}. Build it in DaVinci Resolve and drop the "
            f".cube in luts/ (see spec section 8)."
        )

    stage_dir = ep.work / "grade"
    input_hash = cache.hash_inputs({
        "cut": cache.hash_file(ep.cut) if not ffmpeg.is_dry_run() else "dry",
        "lut": cache.hash_file(lut_file) if lut_file.exists() and not ffmpeg.is_dry_run() else lut_rel,
        "crf": CRF,
    })
    if not force and cache.is_current(stage_dir, input_hash) and ep.graded.exists():
        log.info("grade: cache hit, skipping")
        return

    lut_posix = lut_file.as_posix()
    log.info("grade: applying %s -> %s", lut_rel, ep.graded)
    ffmpeg.run_ffmpeg([
        "-i", ep.cut,
        "-vf", f"lut3d={lut_posix}",
        "-c:v", "libx264", "-preset", "slow", "-crf", CRF,
        "-c:a", "copy",
        ep.graded,
    ])
    cache.mark_done(stage_dir, input_hash, extra={"stage": "grade"})
