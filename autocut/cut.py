"""Stage 4 — cut and concat.

Each kept segment is encoded separately with *identical* parameters, then joined
with a stream copy. Two things are load-bearing:

  * The 25ms in/out audio fades are mandatory. Without them every splice clicks.
  * Every segment uses the same encoder settings, which is the only reason the
    final concat can be a stream copy (no re-encode of the timeline).

Segments are independently cacheable: change one cut, re-encode one segment.
"""

from __future__ import annotations

import json
import logging

from . import cache, edl, ffmpeg
from .paths import Episode

log = logging.getLogger("autocut.cut")

FADE = 0.025  # 25ms — inaudible as a fade, long enough to kill the click.
CRF = 16


def _encode_segment(ep: Episode, seg: dict) -> None:
    seg_in = float(seg["in"])
    seg_out = float(seg["out"])
    dur = seg_out - seg_in
    out_path = ep.segments_dir / f"{seg['id']}.mkv"

    # afade start times are relative to the SEGMENT, not the source, because -ss
    # comes before -i (input seeking resets the segment clock to 0).
    fade_out_start = max(0.0, dur - FADE)
    afade = f"afade=t=in:st=0:d={FADE},afade=t=out:st={fade_out_start:.3f}:d={FADE}"

    ffmpeg.run_ffmpeg([
        "-ss", f"{seg_in:.3f}", "-to", f"{seg_out:.3f}", "-i", ep.mezz,
        "-c:v", "libx264", "-preset", "veryfast", "-crf", CRF,
        "-g", "1", "-pix_fmt", "yuv420p",
        "-af", afade,
        "-c:a", "pcm_s16le", "-ar", "48000",
        out_path,
    ])


def _write_concat_list(ep: Episode, kept: list[dict]) -> None:
    # concat demuxer wants forward slashes and quoted paths, even on Windows.
    lines = []
    for seg in kept:
        seg_path = (ep.segments_dir / f"{seg['id']}.mkv").resolve().as_posix()
        lines.append(f"file '{seg_path}'")
    ep.concat_list.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(ep: Episode, *, force: bool = False) -> None:
    """Execute stage 4. Per-segment caching keyed on segment in/out + mezz hash."""
    if not ep.edl_json.exists():
        raise FileNotFoundError(f"No EDL at {ep.edl_json}. Author it (stage 3) first.")
    if not ep.mezz.exists():
        raise FileNotFoundError(f"No mezzanine at {ep.mezz}. Run probe first.")

    edl_doc = edl.load(ep.edl_json)
    errors = edl.validate(edl_doc)
    if errors:
        raise edl.ValidationError(errors)

    kept = [s for s in edl_doc["segments"] if s["action"] == "keep"]
    ep.segments_dir.mkdir(parents=True, exist_ok=True)
    mezz_hash = cache.hash_file(ep.mezz) if ep.mezz.exists() and not ffmpeg.is_dry_run() else "dry"

    # Encode each kept segment, skipping unchanged ones.
    for seg in kept:
        seg_dir = ep.segments_dir / seg["id"]
        seg_hash = cache.hash_inputs({
            "mezz": mezz_hash, "in": seg["in"], "out": seg["out"],
            "fade": FADE, "crf": CRF,
        })
        seg_file = ep.segments_dir / f"{seg['id']}.mkv"
        if not force and cache.is_current(seg_dir, seg_hash) and seg_file.exists():
            log.info("cut: segment %s cache hit", seg["id"])
            continue
        log.info("cut: encoding segment %s [%.3f, %.3f]", seg["id"], seg["in"], seg["out"])
        _encode_segment(ep, seg)
        cache.mark_done(seg_dir, seg_hash, extra={"stage": "cut", "segment": seg["id"]})

    _write_concat_list(ep, kept)

    log.info("cut: concat %d segments -> %s", len(kept), ep.cut)
    ffmpeg.run_ffmpeg([
        "-f", "concat", "-safe", "0", "-i", ep.concat_list,
        "-c", "copy", ep.cut,
    ])
