"""Stage 8 — composite and encode.

Overlays plus captions in a single pass. The filter graph is written to a file
because Windows caps command lines at 8191 chars and a graph with many overlays
blows past that.

Overlay ``enable`` windows are in the OUTPUT timebase, produced by running each
overlay's source_time through ``source_to_output``. Subtitle paths inside a
filter graph are escaping-hostile on Windows, so we run ffmpeg from the repo root
and reference the .ass with a relative, forward-slashed path.
"""

from __future__ import annotations

import logging

from . import cache, edl, ffmpeg, overlays as overlays_mod
from .paths import Episode

log = logging.getLogger("autocut.composite")

CRF = 18


def _build_filter_graph(ep: Episode, edl_doc: dict, overlay_records: list[dict]) -> str:
    spans = edl.build_time_map(edl_doc["segments"])
    lines: list[str] = []
    cur = "0:v"

    for idx, ov in enumerate(overlay_records):
        out_start = edl.source_to_output(ov["source_time"], spans)
        if out_start is None:
            # Validation should have caught this; skip defensively.
            log.warning("composite: overlay %s source_time is in a cut region; skipping", ov["id"])
            continue
        out_end = out_start + float(ov.get("duration", 4.0))
        x = ov.get("props", {}).get("x", 0)
        y = ov.get("props", {}).get("y", 0)
        label = f"v{idx + 1}"
        # ffmpeg input index: 0 is graded, overlays start at 1.
        lines.append(
            f"[{cur}][{idx + 1}:v]"
            f"overlay=enable='between(t,{out_start:.3f},{out_end:.3f})':x={x}:y={y}"
            f"[{label}]"
        )
        cur = label

    captions = edl_doc.get("captions", {})
    if captions.get("enabled", True) and ep.captions_ass.exists():
        ass_rel = ep.captions_ass.resolve().relative_to(ep.root.resolve()).as_posix()
        fonts_rel = ep.styles_dir.resolve().relative_to(ep.root.resolve()).as_posix()
        lines.append(f"[{cur}]subtitles={ass_rel}:fontsdir={fonts_rel}[vout]")
    else:
        # No captions: alias the last video label to [vout] with a passthrough.
        lines.append(f"[{cur}]null[vout]")

    return ";\n".join(lines) + "\n"


def run(ep: Episode, *, force: bool = False) -> None:
    """Execute stage 8. Produces the final mp4 in out/."""
    if not ep.graded.exists():
        raise FileNotFoundError(f"No graded video at {ep.graded}. Run stage 5 first.")

    edl_doc = edl.load(ep.edl_json)

    # Ensure overlays are rendered (stage 6 is cached, so this is cheap if done).
    overlay_records = overlays_mod.run(ep, force=force)

    graph = _build_filter_graph(ep, edl_doc, overlay_records)
    ep.filter_script.write_text(graph, encoding="utf-8")

    stage_dir = ep.work / "composite"
    input_hash = cache.hash_inputs({
        "graded": cache.hash_file(ep.graded) if not ffmpeg.is_dry_run() else "dry",
        "graph": graph,
        "captions": cache.hash_file(ep.captions_ass) if ep.captions_ass.exists() else None,
        "overlays": [r["webm"] for r in overlay_records],
        "crf": CRF,
    })
    if not force and cache.is_current(stage_dir, input_hash) and ep.output_mp4.exists():
        log.info("composite: cache hit, skipping")
        return

    ep.out_dir.mkdir(parents=True, exist_ok=True)

    inputs = ["-i", ep.graded]
    for r in overlay_records:
        inputs += ["-i", r["webm"]]

    log.info("composite: %d overlays + captions -> %s", len(overlay_records), ep.output_mp4)
    ffmpeg.run_ffmpeg(
        [
            *inputs,
            # -/filter_complex reads the graph from a file (the modern
            # replacement for the removed -filter_complex_script). The graph is
            # file-based because a many-overlay graph exceeds Windows' 8191-char
            # command-line limit.
            "-/filter_complex", ep.filter_script,
            "-map", "[vout]", "-map", "0:a",
            "-c:v", "libx264", "-preset", "slow", "-crf", CRF, "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k",
            "-movflags", "+faststart",
            ep.output_mp4,
        ],
        cwd=str(ep.root),  # keep subtitle path relative for Windows escaping
    )
    cache.mark_done(stage_dir, input_hash, extra={"stage": "composite"})
