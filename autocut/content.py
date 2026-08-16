"""Phase-2 compositor — content track (spec section 4).

Reads ``content.json``, maps each item's ``source_time`` through
``source_to_output`` (the same source-timebase rule as Phase-1 overlays), and
builds a content-rect-sized video layer at full output duration: items placed at
their output times, gap behaviour applied, contain/cover fit with the configured
letterbox, and short crossfades. Built as its own cached step; the composite
just overlays it (spec: don't fold it into one giant graph).

Crossfades are done with overlaid alpha fades rather than the ``xfade`` filter:
that composes cleanly with the exact output-time placement and with both gap
modes (items fade to/from the wallpaper in ``background`` mode), where chaining
``xfade`` would drift the timeline and can't express item -> wallpaper -> item.

``AUTOCUT_CONTENT_SECONDS`` caps the rendered length (a full hour-long track is
an overnight encode); leave it unset for the real render.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from . import cache, edl, ffmpeg, layout as layout_mod
from .paths import Episode

log = logging.getLogger("autocut.content")

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".gif"}
DEFAULT_ITEM_DURATION = 6.0


def _ff_color(value: str) -> str:
    v = str(value).strip()
    return "0x" + v[1:] if v.startswith("#") else v


def load_items(ep: Episode) -> list[dict] | None:
    """Items from content.json, or None if the episode has no content folder."""
    if not ep.content_json.exists():
        return None
    data = json.loads(ep.content_json.read_text(encoding="utf-8"))
    return list(data.get("items") or [])


def _place(items: list[dict], ep: Episode) -> tuple[list[tuple[dict, float]], float]:
    """Map each item's source_time to its output start; drop items that land in
    a cut region. Returns (placed sorted by output time, output_duration)."""
    edl_doc = edl.load(ep.edl_json)
    spans = edl.build_time_map(edl_doc["segments"])
    out_dur = edl.output_duration(spans)
    placed: list[tuple[dict, float]] = []
    for it in items:
        st = float(it["source_time"])
        out = edl.source_to_output(st, spans)
        if out is None:
            log.warning("content: %s at source_time %.3f falls in a cut region; skipping",
                        it.get("file"), st)
            continue
        placed.append((it, out))
    placed.sort(key=lambda p: p[1])
    return placed, out_dur


def _fit_filter(cw: int, ch: int, fit: str, fill: str) -> str:
    if fit == "cover":  # fill the rect, crop the overflow
        return f"scale={cw}:{ch}:force_original_aspect_ratio=increase,crop={cw}:{ch}"
    # contain — letterbox, never crop the subject (spec section 4)
    return (f"scale={cw}:{ch}:force_original_aspect_ratio=decrease,"
            f"pad={cw}:{ch}:(ow-iw)/2:(oh-ih)/2:color={fill}")


def _cap_seconds(seconds: float | None) -> float | None:
    if seconds is not None:
        return seconds
    env = os.environ.get("AUTOCUT_CONTENT_SECONDS")
    return float(env) if env else None


def build_graph(ep: Episode, layout: dict, fps: str, out_dur: float,
                placed: list[tuple[dict, float]], *, seconds: float | None):
    """Return (input_args, graph_text, uses_alpha) for the content track."""
    _, _, cw, ch = layout_mod.rect(layout, "content")
    cfg = layout["content"]
    fit = cfg.get("fit", "contain")
    fill = _ff_color(cfg.get("background", "#000000"))
    gap = cfg.get("gap_behavior", "hold")
    tr = cfg.get("transition") or {}
    crossfade = tr.get("type", "crossfade") == "crossfade" and float(tr.get("duration", 0.35)) > 0
    fade_d = float(tr.get("duration", 0.35))

    dur = out_dur if seconds is None else min(out_dur, float(seconds))
    n = len(placed)

    inputs: list[str] = []
    item_graph: list[str] = []
    labels: list[str] = []
    k = 0  # ffmpeg input index (only for items actually added)
    for i, (it, w_start) in enumerate(placed):
        if w_start >= dur:
            continue
        path = ep.content_dir / it["file"]
        item_dur = float(it.get("duration", DEFAULT_ITEM_DURATION))
        if gap == "background":
            seglen = item_dur
        else:  # hold — extend to the next item (last item holds to the end)
            next_start = placed[i + 1][1] if i + 1 < n else dur
            seglen = max(0.1, next_start - w_start)
            if crossfade and i + 1 < n:
                seglen += fade_d  # persist under the next item's fade-in
        seglen = min(seglen, dur - w_start + (fade_d if crossfade else 0.0))

        if path.suffix.lower() in IMAGE_EXTS:
            inputs += ["-loop", "1", "-t", f"{seglen:.3f}", "-i", str(path)]
        else:
            inputs += ["-t", f"{seglen:.3f}", "-i", str(path)]

        chain = [_fit_filter(cw, ch, fit, fill), "format=yuva420p"]
        if crossfade:
            chain.append(f"fade=t=in:st=0:d={fade_d}:alpha=1")
            if gap == "background":  # fade back to wallpaper (no next item covers it)
                chain.append(f"fade=t=out:st={max(0.0, seglen - fade_d):.3f}:d={fade_d}:alpha=1")
        chain.append(f"setpts=PTS-STARTPTS+{w_start:.3f}/TB")
        item_graph.append(f"[{k}:v]{','.join(chain)}[c{k}]")
        labels.append(f"c{k}")
        k += 1

    if gap == "background":
        base = (f"color=c=black:s={cw}x{ch}:r={fps}:d={dur:.3f},"
                f"format=yuva420p,colorchannelmixer=aa=0[base]")
        uses_alpha = True
    else:
        base = f"color=c={fill}:s={cw}x{ch}:r={fps}:d={dur:.3f},format=yuva420p[base]"
        uses_alpha = False

    graph = [base] + item_graph
    cur = "base"
    for i, lbl in enumerate(labels):
        out_lbl = "ctrack" if i == len(labels) - 1 else f"o{i}"
        graph.append(f"[{cur}][{lbl}]overlay=eof_action=pass[{out_lbl}]")
        cur = out_lbl
    if not labels:
        graph.append("[base]null[ctrack]")
    return inputs, ";\n".join(graph) + "\n", uses_alpha


def render_track(ep: Episode, layout: dict, fps: str, *, force: bool = False,
                 seconds: float | None = None) -> Path | None:
    """Render the content track, cached. Returns its path, or None if the episode
    has no content.json / no items that land in the output."""
    items = load_items(ep)
    if items is None:
        return None
    placed, out_dur = _place(items, ep)
    if not placed:
        log.warning("content: no items map into the output timeline; skipping content track")
        return None

    seconds = _cap_seconds(seconds)
    inputs, graph, uses_alpha = build_graph(ep, layout, fps, out_dur, placed, seconds=seconds)
    dur = out_dur if seconds is None else min(out_dur, float(seconds))
    dry = ffmpeg.is_dry_run()

    stage_dir = ep.compose_dir / "content"
    item_hashes = {
        it["file"]: (cache.hash_file(ep.content_dir / it["file"])
                     if (ep.content_dir / it["file"]).exists() and not dry else "dry")
        for it, _ in placed
    }
    input_hash = cache.hash_inputs({
        "graph": graph,
        "items": item_hashes,
        "edl": cache.hash_file(ep.edl_json) if not dry else "dry",
        "fps": fps,
        "seconds": seconds,
        "uses_alpha": uses_alpha,
    })
    if not force and cache.is_current(stage_dir, input_hash) and ep.content_track.exists():
        log.info("content: track cache hit")
        return ep.content_track

    ep.compose_dir.mkdir(parents=True, exist_ok=True)
    ep.content_filter_script.write_text(graph, encoding="utf-8")
    if uses_alpha:
        vcodec = ["-c:v", "qtrle"]  # lossless alpha, cheap on flat/transparent regions
    else:
        vcodec = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p"]
    log.info("content: %d item(s) -> %s (%.1fs, alpha=%s)", len(placed), ep.content_track, dur, uses_alpha)
    ffmpeg.run_ffmpeg([
        *inputs,
        "-/filter_complex", ep.content_filter_script,
        "-map", "[ctrack]",
        "-t", f"{dur:.3f}",
        "-r", fps,
        *vcodec,
        ep.content_track,
    ])
    cache.mark_done(stage_dir, input_hash, extra={"stage": "compose:content"})
    return ep.content_track
