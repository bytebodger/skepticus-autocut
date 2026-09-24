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

The track is rendered for a given output ``window`` (start, length) — the full
output by default, or a --preview/--range window — so iteration doesn't force a
full-length encode. Item output times are made relative to the window start.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from . import cache, edl, ffmpeg, layout as layout_mod
from .paths import Episode

log = logging.getLogger("autocut.content")

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".gif"}
DEFAULT_ITEM_DURATION = 6.0


def _ff_color(value: str) -> str:
    v = str(value).strip()
    return "0x" + v[1:] if v.startswith("#") else v


def _content_source(ep: Episode) -> tuple[Path, Path] | tuple[None, None]:
    """(content.json path, base dir for item files). Prefer the render stage's
    generated content.json (files under work/<ep>/visuals/); fall back to a manual
    inbox drop (inbox/<ep>_content/)."""
    if ep.visuals_content_json.exists():
        return ep.visuals_content_json, ep.visuals_dir
    if ep.content_json.exists():
        return ep.content_json, ep.content_dir
    return None, None


PREROLL_DURATION = 0.1  # real-decode span before hold takes over freezing frame 1


def _thumb_path(ep: Episode, layout: dict | None) -> Path | None:
    """The thumbnail to show during opening commentary (reaction spec): an
    explicit ``reaction.thumbnail`` config path (relative to the repo root)
    if set and present, else the auto-detected ``inbox/<ep>_thumb.<ext>``,
    else ``None`` (caller falls back to the frozen first frame)."""
    override = ((layout or {}).get("reaction") or {}).get("thumbnail")
    if override:
        path = ep.root / override
        if path.exists():
            return path
        log.warning("reaction: configured thumbnail %s not found; falling back", path)
    return ep.source_thumb


def _reaction_items(ep: Episode, layout: dict | None = None) -> tuple[list[dict], Path] | None:
    """Content items synthesized from the reaction alignment map (reaction spec
    section 7): one video item per playback segment, in host (source) time,
    seeking into the source file at ``clip_in``. The gaps *between* segments —
    commentary — are left to ``gap_behavior: hold`` (the default), which holds
    the previous item's last frame: a held frame of the source, frozen where
    playback stopped.

    The recording opens in commentary (spec §2), so there is no "previous
    item" for hold mode to hold before the first cue — without an explicit
    item there, the content window would show nothing at all for that leading
    span. A pre-roll item covers it, held until the first playback segment
    actually starts (with a crossfade into it, same as any other item
    boundary). Placed at the EDL's first kept instant rather than hardcoded
    host time 0 — true 0 is typically inside the leading-dead-air drop
    autoauthor trims, which would silently drop the item entirely
    (spec-consistent: dead air never survives into output either, so nothing
    is actually lost by anchoring to the first live moment instead).

    The pre-roll item is the source's YouTube thumbnail when one is
    available (``_thumb_path``), else the source's own frame 1 (``clip_in:
    0``) — the prior, thumbnail-less convention.
    """
    if not ep.playback_json.exists():
        return None
    playback = json.loads(ep.playback_json.read_text(encoding="utf-8"))
    segments = playback.get("segments", [])
    if not segments:
        return None
    file = playback["source_file"]
    preroll_time = 0.0
    if ep.edl_json.exists():
        edl_doc = edl.load(ep.edl_json)
        first_keep = next((s for s in edl_doc.get("segments", []) if s.get("action") == "keep"), None)
        if first_keep is not None:
            preroll_time = float(first_keep["in"])
    thumb = _thumb_path(ep, layout)
    if thumb is not None:
        preroll_item = {"file": str(thumb), "source_time": preroll_time, "duration": PREROLL_DURATION}
    else:
        preroll_item = {"file": file, "source_time": preroll_time, "duration": PREROLL_DURATION, "clip_in": 0.0}
    items = [preroll_item]
    items += [
        {
            "file": file,
            "source_time": s["host_in"],
            "duration": s["host_out"] - s["host_in"],
            "clip_in": s["source_in"],
        }
        for s in segments
    ]
    # file is an absolute path (as align.run writes it), so the base dir is
    # irrelevant to path resolution — ep.root is just a harmless placeholder.
    return items, ep.root


def load_items(ep: Episode, layout: dict | None = None) -> tuple[list[dict], Path] | tuple[None, None]:
    """(items, base_dir) from the chosen content.json, or (None, None) if the
    episode has no content source. ``layout`` is only consulted for the
    reaction-format fallback (``reaction.thumbnail`` override)."""
    src, base = _content_source(ep)
    if src is not None:
        data = json.loads(src.read_text(encoding="utf-8"))
        return list(data.get("items") or []), base
    reaction = _reaction_items(ep, layout)
    if reaction is not None:
        return reaction
    return None, None


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


def build_graph(ep: Episode, layout: dict, fps: str, window: tuple,
                placed: list[tuple[dict, float]], base_dir: Path):
    """Return (input_args, graph_text, uses_alpha) for the content track over the
    output window (w0, length). Item output times are made window-relative."""
    w0, dur = window
    _, _, cw, ch = layout_mod.rect(layout, "content")
    cfg = layout["content"]
    fit = cfg.get("fit", "contain")
    fill = _ff_color(cfg.get("background", "#000000"))
    gap = cfg.get("gap_behavior", "hold")
    tr = cfg.get("transition") or {}
    crossfade = tr.get("type", "crossfade") == "crossfade" and float(tr.get("duration", 0.35)) > 0
    fade_d = float(tr.get("duration", 0.35))

    # Window-relative item starts. Include items starting inside the window, plus
    # the single item active at the window start — it began earlier and, in hold
    # mode, is still covering the window — so a --range/--preview window is never
    # empty at t=0 (its w_start is negative; the overlay clips the pre-roll). For a
    # full render (w0=0) there is no earlier item, so this is a preview-only fix.
    in_win = [(it, out - w0) for it, out in placed if w0 - 1e-6 <= out < w0 + dur]
    before = [(it, out - w0) for it, out in placed if out < w0 - 1e-6]
    rel = ([before[-1]] if before else []) + in_win
    n = len(rel)

    inputs: list[str] = []
    item_graph: list[str] = []
    labels: list[str] = []
    k = 0  # ffmpeg input index (only for items actually added)
    for i, (it, w_start) in enumerate(rel):
        if w_start >= dur:
            continue
        path = base_dir / it["file"]
        item_dur = float(it.get("duration", DEFAULT_ITEM_DURATION))
        if gap == "background":
            seglen = item_dur
        else:  # hold — extend to the next item (last item holds to the end)
            next_start = rel[i + 1][1] if i + 1 < n else dur
            seglen = max(0.1, next_start - w_start)
            if crossfade and i + 1 < n:
                seglen += fade_d  # persist under the next item's fade-in
        seglen = min(seglen, dur - w_start + (fade_d if crossfade else 0.0))

        is_image = path.suffix.lower() in IMAGE_EXTS
        if is_image:
            inputs += ["-loop", "1", "-t", f"{seglen:.3f}", "-i", str(path)]
        else:
            clip_in = it.get("clip_in")
            if clip_in is not None:
                inputs += ["-ss", f"{float(clip_in):.3f}"]
            # Cap the input read at the item's OWN duration, not the (possibly
            # hold-extended) seglen: a reaction item's file is the full source
            # video, not a pre-trimmed card, so reading straight through to
            # seglen would keep playing real source content past clip_out
            # instead of freezing there. The tpad/trim below extend the frozen
            # last frame across the rest of seglen.
            inputs += ["-t", f"{item_dur:.3f}", "-i", str(path)]

        chain = [_fit_filter(cw, ch, fit, fill), "format=yuva420p"]
        if gap != "background" and not is_image:
            # hold: freeze the last frame to fill the span so the content window is
            # never empty between cards. A looped still already holds; a finite
            # video (our ProRes cards) needs the clone-pad. trim caps the result to
            # the span whether the card is shorter (padded) or longer (truncated).
            chain.append(f"tpad=stop_mode=clone:stop_duration={seglen:.3f}")
            chain.append(f"trim=duration={seglen:.3f}")
        if crossfade:
            chain.append(f"fade=t=in:st=0:d={fade_d}:alpha=1")
            if gap == "background":  # fade back to wallpaper (no next item covers it)
                chain.append(f"fade=t=out:st={max(0.0, seglen - fade_d):.3f}:d={fade_d}:alpha=1")
        chain.append(f"setpts=PTS-STARTPTS{w_start:+.3f}/TB")
        item_graph.append(f"[{k}:v]{','.join(chain)}[c{k}]")
        labels.append(f"c{k}")
        k += 1

    # Transparent base in BOTH modes so the cards' own alpha (ProRes 4444) reaches
    # the compositor — the show background shows through the cards' transparent
    # regions. The modes differ only in the gaps: `background` lets the base (and
    # so the wallpaper) show between cards; `hold` clone-holds the last card up to
    # the next, so the window is never empty (the fill colour is now unused).
    base = (f"color=c=black:s={cw}x{ch}:r={fps}:d={dur:.3f},"
            f"format=yuva420p,colorchannelmixer=aa=0[base]")
    uses_alpha = True

    graph = [base] + item_graph
    cur = "base"
    for i, lbl in enumerate(labels):
        out_lbl = "ctrack" if i == len(labels) - 1 else f"o{i}"
        graph.append(f"[{cur}][{lbl}]overlay=eof_action=pass[{out_lbl}]")
        cur = out_lbl
    if not labels:
        graph.append("[base]null[ctrack]")
    return inputs, ";\n".join(graph) + "\n", uses_alpha


def render_track(ep: Episode, layout: dict, fps: str, window: tuple,
                 *, force: bool = False) -> Path | None:
    """Render the content track for the output window, cached. Returns its path,
    or None if the episode has no content.json / no items land in the window."""
    items, base_dir = load_items(ep, layout)
    if items is None:
        return None
    placed, _out_dur = _place(items, ep)
    if not placed:
        log.warning("content: no items map into the output timeline; skipping content track")
        return None

    inputs, graph, uses_alpha = build_graph(ep, layout, fps, window, placed, base_dir)
    dur = window[1]
    dry = ffmpeg.is_dry_run()
    # clip_in marks an item as a chunk seeked out of a larger source file (reaction
    # spec section 7) — real, full-motion footage rather than a small pre-rendered
    # card. qtrle is cheap on the flat/mostly-static cards it was chosen for, but
    # balloons on full motion; ProRes 4444 (already used for the speaker layer)
    # handles motion properly at a real cost/quality tradeoff. One codec per track
    # (ffmpeg can't mix codecs within one output stream), so the choice covers the
    # whole track if anything in it needs motion handling.
    has_motion_source = any(it.get("clip_in") is not None for it, _ in placed)

    stage_dir = ep.compose_dir / "content"
    item_hashes = {
        it["file"]: (cache.hash_file(base_dir / it["file"])
                     if (base_dir / it["file"]).exists() and not dry else "dry")
        for it, _ in placed
    }
    input_hash = cache.hash_inputs({
        "graph": graph,
        "items": item_hashes,
        "edl": cache.hash_file(ep.edl_json) if not dry else "dry",
        "fps": fps,
        "window": [round(window[0], 3), round(window[1], 3)],
        "uses_alpha": uses_alpha,
        "has_motion_source": has_motion_source,
    })
    if not force and cache.is_current(stage_dir, input_hash) and ep.content_track.exists():
        log.info("content: track cache hit")
        return ep.content_track

    ep.compose_dir.mkdir(parents=True, exist_ok=True)
    ep.content_filter_script.write_text(graph, encoding="utf-8")
    if uses_alpha and has_motion_source:
        vcodec = ["-c:v", "prores_ks", "-profile:v", "4444", "-pix_fmt", "yuva444p10le"]
    elif uses_alpha:
        vcodec = ["-c:v", "qtrle"]  # lossless alpha, cheap on flat/transparent regions
    else:
        vcodec = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p"]
    log.info("content: %d item(s) -> %s (%.1fs, alpha=%s, codec=%s)",
             len(placed), ep.content_track, dur, uses_alpha, vcodec[1])
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
