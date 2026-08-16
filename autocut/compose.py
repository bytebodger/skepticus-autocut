"""Phase-2 compositor — Step 1: layout config + static composite.

Renders a short static composite — background + one placeholder content image +
keyed speaker — so the geometry in ``config/layout.yaml`` can be checked by eye
(compositor spec section 9, step 1). No content timing, captions, or audio yet.

Two independently-cached steps:
  * speaker layer (spec section 3): crop → chromakey → despill → scale. Cached on
    its own so key parameters can be tuned without re-touching the composite.
  * composite (spec section 5): background + content + speaker overlay.

Every geometry value comes from config/layout.yaml; fps comes from probe.json.
The filter graphs are written to files and passed with ``-/filter_complex`` (the
ffmpeg 9.0 replacement for the removed ``-filter_complex_script``) so a large
graph can never hit Windows' 8191-char command-line limit.
"""

from __future__ import annotations

import logging
from pathlib import Path

from . import cache, ffmpeg, layout as layout_mod
from .paths import Episode

log = logging.getLogger("autocut.compose")

# Step 1 renders a short clip purely to eyeball the geometry (spec section 9).
STEP1_SECONDS = 10.0
COMPOSITE_CRF = 20


def _ff_color(value: str) -> str:
    """Normalise a config colour (``#RRGGBB`` or ``0xRRGGBB``) to ffmpeg's form."""
    v = str(value).strip()
    return "0x" + v[1:] if v.startswith("#") else v


def _speaker_source(ep: Episode) -> Path:
    """Per spec the speaker comes from cut.mkv; fall back to the mezzanine so the
    geometry can be checked before a cut exists."""
    if ep.cut.exists():
        return ep.cut
    if ep.mezz.exists():
        return ep.mezz
    raise FileNotFoundError(
        f"No speaker source: neither {ep.cut} nor {ep.mezz} exists. "
        f"Run 'autocut probe {ep.episode_id}' (and ideally 'cut') first."
    )


def _build_speaker_graph(layout: dict) -> str:
    cx, cy, cw, ch = layout_mod.source_crop(layout)
    _, _, rw, rh = layout_mod.rect(layout, "speaker")
    key = layout["speaker"]["key"]
    color = key.get("color", "0x00b140")
    sim = key.get("similarity", 0.12)
    blend = key.get("blend", 0.05)
    # Crop before keying so we key the strip we keep, not 3x the pixels (spec 3).
    chain = [f"crop={cw}:{ch}:{cx}:{cy}", f"chromakey={color}:{sim}:{blend}"]
    if key.get("despill", True):
        chain.append("despill=type=green")
    chain.append(f"scale={rw}:{rh}:flags=lanczos")
    return f"[0:v]{','.join(chain)}[spk]\n"


def _build_composite_graph(layout: dict) -> str:
    cw, ch = layout_mod.canvas_size(layout)
    cxx, cxy, cwd, chd = layout_mod.rect(layout, "content")
    sx, sy, _, _ = layout_mod.rect(layout, "speaker")
    fill = _ff_color(layout["content"].get("background", "#000000"))
    fit = layout["content"].get("fit", "contain")
    if fit == "cover":
        content = (f"[1:v]scale={cwd}:{chd}:force_original_aspect_ratio=increase,"
                   f"crop={cwd}:{chd}[content]")
    else:  # contain — letterbox, never crop the subject (spec section 4)
        content = (f"[1:v]scale={cwd}:{chd}:force_original_aspect_ratio=decrease,"
                   f"pad={cwd}:{chd}:(ow-iw)/2:(oh-ih)/2:color={fill}[content]")
    # Bottom-to-top: background, content, speaker (spec section 5).
    lines = [
        f"[0:v]scale={cw}:{ch}[bg]",
        content,
        f"[bg][content]overlay=x={cxx}:y={cxy}[tmp]",
        f"[tmp][2:v]overlay=x={sx}:y={sy}[out]",
    ]
    return ";\n".join(lines) + "\n"


def _render_speaker(ep: Episode, layout: dict, fps: str, *, force: bool) -> None:
    stage_dir = ep.compose_dir / "speaker"
    src = _speaker_source(ep)
    graph = _build_speaker_graph(layout)
    input_hash = cache.hash_inputs({
        "source": cache.hash_file(src) if (src.exists() and not ffmpeg.is_dry_run()) else "dry",
        "graph": graph,
        "fps": fps,
        "seconds": STEP1_SECONDS,
    })
    if not force and cache.is_current(stage_dir, input_hash) and ep.speaker_layer.exists():
        log.info("compose: speaker layer cache hit")
        return

    ep.compose_dir.mkdir(parents=True, exist_ok=True)
    ep.speaker_filter_script.write_text(graph, encoding="utf-8")
    log.info("compose: keying speaker layer from %s -> %s", src.name, ep.speaker_layer)
    ffmpeg.run_ffmpeg([
        "-i", src,
        "-/filter_complex", ep.speaker_filter_script,
        "-map", "[spk]",
        "-t", f"{STEP1_SECONDS}",
        "-r", fps,
        # ProRes 4444 keeps the keyed alpha and encodes far faster than VP9 at 4K.
        "-c:v", "prores_ks", "-profile:v", "4444", "-pix_fmt", "yuva444p10le",
        ep.speaker_layer,
    ])
    cache.mark_done(stage_dir, input_hash, extra={"stage": "compose:speaker"})


def _render_composite(ep: Episode, layout: dict, fps: str, *, force: bool) -> None:
    stage_dir = ep.compose_dir / "composite"
    bg = layout_mod.asset(layout, ep.root, "background", "image")
    content = layout_mod.asset(layout, ep.root, "content", "placeholder")
    if not ffmpeg.is_dry_run():
        for path, what in ((bg, "background.image"), (content, "content.placeholder")):
            if not path.exists():
                raise FileNotFoundError(f"layout.yaml {what} not found: {path}")

    graph = _build_composite_graph(layout)
    input_hash = cache.hash_inputs({
        "speaker": cache.hash_file(ep.speaker_layer) if (ep.speaker_layer.exists() and not ffmpeg.is_dry_run()) else "dry",
        "bg": cache.hash_file(bg) if (bg.exists() and not ffmpeg.is_dry_run()) else "dry",
        "content": cache.hash_file(content) if (content.exists() and not ffmpeg.is_dry_run()) else "dry",
        "graph": graph,
        "fps": fps,
        "seconds": STEP1_SECONDS,
        "crf": COMPOSITE_CRF,
    })
    if not force and cache.is_current(stage_dir, input_hash) and ep.compose_preview.exists():
        log.info("compose: composite cache hit")
        return

    ep.compose_dir.mkdir(parents=True, exist_ok=True)
    ep.compose_filter_script.write_text(graph, encoding="utf-8")
    log.info("compose: compositing background + content + speaker -> %s", ep.compose_preview)
    ffmpeg.run_ffmpeg([
        "-loop", "1", "-i", bg,
        "-loop", "1", "-i", content,
        "-i", ep.speaker_layer,
        "-/filter_complex", ep.compose_filter_script,
        "-map", "[out]",
        "-t", f"{STEP1_SECONDS}",
        "-r", fps,
        "-c:v", "libx264", "-preset", "veryfast", "-crf", COMPOSITE_CRF, "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        ep.compose_preview,
    ])
    cache.mark_done(stage_dir, input_hash, extra={"stage": "compose:composite"})


def run(ep: Episode, *, force: bool = False) -> None:
    """Execute the Step-1 compositor: keyed speaker layer + static composite."""
    layout = layout_mod.load(ep.root)
    fps = layout_mod.resolve_fps(layout, ep)
    cw, ch = layout_mod.canvas_size(layout)
    log.info("compose: canvas %dx%d @ %s fps (%.0fs static composite)", cw, ch, fps, STEP1_SECONDS)
    _render_speaker(ep, layout, fps, force=force)
    _render_composite(ep, layout, fps, force=force)
    log.info("compose: done -> %s", ep.compose_preview)
