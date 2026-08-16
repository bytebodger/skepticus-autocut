"""Phase-2 compositor — steps 1-2: layout config + static composite.

Renders a short static composite — background + one placeholder content image +
speaker window — so the geometry in ``config/layout.yaml`` can be checked by eye
(compositor spec sections 3 and 5). No content timing, captions, or audio yet.

Speaker layer (spec section 3), its own cached step so it can be tuned in
isolation:
  * crop -> scale, and (only when ``speaker.key.enabled``) chromakey + despill.
    The default rig records against a black backdrop, so keying is off but the
    code path is kept behind the flag.
  * a rounded-corner window via a pre-rendered mask + alphamerge (faster than a
    per-frame geq), plus an optional drop shadow and border so the window reads
    as intentional framing.

Every geometry value comes from config/layout.yaml; fps comes from probe.json.
Filter graphs are written to files and passed with ``-/filter_complex`` (ffmpeg
9.0's replacement for the removed ``-filter_complex_script``) so a large graph
can never hit Windows' 8191-char command-line limit.
"""

from __future__ import annotations

import logging
from pathlib import Path

from . import cache, content as content_mod, ffmpeg, layout as layout_mod
from .paths import Episode

log = logging.getLogger("autocut.compose")

# Step 1/2 render a short clip purely to eyeball the geometry (spec section 9).
STEP1_SECONDS = 10.0
COMPOSITE_CRF = 20


def _ff_color(value: str) -> str:
    """Normalise a config colour (``#RRGGBB`` or ``0xRRGGBB``) to ffmpeg's form."""
    v = str(value).strip()
    return "0x" + v[1:] if v.startswith("#") else v


def _color_rgb(value: str) -> tuple[int, int, int]:
    """Parse ``#RRGGBB`` / ``0xRRGGBB`` into (r, g, b) integers for geq."""
    v = str(value).strip().lstrip("#")
    if v.lower().startswith("0x"):
        v = v[2:]
    return int(v[0:2], 16), int(v[2:4], 16), int(v[4:6], 16)


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


# --------------------------------------------------------------------------- #
# Rounded-rect chrome, pre-rendered once with geq (commas escaped for the
# filtergraph parser). The distance field saturates within the corner radius, so
# these are only correct for radius > 0 and border width < radius.
# --------------------------------------------------------------------------- #

def _dist_expr(r: int) -> str:
    return (f"hypot(max(abs(X-W/2)-(W/2-{r})\\,0)\\,"
            f"max(abs(Y-H/2)-(H/2-{r})\\,0))")


def _generate_mask(ep: Episode, w: int, h: int, r: int) -> Path:
    """White rounded-rect on black — the speaker window's alpha mask."""
    out = ep.compose_dir / "speaker_mask.png"
    dist = _dist_expr(r)
    ffmpeg.run_ffmpeg([
        "-f", "lavfi", "-i", f"color=c=black:s={w}x{h}",
        "-frames:v", "1",
        "-vf", f"format=gray,geq=lum='255*clip({r}-{dist}+0.5\\,0\\,1)'",
        out,
    ])
    return out


def _generate_border(ep: Episode, w: int, h: int, r: int, bw: int, rgb: tuple[int, int, int]) -> Path:
    """RGBA rounded-rect stroke, ``bw`` px inside the window edge."""
    out = ep.compose_dir / "speaker_border.png"
    red, green, blue = rgb
    dist = _dist_expr(r)
    alpha = (f"255*clip({r}-{dist}+0.5\\,0\\,1)"
             f"*clip({dist}-{r}+{bw}+0.5\\,0\\,1)")
    ffmpeg.run_ffmpeg([
        "-f", "lavfi", "-i", f"color=c=black:s={w}x{h}",
        "-frames:v", "1",
        "-vf", f"format=rgba,geq=r={red}:g={green}:b={blue}:a='{alpha}'",
        out,
    ])
    return out


# --------------------------------------------------------------------------- #
# Filter graphs
# --------------------------------------------------------------------------- #

def _build_speaker_graph(layout: dict, *, use_mask: bool) -> str:
    cx, cy, cw, ch = layout_mod.source_crop(layout)
    _, _, rw, rh = layout_mod.rect(layout, "speaker")
    key = layout["speaker"]["key"]
    scale = f"scale={rw}:{rh}:flags=lanczos"

    if key.get("enabled", False):
        # Crop before keying so we key the strip we keep, not 3x the pixels.
        chain = [f"crop={cw}:{ch}:{cx}:{cy}",
                 f"chromakey={key.get('color', '0x00b140')}:{key.get('similarity', 0.12)}:{key.get('blend', 0.05)}"]
        if key.get("despill", True):
            chain.append("despill=type=green")
        chain.append(scale)
        base = f"[0:v]{','.join(chain)}"
        if not use_mask:
            return f"{base}[spk]\n"
        # Multiply the keyed alpha by the rounded-corner mask so both apply.
        return (f"{base}[keyed];\n"
                f"[keyed]alphaextract[ka];\n"
                f"[ka][1:v]blend=all_mode=multiply[cmb];\n"
                f"[keyed][cmb]alphamerge[spk]\n")

    # No key — black backdrop. crop -> scale, alpha comes only from the mask.
    base = f"[0:v]crop={cw}:{ch}:{cx}:{cy},{scale}"
    if not use_mask:
        return f"{base}[spk]\n"
    return f"{base}[scaled];\n[scaled][1:v]alphamerge[spk]\n"


def _build_composite_graph(layout: dict, *, shadow: dict | None, border_idx: int | None,
                           content_prefit: bool) -> str:
    cw, ch = layout_mod.canvas_size(layout)
    cxx, cxy, cwd, chd = layout_mod.rect(layout, "content")
    sx, sy, rw, rh = layout_mod.rect(layout, "speaker")
    fill = _ff_color(layout["content"].get("background", "#000000"))
    fit = layout["content"].get("fit", "contain")
    if content_prefit:
        # The content track is already content-rect-sized and fitted per item.
        content = "[1:v]null[content]"
    elif fit == "cover":
        content = (f"[1:v]scale={cwd}:{chd}:force_original_aspect_ratio=increase,"
                   f"crop={cwd}:{chd}[content]")
    else:  # contain — letterbox, never crop the subject (spec section 4)
        content = (f"[1:v]scale={cwd}:{chd}:force_original_aspect_ratio=decrease,"
                   f"pad={cwd}:{chd}:(ow-iw)/2:(oh-ih)/2:color={fill}[content]")

    # Bottom-to-top: background, content, [shadow], speaker, [border] (spec 5).
    lines = [
        f"[0:v]scale={cw}:{ch}[bg]",
        content,
        f"[bg][content]overlay=x={cxx}:y={cxy}[base]",
    ]
    cur = "base"
    if shadow is not None:
        blur = shadow["blur"]
        pad = max(1, int(round(3 * blur)))       # room for the blur to bleed out
        pw, ph = rw + 2 * pad, rh + 2 * pad
        dx, dy = shadow["offset"]
        # Silhouette from the speaker's own alpha -> pad -> blur -> tint black at
        # the configured opacity, dropped behind the speaker at an offset.
        lines += [
            f"[2:v]alphaextract,pad={pw}:{ph}:{pad}:{pad},gblur=sigma={blur}[sha]",
            f"color=c=black:s={pw}x{ph}[shbk]",
            f"[shbk][sha]alphamerge,colorchannelmixer=aa={shadow['opacity']}[shadow]",
            f"[{cur}][shadow]overlay=x={sx - pad + dx}:y={sy - pad + dy}[cshadow]",
        ]
        cur = "cshadow"
    lines.append(f"[{cur}][2:v]overlay=x={sx}:y={sy}[cspk]")
    cur = "cspk"
    if border_idx is not None:
        lines.append(f"[{cur}][{border_idx}:v]overlay=x={sx}:y={sy}[out]")
    else:
        lines.append(f"[{cur}]null[out]")
    return ";\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# Stages
# --------------------------------------------------------------------------- #

def _render_speaker(ep: Episode, layout: dict, fps: str, *, force: bool) -> None:
    stage_dir = ep.compose_dir / "speaker"
    src = _speaker_source(ep)
    _, _, rw, rh = layout_mod.rect(layout, "speaker")
    r = int(layout["speaker"].get("corner_radius", 0))
    key = layout["speaker"]["key"]
    use_mask = r > 0
    graph = _build_speaker_graph(layout, use_mask=use_mask)
    input_hash = cache.hash_inputs({
        "source": cache.hash_file(src) if (src.exists() and not ffmpeg.is_dry_run()) else "dry",
        "graph": graph,
        "fps": fps,
        "seconds": STEP1_SECONDS,
        "rect": [rw, rh],
        "corner_radius": r,
        "key": {k: key.get(k) for k in ("enabled", "color", "similarity", "blend", "despill")},
    })
    if not force and cache.is_current(stage_dir, input_hash) and ep.speaker_layer.exists():
        log.info("compose: speaker layer cache hit")
        return

    ep.compose_dir.mkdir(parents=True, exist_ok=True)
    inputs = ["-i", src]
    if use_mask:
        mask = _generate_mask(ep, rw, rh, r)
        inputs += ["-loop", "1", "-i", mask]     # single image, repeated per frame
    ep.speaker_filter_script.write_text(graph, encoding="utf-8")
    keyed = "keyed" if key.get("enabled", False) else "black backdrop, no key"
    log.info("compose: speaker layer (%s, r=%d) from %s -> %s", keyed, r, src.name, ep.speaker_layer)
    ffmpeg.run_ffmpeg([
        *inputs,
        "-/filter_complex", ep.speaker_filter_script,
        "-map", "[spk]",
        "-t", f"{STEP1_SECONDS}",
        "-r", fps,
        # ProRes 4444 keeps the window's alpha and encodes far faster than VP9 at 4K.
        "-c:v", "prores_ks", "-profile:v", "4444", "-pix_fmt", "yuva444p10le",
        ep.speaker_layer,
    ])
    cache.mark_done(stage_dir, input_hash, extra={"stage": "compose:speaker"})


def _shadow_config(layout: dict) -> dict | None:
    cfg = layout["speaker"].get("shadow") or {}
    if not cfg.get("enabled"):
        return None
    dx, dy = (list(cfg.get("offset") or [0, 0]) + [0, 0])[:2]
    return {
        "opacity": float(cfg.get("opacity", 0.5)),
        "blur": float(cfg.get("blur", 16)),
        "offset": (int(dx), int(dy)),
    }


def _render_composite(ep: Episode, layout: dict, fps: str, *, force: bool) -> None:
    stage_dir = ep.compose_dir / "composite"
    bg = layout_mod.asset(layout, ep.root, "background", "image")

    # Content layer: the timed content track when content.json exists, else the
    # static step-1 placeholder image.
    track = content_mod.render_track(ep, layout, fps, force=force)
    content_prefit = track is not None
    content = track if content_prefit else layout_mod.asset(layout, ep.root, "content", "placeholder")
    if not ffmpeg.is_dry_run():
        for path, what in ((bg, "background.image"), (content, "content track/placeholder")):
            if not path.exists():
                raise FileNotFoundError(f"{what} not found: {path}")

    sx, sy, rw, rh = layout_mod.rect(layout, "speaker")
    r = int(layout["speaker"].get("corner_radius", 0))
    border_cfg = layout["speaker"].get("border") or {}
    bw = int(border_cfg.get("width", 0))
    shadow = _shadow_config(layout)

    # The content track is a finite video; the placeholder is a looped still.
    content_in = ["-i", content] if content_prefit else ["-loop", "1", "-i", content]
    inputs = ["-loop", "1", "-i", bg, *content_in, "-i", ep.speaker_layer]
    border_idx = None
    if bw > 0:
        if r <= 0:
            log.warning("compose: border needs corner_radius > 0; skipping border")
        else:
            border_png = _generate_border(ep, rw, rh, r, bw, _color_rgb(border_cfg.get("color", "#ffffff")))
            border_idx = 3
            inputs += ["-loop", "1", "-i", border_png]

    graph = _build_composite_graph(layout, shadow=shadow, border_idx=border_idx, content_prefit=content_prefit)
    input_hash = cache.hash_inputs({
        "speaker": cache.hash_file(ep.speaker_layer) if (ep.speaker_layer.exists() and not ffmpeg.is_dry_run()) else "dry",
        "bg": cache.hash_file(bg) if (bg.exists() and not ffmpeg.is_dry_run()) else "dry",
        "content": cache.hash_file(content) if (content.exists() and not ffmpeg.is_dry_run()) else "dry",
        "content_prefit": content_prefit,
        "graph": graph,
        "border": {"width": bw, "color": border_cfg.get("color")} if border_idx is not None else None,
        "shadow": shadow,
        "fps": fps,
        "seconds": STEP1_SECONDS,
        "crf": COMPOSITE_CRF,
    })
    if not force and cache.is_current(stage_dir, input_hash) and ep.compose_preview.exists():
        log.info("compose: composite cache hit")
        return

    ep.compose_dir.mkdir(parents=True, exist_ok=True)
    ep.compose_filter_script.write_text(graph, encoding="utf-8")
    log.info("compose: compositing (shadow=%s border=%s) -> %s",
             bool(shadow), border_idx is not None, ep.compose_preview)
    ffmpeg.run_ffmpeg([
        *inputs,
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
    """Execute the compositor: speaker window layer + static composite."""
    layout = layout_mod.load(ep.root)
    fps = layout_mod.resolve_fps(layout, ep)
    cw, ch = layout_mod.canvas_size(layout)
    log.info("compose: canvas %dx%d @ %s fps (%.0fs static composite)", cw, ch, fps, STEP1_SECONDS)
    _render_speaker(ep, layout, fps, force=force)
    _render_composite(ep, layout, fps, force=force)
    log.info("compose: done -> %s", ep.compose_preview)
