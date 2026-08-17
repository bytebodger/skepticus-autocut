"""Phase-2 compositor — background, speaker window, content track, captions, audio.

Composes a produced show frame: background wallpaper, a keyed/framed speaker
window, the timed content track, burned captions, and the processed audio chain
(compositor spec sections 3-7).

Rendered for an output ``window`` (start, length): the full output by default,
or a short ``--preview`` (1080p, fast preset) / an explicit ``--range`` window
for iteration without a multi-hour render. Each layer is its own cached step so
one can be tuned without re-rendering the others.

Every geometry value comes from config/layout.yaml; fps comes from probe.json.
Filter graphs are written to files and passed with ``-/filter_complex`` (ffmpeg
9.0's replacement for the removed ``-filter_complex_script``) so a large graph
can never hit Windows' 8191-char command-line limit.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from . import (
    audio as audio_mod,
    cache,
    captions as captions_mod,
    content as content_mod,
    edl,
    ffmpeg,
    layout as layout_mod,
)
from .paths import Episode

log = logging.getLogger("autocut.compose")

PREVIEW_SECONDS = 10.0        # --preview window length
PREVIEW_SCALE = (1920, 1080)  # --preview downscale for a fast iteration render
COMPOSITE_CRF = 20
PREVIEW_CRF = 23


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


def _build_captions(ep: Episode, layout: dict, window: tuple) -> str | None:
    """Generate the config-driven 4K caption ASS from words.json for the output
    window, or None when captions are disabled / there is no transcript. Burned
    at composite time so disabling captions skips the burn entirely rather than
    rendering an empty track (spec section 6)."""
    cfg = layout.get("captions") or {}
    if not cfg.get("enabled", True):
        return None
    if not ep.words_json.exists():
        log.warning("compose: captions enabled but no transcript at %s; skipping", ep.words_json)
        return None
    words_doc = json.loads(ep.words_json.read_text(encoding="utf-8"))
    edl_doc = edl.load(ep.edl_json)
    cw, ch = layout_mod.canvas_size(layout)
    header = captions_mod.style_header_from_config(cfg, cw, ch)
    return captions_mod.build_ass(words_doc, edl_doc, header, window=window)


def _build_composite_graph(layout: dict, *, shadow: dict | None, border_idx: int | None,
                           content_prefit: bool, subtitles: tuple[str, str] | None,
                           preview: bool) -> str:
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
        lines.append(f"[{cur}][{border_idx}:v]overlay=x={sx}:y={sy}[framed]")
        cur = "framed"
    # Burn captions last, over the finished frame (spec section 6).
    if subtitles is not None:
        ass_rel, fonts_rel = subtitles
        lines.append(f"[{cur}]subtitles={ass_rel}:fontsdir={fonts_rel}[vid]")
        cur = "vid"
    # Downscale for a fast preview render (spec section 8).
    if preview:
        pw, ph = PREVIEW_SCALE
        lines.append(f"[{cur}]scale={pw}:{ph}:flags=lanczos[vout]")
    else:
        lines.append(f"[{cur}]null[vout]")
    return ";\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# Stages
# --------------------------------------------------------------------------- #

def _render_speaker(ep: Episode, layout: dict, fps: str, window: tuple, *, force: bool) -> None:
    stage_dir = ep.compose_dir / "speaker"
    src = _speaker_source(ep)
    w0, length = window
    _, _, rw, rh = layout_mod.rect(layout, "speaker")
    r = int(layout["speaker"].get("corner_radius", 0))
    key = layout["speaker"]["key"]
    use_mask = r > 0
    graph = _build_speaker_graph(layout, use_mask=use_mask)
    input_hash = cache.hash_inputs({
        "source": cache.hash_file(src) if (src.exists() and not ffmpeg.is_dry_run()) else "dry",
        "graph": graph,
        "fps": fps,
        "window": [round(w0, 3), round(length, 3)],
        "rect": [rw, rh],
        "corner_radius": r,
        "key": {k: key.get(k) for k in ("enabled", "color", "similarity", "blend", "despill")},
    })
    if not force and cache.is_current(stage_dir, input_hash) and ep.speaker_layer.exists():
        log.info("compose: speaker layer cache hit")
        return

    ep.compose_dir.mkdir(parents=True, exist_ok=True)
    # Seek the (all-intra) source to the window start; the layer's clock is 0..length.
    inputs = ["-ss", f"{w0:.3f}", "-i", src]
    if use_mask:
        mask = _generate_mask(ep, rw, rh, r)
        inputs += ["-loop", "1", "-i", mask]     # single image, repeated per frame
    ep.speaker_filter_script.write_text(graph, encoding="utf-8")
    keyed = "keyed" if key.get("enabled", False) else "black backdrop, no key"
    log.info("compose: speaker layer (%s, r=%d) from %s [%.1fs..%.1fs] -> %s",
             keyed, r, src.name, w0, w0 + length, ep.speaker_layer)
    ffmpeg.run_ffmpeg([
        *inputs,
        "-/filter_complex", ep.speaker_filter_script,
        "-map", "[spk]",
        "-t", f"{length:.3f}",
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


def _render_composite(ep: Episode, layout: dict, fps: str, window: tuple, *,
                      preview: bool, force: bool) -> None:
    stage_dir = ep.compose_dir / "composite"
    w0, length = window
    dry = ffmpeg.is_dry_run()
    bg = layout_mod.asset(layout, ep.root, "background", "image")

    # Content layer: the timed content track when content.json exists, else the
    # static step-1 placeholder image.
    track = content_mod.render_track(ep, layout, fps, window, force=force)
    content_prefit = track is not None
    content = track if content_prefit else layout_mod.asset(layout, ep.root, "content", "placeholder")
    if not dry:
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
    next_idx = 3
    border_idx = None
    if bw > 0:
        if r <= 0:
            log.warning("compose: border needs corner_radius > 0; skipping border")
        else:
            border_png = _generate_border(ep, rw, rh, r, bw, _color_rgb(border_cfg.get("color", "#ffffff")))
            border_idx = next_idx
            next_idx += 1
            inputs += ["-loop", "1", "-i", border_png]

    # Audio: map the output-timeline speech (the speaker source's audio) through
    # the chain, with two-pass loudnorm. Previously the composite had no audio.
    audio_cfg = layout.get("audio") or {}
    audio_src = _speaker_source(ep)
    audio_af = None
    if audio_cfg.get("enabled", True):
        measured = None if dry else audio_mod.analyze(ep, str(audio_src), window, audio_cfg)
        audio_af = audio_mod.final_af(audio_cfg, measured)
        aud_idx = next_idx
        inputs += ["-ss", f"{w0:.3f}", "-t", f"{length:.3f}", "-i", str(audio_src)]

    # Captions: generate the ASS and burn it in-graph. Disabled -> no subtitles
    # filter at all (no empty track). The subtitles path must be relative and
    # forward-slashed, and ffmpeg runs from the repo root, because a Windows
    # drive-letter colon collides with the filter-option separator.
    ass_text = _build_captions(ep, layout, window)
    subtitles = None
    if ass_text is not None:
        ep.captions_ass.write_text(ass_text, encoding="utf-8")
        ass_rel = ep.captions_ass.resolve().relative_to(ep.root.resolve()).as_posix()
        fonts_rel = ep.styles_dir.resolve().relative_to(ep.root.resolve()).as_posix()
        subtitles = (ass_rel, fonts_rel)

    graph = _build_composite_graph(layout, shadow=shadow, border_idx=border_idx,
                                   content_prefit=content_prefit, subtitles=subtitles,
                                   preview=preview)
    if audio_af is not None:
        graph = graph.rstrip("\n") + f";\n[{aud_idx}:a]{audio_af}[aout]\n"

    input_hash = cache.hash_inputs({
        "speaker": cache.hash_file(ep.speaker_layer) if (ep.speaker_layer.exists() and not dry) else "dry",
        "bg": cache.hash_file(bg) if (bg.exists() and not dry) else "dry",
        "content": cache.hash_file(content) if (content.exists() and not dry) else "dry",
        "content_prefit": content_prefit,
        "graph": graph,
        "border": {"width": bw, "color": border_cfg.get("color")} if border_idx is not None else None,
        "shadow": shadow,
        "captions": ass_text,
        "audio": audio_af,
        "fps": fps,
        "window": [round(w0, 3), round(length, 3)],
        "preview": preview,
        "crf": PREVIEW_CRF if preview else COMPOSITE_CRF,
    })
    if not force and cache.is_current(stage_dir, input_hash) and ep.compose_output.exists():
        log.info("compose: composite cache hit")
        return

    ep.compose_dir.mkdir(parents=True, exist_ok=True)
    ep.compose_filter_script.write_text(graph, encoding="utf-8")
    log.info("compose: compositing (shadow=%s border=%s captions=%s audio=%s preview=%s) -> %s",
             bool(shadow), border_idx is not None, subtitles is not None,
             audio_af is not None, preview, ep.compose_output)
    maps = ["-map", "[vout]"] + (["-map", "[aout]"] if audio_af is not None else ["-an"])
    ffmpeg.run_ffmpeg(
        [
            *inputs,
            "-/filter_complex", ep.compose_filter_script,
            *maps,
            "-t", f"{length:.3f}",
            "-r", fps,
            "-c:v", "libx264", "-preset", "veryfast",
            "-crf", PREVIEW_CRF if preview else COMPOSITE_CRF, "-pix_fmt", "yuv420p",
            # loudnorm resamples to 192k internally; bring it back to 48k for delivery.
            "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
            "-movflags", "+faststart",
            ep.compose_output,
        ],
        cwd=str(ep.root),  # keep the subtitles path relative for Windows escaping
    )
    cache.mark_done(stage_dir, input_hash, extra={"stage": "compose:composite"})


def _output_duration(ep: Episode) -> float:
    """Total output duration in seconds, from the EDL time map."""
    edl_doc = edl.load(ep.edl_json)
    return edl.output_duration(edl.build_time_map(edl_doc["segments"]))


def _parse_range(spec: str, out_dur: float) -> tuple:
    """Parse ``START:END`` (output seconds) into a (start, length) window."""
    try:
        start_s, end_s = spec.split(":", 1)
        start = float(start_s)
        end = float(end_s) if end_s.strip() else out_dur
    except ValueError:
        raise ValueError(f"--range must be START:END in output seconds, got {spec!r}")
    end = min(end, out_dur)
    if not (0 <= start < end):
        raise ValueError(f"--range START must be >= 0 and < END (and <= duration {out_dur:.1f}s)")
    return start, end - start


def _resolve_window(ep: Episode, *, preview: bool, render_range: str | None) -> tuple:
    out_dur = _output_duration(ep)
    if render_range:
        return _parse_range(render_range, out_dur)
    if preview:
        return 0.0, min(PREVIEW_SECONDS, out_dur)
    return 0.0, out_dur


def run(ep: Episode, *, force: bool = False, preview: bool = False,
        render_range: str | None = None) -> None:
    """Execute the compositor. Full output by default; --preview renders a short
    1080p window, --range START:END renders an arbitrary output window."""
    layout = layout_mod.load(ep.root)
    fps = layout_mod.resolve_fps(layout, ep)
    window = _resolve_window(ep, preview=preview, render_range=render_range)
    cw, ch = layout_mod.canvas_size(layout)
    mode = "preview 1080p" if preview else ("range" if render_range else "full")
    log.info("compose: canvas %dx%d @ %s fps | %s window [%.1fs..%.1fs]",
             cw, ch, fps, mode, window[0], window[0] + window[1])
    _render_speaker(ep, layout, fps, window, force=force)
    _render_composite(ep, layout, fps, window, preview=preview, force=force)
    log.info("compose: done -> %s", ep.compose_output)
