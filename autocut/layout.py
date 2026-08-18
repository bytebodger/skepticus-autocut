"""Layout config for the Phase-2 compositor (compositor spec section 2).

All geometry is declarative in ``config/layout.yaml`` — nothing is hardcoded.
This module loads and validates that file and resolves the few values that are
not plain literals: ``canvas.fps: from_source`` (read from probe.json) and
``speaker.source_crop: auto`` (a centred vertical strip at the target aspect).
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from .paths import Episode

CONFIG_REL = "config/layout.yaml"


class LayoutError(ValueError):
    """Raised for a missing or malformed layout config."""


def load(root: Path) -> dict:
    path = root / CONFIG_REL
    if not path.exists():
        raise LayoutError(f"No layout config at {path}. See compositor spec section 2.")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    _validate(data)
    return data


def _require(data: dict, *keys: str):
    cur = data
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            raise LayoutError(f"layout.yaml is missing '{'.'.join(keys)}'")
        cur = cur[k]
    return cur


def _rect(value, where: str) -> tuple[int, int, int, int]:
    if not (isinstance(value, (list, tuple)) and len(value) == 4):
        raise LayoutError(f"{where} must be [x, y, w, h]; got {value!r}")
    try:
        x, y, w, h = (int(v) for v in value)
    except (TypeError, ValueError):
        raise LayoutError(f"{where} must be four integers; got {value!r}")
    if w <= 0 or h <= 0:
        raise LayoutError(f"{where} width and height must be positive; got {value!r}")
    return x, y, w, h


def _validate(data: dict) -> None:
    _require(data, "canvas", "width")
    _require(data, "canvas", "height")
    _require(data, "background", "image")
    _rect(_require(data, "speaker", "rect"), "speaker.rect")
    _require(data, "speaker", "key", "color")
    _rect(_require(data, "content", "rect"), "content.rect")
    sc = data["speaker"].get("source_crop", "auto")
    if sc != "auto":
        _rect(sc, "speaker.source_crop")


def canvas_size(layout: dict) -> tuple[int, int]:
    return int(layout["canvas"]["width"]), int(layout["canvas"]["height"])


def _horizontal_origins(layout: dict) -> dict[str, int] | None:
    """Left-edge x for the content and speaker windows, derived from
    ``speaker.side`` — or ``None`` when both rects aren't present to derive from.

    The two windows tile the canvas horizontally as
    ``[margin | left window | gutter | right window | margin]`` sharing one
    vertical band. Each window's width comes from its own rect; the outer margin
    and the gutter are read from the authored rects (they are side-invariant — the
    left margin is just the smaller of the two authored x's, and the gutter is the
    space the author left between the windows). ``speaker.side`` alone decides
    which window sits on the left, so flipping it swaps the two windows with no
    overlap and no gap — the spacing and the widths are unchanged, only the order.
    """
    speaker, content = layout.get("speaker") or {}, layout.get("content") or {}
    if "rect" not in speaker or "rect" not in content:
        return None  # a partial layout (e.g. content-only) can't derive placement
    sx, _, sw, _ = _rect(speaker["rect"], "speaker.rect")
    cx, _, cw, _ = _rect(content["rect"], "content.rect")
    left_margin = min(sx, cx)
    gutter = (cx - (sx + sw)) if sx < cx else (sx - (cx + cw))
    side = speaker.get("side", "right")
    if side == "left":  # speaker on the left, content on the right
        return {"speaker": left_margin, "content": left_margin + sw + gutter}
    return {"content": left_margin, "speaker": left_margin + cw + gutter}


def rect(layout: dict, section: str) -> tuple[int, int, int, int]:
    """The (x, y, w, h) rect for 'speaker' or 'content', in canvas space.

    Size (w, h) and the vertical band (y) come straight from the section's rect;
    the horizontal origin (x) is derived from ``speaker.side`` so a single config
    flip moves both windows together (see ``_horizontal_origins``). If the layout
    is partial (only one of the two rects present, as when the content track is
    built in isolation), the authored x is used as-is."""
    x, y, w, h = _rect(layout[section]["rect"], f"{section}.rect")
    origins = _horizontal_origins(layout)
    if origins is not None:
        x = origins[section]
    return x, y, w, h


def source_crop(layout: dict) -> tuple[int, int, int, int]:
    """The crop taken from the source frame before keying.

    Explicit ``source_crop: [x, y, w, h]`` is used as-is (measure it once from a
    real frame — see spec section 3). ``auto`` centres a full-height vertical
    strip at the speaker rect's aspect ratio.
    """
    sc = layout["speaker"].get("source_crop", "auto")
    if sc != "auto":
        return _rect(sc, "speaker.source_crop")
    cw, ch = canvas_size(layout)
    _, _, rw, rh = rect(layout, "speaker")
    crop_w = round(ch * (rw / rh))
    crop_x = (cw - crop_w) // 2
    return crop_x, 0, crop_w, ch


def resolve_fps(layout: dict, ep: Episode) -> str:
    """The frame rate for the composite, as an exact rational string for ffmpeg
    ``-r``. ``canvas.fps: from_source`` reads it from probe.json — never assumed.
    """
    fps = layout["canvas"].get("fps", "from_source")
    if fps != "from_source":
        return str(fps)
    if not ep.probe_json.exists():
        raise LayoutError(
            f"canvas.fps is 'from_source' but no probe.json at {ep.probe_json}; "
            f"run 'autocut probe {ep.episode_id}' first."
        )
    probe = json.loads(ep.probe_json.read_text(encoding="utf-8"))
    fps_val = probe.get("fps_rational") or probe.get("fps")
    if not fps_val:
        raise LayoutError(f"probe.json for {ep.episode_id} has no fps/fps_rational")
    return str(fps_val)


def asset(layout: dict, root: Path, *keys: str) -> Path:
    """Resolve a config-declared asset path (e.g. background.image) against the
    repo root."""
    cur = layout
    for k in keys:
        cur = cur[k]
    return root / str(cur)
