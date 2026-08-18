"""Layout config parsing + compositor filter-graph construction (pure logic).

No ffmpeg here — the actual render is verified end-to-end separately."""

import json

import pytest

from autocut import compose, layout as layout_mod
from autocut.paths import resolve

_YAML = """
canvas: {width: 3840, height: 2160, fps: from_source}
background: {image: assets/bg.png}
speaker:
  side: right
  rect: [2640, 240, 1080, 1680]
  source_crop: [636, 0, 1389, 2160]
  key: {color: "0x00b140", similarity: 0.12, blend: 0.05, despill: true}
content:
  rect: [160, 240, 2400, 1680]
  fit: contain
  background: "#0b0b0d"
  placeholder: assets/content.png
"""


def _root(tmp_path, yaml_text=_YAML, fps_rational="24000/1001"):
    (tmp_path / "config").mkdir(parents=True, exist_ok=True)
    (tmp_path / "config" / "layout.yaml").write_text(yaml_text, encoding="utf-8")
    ep = resolve("ep001", root=tmp_path)
    ep.work.mkdir(parents=True, exist_ok=True)
    if fps_rational is not None:
        ep.probe_json.write_text(json.dumps({"fps": 23.976, "fps_rational": fps_rational}),
                                 encoding="utf-8")
    return tmp_path, ep


def test_load_and_accessors(tmp_path):
    root, _ = _root(tmp_path)
    lay = layout_mod.load(root)
    assert layout_mod.canvas_size(lay) == (3840, 2160)
    assert layout_mod.rect(lay, "speaker") == (2640, 240, 1080, 1680)
    assert layout_mod.rect(lay, "content") == (160, 240, 2400, 1680)


def test_side_left_swaps_both_windows(tmp_path):
    # Same authored geometry, speaker.side flipped to left: the speaker takes the
    # left slot, content the right, derived from the one knob.
    yaml_text = _YAML.replace("side: right", "side: left")
    lay = layout_mod.load(_root(tmp_path, yaml_text)[0])
    sx, sy, sw, sh = layout_mod.rect(lay, "speaker")
    cx, cy, cw, ch = layout_mod.rect(lay, "content")
    # widths / vertical band are untouched; only the x origins move
    assert (sw, sh, sy) == (1080, 1680, 240)
    assert (cw, ch, cy) == (2400, 1680, 240)
    # speaker now on the left at the outer margin; content to its right
    assert sx == 160
    assert cx == 160 + sw + 80          # left_margin + speaker_w + gutter
    # no overlap and no gap: speaker's right edge + gutter == content's left edge
    assert cx - (sx + sw) == 80


def test_side_flip_preserves_spacing_and_fills_canvas(tmp_path):
    # Both orientations keep the 80px gutter and stay within the 3840 canvas with
    # the two windows abutting the same gutter — proof the flip introduces neither
    # overlap nor gap.
    for side in ("right", "left"):
        yaml_text = _YAML.replace("side: right", f"side: {side}")
        lay = layout_mod.load(_root(tmp_path, yaml_text)[0])
        sx, _, sw, _ = layout_mod.rect(lay, "speaker")
        cx, _, cw, _ = layout_mod.rect(lay, "content")
        left, right = sorted([(sx, sw), (cx, cw)])
        assert left[0] == 160                       # left margin unchanged
        assert right[0] - (left[0] + left[1]) == 80  # gutter unchanged
        assert right[0] + right[1] == 3720           # right edge unchanged
        assert (sx < cx) == (side == "left")         # speaker leads only when left


def test_explicit_source_crop_used_verbatim(tmp_path):
    root, _ = _root(tmp_path)
    lay = layout_mod.load(root)
    assert layout_mod.source_crop(lay) == (636, 0, 1389, 2160)


def test_auto_source_crop_centres_strip(tmp_path):
    yaml_text = _YAML.replace("source_crop: [636, 0, 1389, 2160]", "source_crop: auto")
    root, _ = _root(tmp_path, yaml_text)
    lay = layout_mod.load(root)
    x, y, w, h = layout_mod.source_crop(lay)
    assert (y, h) == (0, 2160)
    assert w == round(2160 * (1080 / 1680))      # target aspect strip
    assert x == (3840 - w) // 2                    # centred


def test_fps_from_probe(tmp_path):
    root, ep = _root(tmp_path)
    assert layout_mod.resolve_fps(layout_mod.load(root), ep) == "24000/1001"


def test_fps_literal_override(tmp_path):
    yaml_text = _YAML.replace("fps: from_source", "fps: 30")
    root, ep = _root(tmp_path, yaml_text, fps_rational=None)
    assert layout_mod.resolve_fps(layout_mod.load(root), ep) == "30"


def test_fps_from_source_without_probe_errors(tmp_path):
    root, ep = _root(tmp_path, fps_rational=None)  # no probe.json written
    with pytest.raises(layout_mod.LayoutError, match="probe.json"):
        layout_mod.resolve_fps(layout_mod.load(root), ep)


def test_missing_config_errors(tmp_path):
    with pytest.raises(layout_mod.LayoutError, match="No layout config"):
        layout_mod.load(tmp_path)


def test_bad_rect_errors(tmp_path):
    yaml_text = _YAML.replace("rect: [2640, 240, 1080, 1680]", "rect: [2640, 240, 1080]")
    root, _ = _root(tmp_path, yaml_text)
    with pytest.raises(layout_mod.LayoutError, match="speaker.rect"):
        layout_mod.load(root)


# --- filter graph construction ---

def test_speaker_graph_no_key_crops_scales_alphamerges(tmp_path):
    # Default rig: key disabled (black backdrop) -> crop -> scale -> mask.
    lay = layout_mod.load(_root(tmp_path)[0])
    g = compose._build_speaker_graph(lay, use_mask=True)
    assert "crop=1389:2160:636:0" in g          # explicit source_crop
    assert "chromakey" not in g and "despill" not in g
    assert g.index("crop=") < g.index("scale=1080:1680")
    assert "alphamerge[spk]" in g               # rounded-corner mask applied


def test_speaker_graph_no_key_no_mask(tmp_path):
    lay = layout_mod.load(_root(tmp_path)[0])
    g = compose._build_speaker_graph(lay, use_mask=False)
    assert "alphamerge" not in g
    assert g.strip().endswith("[spk]")


def test_speaker_graph_keyed_when_enabled(tmp_path):
    # Keeping the key path behind the flag: enabling it restores crop->key->scale.
    yaml_text = _YAML.replace("key: {color:", "key: {enabled: true, color:")
    lay = layout_mod.load(_root(tmp_path, yaml_text)[0])
    g = compose._build_speaker_graph(lay, use_mask=True)
    assert g.index("crop=") < g.index("chromakey=0x00b140:0.12:0.05") < g.index("scale=1080:1680")
    assert "despill=type=green" in g
    assert "blend=all_mode=multiply" in g       # keyed alpha combined with mask


def test_composite_graph_overlays_at_rect_origins(tmp_path):
    lay = layout_mod.load(_root(tmp_path)[0])
    g = compose._build_composite_graph(lay, shadow=None, border_idx=None,
                                       content_prefit=False, subtitles=None, preview=False)
    assert "scale=3840:2160[bg]" in g
    assert "overlay=x=160:y=240[base]" in g      # content rect origin
    assert "overlay=x=2640:y=240[cspk]" in g     # speaker rect origin
    assert "color=0x0b0b0d" in g                  # letterbox fill normalised from #
    assert "subtitles=" not in g                  # no captions -> no burn
    assert "[vout]" in g and "scale=1920:1080" not in g   # full res


def test_composite_graph_prefit_content_passes_through(tmp_path):
    # A pre-fitted content track is overlaid as-is (no scale/pad in the composite).
    lay = layout_mod.load(_root(tmp_path)[0])
    g = compose._build_composite_graph(lay, shadow=None, border_idx=None,
                                       content_prefit=True, subtitles=None, preview=False)
    assert "[1:v]null[content]" in g
    assert "force_original_aspect_ratio" not in g


def test_composite_graph_shadow_border_and_captions(tmp_path):
    lay = layout_mod.load(_root(tmp_path)[0])
    shadow = {"opacity": 0.5, "blur": 18.0, "offset": (0, 10)}
    g = compose._build_composite_graph(lay, shadow=shadow, border_idx=3,
                                       content_prefit=False,
                                       subtitles=("work/ep/captions.ass", "styles"),
                                       preview=False)
    assert "gblur=sigma=18.0" in g                       # shadow blur
    assert "colorchannelmixer=aa=0.5" in g               # shadow opacity
    assert "[3:v]overlay=x=2640:y=240[framed]" in g      # border, then captions on top
    assert "subtitles=work/ep/captions.ass:fontsdir=styles[vid]" in g


def test_composite_graph_preview_downscales(tmp_path):
    lay = layout_mod.load(_root(tmp_path)[0])
    g = compose._build_composite_graph(lay, shadow=None, border_idx=None,
                                       content_prefit=False, subtitles=None, preview=True)
    assert "scale=1920:1080:flags=lanczos[vout]" in g


# --- window resolution ---

def test_resolve_window_full_preview_range(tmp_path, monkeypatch):
    root, ep = _root(tmp_path)
    monkeypatch.setattr(compose, "_output_duration", lambda e: 3000.0)
    assert compose._resolve_window(ep, preview=False, render_range=None) == (0.0, 3000.0)
    assert compose._resolve_window(ep, preview=True, render_range=None) == (0.0, compose.PREVIEW_SECONDS)
    assert compose._resolve_window(ep, preview=False, render_range="300:360") == (300.0, 60.0)
    # END past the duration clamps to it
    assert compose._resolve_window(ep, preview=False, render_range="2990:9999") == (2990.0, 10.0)


def test_resolve_range_rejects_bad_spec(tmp_path, monkeypatch):
    root, ep = _root(tmp_path)
    monkeypatch.setattr(compose, "_output_duration", lambda e: 3000.0)
    with pytest.raises(ValueError):
        compose._resolve_window(ep, preview=False, render_range="360:300")  # start >= end


def test_color_helpers():
    assert compose._ff_color("#0b0b0d") == "0x0b0b0d"
    assert compose._ff_color("0x00b140") == "0x00b140"
    assert compose._color_rgb("#f0f0f4") == (0xf0, 0xf0, 0xf4)
    assert compose._color_rgb("0xff8040") == (255, 128, 64)
