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

def test_speaker_graph_crops_then_keys_then_scales(tmp_path):
    lay = layout_mod.load(_root(tmp_path)[0])
    g = compose._build_speaker_graph(lay)
    # crop uses the explicit source_crop; order is crop -> chromakey -> ... -> scale.
    assert "crop=1389:2160:636:0" in g
    assert g.index("crop=") < g.index("chromakey=") < g.index("despill=") < g.index("scale=1080:1680")
    assert "chromakey=0x00b140:0.12:0.05" in g
    assert g.strip().endswith("[spk]")


def test_composite_graph_overlays_at_rect_origins(tmp_path):
    lay = layout_mod.load(_root(tmp_path)[0])
    g = compose._build_composite_graph(lay)
    assert "scale=3840:2160[bg]" in g
    assert "overlay=x=160:y=240[tmp]" in g      # content rect origin
    assert "overlay=x=2640:y=240[out]" in g     # speaker rect origin
    assert "color=0x0b0b0d" in g                 # letterbox fill normalised from #


def test_ff_color_normalises_hash():
    assert compose._ff_color("#0b0b0d") == "0x0b0b0d"
    assert compose._ff_color("0x00b140") == "0x00b140"
