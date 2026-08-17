"""Content track: source->output placement and filter-graph construction.

The source-timebase mapping is the load-bearing part (same rule as Phase-1
overlays); the actual render is verified end-to-end separately."""

import json

import pytest

from autocut import content
from autocut.paths import resolve


def _setup(tmp_path, items, segments):
    ep = resolve("ep001", root=tmp_path)
    ep.work.mkdir(parents=True, exist_ok=True)
    ep.edl_json.write_text(json.dumps({
        "version": 1, "fps": 24, "segments": segments, "overlays": [],
    }), encoding="utf-8")
    ep.content_dir.mkdir(parents=True, exist_ok=True)
    ep.content_json.write_text(json.dumps({"items": items}), encoding="utf-8")
    for it in items:
        (ep.content_dir / it["file"]).write_bytes(b"x")
    return ep


def _layout(gap="hold", fit="contain", ttype="crossfade"):
    return {"content": {"rect": [160, 240, 2400, 1680], "fit": fit,
                        "background": "#0b0b0d", "gap_behavior": gap,
                        "transition": {"type": ttype, "duration": 0.35}}}


# --- placement / mapping ---

_SEGS = [
    {"id": "s1", "in": 0.0, "out": 10.0, "action": "keep"},    # out [0,10)
    {"id": "s2", "in": 10.0, "out": 20.0, "action": "drop"},   # cut
    {"id": "s3", "in": 20.0, "out": 30.0, "action": "keep"},   # out [10,20)
]


def test_place_maps_sorts_and_drops_cut_items(tmp_path):
    items = [
        {"file": "a.png", "source_time": 25.0, "duration": 5},  # in s3 -> out 15
        {"file": "b.png", "source_time": 5.0, "duration": 5},   # in s1 -> out 5
        {"file": "c.png", "source_time": 15.0, "duration": 5},  # in the drop -> skipped
    ]
    ep = _setup(tmp_path, items, _SEGS)
    placed, out_dur = content._place(content.load_items(ep), ep)
    assert out_dur == 20.0
    assert [it["file"] for it, _ in placed] == ["b.png", "a.png"]  # sorted; c dropped
    assert {it["file"]: round(o, 3) for it, o in placed} == {"b.png": 5.0, "a.png": 15.0}


def test_no_content_json_is_none(tmp_path):
    ep = resolve("ep001", root=tmp_path)
    ep.work.mkdir(parents=True, exist_ok=True)
    assert content.load_items(ep) is None


# --- graph construction ---

def test_graph_hold_opaque_base_and_placement(tmp_path):
    ep = _setup(tmp_path, [{"file": "a.png"}, {"file": "b.mp4"}], _SEGS)
    placed = [({"file": "a.png", "duration": 6}, 5.0), ({"file": "b.mp4", "duration": 6}, 15.0)]
    inputs, graph, alpha = content.build_graph(ep, _layout("hold"), "24", (0.0, 40.0), placed)
    assert alpha is False
    assert "color=c=0x0b0b0d" in graph                         # hold base = letterbox fill
    assert "setpts=PTS-STARTPTS+5.000/TB" in graph             # item placed at out time
    assert "setpts=PTS-STARTPTS+15.000/TB" in graph
    assert "fade=t=in:st=0:d=0.35:alpha=1" in graph            # crossfade in
    assert graph.rstrip().endswith("[ctrack]")
    assert "-loop" in inputs                                    # image looped
    assert inputs.count("-i") == 2                              # one input per item


def test_graph_window_offsets_and_filters_items(tmp_path):
    # A window starting at 10s makes item output-times relative and drops items
    # outside it. Item at out 5.0 is excluded; item at 15.0 -> rel 5.0.
    ep = _setup(tmp_path, [{"file": "a.png"}, {"file": "b.mp4"}], _SEGS)
    placed = [({"file": "a.png", "duration": 6}, 5.0), ({"file": "b.mp4", "duration": 6}, 15.0)]
    inputs, graph, _ = content.build_graph(ep, _layout("hold"), "24", (10.0, 30.0), placed)
    assert "setpts=PTS-STARTPTS+5.000/TB" in graph   # 15.0 - 10.0
    assert "setpts=PTS-STARTPTS+0.000/TB" not in graph
    assert inputs.count("-i") == 1                    # only the in-window item


def test_graph_background_transparent_base_and_fade_out(tmp_path):
    ep = _setup(tmp_path, [{"file": "a.png"}], _SEGS)
    placed = [({"file": "a.png", "duration": 6}, 5.0)]
    _, graph, alpha = content.build_graph(ep, _layout("background"), "24", (0.0, 40.0), placed)
    assert alpha is True
    assert "colorchannelmixer=aa=0" in graph                   # transparent base
    assert "fade=t=out" in graph                               # items fade back to wallpaper


def test_graph_cut_transition_has_no_fades(tmp_path):
    ep = _setup(tmp_path, [{"file": "a.png"}], _SEGS)
    placed = [({"file": "a.png", "duration": 6}, 5.0)]
    _, graph, _ = content.build_graph(ep, _layout("hold", ttype="cut"), "24", (0.0, 40.0), placed)
    assert "fade=" not in graph


def test_fit_filter_contain_vs_cover():
    assert "pad=" in content._fit_filter(2400, 1680, "contain", "0x0b0b0d")
    assert "decrease" in content._fit_filter(2400, 1680, "contain", "0x0b0b0d")
    assert "crop=2400:1680" in content._fit_filter(2400, 1680, "cover", "0x0b0b0d")
    assert "increase" in content._fit_filter(2400, 1680, "cover", "0x0b0b0d")
