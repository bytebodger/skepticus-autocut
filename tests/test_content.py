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
    loaded, _base = content.load_items(ep)
    placed, out_dur = content._place(loaded, ep)
    assert out_dur == 20.0
    assert [it["file"] for it, _ in placed] == ["b.png", "a.png"]  # sorted; c dropped
    assert {it["file"]: round(o, 3) for it, o in placed} == {"b.png": 5.0, "a.png": 15.0}


def test_no_content_json_is_none(tmp_path):
    ep = resolve("ep001", root=tmp_path)
    ep.work.mkdir(parents=True, exist_ok=True)
    assert content.load_items(ep) == (None, None)


# --- reaction format: items synthesized from playback.json ---

def test_load_items_falls_back_to_reaction_playback(tmp_path):
    ep = resolve("ep001", root=tmp_path)
    ep.work.mkdir(parents=True, exist_ok=True)
    ep.playback_json.write_text(json.dumps({
        "version": 1, "episode_id": "ep001", "source_file": "/abs/source.mp4",
        "segments": [
            {"id": "pb001", "host_in": 10.0, "host_out": 25.0,
             "source_in": 0.0, "source_out": 15.0},
            {"id": "pb002", "host_in": 40.0, "host_out": 42.5,
             "source_in": 15.0, "source_out": 17.5},
        ],
    }), encoding="utf-8")
    items, _base = content.load_items(ep)
    assert items == [
        # pre-roll: frozen on the source's own frame 1 until playback starts
        # (spec §2: the recording opens in commentary — no prior item to hold).
        {"file": "/abs/source.mp4", "source_time": 0.0, "duration": content.PREROLL_DURATION, "clip_in": 0.0},
        {"file": "/abs/source.mp4", "source_time": 10.0, "duration": 15.0, "clip_in": 0.0},
        {"file": "/abs/source.mp4", "source_time": 40.0, "duration": 2.5, "clip_in": 15.0},
    ]


def test_reaction_preroll_anchors_to_the_edls_first_kept_instant(tmp_path):
    # True host time 0 is typically inside the leading-dead-air drop autoauthor
    # trims (spec §6) — anchoring the pre-roll there would place an item in a
    # cut region and silently lose it. It must anchor to wherever output
    # actually starts instead.
    ep = resolve("ep001", root=tmp_path)
    ep.work.mkdir(parents=True, exist_ok=True)
    ep.edl_json.write_text(json.dumps({
        "version": 1, "fps": 24,
        "segments": [
            {"id": "s1", "in": 0.0, "out": 2.083, "action": "drop", "reason": "dead_air"},
            {"id": "s2", "in": 2.083, "out": 100.0, "action": "keep"},
        ],
        "overlays": [],
    }), encoding="utf-8")
    ep.playback_json.write_text(json.dumps({
        "version": 1, "episode_id": "ep001", "source_file": "/abs/source.mp4",
        "segments": [{"id": "pb001", "host_in": 10.0, "host_out": 25.0,
                      "source_in": 0.0, "source_out": 15.0}],
    }), encoding="utf-8")
    items, _base = content.load_items(ep)
    assert items[0] == {"file": "/abs/source.mp4", "source_time": 2.083,
                        "duration": content.PREROLL_DURATION, "clip_in": 0.0}


def _reaction_setup(tmp_path):
    ep = resolve("ep001", root=tmp_path)
    ep.work.mkdir(parents=True, exist_ok=True)
    ep.playback_json.write_text(json.dumps({
        "version": 1, "episode_id": "ep001", "source_file": "/abs/source.mp4",
        "segments": [{"id": "pb001", "host_in": 10.0, "host_out": 25.0,
                      "source_in": 0.0, "source_out": 15.0}],
    }), encoding="utf-8")
    return ep


def test_reaction_preroll_uses_auto_detected_thumbnail(tmp_path):
    ep = _reaction_setup(tmp_path)
    ep.inbox.mkdir(parents=True, exist_ok=True)
    thumb = ep.inbox / "ep001_thumb.jpg"
    thumb.write_bytes(b"x")
    items, _base = content.load_items(ep)
    assert items[0] == {"file": str(thumb), "source_time": 0.0, "duration": content.PREROLL_DURATION}
    assert items[1]["file"] == "/abs/source.mp4"  # pb001 unaffected


def test_reaction_preroll_prefers_configured_thumbnail_override(tmp_path):
    ep = _reaction_setup(tmp_path)
    ep.inbox.mkdir(parents=True, exist_ok=True)
    (ep.inbox / "ep001_thumb.jpg").write_bytes(b"x")  # would be auto-detected...
    override = tmp_path / "custom_thumb.png"
    override.write_bytes(b"x")
    layout = {"reaction": {"thumbnail": "custom_thumb.png"}}
    items, _base = content.load_items(ep, layout)
    assert items[0]["file"] == str(override)  # ...but the override wins


def test_reaction_preroll_falls_back_when_configured_thumbnail_missing(tmp_path):
    ep = _reaction_setup(tmp_path)
    layout = {"reaction": {"thumbnail": "does_not_exist.png"}}
    items, _base = content.load_items(ep, layout)
    assert items[0] == {"file": "/abs/source.mp4", "source_time": 0.0,
                        "duration": content.PREROLL_DURATION, "clip_in": 0.0}


def test_load_items_prefers_content_json_over_reaction(tmp_path):
    # A manually authored content.json takes priority even if playback.json exists.
    ep = _setup(tmp_path, [{"file": "a.png"}], _SEGS)
    ep.playback_json.write_text(json.dumps({
        "segments": [{"id": "pb001", "host_in": 0.0, "host_out": 1.0,
                      "source_in": 0.0, "source_out": 1.0}],
        "source_file": "/abs/source.mp4",
    }), encoding="utf-8")
    items, _base = content.load_items(ep)
    assert items == [{"file": "a.png"}]


# --- graph construction ---

def test_graph_hold_is_alpha_aware_and_holds_last_frame(tmp_path):
    ep = _setup(tmp_path, [{"file": "a.png"}, {"file": "b.mov"}], _SEGS)
    placed = [({"file": "a.png", "duration": 6}, 5.0), ({"file": "b.mov", "duration": 6}, 15.0)]
    inputs, graph, alpha = content.build_graph(ep, _layout("hold"), "24", (0.0, 40.0), placed, ep.content_dir)
    assert alpha is True                                        # transparent base -> card alpha survives
    assert "colorchannelmixer=aa=0" in graph                   # transparent base, not a flat fill
    assert "tpad=stop_mode=clone" in graph                     # the video card holds its last frame
    assert "setpts=PTS-STARTPTS+5.000/TB" in graph             # item placed at out time
    assert "setpts=PTS-STARTPTS+15.000/TB" in graph
    assert "fade=t=in:st=0:d=0.35:alpha=1" in graph            # crossfade in
    assert graph.rstrip().endswith("[ctrack]")
    assert "-loop" in inputs                                    # still image looped
    assert inputs.count("-i") == 2                              # one input per item


def test_graph_window_includes_item_active_at_start(tmp_path):
    # A window starting at 10s makes item output-times relative. Item at out 15
    # -> rel 5. Item at out 5 started before the window but holds into it, so it is
    # included with a negative (pre-roll) start rather than dropped — the window is
    # never empty at t=0.
    ep = _setup(tmp_path, [{"file": "a.png"}, {"file": "b.mov"}], _SEGS)
    placed = [({"file": "a.png", "duration": 6}, 5.0), ({"file": "b.mov", "duration": 6}, 15.0)]
    inputs, graph, _ = content.build_graph(ep, _layout("hold"), "24", (10.0, 30.0), placed, ep.content_dir)
    assert "setpts=PTS-STARTPTS+5.000/TB" in graph    # 15.0 - 10.0
    assert "setpts=PTS-STARTPTS-5.000/TB" in graph    # 5.0 - 10.0, holds into the window
    assert inputs.count("-i") == 2


def test_graph_background_transparent_base_and_fade_out(tmp_path):
    ep = _setup(tmp_path, [{"file": "a.png"}], _SEGS)
    placed = [({"file": "a.png", "duration": 6}, 5.0)]
    _, graph, alpha = content.build_graph(ep, _layout("background"), "24", (0.0, 40.0), placed, ep.content_dir)
    assert alpha is True
    assert "colorchannelmixer=aa=0" in graph                   # transparent base
    assert "fade=t=out" in graph                               # items fade back to wallpaper


def test_graph_video_item_seeks_to_clip_in_and_caps_read_at_its_own_duration(tmp_path):
    # A reaction content item points at the full source file with clip_in as its
    # in-point; the input read must be capped at the item's own duration (not the
    # hold-extended seglen), or it would keep playing real source content past
    # where the playback segment actually ends instead of freezing there.
    ep = _setup(tmp_path, [{"file": "source.mp4"}], _SEGS)
    placed = [({"file": "source.mp4", "duration": 5.0, "clip_in": 52.583}, 5.0)]
    inputs, graph, _ = content.build_graph(ep, _layout("hold"), "24", (0.0, 40.0), placed, ep.content_dir)
    assert "-ss" in inputs
    assert inputs[inputs.index("-ss") + 1] == "52.583"
    assert "-t" in inputs
    assert inputs[inputs.index("-t") + 1] == "5.000"    # item_dur, not the 35s hold-extended seglen
    assert "tpad=stop_mode=clone:stop_duration=35.000" in graph  # holds the frozen frame the rest of the way


def test_render_track_uses_prores_for_a_clip_in_item(tmp_path, caplog):
    # qtrle is cheap on flat/static cards but balloons on full-motion footage —
    # an item with clip_in (a chunk seeked out of a larger source, reaction spec
    # section 7) must render with ProRes 4444 instead, matching the speaker layer.
    from autocut import ffmpeg
    ffmpeg.set_dry_run(True)
    try:
        ep = _setup(tmp_path, [{"file": "source.mp4", "source_time": 5.0,
                                "duration": 4.0, "clip_in": 52.583}], _SEGS)
        with caplog.at_level("INFO", logger="autocut.content"):
            content.render_track(ep, _layout("hold"), "24", (0.0, 20.0))
        assert "codec=prores_ks" in caplog.text
    finally:
        ffmpeg.set_dry_run(False)


def test_render_track_uses_qtrle_for_a_plain_card(tmp_path, caplog):
    from autocut import ffmpeg
    ffmpeg.set_dry_run(True)
    try:
        ep = _setup(tmp_path, [{"file": "a.png", "source_time": 5.0, "duration": 4.0}], _SEGS)
        with caplog.at_level("INFO", logger="autocut.content"):
            content.render_track(ep, _layout("hold"), "24", (0.0, 20.0))
        assert "codec=qtrle" in caplog.text
    finally:
        ffmpeg.set_dry_run(False)


def test_graph_cut_transition_has_no_fades(tmp_path):
    ep = _setup(tmp_path, [{"file": "a.png"}], _SEGS)
    placed = [({"file": "a.png", "duration": 6}, 5.0)]
    _, graph, _ = content.build_graph(ep, _layout("hold", ttype="cut"), "24", (0.0, 40.0), placed, ep.content_dir)
    assert "fade=" not in graph


def test_fit_filter_contain_vs_cover():
    assert "pad=" in content._fit_filter(2400, 1680, "contain", "0x0b0b0d")
    assert "decrease" in content._fit_filter(2400, 1680, "contain", "0x0b0b0d")
    assert "crop=2400:1680" in content._fit_filter(2400, 1680, "cover", "0x0b0b0d")
    assert "increase" in content._fit_filter(2400, 1680, "cover", "0x0b0b0d")
