"""Reaction audio track: window placement and filter-graph construction
(reaction spec section 8). The actual render is verified end-to-end
separately (align-check style: lip-sync/level checks by ear)."""

import json

from autocut import audiotrack, edl
from autocut.paths import resolve


def _setup(tmp_path, segments, source_file="/abs/source.mp4", edl_segments=None):
    ep = resolve("ep001", root=tmp_path)
    ep.work.mkdir(parents=True, exist_ok=True)
    ep.edl_json.write_text(json.dumps({
        "version": 1, "fps": 24,
        "segments": edl_segments or [{"id": "s1", "in": 0.0, "out": 10000.0, "action": "keep"}],
        "overlays": [],
    }), encoding="utf-8")
    ep.playback_json.write_text(json.dumps({
        "version": 1, "episode_id": "ep001", "source_file": source_file,
        "segments": segments,
    }), encoding="utf-8")
    return ep


def _spans(ep):
    return edl.build_time_map(edl.load(ep.edl_json)["segments"])


# host_in/host_out are also the output times here (single keep segment from 0).
_SEGS = [
    {"id": "pb001", "host_in": 10.0, "host_out": 20.0, "source_in": 100.0, "source_out": 110.0},
    {"id": "pb002", "host_in": 30.0, "host_out": 35.0, "source_in": 110.0, "source_out": 115.0},
]


def test_windowed_segments_maps_and_clips_to_the_window(tmp_path):
    ep = _setup(tmp_path, _SEGS)
    segs = audiotrack._windowed_segments(_spans(ep), _SEGS, (0.0, 40.0))
    assert len(segs) == 2
    assert segs[0]["window_rel_start"] == 10.0
    assert segs[0]["clip_dur"] == 10.0
    assert segs[0]["source_in"] == 100.0
    assert segs[0]["fade_in"] and segs[0]["fade_out"]


def test_windowed_segments_excludes_segments_outside_the_window(tmp_path):
    ep = _setup(tmp_path, _SEGS)
    segs = audiotrack._windowed_segments(_spans(ep), _SEGS, (0.0, 22.0))
    assert len(segs) == 1
    assert segs[0]["window_rel_start"] == 10.0


def test_windowed_segments_clips_a_segment_straddling_the_window_end(tmp_path):
    ep = _setup(tmp_path, _SEGS)
    # Window ends at 15 -> pb001 [10,20) is truncated to [10,15); the tail was
    # trimmed so its own fade-out doesn't belong to this render.
    segs = audiotrack._windowed_segments(_spans(ep), _SEGS, (0.0, 15.0))
    assert len(segs) == 1
    seg = segs[0]
    assert seg["clip_dur"] == 5.0
    assert seg["fade_in"] is True
    assert seg["fade_out"] is False


def test_windowed_segments_clips_a_segment_straddling_the_window_start(tmp_path):
    # Window starts at 12 -> pb001 [10,20) is truncated to [12,20); source_in
    # shifts forward by the trimmed head, and no fade-in belongs to this render.
    ep = _setup(tmp_path, _SEGS)
    segs = audiotrack._windowed_segments(_spans(ep), _SEGS, (12.0, 8.0))
    assert len(segs) == 1
    seg = segs[0]
    assert seg["window_rel_start"] == 0.0
    assert seg["clip_dur"] == 8.0
    assert seg["source_in"] == 102.0
    assert seg["fade_in"] is False
    assert seg["fade_out"] is True


# --- kept EDL spans (the host bed's actual audio source) ---

def test_windowed_keep_spans_reads_source_time_not_output_time(tmp_path):
    # A drop before the window shifts source and output time apart: output 0
    # is source 5.0 here (a 5s leading drop), not source 0.
    edl_segments = [
        {"id": "s1", "in": 0.0, "out": 5.0, "action": "drop", "reason": "dead_air"},
        {"id": "s2", "in": 5.0, "out": 100.0, "action": "keep"},
    ]
    ep = _setup(tmp_path, _SEGS, edl_segments=edl_segments)
    spans = _spans(ep)
    keep = audiotrack._windowed_keep_spans(spans, (0.0, 10.0))
    assert keep == [{"source_in": 5.0, "source_out": 15.0}]


def test_windowed_keep_spans_covers_multiple_fragmented_kept_segments(tmp_path):
    # Several small drops inside the window -> several kept spans to concatenate,
    # exactly what a real cut.mkv would splice together for this slice.
    edl_segments = [
        {"id": "s1", "in": 0.0, "out": 10.0, "action": "keep"},
        {"id": "s2", "in": 10.0, "out": 12.0, "action": "drop", "reason": "long_silence"},
        {"id": "s3", "in": 12.0, "out": 20.0, "action": "keep"},
        {"id": "s4", "in": 20.0, "out": 21.0, "action": "drop", "reason": "filler"},
        {"id": "s5", "in": 21.0, "out": 40.0, "action": "keep"},
    ]
    ep = _setup(tmp_path, _SEGS, edl_segments=edl_segments)
    spans = _spans(ep)  # output: [0,10) [10,18) [18,37)
    keep = audiotrack._windowed_keep_spans(spans, (0.0, 20.0))
    assert keep == [
        {"source_in": 0.0, "source_out": 10.0},
        {"source_in": 12.0, "source_out": 20.0},
        {"source_in": 21.0, "source_out": 23.0},  # clipped: output window ends at 20
    ]


def test_windowed_keep_spans_excludes_spans_outside_the_window(tmp_path):
    edl_segments = [
        {"id": "s1", "in": 0.0, "out": 5.0, "action": "keep"},
        {"id": "s2", "in": 5.0, "out": 6.0, "action": "drop", "reason": "filler"},
        {"id": "s3", "in": 6.0, "out": 100.0, "action": "keep"},
    ]
    ep = _setup(tmp_path, _SEGS, edl_segments=edl_segments)
    spans = _spans(ep)
    keep = audiotrack._windowed_keep_spans(spans, (0.0, 3.0))
    assert keep == [{"source_in": 0.0, "source_out": 3.0}]


# --- graph construction ---

def _seg(rel, dur, source_in=0.0, fade_in=True, fade_out=True):
    return {"window_rel_start": rel, "clip_dur": dur, "source_in": source_in,
            "fade_in": fade_in, "fade_out": fade_out}


_ONE_SPAN = [{"source_in": 0.0, "source_out": 20.0}]


def test_graph_no_segments_is_hostbed_passthrough():
    graph = audiotrack._build_graph(_ONE_SPAN, [], (0.0, 20.0), 0.075, 0.0)
    assert "amix" not in graph
    assert "[hostbed]anull[mix]" in graph


def test_graph_uses_a_single_volume_expression_not_chained_afade():
    # afade=t=in silences the *entire* stream before its own start_time
    # (confirmed against real ffmpeg output) — chaining it after an out-fade
    # to "restore" the host mid-stream silenced the whole track, including
    # everything before either fade. Regression guard: no afade on the host
    # envelope line (the keep-span splice fades are a separate, unrelated line).
    graph = audiotrack._build_graph(_ONE_SPAN, [_seg(10.0, 5.0)], (0.0, 20.0), 0.075, 0.0)
    envelope_line = next(l for l in graph.splitlines() if "volume=eval=frame" in l)
    assert "afade" not in envelope_line
    assert "amix=inputs=2:duration=longest:normalize=0[mix]" in graph


def test_graph_concatenates_multiple_keep_spans_into_the_host_bed():
    spans = [{"source_in": 0.0, "source_out": 10.0}, {"source_in": 12.0, "source_out": 20.0}]
    graph = audiotrack._build_graph(spans, [], (0.0, 18.0), 0.075, 0.0)
    assert "[hk0][hk1]concat=n=2:v=0:a=1[hostraw]" in graph
    assert graph.splitlines()[0].startswith("[0:a]")
    assert graph.splitlines()[1].startswith("[1:a]")
    # Source clips (if any) must index after the keep spans, not before.
    assert "[2:a]" not in graph  # none here (no segments), but no stray reference either


def test_graph_source_clip_seeks_delays_pads_and_gain_matches():
    graph = audiotrack._build_graph(_ONE_SPAN, [_seg(10.0, 5.0, source_in=52.583)], (0.0, 20.0), 0.075, 3.5)
    src_line = next(l for l in graph.splitlines() if l.startswith("[1:a]"))  # after keep-span input 0
    assert "volume=3.50dB" in src_line
    assert "afade=t=in:st=0:d=0.075" in src_line
    assert "afade=t=out:st=4.925:d=0.075" in src_line
    assert "adelay=10000|10000" in src_line
    assert "apad=whole_dur=20.000" in src_line


def test_graph_clipped_segment_skips_its_own_edge_fade():
    # Segment starts at the window's own start (already mid-span) - no clip
    # "in" fade (there's nothing to ramp up from within this render).
    graph = audiotrack._build_graph(_ONE_SPAN, [_seg(0.0, 5.0, fade_in=False)], (0.0, 20.0), 0.075, 0.0)
    src_line = next(l for l in graph.splitlines() if l.startswith("[1:a]"))
    assert "afade=t=in" not in src_line
    assert "afade=t=out" in src_line


# --- host volume envelope expression ---

def test_host_volume_expr_full_volume_with_no_segments():
    assert audiotrack._host_volume_expr([], 20.0, 0.075) == "1"


def test_host_volume_expr_mutes_the_playback_span_with_ramps_at_both_edges():
    expr = audiotrack._host_volume_expr([_seg(10.0, 5.0)], 20.0, 0.075)
    assert "between(t,9.925000,10.000000)" in expr    # ramp down into playback
    assert "between(t,10.000000,15.000000)" in expr   # muted for the whole span
    assert "between(t,15.000000,15.075000)" in expr   # ramp back up after
    assert expr.rstrip(")").endswith("1")             # falls back to full volume


def test_host_volume_expr_skips_ramp_when_start_is_window_clipped():
    # Window starts mid-playback: host must be muted from t=0, no preceding
    # ramp to build (there's no earlier in-window audio to ramp down from).
    expr = audiotrack._host_volume_expr([_seg(0.0, 5.0, fade_in=False)], 20.0, 0.075)
    assert "between(t,0.000000,5.000000),0" in expr
    assert "between(t,-0.075000" not in expr


def test_host_volume_expr_skips_ramp_when_end_is_window_clipped():
    # Window ends mid-playback: host stays muted through window end, no
    # trailing ramp back up (real playback continues past this render).
    expr = audiotrack._host_volume_expr([_seg(10.0, 10.0, fade_out=False)], 20.0, 0.075)
    assert "between(t,10.000000,20.000000),0" in expr
    assert "between(t,20.000000" not in expr


def test_audio_config_defaults(tmp_path):
    from autocut.paths import resolve as _resolve
    ep = _resolve("ep001", root=tmp_path)
    cfg = audiotrack._audio_config(ep)
    assert cfg["crossfade_s"] == 0.075
    assert cfg["match_levels"] is True
    assert cfg["source_gain_db"] == 0.0
