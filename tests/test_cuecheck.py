"""Cue-leak checker: pure-logic pieces (transition placement, fragment
matching). The actual re-transcription is verified end-to-end separately —
this covers the parts that don't need real ffmpeg/Whisper."""

import json

from autocut import cuecheck, edl
from autocut.paths import resolve


def test_transitions_one_start_and_one_stop_per_segment():
    segs = [
        {"id": "pb001", "host_in": 10.0, "host_out": 20.0},
        {"id": "pb002", "host_in": 30.0, "host_out": 35.0},
    ]
    t = cuecheck._transitions(segs)
    assert len(t) == 4
    assert {(x["id"], x["kind"]) for x in t} == {
        ("pb001", "start"), ("pb001", "stop"), ("pb002", "start"), ("pb002", "stop"),
    }
    pb001_start = next(x for x in t if x["id"] == "pb001" and x["kind"] == "start")
    assert pb001_start["source_t"] == 10.0


def test_output_position_finds_a_start_cues_direct_hit():
    # host_in is itself a kept span's source_in -> a direct hit.
    edl_doc = {"segments": [
        {"id": "s1", "in": 0.0, "out": 10.0, "action": "keep"},
        {"id": "s2", "in": 10.0, "out": 12.0, "action": "drop"},
        {"id": "s3", "in": 12.0, "out": 20.0, "action": "keep"},
    ]}
    spans = edl.build_time_map(edl_doc["segments"])
    assert cuecheck._output_position_at_or_after(spans, 12.0) == 10.0


def test_output_position_finds_a_stop_cues_next_span_when_not_a_direct_hit():
    # host_out (10.0) is a kept span's source *end*, not any span's source_in
    # -> must land on the NEXT kept span's output start, not return None.
    edl_doc = {"segments": [
        {"id": "s1", "in": 0.0, "out": 10.0, "action": "keep"},
        {"id": "s2", "in": 10.0, "out": 12.5, "action": "drop"},
        {"id": "s3", "in": 12.5, "out": 20.0, "action": "keep"},
    ]}
    spans = edl.build_time_map(edl_doc["segments"])
    assert cuecheck._output_position_at_or_after(spans, 10.0) == 10.0  # s3's own output start


def test_output_position_none_past_the_end_of_the_edl():
    edl_doc = {"segments": [{"id": "s1", "in": 0.0, "out": 10.0, "action": "keep"}]}
    spans = edl.build_time_map(edl_doc["segments"])
    assert cuecheck._output_position_at_or_after(spans, 50.0) is None


def test_leaks_catches_a_fragment_substring():
    hits = cuecheck._leaks([{"word": "ary", "start": 1.0}], {"end", "my", "commentary", "begin"})
    assert len(hits) == 1
    assert hits[0]["matched"] == "commentary"


def test_leaks_catches_the_whole_cue_word():
    hits = cuecheck._leaks([{"word": "Begin.", "start": 1.0}], {"end", "my", "commentary", "begin"})
    assert len(hits) == 1
    assert hits[0]["matched"] == "begin"


def test_leaks_ignores_unrelated_words():
    hits = cuecheck._leaks(
        [{"word": "so", "start": 1.0}, {"word": "anyway", "start": 1.2}],
        {"end", "my", "commentary", "begin"})
    assert hits == []


def test_leaks_ignores_fragments_shorter_than_the_minimum():
    hits = cuecheck._leaks([{"word": "a", "start": 1.0}], {"end", "my", "commentary", "begin"})
    assert hits == []


def test_cue_words_from_config(tmp_path):
    ep = resolve("ep001", root=tmp_path)
    ep.work.mkdir(parents=True, exist_ok=True)
    words = cuecheck._cue_words(ep)
    # Defaults: "end my commentary" / "begin my commentary" + the "and my
    # commentary" Whisper-mishearing variant.
    assert words == {"end", "my", "commentary", "begin", "and"}


def test_run_reports_zero_checked_when_nothing_falls_in_window(tmp_path, monkeypatch):
    ep = resolve("ep001", root=tmp_path)
    ep.work.mkdir(parents=True, exist_ok=True)
    ep.playback_json.write_text(json.dumps({
        "version": 1, "episode_id": "ep001", "source_file": "/abs/source.mp4",
        "segments": [{"id": "pb001", "host_in": 500.0, "host_out": 510.0,
                      "source_in": 0.0, "source_out": 10.0}],
    }), encoding="utf-8")
    ep.edl_json.write_text(json.dumps({
        "version": 1, "fps": 24,
        "segments": [{"id": "s1", "in": 0.0, "out": 1000.0, "action": "keep"}],
        "overlays": [],
    }), encoding="utf-8")
    output = tmp_path / "out.mp4"
    output.write_bytes(b"x")
    from autocut import ffmpeg
    monkeypatch.setattr(ffmpeg, "ffprobe_json", lambda args: {"format": {"duration": "5.0"}})
    report = cuecheck.run(ep, output_path=output)
    assert report == {"checked": 0, "skipped": 2, "violations": []}
    assert json.loads(ep.cuecheck_report_json.read_text()) == report
