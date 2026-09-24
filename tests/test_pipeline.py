"""``go``'s orchestration helpers: summary math and formatting (pure logic).
The actual stage sequencing is exercised end-to-end separately (it's just
calling the same functions the individual CLI commands call)."""

import json

import pytest

from autocut import pipeline
from autocut.paths import resolve


def test_fmt_dur_sub_minute():
    assert pipeline._fmt_dur(4.2) == "00:04.2"


def test_fmt_dur_minutes():
    assert pipeline._fmt_dur(65.0) == "01:05.0"


def test_fmt_dur_hours():
    assert pipeline._fmt_dur(3725.0) == "1:02:05.0"


def test_fmt_size_units():
    assert pipeline._fmt_size(500) == "500.0 B"
    assert pipeline._fmt_size(2048) == "2.0 KB"
    assert pipeline._fmt_size(5 * 1024 * 1024) == "5.0 MB"


def test_flagged_drop_count_only_counts_low_confidence_drops():
    edl_doc = {"segments": [
        {"action": "drop", "confidence": 0.5},
        {"action": "drop", "confidence": 0.95},
        {"action": "drop"},  # no confidence -> treated as 1.0, not flagged
        {"action": "keep", "confidence": 0.1},  # kept, not a drop -> not counted
    ]}
    assert pipeline._flagged_drop_count(edl_doc) == 1


def test_drop_seconds_by_reason_sums_only_drops():
    edl_doc = {"segments": [
        {"action": "drop", "reason": "filler", "in": 1.0, "out": 2.5},
        {"action": "drop", "reason": "filler", "in": 10.0, "out": 10.5},
        {"action": "drop", "reason": "dead_air", "in": 20.0, "out": 21.0},
        {"action": "keep", "in": 30.0, "out": 40.0},
    ]}
    totals = pipeline._drop_seconds_by_reason(edl_doc)
    assert totals["filler"] == pytest.approx(2.0)
    assert totals["dead_air"] == pytest.approx(1.0)
    assert "keep" not in totals


def test_playback_split_reads_probe_and_playback_json(tmp_path):
    ep = resolve("ep001", root=tmp_path)
    ep.work.mkdir(parents=True, exist_ok=True)
    ep.probe_json.write_text(json.dumps({"source_duration": 100.0}), encoding="utf-8")
    ep.playback_json.write_text(json.dumps({
        "segments": [{"host_in": 0.0, "host_out": 30.0}, {"host_in": 50.0, "host_out": 60.0}],
    }), encoding="utf-8")
    play, commentary = pipeline._playback_split(ep)
    assert play == pytest.approx(40.0)
    assert commentary == pytest.approx(60.0)


def test_playback_split_none_for_a_monologue_episode(tmp_path):
    ep = resolve("ep001", root=tmp_path)
    ep.work.mkdir(parents=True, exist_ok=True)
    assert pipeline._playback_split(ep) is None
