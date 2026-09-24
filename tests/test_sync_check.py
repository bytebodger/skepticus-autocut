"""Playback sync checker: word-alignment offset math (pure logic). The
transcription itself is exercised end-to-end separately."""

import json

from autocut import sync_check
from autocut.paths import resolve


def _w(word, start):
    return {"word": word, "start": start}


def test_aligned_offsets_zero_when_arithmetic_is_exact():
    # host word at host_in+1.0 should predict source_in+1.0 exactly.
    host_words = [_w("hello", 11.0), _w("world", 11.5)]
    source_words = [_w("hello", 1.0), _w("world", 1.5)]
    offsets = sync_check._aligned_offsets(host_words, source_words, host_in=10.0, source_in=0.0)
    assert offsets == [0.0, 0.0]


def test_aligned_offsets_detects_a_constant_drift():
    # source words all land 2s later than the arithmetic predicts.
    host_words = [_w("hello", 11.0), _w("world", 11.5), _w("again", 12.0)]
    source_words = [_w("hello", 3.0), _w("world", 3.5), _w("again", 4.0)]
    offsets = sync_check._aligned_offsets(host_words, source_words, host_in=10.0, source_in=0.0)
    assert offsets == [2.0, 2.0, 2.0]


def test_aligned_offsets_skips_words_only_one_side_heard():
    # Bleed degrades "world" into something unrecognisable -> no match for
    # it, but "hello"/"again" still align correctly around it.
    host_words = [_w("hello", 11.0), _w("wrld", 11.5), _w("again", 12.0)]
    source_words = [_w("hello", 1.0), _w("world", 1.5), _w("again", 2.0)]
    offsets = sync_check._aligned_offsets(host_words, source_words, host_in=10.0, source_in=0.0)
    assert offsets == [0.0, 0.0]


# --- window filtering (go --preview: check only what was actually rendered) ---

def _setup_ep(tmp_path):
    ep = resolve("ep001", root=tmp_path)
    ep.work.mkdir(parents=True, exist_ok=True)
    ep.edl_json.write_text(json.dumps({
        "version": 1, "fps": 24,
        "segments": [{"id": "s1", "in": 0.0, "out": 1000.0, "action": "keep"}],
        "overlays": [],
    }), encoding="utf-8")
    return ep


def test_segments_in_window_keeps_only_overlapping_segments(tmp_path):
    ep = _setup_ep(tmp_path)
    segments = [
        {"id": "pb001", "host_in": 5.0, "host_out": 8.0},   # output 5-8s: in a 0-10s window
        {"id": "pb002", "host_in": 50.0, "host_out": 55.0},  # output 50-55s: outside it
    ]
    kept = sync_check._segments_in_window(segments, ep, (0.0, 10.0))
    assert [s["id"] for s in kept] == ["pb001"]


def test_segments_in_window_none_means_no_filtering(tmp_path):
    ep = _setup_ep(tmp_path)
    segments = [{"id": "pb001", "host_in": 5.0, "host_out": 8.0},
               {"id": "pb002", "host_in": 500.0, "host_out": 505.0}]
    # run()'s default path (window=None) never calls _segments_in_window at
    # all -- this just documents that every segment is a legitimate "in
    # window" candidate absent an explicit window.
    kept = sync_check._segments_in_window(segments, ep, (0.0, 1000.0))
    assert [s["id"] for s in kept] == ["pb001", "pb002"]
