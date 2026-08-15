"""Tests for the deterministic baseline EDL author and the cache."""

import json

import pytest

from autocut import analyze, cache, edl
from autocut.paths import resolve


def _make_episode(tmp_path, words, duration, fps=30):
    ep = resolve("ep001", root=tmp_path)
    ep.transcript_dir.mkdir(parents=True, exist_ok=True)
    ep.work.mkdir(parents=True, exist_ok=True)
    ep.words_json.write_text(json.dumps({
        "language": "en",
        "words": [{"i": i, "start": s, "end": e, "word": w, "prob": 1.0}
                  for i, (s, e, w) in enumerate(words)],
    }), encoding="utf-8")
    ep.silence_json.write_text(json.dumps({"silences": []}), encoding="utf-8")
    ep.probe_json.write_text(json.dumps({
        "fps": fps, "source_duration": duration, "raw_sha256": "abc",
    }), encoding="utf-8")
    return ep


def test_autoauthor_trims_leading_and_trailing_dead_air(tmp_path):
    # 5s of silence, speech 5-8, then 4s trailing silence.
    ep = _make_episode(tmp_path, [(5.0, 5.5, "hello"), (5.6, 8.0, "world")], duration=12.0)
    result = analyze.autoauthor(ep)
    drops = [s for s in result["segments"] if s["action"] == "drop"]
    reasons = {s.get("reason") for s in drops}
    assert "dead_air" in reasons
    # First kept segment starts near the first word (with padding), not at 0.
    first_keep = next(s for s in result["segments"] if s["action"] == "keep")
    assert first_keep["in"] > 0
    assert edl.validate(result, source_duration=12.0) == []


def test_autoauthor_trims_long_silence(tmp_path):
    ep = _make_episode(tmp_path, [
        (0.5, 1.0, "start"),
        (3.0, 3.5, "resume"),  # 2.0s gap -> long silence
        (3.6, 4.0, "end"),
    ], duration=5.0)
    result = analyze.autoauthor(ep)
    assert any(s.get("reason") == "long_silence" for s in result["segments"] if s["action"] == "drop")


def test_autoauthor_removes_filler_midsentence(tmp_path):
    ep = _make_episode(tmp_path, [
        (0.5, 0.9, "the"),
        (0.95, 1.15, "um"),    # filler, not sentence start
        (1.2, 1.6, "point"),
    ], duration=3.0)
    result = analyze.autoauthor(ep)
    fillers = [s for s in result["segments"] if s.get("reason") == "filler"]
    assert len(fillers) == 1
    assert edl.validate(result, source_duration=3.0) == []


def test_autoauthor_keeps_filler_at_sentence_start(tmp_path):
    # A filler as the very first word is a sentence start -> conservative, kept.
    ep = _make_episode(tmp_path, [
        (0.5, 0.9, "um"),
        (1.0, 1.4, "hello"),
    ], duration=3.0)
    result = analyze.autoauthor(ep)
    assert not any(s.get("reason") == "filler" for s in result["segments"])


def test_autoauthor_is_deterministic(tmp_path):
    words = [(5.0, 5.5, "hello"), (5.6, 8.0, "world")]
    ep1 = _make_episode(tmp_path / "a", words, duration=12.0)
    ep2 = _make_episode(tmp_path / "b", words, duration=12.0)
    r1 = analyze.autoauthor(ep1)
    r2 = analyze.autoauthor(ep2)
    # Segments identical (episode_id/source differ, so compare segments only).
    assert r1["segments"] == r2["segments"]


def test_autoauthor_preserves_override(tmp_path):
    ep = _make_episode(tmp_path, [
        (0.5, 0.9, "the"),
        (0.95, 1.15, "um"),
        (1.2, 1.6, "point"),
    ], duration=3.0)
    first = analyze.autoauthor(ep)
    filler = next(s for s in first["segments"] if s.get("reason") == "filler")
    # Human vetoes the filler drop via the review gate.
    filler["action"] = "keep"
    filler["override"] = True
    edl.save(first, ep.edl_json)

    # Re-author must not clobber the veto.
    second = analyze.autoauthor(ep)
    kept = next(s for s in second["segments"] if s["id"] == filler["id"])
    assert kept["action"] == "keep"
    assert kept["override"] is True


def test_no_drop_shorter_than_minimum(tmp_path):
    # A 0.2s gap is below LONG_SILENCE; a tiny filler after padding would be < MIN_DROP.
    ep = _make_episode(tmp_path, [
        (0.5, 0.9, "a"),
        (1.0, 1.4, "b"),   # 0.1s gap, no silence drop
    ], duration=2.0)
    result = analyze.autoauthor(ep)
    for s in result["segments"]:
        if s["action"] == "drop":
            assert s["out"] - s["in"] >= analyze.MIN_DROP - 1e-6


# --------------------------------------------------------------------------- #
# Cache
# --------------------------------------------------------------------------- #

def test_cache_roundtrip(tmp_path):
    stage = tmp_path / "stage"
    h = cache.hash_inputs({"a": 1, "b": 2})
    assert not cache.is_current(stage, h)
    cache.mark_done(stage, h, extra={"stage": "x"})
    assert cache.is_current(stage, h)
    assert not cache.is_current(stage, cache.hash_inputs({"a": 1, "b": 3}))


def test_hash_inputs_order_independent():
    assert cache.hash_inputs({"a": 1, "b": 2}) == cache.hash_inputs({"b": 2, "a": 1})


def test_invalidate(tmp_path):
    stage = tmp_path / "stage"
    h = cache.hash_inputs({"x": 1})
    cache.mark_done(stage, h)
    cache.invalidate(stage)
    assert not cache.is_current(stage, h)
