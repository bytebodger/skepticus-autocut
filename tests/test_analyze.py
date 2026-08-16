"""Tests for the deterministic baseline EDL author and the cache."""

import json

import pytest

from autocut import analyze, cache, edl
from autocut.paths import resolve


def _make_episode(tmp_path, words, duration, fps=30, silences=None):
    ep = resolve("ep001", root=tmp_path)
    ep.transcript_dir.mkdir(parents=True, exist_ok=True)
    ep.work.mkdir(parents=True, exist_ok=True)
    ep.words_json.write_text(json.dumps({
        "language": "en",
        "words": [{"i": i, "start": s, "end": e, "word": w, "prob": 1.0}
                  for i, (s, e, w) in enumerate(words)],
    }), encoding="utf-8")
    ep.silence_json.write_text(json.dumps({
        "silences": [{"start": s, "end": e, "duration": round(e - s, 3)}
                     for s, e in (silences or [])],
    }), encoding="utf-8")
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
    # 2.0s word gap that silencedetect confirms is real dead air -> trimmed.
    ep = _make_episode(tmp_path, [
        (0.5, 1.0, "start"),
        (3.0, 3.5, "resume"),  # 2.0s gap -> long silence
        (3.6, 4.0, "end"),
    ], duration=5.0, silences=[(1.0, 3.0)])
    result = analyze.autoauthor(ep)
    assert any(s.get("reason") == "long_silence" for s in result["segments"] if s["action"] == "drop")


def test_word_gap_without_acoustic_silence_is_not_cut(tmp_path):
    # The dialogue-loss bug: Whisper left a 2.0s gap between words (a re-take it
    # failed to transcribe), but silencedetect finds NO dead air there. That gap
    # is real speech and must never be trimmed as a "long silence".
    ep = _make_episode(tmp_path, [
        (0.5, 1.0, "start"),
        (3.0, 3.5, "resume"),  # 2.0s word gap, but the audio is not silent
        (3.6, 4.0, "end"),
    ], duration=5.0, silences=[])  # silencedetect found no silence in the gap
    result = analyze.autoauthor(ep)
    assert not any(s.get("reason") == "long_silence" for s in result["segments"] if s["action"] == "drop")


def test_long_silence_clamped_to_confirmed_silent_span(tmp_path):
    # A 3.0s word gap where only the middle 1.0s is truly silent (Whisper missed
    # speech at the edges). Only the confirmed-silent span may be cut, so the
    # surrounding speech survives.
    ep = _make_episode(tmp_path, [
        (0.5, 1.0, "a"),
        (4.0, 4.5, "b"),   # gap 1.0 -> 4.0; silence only 2.0-3.0
    ], duration=6.0, silences=[(2.0, 3.0)])
    result = analyze.autoauthor(ep)
    drop = next(s for s in result["segments"] if s.get("reason") == "long_silence")
    # Cut stays within the silent span (padded inward), never into 1.0-2.0 or
    # 3.0-4.0 where Whisper-missed speech lives.
    assert drop["in"] >= 2.0 - 1e-6
    assert drop["out"] <= 3.0 + 1e-6


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


def test_autoauthor_refuses_implausibly_sparse_edl(tmp_path):
    # A long source that yields almost no cuts is the silent-failure symptom.
    ep = _make_episode(tmp_path, [(1.0, 2.0, "hello"), (1400.0, 1401.0, "bye")],
                       duration=1500.0, silences=[])
    with pytest.raises(RuntimeError, match="implausibly few"):
        analyze.autoauthor(ep)
    assert not ep.edl_json.exists()  # refused to emit


def test_sparse_edl_message_flags_empty_silence(tmp_path):
    ep = _make_episode(tmp_path, [(1.0, 2.0, "hello"), (1400.0, 1401.0, "bye")],
                       duration=1500.0, silences=[])
    with pytest.raises(RuntimeError, match="silence.json is empty"):
        analyze.autoauthor(ep)


def test_sparse_edl_bypass_env_emits_with_warning(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOCUT_ALLOW_SPARSE_EDL", "1")
    ep = _make_episode(tmp_path, [(1.0, 2.0, "hello"), (1400.0, 1401.0, "bye")],
                       duration=1500.0, silences=[])
    result = analyze.autoauthor(ep)  # warns, does not raise
    assert ep.edl_json.exists()
    assert result["segments"]


def test_short_source_skips_sparse_check(tmp_path):
    # Under the duration floor, a sparse EDL is fine (short clip legitimately has
    # few cuts) — must not raise.
    ep = _make_episode(tmp_path, [(5.0, 5.5, "hi"), (6.0, 8.0, "there")], duration=90.0)
    result = analyze.autoauthor(ep)
    assert result["segments"]


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
