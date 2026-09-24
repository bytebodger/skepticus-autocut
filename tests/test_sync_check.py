"""Playback sync checker: word-alignment offset math (pure logic). The
transcription itself is exercised end-to-end separately."""

from autocut import sync_check


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
