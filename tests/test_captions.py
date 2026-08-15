"""Tests for ASS caption generation.

Captions must map word timing through source_to_output (dropping cut words) and
emit valid karaoke timing. These guard the second place the source/output bug can
hide.
"""

from autocut import captions


TEMPLATE = """[Script Info]
ScriptType: v4.00+

[V4+ Styles]
Format: Name, Fontname
Style: Skepticus,Montserrat

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def _words(*triples):
    return {"words": [{"i": i, "start": s, "end": e, "word": w, "prob": 1.0}
                      for i, (s, e, w) in enumerate(triples)]}


def _edl(segments):
    return {"version": 1, "fps": 30, "segments": segments}


def test_fmt_time():
    assert captions._fmt_time(0.0) == "0:00:00.00"
    assert captions._fmt_time(12.48) == "0:00:12.48"
    assert captions._fmt_time(3661.5) == "1:01:01.50"


def test_dropped_words_are_excluded():
    words = _words(
        (0.0, 0.5, "kept"),
        (11.0, 11.5, "cut"),      # falls in dropped region
        (15.0, 15.5, "also"),
    )
    e = _edl([
        {"id": "s001", "in": 0.0, "out": 10.0, "action": "keep"},
        {"id": "s002", "in": 10.0, "out": 14.0, "action": "drop"},
        {"id": "s003", "in": 14.0, "out": 20.0, "action": "keep"},
    ])
    ass = captions.build_ass(words, e, TEMPLATE)
    assert "kept" in ass
    assert "also" in ass
    assert "cut" not in ass


def test_word_times_are_output_relative():
    """A word at source 15s, after a 4s cut, should render at output 11s."""
    words = _words((15.0, 15.4, "hello"))
    e = _edl([
        {"id": "s001", "in": 0.0, "out": 10.0, "action": "keep"},
        {"id": "s002", "in": 10.0, "out": 14.0, "action": "drop"},
        {"id": "s003", "in": 14.0, "out": 20.0, "action": "keep"},
    ])
    ass = captions.build_ass(words, e, TEMPLATE)
    # source 15 -> output 15 - 4 = 11.0
    assert "0:00:11.00" in ass


def test_karaoke_tags_present_and_positive():
    words = _words((0.0, 0.3, "The"), (0.3, 0.7, "argument"), (0.7, 1.1, "here"))
    e = _edl([{"id": "s001", "in": 0.0, "out": 5.0, "action": "keep"}])
    ass = captions.build_ass(words, e, TEMPLATE)
    assert r"{\k" in ass
    # Each word rendered
    for w in ("The", "argument", "here"):
        assert w in ass


def test_chunking_breaks_on_terminal_punctuation():
    # Two sentences separated by a period past the min-break length.
    words = _words(
        (0.0, 0.4, "First"), (0.5, 1.0, "clause"), (1.1, 2.0, "here."),
        (2.1, 2.6, "Second"), (2.7, 3.2, "clause"), (3.3, 3.9, "now."),
    )
    e = _edl([{"id": "s001", "in": 0.0, "out": 5.0, "action": "keep"}])
    ass = captions.build_ass(words, e, TEMPLATE)
    dialogue = [l for l in ass.splitlines() if l.startswith("Dialogue:")]
    assert len(dialogue) == 2


def test_chunking_breaks_on_long_pause():
    words = _words(
        (0.0, 0.4, "before"),
        (3.0, 3.4, "after"),   # 2.6s gap > PAUSE_BREAK
    )
    e = _edl([{"id": "s001", "in": 0.0, "out": 5.0, "action": "keep"}])
    ass = captions.build_ass(words, e, TEMPLATE)
    dialogue = [l for l in ass.splitlines() if l.startswith("Dialogue:")]
    assert len(dialogue) == 2


def test_braces_in_words_are_escaped():
    words = _words((0.0, 0.5, "a{b}c"))
    e = _edl([{"id": "s001", "in": 0.0, "out": 5.0, "action": "keep"}])
    ass = captions.build_ass(words, e, TEMPLATE)
    # No literal word-braces survive (only the \k override braces).
    assert "a(b)c" in ass


def test_template_header_preserved():
    words = _words((0.0, 0.5, "hi"))
    e = _edl([{"id": "s001", "in": 0.0, "out": 5.0, "action": "keep"}])
    ass = captions.build_ass(words, e, TEMPLATE)
    assert "[V4+ Styles]" in ass
    assert ass.count("[Events]") == 1
