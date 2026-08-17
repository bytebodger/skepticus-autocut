"""Retake detection (retakes spec, steps 1/2/5/10).

Deterministic, model-free: utterance segmentation, explicit-marker cue detection
across the word stream and the isolated sidecar, and silence-snapped drops.
"""

from autocut import retakes


def _w(word, start, end):
    return {"start": start, "end": end, "word": word}


def _line(text, t0, wdur=0.3, gap=0.05):
    out, t = [], t0
    for tok in text.split():
        out.append(_w(tok, round(t, 3), round(t + wdur, 3)))
        t += wdur + gap
    return out


# --- utterance segmentation ------------------------------------------------ #

def test_utterances_split_on_punctuation_and_pause():
    words = _line("First thought here.", 0.0) + _line("After a pause now.", 5.0)
    utts = retakes.utterances(words)
    assert len(utts) == 2
    assert utts[0]["text"].startswith("First") and utts[1]["start"] == 5.0
    assert utts[0]["w0"] == 0 and utts[1]["w1"] == len(words) - 1


# --- explicit-marker cue detection ----------------------------------------- #

def test_cue_from_single_word_marker_in_stream():
    words = _line("the council of nicaea", 0.0) + _line("mulligan", 3.0) + _line("the council of nicaea convened", 5.0)
    cs = retakes.cues({"words": words})
    assert len(cs) == 1 and cs[0]["source"] == "word"
    assert abs(cs[0]["start"] - 3.0) < 1e-6


def test_cue_from_isolated_sidecar_when_stream_misheard():
    # The cue was smoothed into a sentence in the stream ("aggressive"), but the
    # isolated sidecar recovered it — detection must still fire.
    words = _line("opposes the more aggressive parliamentary rules", 0.0)
    doc = {"words": words, "isolated": [{"start": 3.1, "end": 4.3, "text": "Mulligan."}]}
    cs = retakes.cues(doc)
    assert len(cs) == 1 and cs[0]["source"] == "isolated"


def test_cue_dedups_stream_and_isolated_twins():
    words = _line("mulligan", 10.0)
    doc = {"words": words, "isolated": [{"start": 10.1, "end": 11.0, "text": "mulligan"}]}
    assert len(retakes.cues(doc)) == 1          # same cue seen twice -> one


def test_phrase_marker_detected():
    words = _line("let me try that again", 20.0)
    cs = retakes.cues({"words": words})
    assert len(cs) == 1 and cs[0]["source"] == "phrase"


def test_no_marker_no_cue():
    words = _line("just an ordinary sentence about context", 0.0)
    assert retakes.cues({"words": words}) == []


# --- drops + boundary resolution ------------------------------------------- #

def _doc():
    # good content we KEEP, then flub, [pause] mulligan [pause], redo.
    words = (_line("we keep this earlier sentence entirely.", 0.0)   # KEEP (ends ~2.4)
             + _line("the council of nicea convened.", 5.0)          # flub (~5-6.7)
             + _line("mulligan.", 8.0)                                # marker
             + _line("the council of nicaea convened in 325.", 10.0))  # redo repeats the flub
    return {"words": words}


def _redo_start(doc):
    # first redo word starts at 10.0; the pause before it is [8.3, 10.0]-ish
    return 10.0


def test_retake_drop_excludes_earlier_kept_content():
    doc = _doc()
    silences = {"silences": [{"start": 9.4, "end": 10.0}]}   # pause before the redo
    drops = retakes.retake_drops(doc, silences)
    assert len(drops) == 1
    d = drops[0]
    assert d["reason"] == "retake"
    # flub is prefix-bounded to "the council of..." at 5.0 — the earlier kept
    # sentence (starts at 0.0) is NOT swallowed.
    assert d["start"] >= 4.9                                 # excludes the earlier kept sentence
    assert abs(d["end"] - 10.0) < 1e-6                       # ends where the redo begins
    assert d["confidence"] == retakes.MARKER_CONFIDENCE
    assert d["needs_review"] is False


def test_unbounded_flub_uses_conservative_boundary_and_still_cuts():
    # Redo shares no opening with the flub -> can't prefix-bound. Per spec 3.5 the
    # cue STILL cuts, with a conservative boundary (previous utterance start) that
    # over-cuts the flub rather than leaving it in, flagged for review.
    words = (_line("some earlier sentence we keep.", 0.0)
             + _line("a totally different flubbed thing.", 5.0)
             + _line("mulligan.", 8.0)
             + _line("an unrelated fresh restart entirely.", 10.0))
    doc = {"words": words}
    silences = {"silences": [{"start": 7.6, "end": 8.0}, {"start": 9.4, "end": 10.0}]}
    d = retakes.retake_drops(doc, silences)[0]
    assert d["needs_review"] is True
    assert d["confidence"] < retakes.MARKER_CONFIDENCE
    assert 4.9 <= d["start"] <= 5.0                         # falls back to the previous utterance (minus the snap nudge)
    assert d["end"] >= 8.0                                   # and still removes the cue


def test_every_cue_cuts_even_with_no_silence():
    # No silence anywhere, no prefix repetition -> still cut (never suppressed).
    words = (_line("earlier content here now.", 0.0)
             + _line("mulligan.", 5.0)
             + _line("fresh unrelated wording follows.", 6.5))
    d = retakes.retake_drops({"words": words}, {"silences": []})
    assert len(d) == 1 and d[0]["end"] > d[0]["start"]      # a cut is produced regardless


def test_summary_metrics():
    doc = _doc()
    drops = retakes.retake_drops(doc, {"silences": [{"start": 9.4, "end": 10.0}]})
    s = retakes.summary(drops, duration=100.0)
    assert s["retakes"] == 1 and s["seconds_removed"] > 0
    assert 0 <= s["share_of_source"] <= 1
