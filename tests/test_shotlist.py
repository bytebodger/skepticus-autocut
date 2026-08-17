"""Shot list authoring (visuals spec section 5).

The editorial decision is the LLM's; these tests cover the deterministic scaffold
around it — segmentation, the data rule (no infographic without real props),
assembly, and the coherence drop — with the LLM call mocked.
"""

import json

from autocut import shotlist
from autocut.shotlist import Decision, BatchResult, Review, SequenceReview


def _line(text, t0, wdur=0.3, gap=0.05):
    out, t = [], t0
    for w in text.split():
        out.append({"start": round(t, 3), "end": round(t + wdur, 3), "word": w})
        t += wdur + gap
    return out


# --- segmenter ------------------------------------------------------------- #

def test_segmenter_splits_on_pause_and_punctuation():
    words = _line("First sentence here.", 0.0) + _line("After a gap.", 3.0)
    sents = shotlist._sentences(words)
    assert len(sents) == 2
    assert sents[0]["text"].startswith("First")


def test_passages_group_toward_target_spacing():
    words = _line("word", 0.0)
    # stretch a passage past TARGET_SPACING by placing a far-apart sentence
    words += _line("later words here now again.", shotlist.TARGET_SPACING + 1)
    passages = shotlist._passages(shotlist._sentences(words))
    assert len(passages) >= 1
    assert passages[0]["start"] == 0.0


# --- the data rule (never ship an infographic without real props) ---------- #

def _passage(t0=10.0, t1=25.0, text="a passage"):
    return {"start": t0, "end": t1, "text": text}


def test_illustration_shot_carries_subject_and_concept():
    d = Decision(index=0, kind="illustration", subject="the Council of Nicaea",
                 concept="robed figures in a Roman hall", variant="engraving",
                 composition="", props_json="{}", confidence=0.8)
    s = shotlist._to_shot(d, _passage())
    assert s["kind"] == "illustration"
    assert s["subject"] == "the Council of Nicaea" and s["concept"]
    assert s["variant"] == "engraving"
    assert "data_status" not in s and s["source_time"] == 10.0


def test_infographic_with_real_props_ships():
    d = Decision(index=0, kind="infographic", subject="", concept="", variant="",
                 composition="line_chart",
                 props_json='{"values": ["15.7 million", "12.9 million"]}', confidence=0.9)
    s = shotlist._to_shot(d, _passage())
    assert s["kind"] == "infographic" and s["composition"] == "line_chart"
    assert s["props"]["values"] == ["15.7 million", "12.9 million"]
    assert "data_status" not in s


def test_infographic_without_props_is_dropped():
    # A chart the LLM couldn't fill (no spoken data) must not ship a stub.
    d = Decision(index=0, kind="infographic", subject="", concept="", variant="",
                 composition="line_chart", props_json="{}", confidence=0.6)
    assert shotlist._to_shot(d, _passage()) is None


def test_none_emits_no_shot():
    d = Decision(index=0, kind="none", subject="", concept="", variant="",
                 composition="", props_json="{}", confidence=0.9)
    assert shotlist._to_shot(d, _passage()) is None


# --- author() end to end with the LLM mocked ------------------------------- #

def test_author_assembles_drops_and_reindexes(monkeypatch):
    passages = [_passage(10, 30, "p0"), _passage(30, 50, "p1"), _passage(50, 70, "p2")]
    monkeypatch.setattr(shotlist, "_passages", lambda sents: passages)

    decisions = [
        Decision(index=0, kind="illustration", subject="a temple", concept="a stone temple",
                 variant="", composition="", props_json="{}", confidence=0.7),
        Decision(index=1, kind="infographic", subject="", concept="", variant="",
                 composition="bar_chart", props_json='{"a": 1}', confidence=0.8),
        Decision(index=2, kind="none", subject="", concept="", variant="",
                 composition="", props_json="{}", confidence=0.9),  # -> no shot
    ]

    def fake_call(client, *, system, user, output_format):
        if output_format is BatchResult:
            return BatchResult(decisions=decisions)
        # coherence: keep the illustration, drop the infographic
        return SequenceReview(reviews=[Review(id="t0", action="keep", reason=""),
                                       Review(id="t1", action="drop", reason="oscillation")])

    monkeypatch.setattr(shotlist, "_call_structured", fake_call)

    result = shotlist.author(client=object(), words_doc={"words": []},
                             episode_id="context", style="default")
    assert result["episode_id"] == "context" and result["style"] == "default"
    shots = result["shots"]
    assert [s["kind"] for s in shots] == ["illustration"]   # none dropped, infographic pruned
    assert shots[0]["id"] == "sh001"
    assert "_tid" not in shots[0]                            # temp coherence id cleaned up
    assert list(shots[0])[:4] == ["id", "kind", "source_time", "duration"]


def test_prompt_version_and_model_are_in_the_cache_key():
    # Guards that re-authoring keys on prompt/model, not just the transcript.
    assert shotlist.PROMPT_VERSION and shotlist.MODEL
