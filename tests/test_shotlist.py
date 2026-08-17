"""Shot list authoring (visuals spec sections 2, 5, 12).

The editorial decision is the LLM's; these tests cover the deterministic scaffold
around it — segmentation, the composition taxonomy assembly, the data rule (no
chart without spoken numbers), the generated_image justification requirement, the
distribution buckets, and the coherence drop — with the LLM call mocked.
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


def _decision(index=0, kind="none", props_json="{}", concept="", why_generated="", confidence=0.8):
    return Decision(index=index, kind=kind, props_json=props_json, concept=concept,
                    why_generated=why_generated, confidence=confidence)


# --- segmenter ------------------------------------------------------------- #

def test_segmenter_splits_on_pause_and_punctuation():
    words = _line("First sentence here.", 0.0) + _line("After a gap.", 3.0)
    sents = shotlist._sentences(words)
    assert len(sents) == 2
    assert sents[0]["text"].startswith("First")


def test_passages_group_toward_target_spacing():
    words = _line("word", 0.0)
    words += _line("later words here now again.", shotlist.TARGET_SPACING + 1)
    passages = shotlist._passages(shotlist._sentences(words))
    assert len(passages) >= 1
    assert passages[0]["start"] == 0.0


# --- the taxonomy: content compositions carry props ------------------------ #

def _passage(t0=10.0, t1=25.0, text="a passage"):
    return {"start": t0, "end": t1, "text": text}


def test_content_composition_carries_props():
    d = _decision(kind="pull_quote",
                  props_json='{"quote": "The virgin shall conceive", "attribution": "Isaiah 7:14"}')
    s = shotlist._to_shot(d, _passage())
    assert s["kind"] == "pull_quote"
    assert s["props"]["attribution"] == "Isaiah 7:14"
    assert s["source_time"] == 10.0 and "data_status" not in s


def test_chart_with_real_spoken_numbers_ships():
    d = _decision(kind="chart",
                  props_json='{"chart_type": "bar", "series": [{"label": "KJV", "value": "15.7 million"}]}')
    s = shotlist._to_shot(d, _passage())
    assert s["kind"] == "chart" and s["props"]["series"][0]["value"] == "15.7 million"


def test_chart_without_data_is_dropped():
    # A chart the LLM couldn't fill with spoken numbers must not ship a stub.
    assert shotlist._to_shot(_decision(kind="chart", props_json="{}"), _passage()) is None


def test_content_composition_without_props_is_dropped():
    assert shotlist._to_shot(_decision(kind="term_card", props_json="{}"), _passage()) is None


# --- vector_scene uses concept, not props ---------------------------------- #

def test_vector_scene_carries_concept():
    d = _decision(kind="vector_scene", concept="robed figures seated on a hillside at dawn")
    s = shotlist._to_shot(d, _passage())
    assert s["kind"] == "vector_scene" and s["concept"].startswith("robed figures")
    assert "why_generated" not in s


def test_vector_scene_without_concept_is_dropped():
    assert shotlist._to_shot(_decision(kind="vector_scene", concept=""), _passage()) is None


# --- generated_image must justify itself ----------------------------------- #

def test_generated_image_with_justification_ships():
    d = _decision(kind="generated_image",
                  concept="a crowded first-century Alexandrian library interior",
                  why_generated="dense architectural detail the flat component library can't compose")
    s = shotlist._to_shot(d, _passage())
    assert s["kind"] == "generated_image"
    assert s["concept"] and s["why_generated"]


def test_generated_image_without_why_is_dropped():
    # No justification -> almost certainly a misclassified vector_scene.
    d = _decision(kind="generated_image", concept="a temple", why_generated="")
    assert shotlist._to_shot(d, _passage()) is None


def test_none_emits_no_shot():
    assert shotlist._to_shot(_decision(kind="none"), _passage()) is None


# --- distribution buckets (the 70/20/10 health check) ---------------------- #

def test_distribution_buckets_structured_vector_generated():
    shots = [
        {"kind": "pull_quote"}, {"kind": "chart"}, {"kind": "map"},        # structured
        {"kind": "vector_scene"},                                          # vector
        {"kind": "generated_image"},                                       # generated
    ]
    dist = shotlist.distribution(shots)
    assert dist["total"] == 5
    assert dist["buckets"] == {"structured": 3, "vector_scene": 1, "generated_image": 1}


# --- author() end to end with the LLM mocked ------------------------------- #

def test_author_assembles_drops_and_reindexes(monkeypatch):
    passages = [_passage(10, 30, "p0"), _passage(30, 50, "p1"), _passage(50, 70, "p2")]
    monkeypatch.setattr(shotlist, "_passages", lambda sents: passages)

    decisions = [
        _decision(index=0, kind="pull_quote", props_json='{"quote": "q", "attribution": "a"}'),
        _decision(index=1, kind="vector_scene", concept="a stone temple at dusk"),
        _decision(index=2, kind="none"),  # -> no shot
    ]

    def fake_call(client, *, system, user, output_format):
        if output_format is BatchResult:
            return BatchResult(decisions=decisions)
        # coherence: keep the pull_quote, drop the vector_scene
        return SequenceReview(reviews=[Review(id="t0", action="keep", reason=""),
                                       Review(id="t1", action="drop", reason="run of scenes")])

    monkeypatch.setattr(shotlist, "_call_structured", fake_call)

    result = shotlist.author(client=object(), words_doc={"words": []},
                             episode_id="context", style="default")
    assert result["episode_id"] == "context" and result["style"] == "default"
    shots = result["shots"]
    assert [s["kind"] for s in shots] == ["pull_quote"]   # none dropped, scene pruned
    assert shots[0]["id"] == "sh001"
    assert "_tid" not in shots[0]                          # temp coherence id cleaned up
    assert list(shots[0])[:4] == ["id", "kind", "source_time", "duration"]


def test_taxonomy_includes_statement_plus_escape_and_none():
    assert "statement" in shotlist.COMPOSITIONS
    assert len(shotlist.COMPOSITIONS) == 13
    assert shotlist.KINDS[-2:] == ("generated_image", "none")
    assert "vector_scene" not in shotlist.STRUCTURED
    assert "statement" in shotlist.STRUCTURED


def test_statement_ships_with_text_and_no_attribution():
    # The host's own words -> statement, never a self-attributed pull_quote.
    d = _decision(kind="statement", props_json='{"text": "Context is the bug spray of theology."}')
    s = shotlist._to_shot(d, _passage())
    assert s["kind"] == "statement"
    assert s["props"]["text"].startswith("Context")
    assert "attribution" not in s["props"]


def _s(kind, t, conf=0.8):
    return {"kind": kind, "source_time": float(t), "duration": 6.0, "confidence": conf}


def test_rebalance_caps_statement_flood():
    # 20 statements + 5 other -> statement must fall to ~10% of the final count.
    shots = [_s("statement", i * 10, conf=0.5 + i * 0.01) for i in range(20)]
    shots += [_s("comparison", 5 + i * 40) for i in range(5)]
    kept = shotlist._rebalance(shots)
    n_stmt = sum(1 for s in kept if s["kind"] == "statement")
    assert n_stmt <= max(1, round(len(kept) * shotlist.STATEMENT_CAP)) + 1
    assert n_stmt < 20                                  # the flood is cut
    # the strongest statements survive (higher source_time had higher conf here)
    kept_ts = {s["source_time"] for s in kept if s["kind"] == "statement"}
    assert max(kept_ts) == 190.0


def test_rebalance_never_two_statements_in_a_row():
    # statements interleaved so the cap alone wouldn't separate them
    shots = [_s("statement", 0, 0.9), _s("statement", 10, 0.9),
             _s("comparison", 20), _s("statement", 30, 0.9), _s("statement", 40, 0.9)]
    kept = shotlist._rebalance(shots)
    kinds = [s["kind"] for s in sorted(kept, key=lambda s: s["source_time"])]
    assert not any(a == "statement" and b == "statement" for a, b in zip(kinds, kinds[1:]))


def test_rebalance_caps_any_type_at_25pct():
    # 8 comparisons flooding among 24 shots (33%) -> capped toward ~25% of shots.
    shots = [_s("comparison", i * 10, conf=0.5 + i * 0.01) for i in range(8)]
    shots += [_s("bullet_reveal", 5 + i * 20) for i in range(16)]
    n0 = len(shots)
    kept = shotlist._rebalance(shots)
    n_comp = sum(1 for s in kept if s["kind"] == "comparison")
    assert n_comp <= max(1, int(shotlist.TYPE_CAP * n0))    # capped to ~25% of shots
    assert n_comp < 8                                        # the flood is cut
    # the weakest comparisons are the ones dropped
    kept_comp_conf = [s["confidence"] for s in kept if s["kind"] == "comparison"]
    assert min(kept_comp_conf) >= 0.5 + (8 - n_comp) * 0.01 - 1e-9


def test_make_contiguous_extends_each_shot_to_the_next():
    shots = [
        {"kind": "statement", "source_time": 10.0, "duration": 6.0},
        {"kind": "pull_quote", "source_time": 30.0, "duration": 6.0},
        {"kind": "bullet_reveal", "source_time": 55.0, "duration": 6.0},
    ]
    shotlist._make_contiguous(shots)
    assert [s["duration"] for s in shots] == [20.0, 25.0, 6.0]  # last keeps its own


def test_prompt_version_and_model_are_in_the_cache_key():
    assert shotlist.PROMPT_VERSION and shotlist.MODEL
