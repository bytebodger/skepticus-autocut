"""Phase-3 visuals — shot list authoring (visuals spec section 5).

Reads ``words.json`` and writes ``shotlist.json``. A heuristic segmenter splits
the transcript into ~20s passages; an LLM then makes the editorial decision for
each passage — illustration, infographic, or neither — writes a real subject
(a noun phrase with context) and a rich concept (what the image should depict,
composition and all), and for infographics extracts the data.

A rule-based matcher can't tell that "English" refers to a translation, that
"Each" isn't a subject, or that "Psalm 23" is a citation and not chart data.
That judgment is the LLM's job here; the code owns segmentation, batching,
caching, and the hard data rule.

The data rule (this supersedes the spec's ``data_status`` model — removed):
only numbers actually spoken in a passage may be charted, extracted verbatim.
Scripture citations, dates in prose, and rhetorical numbers are not data. If a
chart is warranted but the figures weren't spoken, the shot is demoted to an
illustration (or dropped) at authoring time — never a placeholder, never a
render-blocking stub. Every infographic in the output has real props.

Passages are batched per API call; ``shotlist.json`` is cached on the transcript
hash + prompt version + model, so it is a durable artifact — re-running does not
re-author unless ``--force`` or an input changes. A final sequence-level pass,
with the whole list in context, prunes repetition and bad pacing.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Literal

import yaml
from pydantic import BaseModel

from . import cache
from .paths import Episode

log = logging.getLogger("autocut.shotlist")

# Bump when the prompts or schema change, so cached shot lists re-author.
PROMPT_VERSION = "llm-1"
MODEL = os.environ.get("AUTOCUT_SHOTLIST_MODEL", "claude-opus-4-8")
BATCH_SIZE = 14          # passages per authoring call
MAX_OUTPUT_TOKENS = 10000

# Segmentation: one visual every ~15-25s (spec section 5). Passages target the
# middle of that band; the LLM decides which actually warrant a visual.
TARGET_SPACING = 20.0
SENT_PAUSE = 0.65
MIN_SHOT_DUR = 4.0
MAX_SHOT_DUR = 25.0
TERMINAL = (".", "?", "!")

COMPOSITIONS = ("line_chart", "bar_chart", "stacked_area", "timeline",
                "definition_card", "pull_quote", "comparison_table",
                "argument_diagram", "bullet_reveal")


# --------------------------------------------------------------------------- #
# Heuristic segmenter (the passage-boundary pass underneath the LLM)
# --------------------------------------------------------------------------- #

def _sentences(words: list[dict]) -> list[dict]:
    sents, cur = [], []
    for i, w in enumerate(words):
        cur.append(w)
        nxt = words[i + 1] if i + 1 < len(words) else None
        gap = (nxt["start"] - w["end"]) if nxt else 0.0
        if w["word"].rstrip().endswith(TERMINAL) or gap >= SENT_PAUSE or nxt is None:
            sents.append(cur)
            cur = []
    if cur:
        sents.append(cur)
    return [{"start": s[0]["start"], "end": s[-1]["end"],
             "text": " ".join(w["word"].strip() for w in s)} for s in sents]


def _passages(sents: list[dict]) -> list[dict]:
    passages, cur = [], []
    for s in sents:
        cur.append(s)
        if cur[-1]["end"] - cur[0]["start"] >= TARGET_SPACING:
            passages.append(cur)
            cur = []
    if cur:
        passages.append(cur)
    return [{"start": p[0]["start"], "end": p[-1]["end"],
             "text": " ".join(s["text"] for s in p)} for p in passages]


# --------------------------------------------------------------------------- #
# LLM I/O — structured outputs (props carried as a JSON string so the strict
# schema doesn't need to enumerate every composition's prop shape)
# --------------------------------------------------------------------------- #

class Decision(BaseModel):
    index: int
    kind: Literal["illustration", "infographic", "none"]
    subject: str      # illustration: a noun phrase WITH context ("" otherwise)
    concept: str      # illustration: what the image depicts, composition and all
    variant: str      # illustration: named stylistic wrapper, else ""
    composition: str  # infographic: one of COMPOSITIONS, else ""
    props_json: str   # infographic: verbatim data as a JSON object, else "{}"
    confidence: float


class BatchResult(BaseModel):
    decisions: list[Decision]


class Review(BaseModel):
    id: str
    action: Literal["keep", "drop"]
    reason: str


class SequenceReview(BaseModel):
    reviews: list[Review]


AUTHOR_SYSTEM = f"""\
You are the shot-list author for Skepticus, a video-essay channel about biblical
criticism and Christian apologetics. You decide what on-screen visual, if any, a
passage of the transcript should get. The channel's look is flat, simple,
obviously-drawn illustration — never photorealism — plus deterministically
rendered infographics.

For each passage, choose exactly one kind:

- "illustration" — the passage has a concrete visual referent: a place, object,
  building, document, historical scene, landscape, or figure. Provide:
    - subject: a short NOUN PHRASE WITH CONTEXT, not a bare token. Good:
      "the translation of the Hebrew Bible into English"; "the Council of Nicaea
      in 325". Bad: "English", "Council".
    - concept: one sentence describing what the image should DEPICT — the
      composition, not a label. Concrete, simple, no legible text, never
      photorealistic. Good: "a weathered printing press stamping English words
      over faded Hebrew letters, flat editorial illustration".
    - variant: usually "". Only set a stylistic wrapper (engraving, woodcut,
      diagram) when it clearly suits the subject (e.g. a confronting scene).

- "infographic" — ONLY when the passage states real data aloud: actual numbers
  or statistics to chart, a definition stated in words, a quotation worth
  showing, or named categories compared with figures. Provide:
    - composition: one of {", ".join(COMPOSITIONS)}.
    - props_json: the REAL content, extracted VERBATIM from the passage, as a
      JSON object. For a chart, the actual spoken numbers with their labels. For
      definition_card, the term and its definition. For pull_quote, the quote
      and any attribution.
    NEVER invent, estimate, or interpolate data. Scripture citations (Psalm 23,
    John 3:16, Genesis 1), chapter/verse numbers, dates mentioned in prose,
    ordinals, and rhetorical numbers ("one word", "a thousand times") are NOT
    chartable data. If a chart would be warranted but the figures were not
    actually spoken, do NOT emit an infographic — choose "illustration" if the
    passage supports a concrete visual, otherwise "none".

- "none" — abstract argument with no concrete visual and no data. Emit "none".
  Long stretches of pure argument are normal and correct; prefer fewer visuals
  and longer holds over forcing an image onto an abstract point.

Fill every field on every decision. Use "" / "{{}}" for fields that don't apply
to the chosen kind. confidence is 0-1. Return one decision per passage index.\
"""

COHERENCE_SYSTEM = """\
You are reviewing a finished shot list as a whole sequence, in order. Drop shots
that hurt the sequence:
- two near-identical subjects or the same composition back-to-back,
- rapid oscillation between chart and illustration every few seconds,
- runs that are too dense (aim for roughly one visual every 15-25 seconds).
Keep the strong, well-paced shots; don't thin the list below that density. For
each shot id, return keep or drop with a one-line reason.\
"""


def _call_structured(client, *, system: str, user: str, output_format):
    """One structured-output call with adaptive thinking; the stable system
    prompt is cached across batches."""
    resp = client.messages.parse(
        model=MODEL,
        max_tokens=MAX_OUTPUT_TOKENS,
        thinking={"type": "adaptive"},
        system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": user}],
        output_format=output_format,
    )
    return resp.parsed_output


def _author_batch(client, passages: list[dict], offset: int) -> list[Decision]:
    lines = []
    for i, p in enumerate(passages):
        lines.append(f'[{offset + i}] source_time={p["start"]:.1f}s\n"{p["text"]}"')
    user = ("Decide one shot for each passage below (by its index).\n\n"
            + "\n\n".join(lines))
    return _call_structured(client, system=AUTHOR_SYSTEM, user=user, output_format=BatchResult).decisions


def _to_shot(d: Decision, passage: dict) -> dict | None:
    """Assemble a shot from a decision + its passage, enforcing the data rule.
    Returns None to drop (neither, or an infographic without real props)."""
    if d.kind == "none":
        return None
    source_time = round(passage["start"], 3)
    duration = round(max(MIN_SHOT_DUR, min(MAX_SHOT_DUR, passage["end"] - passage["start"])), 3)
    shot = {"kind": d.kind, "source_time": source_time, "duration": duration,
            "transcript": passage["text"], "confidence": round(float(d.confidence), 2)}

    if d.kind == "illustration":
        if not d.subject.strip() or not d.concept.strip():
            return None
        shot.update(subject=d.subject.strip(), concept=d.concept.strip(),
                    variant=d.variant.strip())
        return shot

    # infographic — must carry real, non-empty props or it doesn't ship
    try:
        props = json.loads(d.props_json) if d.props_json.strip() else {}
    except json.JSONDecodeError:
        props = {}
    if d.composition.strip() not in COMPOSITIONS or not props:
        log.warning("shotlist: dropping infographic at %.1fs with no usable data", source_time)
        return None
    shot.update(composition=d.composition.strip(), props=props)
    return shot


def _coherence(client, shots: list[dict]) -> list[dict]:
    """Sequence-level review of the whole list (spec section 5)."""
    if len(shots) <= 1:
        return shots
    lines = []
    for i, s in enumerate(shots):
        s["_tid"] = f"t{i}"
        desc = s.get("subject") or f'{s.get("composition")}: {list(s.get("props", {}))}'
        lines.append(f'{s["_tid"]} @ {s["source_time"]:.0f}s [{s["kind"]}] {desc}')
    user = ("Full shot list, in playback order. Return keep/drop for each id.\n\n"
            + "\n".join(lines))
    review = _call_structured(client, system=COHERENCE_SYSTEM, user=user, output_format=SequenceReview)
    dropped = {r.id for r in review.reviews if r.action == "drop"}
    kept = [s for s in shots if s["_tid"] not in dropped]
    for s in kept:
        s.pop("_tid", None)
    log.info("shotlist: coherence pass kept %d of %d", len(kept), len(shots))
    return kept


def _order_keys(shot: dict) -> dict:
    head = {k: shot[k] for k in ("id", "kind", "source_time", "duration")}
    return {**head, **{k: v for k, v in shot.items() if k not in head}}


def author(client, words_doc: dict, episode_id: str, style: str) -> dict:
    """Author the shot list. ``client`` is an anthropic client (or a stand-in
    with the same ``messages.parse`` surface)."""
    passages = _passages(_sentences(words_doc.get("words", [])))

    decisions: list[Decision] = []
    for start in range(0, len(passages), BATCH_SIZE):
        batch = passages[start:start + BATCH_SIZE]
        log.info("shotlist: authoring passages %d-%d of %d", start, start + len(batch) - 1, len(passages))
        decisions.extend(_author_batch(client, batch, start))

    by_index = {d.index: d for d in decisions}
    shots: list[dict] = []
    for i, passage in enumerate(passages):
        d = by_index.get(i)
        if d is None:
            continue
        shot = _to_shot(d, passage)
        if shot is not None:
            shots.append(shot)

    shots = _coherence(client, shots)
    for n, shot in enumerate(shots, 1):
        shot["id"] = f"sh{n:03d}"

    return {"version": 1, "episode_id": episode_id, "style": style,
            "shots": [_order_keys(s) for s in shots]}


# --------------------------------------------------------------------------- #
# Stage
# --------------------------------------------------------------------------- #

def _style_name(ep: Episode) -> str:
    cfg = ep.root / "config" / "layout.yaml"
    if cfg.exists():
        try:
            return str((yaml.safe_load(cfg.read_text(encoding="utf-8")) or {}).get("style", "default"))
        except yaml.YAMLError:
            pass
    return "default"


def _client():
    try:
        import anthropic
    except ImportError as e:
        raise RuntimeError(
            "shot-list authoring needs the anthropic SDK (a project dependency): "
            "install with '.venv\\Scripts\\python -m pip install -e .'"
        ) from e
    return anthropic.Anthropic()  # resolves ANTHROPIC_API_KEY / ant profile


def run(ep: Episode, *, force: bool = False) -> dict:
    """Author shotlist.json from words.json (LLM). Cached on transcript + prompt
    version + model, so it stays a durable artifact; re-run needs --force."""
    if not ep.words_json.exists():
        raise FileNotFoundError(
            f"No transcript at {ep.words_json}. Run 'autocut transcribe {ep.episode_id}' first."
        )
    style = _style_name(ep)
    stage_dir = ep.work / "shotlist"
    input_hash = cache.hash_inputs({
        "words": cache.hash_file(ep.words_json),
        "style": style,
        "prompt_version": PROMPT_VERSION,
        "model": MODEL,
    })
    if not force and cache.is_current(stage_dir, input_hash) and ep.shotlist_json.exists():
        log.info("shotlist: cache hit for %s, skipping", ep.episode_id)
        return json.loads(ep.shotlist_json.read_text(encoding="utf-8"))

    words_doc = json.loads(ep.words_json.read_text(encoding="utf-8"))
    log.info("shotlist: authoring %s with %s", ep.episode_id, MODEL)
    result = author(_client(), words_doc, ep.episode_id, style)
    ep.shotlist_json.write_text(json.dumps(result, indent=2), encoding="utf-8")

    kinds: dict[str, int] = {}
    for s in result["shots"]:
        kinds[s["kind"]] = kinds.get(s["kind"], 0) + 1
    log.info("shotlist: %d shots (%s) -> %s", len(result["shots"]), kinds, ep.shotlist_json)
    cache.mark_done(stage_dir, input_hash, extra={"stage": "shotlist", "model": MODEL})
    return result
