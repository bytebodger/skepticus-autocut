"""Phase-3 visuals — shot list authoring (visuals spec sections 2, 5, 12).

Reads ``words.json`` and writes ``shotlist.json``. A heuristic segmenter splits
the transcript into ~20s passages; an LLM then picks, for each passage, the one
composition that best carries it from the full taxonomy (spec section 2), or
``none``.

The taxonomy replaces the old illustration/infographic binary — that binary is
why 57 of 60 ``context`` shots came back as illustrations: anything that wasn't a
chart fell through to the image model. Now the LLM chooses among twelve rendered
HyperFrames compositions (``pull_quote``, ``term_card``, ``comparison``,
``bullet_reveal``, ``argument_diagram``, ``title_card``, ``chart``, ``timeline``,
``table``, ``map``, ``diagram``, ``vector_scene``) plus ``generated_image`` (the
escape hatch) and ``none``. It is told to prefer the most structured type the
passage supports; a healthy episode lands near 70% structured / 20% vector_scene
/ 10% generated.

``generated_image`` is the exception, not the engine. Every one must carry a
``why_generated`` justifying what made a composed vector scene impossible; a weak
justification means the classification is probably wrong.

The data rule is unchanged: only numbers actually spoken in a passage may be
charted, extracted verbatim. Scripture citations, dates in prose, and rhetorical
numbers are not data. If a chart is warranted but the figures weren't spoken, the
shot is demoted to another type (or dropped) at authoring time — never a
placeholder, never a render-blocking stub. Every ``chart`` in the output has real
spoken numbers.

Passages are batched per API call; ``shotlist.json`` is cached on the transcript
hash + prompt version + model, so it is a durable artifact — re-running does not
re-author unless ``--force`` or an input changes. A final sequence-level pass,
with the whole list in context, prunes repetition and bad pacing, and breaks up
runs of the same composition type (the lecture-slide-deck failure, section 12).
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
PROMPT_VERSION = "llm-2"
MODEL = os.environ.get("AUTOCUT_SHOTLIST_MODEL", "claude-opus-4-8")
BATCH_SIZE = 14          # passages per authoring call
MAX_OUTPUT_TOKENS = 16000

# Segmentation: one visual every ~15-25s (spec section 5). Passages target the
# middle of that band; the LLM decides which actually warrant a visual.
TARGET_SPACING = 20.0
SENT_PAUSE = 0.65
MIN_SHOT_DUR = 4.0
MAX_SHOT_DUR = 25.0
TERMINAL = (".", "?", "!")

# The twelve rendered HyperFrames compositions (spec section 2/3). Order is the
# taxonomy's own grouping: text/argument, data/time, space/structure, pictorial.
COMPOSITIONS = (
    "pull_quote", "term_card", "comparison", "bullet_reveal",
    "argument_diagram", "title_card",     # text and argument
    "chart", "timeline", "table",          # data and time
    "map", "diagram",                      # space and structure
    "vector_scene",                        # pictorial, library-composed
)
# Everything the LLM may choose: the twelve + the generated escape hatch + none.
KINDS = COMPOSITIONS + ("generated_image", "none")
# Distribution buckets for the 70/20/10 health check (spec section 2).
STRUCTURED = frozenset(COMPOSITIONS) - {"vector_scene"}


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
    kind: Literal[
        "pull_quote", "term_card", "comparison", "bullet_reveal",
        "argument_diagram", "title_card", "chart", "timeline", "table",
        "map", "diagram", "vector_scene", "generated_image", "none",
    ]
    props_json: str        # rendered compositions: content as a JSON object, else "{}"
    concept: str           # vector_scene / generated_image: the scene to depict, else ""
    why_generated: str     # generated_image ONLY: what made a vector scene impossible
    confidence: float


class BatchResult(BaseModel):
    decisions: list[Decision]


class Review(BaseModel):
    id: str
    action: Literal["keep", "drop"]
    reason: str


class SequenceReview(BaseModel):
    reviews: list[Review]


AUTHOR_SYSTEM = """\
You are the shot-list author for Skepticus, a video-essay channel about biblical
criticism and Christian apologetics. For each passage of the transcript you
choose the single on-screen composition that best carries it — or "none".

The content window is a RENDERED SLIDE SURFACE, not an image slot. The visual
thread is composed from deterministic HyperFrames compositions — text, data,
diagrams, and flat vector scenes. Generated imagery is a rare escape hatch, not
the default. PREFER THE MOST STRUCTURED TYPE THE PASSAGE SUPPORTS.

Choose exactly one `kind`. Put the composition's content in props_json as a JSON
object (shapes below are guidance — extract what the passage actually gives you):

TEXT AND ARGUMENT
- pull_quote      the passage quotes something (scripture, a scholar, an
                  apologist) accurately. {"quote": "...", "attribution": "..."}
- term_card       a word and its definition (Hebrew/Greek terms, technical
                  vocabulary). {"term": "...", "definition": "..."}
- comparison      two or three things side by side (almah vs betulah, competing
                  translations, two positions).
                  {"items": [{"label": "...", "detail": "..."}, ...]}
- bullet_reveal   enumerated points revealed in sequence. {"bullets": ["...", ...]}
- argument_diagram premises leading to a conclusion; branches for a dilemma,
                  chains for causal claims.
                  {"premises": ["..."], "conclusion": "..."}
- title_card      a section marker. {"title": "...", "subtitle": "..."}

DATA AND TIME
- chart           line/bar/stacked, ONLY from numbers spoken in THIS passage.
                  {"chart_type": "bar|line|stacked",
                   "series": [{"label": "...", "value": "15.7 million"}, ...]}
- timeline        dated events in sequence.
                  {"events": [{"date": "...", "label": "..."}, ...]}
- table           structured comparison across several rows.
                  {"columns": ["..."], "rows": [["...", "..."], ...]}

SPACE AND STRUCTURE
- map             geography (the ancient Near East, the spread of a movement,
                  council locations). {"region": "...", "markers": ["..."]}
- diagram         relationships/hierarchies/structures (a manuscript stemma, a
                  canon's formation). {"nodes": ["..."], "edges": [["a", "b"]]}

PICTORIAL
- vector_scene    a scene composed from flat, on-brand SVG parts (figures,
                  buildings, landscape, objects). Leave props_json "{}"; put ONE
                  sentence describing the scene in `concept`. This is the default
                  for a concrete visual referent that isn't better served above.
- generated_image THE ESCAPE HATCH. Only when a scene genuinely needs depicting
                  and CANNOT be composed from flat vector parts. Put what to
                  depict (flat/illustrative, never photorealistic) in `concept`,
                  AND set `why_generated` to one sentence naming what made a
                  vector scene impossible — specific historical detail, an
                  atmospheric establishing shot, a one-off subject not worth new
                  components. If you cannot write a strong why_generated, it is
                  not a generated_image — use vector_scene.

NONE
- none            abstract argument with no concrete visual and no data. Long
                  stretches of pure argument are normal; prefer fewer visuals and
                  longer holds over forcing something on-screen.

SELECTION PRIORITY — walk this in order and take the FIRST that fits:
  1  quotes something                 -> pull_quote
  2  defines a term                   -> term_card
  3  contrasts two or three things    -> comparison
  4  enumerates points                -> bullet_reveal
  5  structured argument (premises->conclusion) -> argument_diagram
  6  states numbers ALOUD             -> chart
  7  walks through time               -> timeline
  8  is about a place                 -> map
  9  a scene the SVG library can compose -> vector_scene
 10  a scene it CANNOT compose        -> generated_image
 11  none of the above                -> none
Also reach for title_card (a section marker), table (a multi-row comparison), or
diagram (an explicit structure/hierarchy) when they clearly fit.

DATA RULE — only numbers SPOKEN ALOUD in the passage may be charted, extracted
verbatim with their labels. Scripture citations (Psalm 23, John 3:16, Genesis 1),
chapter/verse numbers, dates in prose, ordinals, and rhetorical numbers ("a
thousand times") are NOT data. If a chart would be warranted but the figures were
not actually spoken, DEMOTE to another type (or "none") — never an empty chart,
never a placeholder. NEVER invent, estimate, or interpolate data.

generated_image should be RARE. Reach for a rendered composition or vector_scene
first; the image model is the last resort. Fill only the fields the chosen kind
needs; leave the rest "" or "{}". confidence is 0-1. Return one decision per
passage index.\
"""

COHERENCE_SYSTEM = """\
You are reviewing a finished shot list as one sequence, in playback order. Drop
shots that hurt the flow. Watch especially for:
- RUNS OF THE SAME COMPOSITION TYPE. Several text cards in a row (pull_quote,
  term_card, bullet_reveal, argument_diagram, title_card ...) read as a lecture
  slide deck. Break up a long run of one type by dropping its weakest members so
  the sequence varies; a map, chart, or vector_scene between them restores rhythm.
- two near-identical subjects, or the same composition back-to-back.
- rapid oscillation between types every few seconds.
- density that is too high (aim for roughly one visual every 15-25 seconds).
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


def _parse_props(props_json: str) -> dict:
    """Parse the decision's props into a non-empty dict, or {} if unusable."""
    try:
        v = json.loads(props_json) if props_json and props_json.strip() else {}
    except json.JSONDecodeError:
        return {}
    return v if isinstance(v, dict) and v else {}


def _to_shot(d: Decision, passage: dict) -> dict | None:
    """Assemble a shot from a decision + its passage, enforcing the data rule and
    the generated_image justification. Returns None to drop the shot ("none", a
    content composition with no usable props, or a generated_image that can't
    justify itself)."""
    if d.kind == "none":
        return None
    source_time = round(passage["start"], 3)
    duration = round(max(MIN_SHOT_DUR, min(MAX_SHOT_DUR, passage["end"] - passage["start"])), 3)
    shot = {"kind": d.kind, "source_time": source_time, "duration": duration,
            "transcript": passage["text"], "confidence": round(float(d.confidence), 2)}

    if d.kind == "generated_image":
        # The escape hatch must justify why a vector scene couldn't do it. A
        # generated_image without both a concept and a real why_generated is
        # almost always a misclassification — drop it rather than ship it.
        if not d.concept.strip() or not d.why_generated.strip():
            log.warning("shotlist: dropping generated_image at %.1fs without concept/why_generated",
                        source_time)
            return None
        shot.update(concept=d.concept.strip(), why_generated=d.why_generated.strip())
        return shot

    if d.kind == "vector_scene":
        # Composed from the SVG library; the scene lives in `concept`. Props
        # (element hints) are optional here — the render stage owns placement.
        if not d.concept.strip():
            return None
        shot["concept"] = d.concept.strip()
        props = _parse_props(d.props_json)
        if props:
            shot["props"] = props
        return shot

    # A rendered content composition (text / data / space). It must carry real,
    # non-empty props or it doesn't ship — this is where the data rule bites for
    # `chart`: a chart the LLM couldn't fill with spoken numbers is dropped, not
    # placeheld.
    props = _parse_props(d.props_json)
    if not props:
        log.warning("shotlist: dropping %s at %.1fs with no usable content", d.kind, source_time)
        return None
    shot["props"] = props
    return shot


def _coherence(client, shots: list[dict]) -> list[dict]:
    """Sequence-level review of the whole list (spec section 5)."""
    if len(shots) <= 1:
        return shots
    lines = []
    for i, s in enumerate(shots):
        s["_tid"] = f"t{i}"
        desc = s.get("concept") or ", ".join(str(k) for k in list(s.get("props", {}))[:4])
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


def distribution(shots: list[dict]) -> dict:
    """Per-type counts plus the structured / vector_scene / generated_image
    buckets used for the 70/20/10 health check (spec section 2)."""
    by_type: dict[str, int] = {}
    for s in shots:
        by_type[s["kind"]] = by_type.get(s["kind"], 0) + 1
    buckets = {"structured": 0, "vector_scene": 0, "generated_image": 0}
    for kind, n in by_type.items():
        if kind in STRUCTURED:
            buckets["structured"] += n
        elif kind in buckets:
            buckets[kind] += n
    return {"total": len(shots), "by_type": by_type, "buckets": buckets}


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

    dist = distribution(result["shots"])
    total = dist["total"] or 1
    b = dist["buckets"]
    log.info("shotlist: %d shots by type: %s", dist["total"], dist["by_type"])
    log.info("shotlist: structured %d (%.0f%%) / vector_scene %d (%.0f%%) / generated_image %d (%.0f%%)"
             " [target ~70/20/10] -> %s",
             b["structured"], 100 * b["structured"] / total,
             b["vector_scene"], 100 * b["vector_scene"] / total,
             b["generated_image"], 100 * b["generated_image"] / total, ep.shotlist_json)
    if b["generated_image"] > 0.20 * total:
        log.warning("shotlist: generated_image is %.0f%% (>20%%) — taxonomy guidance needs tightening",
                    100 * b["generated_image"] / total)
    cache.mark_done(stage_dir, input_hash, extra={"stage": "shotlist", "model": MODEL})
    return result
