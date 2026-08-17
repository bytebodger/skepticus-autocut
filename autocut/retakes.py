"""Retake detection (retakes spec) — build-order steps 1, 2, 5, 10.

Detects flubbed lines that were re-recorded in place and drops the discarded
attempt. Runs inside ``autoauthor``, emitting drops into the EDL; changes nothing
downstream.

This ships the high-precision core: explicit-marker detection. When the host says
a retake marker — an explicit "let me try that again", or their personal cue word
"mulligan" — the attempt before it is dropped and the redo after it kept. Per the
spec's governing asymmetry (a false positive deletes published content; a miss
costs one manual cut), everything here biases toward leaving audio alone and goes
to the review gate.

Two marker sources are read from ``words.json``:
  * the main word stream (multi-word phrases, and single-word markers Whisper
    heard in context), and
  * the ``isolated`` sidecar — lone words re-transcribed from silence-bounded VAD
    chunks, which is where a cue spoken into a pause actually lives (Whisper
    smooths it into a neighbouring sentence in the full-audio pass).

Similarity/false-start candidates and LLM adjudication (spec steps 3-4, 7) are the
next increment; this pass emits only explicit-marker retakes.
"""

from __future__ import annotations

import logging
import re

log = logging.getLogger("autocut.retakes")

# A pause at least this long ends an utterance (spec section 5, ~400ms).
UTTERANCE_GAP = 0.40
# Two cues closer than this are the same detection seen in both sources.
CUE_DEDUP = 2.0
MARKER_CONFIDENCE = 0.9        # explicit markers are the highest-precision signal
NO_SILENCE_PENALTY = 0.3       # reduce when a boundary has no acoustic silence

# The host's personal cue words (single tokens). "mulligan" = a do-over.
SINGLE_MARKERS = {"mulligan"}
# Explicit retake phrases (matched on normalised words).
PHRASE_MARKERS = [
    "let me try that again", "let me try again", "let me do that again",
    "let me redo that", "let me start over", "let me say that again",
    "one more time", "take two", "take three", "scratch that",
    "sorry one more time",
]


def _norm(token: str) -> str:
    return re.sub(r"[^a-z0-9]", "", token.lower())


# --------------------------------------------------------------------------- #
# Step 1 — utterance segmentation
# --------------------------------------------------------------------------- #

def utterances(words: list[dict]) -> list[dict]:
    """Segment words into utterances on sentence-final punctuation or a pause
    above UTTERANCE_GAP. Each records its word-index range, times, and text."""
    utts: list[dict] = []
    cur: list[tuple[int, dict]] = []
    for i, w in enumerate(words):
        cur.append((i, w))
        nxt = words[i + 1] if i + 1 < len(words) else None
        ends = (w["word"].rstrip().endswith((".", "?", "!"))
                or nxt is None
                or (nxt["start"] - w["end"]) >= UTTERANCE_GAP)
        if ends:
            utts.append({
                "index": len(utts),
                "w0": cur[0][0], "w1": cur[-1][0],
                "start": float(cur[0][1]["start"]), "end": float(cur[-1][1]["end"]),
                "text": " ".join(x[1]["word"] for x in cur),
            })
            cur = []
    return utts


# --------------------------------------------------------------------------- #
# Step 2 — explicit-marker cue detection
# --------------------------------------------------------------------------- #

def _phrase_hits(words: list[dict], phrases: list[list[str]]) -> list[tuple[float, float]]:
    """Find each phrase (as a normalised token sequence) in the word stream and
    return its (start, end) time span."""
    norm = [_norm(w["word"]) for w in words]
    hits = []
    for toks in phrases:
        n = len(toks)
        for i in range(len(norm) - n + 1):
            if norm[i:i + n] == toks:
                hits.append((float(words[i]["start"]), float(words[i + n - 1]["end"])))
    return hits


def cues(words_doc: dict) -> list[dict]:
    """Explicit-marker cue spans, de-duplicated across the word stream and the
    isolated sidecar. Each cue is ``{start, end, source}``."""
    words = words_doc.get("words", [])
    raw: list[tuple[float, float, str]] = []

    for w in words:                                    # single-word markers, in context
        if _norm(w["word"]) in SINGLE_MARKERS:
            raw.append((float(w["start"]), float(w["end"]), "word"))
    for x in words_doc.get("isolated", []):            # lone words from silence-bounded chunks
        if any(_norm(t) in SINGLE_MARKERS for t in x["text"].split()):
            raw.append((float(x["start"]), float(x["end"]), "isolated"))
    phrases = [[_norm(t) for t in p.split()] for p in PHRASE_MARKERS]
    for s, e in _phrase_hits(words, phrases):
        raw.append((s, e, "phrase"))

    raw.sort()
    deduped: list[dict] = []
    for s, e, src in raw:
        if deduped and s - deduped[-1]["start"] < CUE_DEDUP:
            # Same cue seen twice (word + isolated). Keep the FIRST span — do not
            # extend the end, or a slightly-longer isolated detection swallows the
            # redo's first word and defeats prefix alignment (52.3s).
            continue
        deduped.append({"start": s, "end": e, "source": src})
    return deduped


# --------------------------------------------------------------------------- #
# Steps 5/10 — boundary resolution + drops
# --------------------------------------------------------------------------- #

# A flubbed attempt longer than this is not dropped in full — cap the lookback and
# flag for review, so a runaway match can never delete a large span.
MAX_FLUB = 20.0
REDO_PREFIX_WORDS = 6       # how many of the redo's opening words to look back for
MIN_PREFIX_MATCH = 2        # require >= this many repeated words to bound the flub


def _prev_silence(silences: list[dict], t: float):
    """Last silence interval ending at/before t (silences sorted by start)."""
    best = None
    for iv in silences:
        if float(iv["end"]) <= t + 1e-6:
            best = iv
    return best


def _next_silence(silences: list[dict], t: float):
    """First silence interval starting at/after t."""
    for iv in silences:
        if float(iv["start"]) >= t - 1e-6:
            return iv
    return None


def _flub_start(pre_words: list[dict], post_words: list[dict]) -> tuple[float | None, int]:
    """The flub is the abandoned attempt whose opening the redo repeats (spec §2/§6
    prefix repetition). Find the LAST place before the cue where the redo's opening
    words recur — that's where the flubbed attempt began. Returns (start, match_len)
    or (None, 0) if the redo doesn't repeat a pre-cue phrase."""
    redo_pre = [t for t in (_norm(w["word"]) for w in post_words[:REDO_PREFIX_WORDS]) if t]
    pren = [_norm(w["word"]) for w in pre_words]
    for plen in range(min(len(redo_pre), len(pren)), MIN_PREFIX_MATCH - 1, -1):
        target = redo_pre[:plen]
        for i in range(len(pren) - plen, -1, -1):
            if pren[i:i + plen] == target:
                return float(pre_words[i]["start"]), plen
    return None, 0


def _conservative_start(cs: float, silences: list[dict], utts: list[dict]) -> float:
    """Low-confidence flub boundary (spec §3.5): the flub can't be prefix-bounded,
    so fall back to the nearest confirmed silence BEFORE the flub (the pause before
    the pause-before-the-cue), or the previous utterance's start. Over-cuts slightly
    — the correct failure direction — while keeping the drop start on/at silence."""
    pre_cue = _prev_silence(silences, cs)                   # pause between flub and cue
    if pre_cue is not None:
        pre_flub = _prev_silence(silences, float(pre_cue["start"]))  # pause before the flub
        if pre_flub is not None and float(pre_flub["end"]) >= cs - MAX_FLUB:
            return float(pre_flub["end"])                   # drop starts at the flub, keep-before ends in silence
    prev = [u for u in utts if u["end"] <= cs + 1e-6]
    if prev:
        return max(prev[-1]["start"], cs - MAX_FLUB)
    if pre_cue is not None:
        return float(pre_cue["end"])                        # last resort: just the cue (still cuts it)
    return cs


def retake_drops(words_doc: dict, silence_doc: dict) -> list[dict]:
    """Explicit-marker retake drops. EVERY detected cue produces a cut (spec §3.5)
    — there is no "detected but not cut" state. Confidence governs only WHERE the
    boundary goes:

      good content ... [pause] FLUB(=redo's opening) [pause] MARKER [pause] redo ...
                                ^-- drop_start                       drop_end --^

    * High confidence — the redo repeats the flub's opening, so prefix alignment
      bounds the flub tightly (drop = flub + cue), and the drop ends at the redo's
      first word so the repeated phrase is kept exactly once (no duplication).
    * Low confidence — no prefix repetition; fall back to a conservative boundary
      (nearest confirmed silence before the flub, or the previous utterance start).
      This may over-cut slightly, which is correct; leaving the flub in is not.
      The row is flagged (``needs_review``) so the review gate can nudge its scope."""
    words = words_doc.get("words", [])
    if not words:
        return []
    silences = sorted(silence_doc.get("silences", []), key=lambda iv: float(iv["start"]))
    utts = utterances(words)
    drops: list[dict] = []

    for c in cues(words_doc):
        cs, ce = c["start"], c["end"]
        pre_words = [w for w in words if float(w["end"]) <= cs + 1e-6]
        post_words = [w for w in words if float(w["start"]) >= ce - 1e-6]
        # Redo boundary: the redo's first word (word cue) so the repeated phrase is
        # kept once; the cue end (isolated cue) since the base word stream is smeared
        # there and its word boundaries can't be trusted.
        if c["source"] == "isolated":
            redo_start = ce
        else:
            redo_start = float(post_words[0]["start"]) if post_words else ce

        fs, match_len = _flub_start(pre_words, post_words)
        bounded = fs is not None and cs - MAX_FLUB <= fs <= cs
        if bounded:
            flub_start, conf = fs, MARKER_CONFIDENCE
            note = f"explicit marker ({c['source']}) at {cs:.1f}s, flub prefix x{match_len}"
        else:
            flub_start, conf = _conservative_start(cs, silences, utts), MARKER_CONFIDENCE - NO_SILENCE_PENALTY
            note = f"explicit marker ({c['source']}) at {cs:.1f}s, conservative boundary"

        start, end = round(max(0.0, flub_start), 3), round(redo_start, 3)
        # Join de-dup: if the word kept just before the drop equals the redo's first
        # kept word, the repeated phrase would be spoken twice ("underneath
        # underneath..."). Pull the drop start back over each such word so the phrase
        # is kept exactly once (over-cutting is the correct direction, spec §3.5).
        for _ in range(12):
            before = [w for w in words if float(w["end"]) <= start + 1e-6]
            after = [w for w in words if float(w["start"]) >= end - 1e-6]
            if (before and after and _norm(before[-1]["word"])
                    and _norm(before[-1]["word"]) == _norm(after[0]["word"])):
                new_start = round(float(before[-1]["start"]), 3)
                if new_start < start - 1e-6:
                    start = new_start
                    continue
            break
        # Nudge the start a hair earlier so frame-snapping (which rounds to nearest)
        # can't leave a sliver of the flub's first word in the keep. The lost ~30ms
        # of the previous word's tail is under the cut stage's 25ms fade.
        start = round(max(0.0, start - 0.03), 3)
        if end - start < 1e-3:                              # never a zero cut: at least the cue
            start, end = round(cs, 3), round(max(ce, cs + 0.05), 3)
        drops.append({
            "start": start, "end": end,
            "reason": "retake", "confidence": round(max(0.1, conf), 2),
            "note": note,
            "needs_review": not bounded,                    # low-confidence boundaries get review
        })
    return drops


def summary(drops: list[dict], duration: float) -> dict:
    """Metrics for the run (spec section 12)."""
    removed = sum(d["end"] - d["start"] for d in drops)
    return {
        "retakes": len(drops),
        "seconds_removed": round(removed, 2),
        "share_of_source": round(removed / duration, 4) if duration else 0.0,
        "needs_review": sum(1 for d in drops if d.get("needs_review")),
    }
