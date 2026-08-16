"""Stage 3 helpers.

Claude Code authors the EDL (that is the whole point of the pipeline — a model
reads the transcript and makes editorial decisions). This module gives it, and
the CLI, the deterministic pieces around that decision:

  * ``load_inputs`` — read words/silence/probe.
  * ``autoauthor`` — a deterministic baseline EDL: leading/trailing dead air,
    long-silence trims, and single-word filler removal. This is the "ship after
    step 2" path — useful on its own, and a floor Claude can build on.
  * ``validate`` — structural + source-timebase validation (delegates to edl.py).

Everything here obeys the section-5 rules: word boundaries only, padding on keep
boundaries, and no drop shorter than the minimum-drop threshold.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

from . import edl
from .paths import Episode

log = logging.getLogger("autocut.analyze")

# Editorial constants (spec section 5). Tune to taste, then leave alone.
PAD = 0.10            # 100ms padding kept on each side of a keep boundary
LONG_SILENCE = 0.70   # mid-sentence silence over this gets trimmed
MIN_DROP = 0.15       # never make a drop shorter than this — the cut costs more
SENTENCE_GAP = 0.50   # a gap this long implies a sentence boundary

# Single-word fillers only. Multi-word fillers ("you know", "I mean") are left
# for Claude Code to judge in context — automating them is not reliably safe.
FILLERS = {"um", "uh", "uhh", "umm", "erm", "hmm", "mmm", "er"}


def load_inputs(ep: Episode) -> tuple[dict, dict, dict]:
    """Return (words, silence, probe). Raises if a prerequisite is missing."""
    for path in (ep.words_json, ep.silence_json, ep.probe_json):
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path.name}. Run probe + transcribe for {ep.episode_id} first."
            )
    words = json.loads(ep.words_json.read_text(encoding="utf-8"))
    silence = json.loads(ep.silence_json.read_text(encoding="utf-8"))
    probe = json.loads(ep.probe_json.read_text(encoding="utf-8"))
    return words, silence, probe


def _norm(token: str) -> str:
    return "".join(c for c in token.lower() if c.isalpha())


def _is_sentence_start(words: list[dict], idx: int) -> bool:
    """Conservative: treat a word as a sentence start if it's first, follows
    terminal punctuation, or follows a long gap. Removal there sounds abrupt."""
    if idx == 0:
        return True
    prev = words[idx - 1]
    if prev["word"].rstrip().endswith((".", "?", "!")):
        return True
    return (words[idx]["start"] - prev["end"]) >= SENTENCE_GAP


# --------------------------------------------------------------------------- #
# Cut discovery
# --------------------------------------------------------------------------- #

class _Cut:
    __slots__ = ("start", "end", "reason", "confidence", "note", "pad")

    def __init__(self, start, end, reason, confidence, note, pad):
        self.start = start
        self.end = end
        self.reason = reason
        self.confidence = confidence
        self.note = note
        self.pad = pad  # whether internal edges should be padded


def _discover_cuts(words: list[dict], silences: list[dict], duration: float) -> list[_Cut]:
    cuts: list[_Cut] = []
    if not words:
        return cuts

    first, last = words[0], words[-1]

    # 1. Leading / trailing dead air. Trivial and always safe.
    if first["start"] > PAD:
        cuts.append(_Cut(0.0, first["start"], "dead_air", 1.0, "leading dead air", pad="right"))
    if duration - last["end"] > PAD:
        cuts.append(_Cut(last["end"], duration, "dead_air", 1.0, "trailing dead air", pad="left"))

    # 2. Long mid-sentence silences. A gap between consecutive Whisper word
    #    timestamps is trimmed ONLY where the independent, energy-based
    #    silencedetect pass confirms real dead air. Whisper routinely drops or
    #    stretches words across a re-take or fast speech, leaving a multi-second
    #    word gap that is actually full of dialogue — cutting those phantom gaps
    #    silently deletes speech. Intersecting with silencedetect (and clamping
    #    to the gap, which lies between two words) keeps every cut on real
    #    silence. Padding leaves ~PAD on each side so pacing stays human.
    for a, b in zip(words, words[1:]):
        if b["start"] - a["end"] <= LONG_SILENCE:
            continue
        for iv in silences:
            s = max(a["end"], float(iv["start"]))
            e = min(b["start"], float(iv["end"]))
            # Only trim a confirmed-silent span that itself exceeds the editorial
            # threshold; a sub-threshold sliver of real silence is left alone.
            if e - s >= LONG_SILENCE:
                cuts.append(_Cut(
                    s, e, "long_silence", 1.0,
                    f"{e - s:.2f}s silence trimmed", pad="both",
                ))

    # 3. Single-word fillers. Word boundaries only; skip near sentence starts.
    for idx, w in enumerate(words):
        if _norm(w["word"]) in FILLERS and not _is_sentence_start(words, idx):
            cuts.append(_Cut(
                w["start"], w["end"], "filler", 0.95,
                f"filler {w['word']!r}", pad="none",
            ))

    return cuts


def _apply_padding(cut: _Cut, duration: float) -> tuple[float, float]:
    """Shrink a cut inward on internal edges so the neighbouring keep retains
    padding. Edges at 0 / duration are not padded (no keep beyond them)."""
    start, end = cut.start, cut.end
    if cut.pad in ("both", "left") and start > 0:
        start += PAD
    if cut.pad in ("both", "right") and end < duration:
        end -= PAD
    return start, end


def _merge_and_snap(cuts: list[_Cut], duration: float, fps: float) -> list[dict]:
    """Pad, snap to frames, drop sub-minimum cuts, and merge overlaps.

    Returns a list of drop dicts ``{start, end, reason, confidence, note}``.
    """
    padded: list[dict] = []
    for c in cuts:
        start, end = _apply_padding(c, duration)
        start = edl.snap(max(0.0, start), fps)
        end = edl.snap(min(duration, end), fps)
        if end - start < MIN_DROP:
            continue
        padded.append({
            "start": start, "end": end, "reason": c.reason,
            "confidence": c.confidence, "note": c.note,
        })

    padded.sort(key=lambda d: d["start"])

    merged: list[dict] = []
    for d in padded:
        if merged and d["start"] <= merged[-1]["end"] + 1e-9:
            prev = merged[-1]
            prev["end"] = max(prev["end"], d["end"])
            if d["reason"] != prev["reason"]:
                prev["reason"] = "multiple"
                prev["note"] = "merged cuts"
            prev["confidence"] = min(prev["confidence"], d["confidence"])
        else:
            merged.append(dict(d))
    return merged


def _build_segments(drops: list[dict], duration: float, fps: float) -> list[dict]:
    """Tile [0, duration] with alternating keep/drop segments in source order."""
    segments: list[dict] = []
    cursor = edl.snap(0.0, fps)
    counter = 1

    def add(seg_in, seg_out, action, **extra):
        nonlocal counter
        if seg_out - seg_in < 1e-6:
            return
        out_val = round(seg_out, 3)
        # The final keep lands on the last frame boundary, which sits a hair
        # below the source duration; rounding its out up to 3dp can nudge it a
        # fraction of a millisecond *past* the (6dp) source duration and trip
        # validate's out-vs-duration check. Never emit an out beyond the source
        # — floor to 3dp in that case instead of rounding to nearest.
        if out_val > duration:
            out_val = math.floor(seg_out * 1000) / 1000
        seg = {"id": f"s{counter:03d}", "in": round(seg_in, 3),
               "out": out_val, "action": action}
        seg.update(extra)
        segments.append(seg)
        counter += 1

    for d in drops:
        if d["start"] > cursor + 1e-6:
            add(cursor, d["start"], "keep")
        add(d["start"], d["end"], "drop",
            reason=d["reason"], confidence=round(d["confidence"], 2), note=d["note"])
        cursor = d["end"]

    # Keep to the end of the source: snap to a frame boundary, but clamp to the
    # source duration so a frame boundary that rounds up can never overshoot it.
    end = min(edl.snap(duration, fps), duration)
    if end > cursor + 1e-6:
        add(cursor, end, "keep")
    return segments


def autoauthor(ep: Episode) -> dict[str, Any]:
    """Produce a deterministic baseline EDL and write it to ``edl.json``.

    Existing human overrides (``override: true``) are preserved — re-running
    stage 3 must never clobber a review veto (spec section 6).
    """
    words_doc, silence_doc, probe = load_inputs(ep)
    words = words_doc.get("words", [])
    silences = silence_doc.get("silences", [])
    fps = float(probe["fps"])
    duration = float(probe["source_duration"])

    cuts = _discover_cuts(words, silences, duration)
    drops = _merge_and_snap(cuts, duration, fps)
    segments = _build_segments(drops, duration, fps)

    new_edl: dict[str, Any] = {
        "version": edl.SCHEMA_VERSION,
        "episode_id": ep.episode_id,
        "source": str(ep.mezz).replace("\\", "/"),
        "fps": fps,
        "segments": segments,
        "overlays": [],
        "grade": "luts/skepticus_v1.cube",
        "captions": {"style": "styles/captions.ass.template", "enabled": True},
    }

    # Preserve overrides from any prior EDL.
    if ep.edl_json.exists():
        try:
            prior = edl.load(ep.edl_json)
            overrides = edl.collect_overrides(prior)
            new_edl = edl.apply_overrides(new_edl, overrides)
            if overrides:
                log.info("autoauthor: preserved %d override(s)", len(overrides))
        except (json.JSONDecodeError, OSError):
            log.warning("autoauthor: could not read prior EDL; ignoring overrides")

    edl.save(new_edl, ep.edl_json)
    log.info(
        "autoauthor: %d segments (%d drops) -> %s",
        len(segments), sum(1 for s in segments if s["action"] == "drop"), ep.edl_json,
    )
    return new_edl


def validate(ep: Episode) -> list[str]:
    """Validate ``edl.json`` against schema + source-timebase invariants."""
    if not ep.edl_json.exists():
        raise FileNotFoundError(f"No EDL at {ep.edl_json}.")
    edl_doc = edl.load(ep.edl_json)
    source_duration = None
    if ep.probe_json.exists():
        source_duration = json.loads(ep.probe_json.read_text(encoding="utf-8")).get("source_duration")
    return edl.validate(edl_doc, source_duration=source_duration)
