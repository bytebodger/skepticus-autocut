"""Stage 7 — captions.

Generate ASS subtitles directly from words.json. Not HyperFrames: FFmpeg's
subtitle burner does a full episode in ~90s, versus tens of thousands of browser
frames for a visually similar result.

Word timing is mapped through ``source_to_output`` so captions stay aligned after
cuts. Words that fell inside a dropped region are skipped — they no longer exist
on the output timeline. ASS ``\\k`` karaoke tags give word-by-word highlighting;
durations are in centiseconds.
"""

from __future__ import annotations

import json
import logging

from . import cache, edl
from .paths import Episode

log = logging.getLogger("autocut.captions")

STYLE_NAME = "Skepticus"

# Chunking: group words into 2-4s display chunks, breaking on punctuation and
# natural pauses. Never break mid-clause if it can be helped.
MAX_CHUNK = 4.0       # seconds of output time
MIN_FOR_BREAK = 1.8   # only honour a punctuation break past this length
PAUSE_BREAK = 0.6     # a silent gap this long forces a chunk break
CLAUSE_PUNCT = (",", ";", ":")
TERMINAL_PUNCT = (".", "?", "!")


def _fmt_time(t: float) -> str:
    """Format seconds as ASS H:MM:SS.cc (centisecond precision)."""
    if t < 0:
        t = 0.0
    cs = int(round(t * 100))
    h, cs = divmod(cs, 360000)
    m, cs = divmod(cs, 6000)
    s, cs = divmod(cs, 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def _escape(text: str) -> str:
    # Braces start override blocks in ASS; neutralise any that appear in words.
    return text.replace("{", "(").replace("}", ")")


def _map_words(words: list[dict], spans: list[edl.Span]) -> list[dict]:
    """Return words with output-time start/end, dropping any that were cut."""
    out = []
    for w in words:
        os_ = edl.source_to_output(w["start"], spans)
        oe = edl.source_to_output(max(w["start"], w["end"] - 1e-6), spans)
        if os_ is None or oe is None:
            continue
        out.append({"word": w["word"].strip(), "start": os_, "end": max(oe, os_)})
    return out


def _chunk(words: list[dict]) -> list[list[dict]]:
    """Split output-time words into display chunks."""
    chunks: list[list[dict]] = []
    cur: list[dict] = []
    for i, w in enumerate(words):
        cur.append(w)
        chunk_len = w["end"] - cur[0]["start"]
        nxt = words[i + 1] if i + 1 < len(words) else None
        gap_next = (nxt["start"] - w["end"]) if nxt else 0.0
        ends_terminal = w["word"].endswith(TERMINAL_PUNCT)
        ends_clause = w["word"].endswith(CLAUSE_PUNCT)

        should_break = (
            nxt is None
            or chunk_len >= MAX_CHUNK
            or gap_next >= PAUSE_BREAK
            or (ends_terminal and chunk_len >= MIN_FOR_BREAK)
            or (ends_clause and chunk_len >= MAX_CHUNK * 0.75)
        )
        if should_break:
            chunks.append(cur)
            cur = []
    if cur:
        chunks.append(cur)
    return chunks


def _dialogue_line(chunk: list[dict]) -> str:
    start = chunk[0]["start"]
    end = chunk[-1]["end"]
    parts = []
    for i, w in enumerate(chunk):
        if i + 1 < len(chunk):
            # Absorb the gap to the next word so the karaoke sweep tracks timing.
            k = round((chunk[i + 1]["start"] - w["start"]) * 100)
        else:
            k = round((w["end"] - w["start"]) * 100)
        k = max(1, k)  # \k must be positive
        parts.append(f"{{\\k{k}}}{_escape(w['word'])}")
    text = " ".join(parts)
    return f"Dialogue: 0,{_fmt_time(start)},{_fmt_time(end)},{STYLE_NAME},,0,0,0,,{text}"


def build_ass(words_doc: dict, edl_doc: dict, template: str) -> str:
    """Build a full ASS document. Pure function — unit-tested."""
    spans = edl.build_time_map(edl_doc["segments"])
    mapped = _map_words(words_doc.get("words", []), spans)
    chunks = _chunk(mapped)
    dialogue = "\n".join(_dialogue_line(c) for c in chunks)
    body = template.rstrip("\n") + "\n"
    return body + dialogue + "\n"


def run(ep: Episode, *, force: bool = False) -> None:
    """Execute stage 7. Writes captions.ass. Cached on words + edl + template."""
    edl_doc = edl.load(ep.edl_json)
    if not edl_doc.get("captions", {}).get("enabled", True):
        log.info("captions: disabled in EDL; skipping")
        return
    if not ep.words_json.exists():
        raise FileNotFoundError(f"No transcript at {ep.words_json}.")

    template_rel = edl_doc.get("captions", {}).get("style", "styles/captions.ass.template")
    template_file = ep.root / template_rel
    if not template_file.exists():
        raise FileNotFoundError(f"Caption style template not found: {template_file}")

    words_doc = json.loads(ep.words_json.read_text(encoding="utf-8"))
    template = template_file.read_text(encoding="utf-8")

    stage_dir = ep.work / "captions"
    input_hash = cache.hash_inputs({
        "words": cache.hash_file(ep.words_json),
        "segments": edl_doc["segments"],
        "template": template,
    })
    if not force and cache.is_current(stage_dir, input_hash) and ep.captions_ass.exists():
        log.info("captions: cache hit, skipping")
        return

    ass = build_ass(words_doc, edl_doc, template)
    ep.captions_ass.write_text(ass, encoding="utf-8")
    log.info("captions: wrote %s", ep.captions_ass)
    cache.mark_done(stage_dir, input_hash, extra={"stage": "captions"})
