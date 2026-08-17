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


def _ass_color(value: str) -> str:
    """Normalise a config colour to ASS ``&HAABBGGRR`` (opaque). Accepts an RGB
    ``#RRGGBB`` (converted — ASS is BBGGRR, not RGB) or a literal ``&H...``."""
    v = str(value).strip()
    if v.upper().startswith("&H"):
        h = v[2:]
        return "&H" + (("00" + h) if len(h) == 6 else h).upper()
    h = v.lstrip("#")
    r, g, b = h[0:2], h[2:4], h[4:6]
    return f"&H00{b}{g}{r}".upper()


def style_header_from_config(cfg: dict, play_w: int, play_h: int) -> str:
    """Build the ASS header + style block from the layout ``captions`` config,
    sized for the given canvas (4K). build_ass appends the Dialogue lines.

    Karaoke colours: the current/spoken word is PrimaryColour (highlight), the
    not-yet-spoken text is SecondaryColour (base) — per spec section 6.
    """
    wh = cfg.get("word_highlight") or {}
    primary = _ass_color(wh.get("highlight_color", "#40FF40"))   # bright green
    secondary = _ass_color(wh.get("base_color", "#FFFFFF"))       # white
    outline_c = _ass_color(cfg.get("outline_color", "#000000"))
    font = cfg.get("font", "Arial")
    size = int(cfg.get("font_size", 100))
    outline = cfg.get("outline", 4)
    shadow = cfg.get("shadow", 3)
    margin_v = int(cfg.get("margin_bottom", 100))
    return (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        f"PlayResX: {play_w}\n"
        f"PlayResY: {play_h}\n"
        "WrapStyle: 2\n"
        "ScaledBorderAndShadow: yes\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, "
        "ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, "
        "MarginL, MarginR, MarginV, Encoding\n"
        f"Style: {STYLE_NAME},{font},{size},{primary},{secondary},{outline_c},"
        f"&H64000000,-1,0,0,0,100,100,0,0,1,{outline},{shadow},2,120,120,{margin_v},1\n\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )


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
    # \k durations are centiseconds and accumulate across the line. Rounding each
    # word's duration independently lets the error compound, and the highlight
    # slides off the audio by the end of a long line. Instead compute each word's
    # cumulative position (rounded once from the absolute time) and take
    # differences, so the sweep lands exactly on the line end.
    bounds = []
    for i, w in enumerate(chunk):
        edge = chunk[i + 1]["start"] if i + 1 < len(chunk) else w["end"]
        bounds.append(round((edge - start) * 100))
    parts = []
    prev = 0
    for i, w in enumerate(chunk):
        k = max(1, bounds[i] - prev)  # \k must be positive
        prev = bounds[i]              # track the true boundary, not the clamped sum
        parts.append(f"{{\\k{k}}}{_escape(w['word'])}")
    text = " ".join(parts)
    return f"Dialogue: 0,{_fmt_time(start)},{_fmt_time(end)},{STYLE_NAME},,0,0,0,,{text}"


def build_ass(words_doc: dict, edl_doc: dict, template: str, *, window: tuple | None = None) -> str:
    """Build a full ASS document. Pure function — unit-tested.

    ``window`` = (start, length) in output time renders only that window, with
    caption times shifted so the window start is 0 (so a --range/--preview clip's
    captions line up with its shifted footage)."""
    spans = edl.build_time_map(edl_doc["segments"])
    mapped = _map_words(words_doc.get("words", []), spans)
    if window is not None:
        w0, length = window
        clipped = []
        for m in mapped:
            if not (w0 - 1e-6 <= m["start"] < w0 + length + 1e-6):
                continue
            s = max(0.0, m["start"] - w0)
            clipped.append({"word": m["word"], "start": s,
                            "end": max(s, min(m["end"] - w0, length))})
        mapped = clipped
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
