"""Reaction format — playback sync verification (reaction spec section 4).

Same idea as cuecheck.py, aimed at a different failure mode: instead of
checking whether a cue phrase survived a cut, this checks whether the
cumulative source position (align.py's arithmetic, refined or not) is
actually right. Captions during playback come from the host bleed
transcript mapped straight through the EDL — correct by construction,
regardless of any playback-arithmetic bug — so they're a source of ground
truth: for every host bleed word inside a playback segment, the arithmetic
mapping predicts a source time; the source's own transcript says what time
that word actually occurs at. The difference is the arithmetic's error.

Per segment: transcribe the source audio for its own ``[source_in,
source_out)`` span, align it against the host bleed words already in
``words.json`` for ``[host_in, host_out)`` by normalised-token sequence
matching (the two transcripts of the same content won't match word-for-word
— bleed is degraded, going through speakers/a room/a mic — but they're in
the same order, so a longest-common-subsequence-style alignment finds the
genuine correspondences), and report the MEDIAN predicted-vs-actual offset.
Nea-zero everywhere is the goal; a growing offset across segments is exactly
the accumulating arithmetic-drift bug this exists to catch.
"""

from __future__ import annotations

import difflib
import json
import logging
import re
from pathlib import Path

from . import edl
from .paths import Episode

log = logging.getLogger("autocut.sync_check")

MIN_MATCHED_WORDS = 4  # below this, a segment's median offset is too noisy to trust


def _norm(token: str) -> str:
    return re.sub(r"[^a-z]", "", token.lower())


def _aligned_offsets(host_words: list[dict], source_words: list[dict],
                     host_in: float, source_in: float) -> list[float]:
    """predicted-vs-actual offset (seconds) for each word the two transcripts
    agree on, via normalised-token sequence alignment (the transcripts are
    the same content in the same order, just not word-for-word identical —
    bleed degrades some words Whisper gets right in the clean source)."""
    host_tokens = [_norm(w["word"]) for w in host_words]
    source_tokens = [_norm(w["word"]) for w in source_words]
    matcher = difflib.SequenceMatcher(a=host_tokens, b=source_tokens, autojunk=False)
    offsets = []
    for block in matcher.get_matching_blocks():
        for k in range(block.size):
            hw = host_words[block.a + k]
            sw = source_words[block.b + k]
            predicted = source_in + (float(hw["start"]) - host_in)
            offsets.append(float(sw["start"]) - predicted)
    return offsets


def _segments_in_window(segments: list[dict], ep: Episode, window: tuple[float, float]) -> list[dict]:
    """Segments whose output-time position (via the EDL's source->output map)
    falls inside ``window`` — the same output-time window a ``compose
    --preview``/``--range`` render covers. Lets a quick preview run sync-check
    only against what was actually rendered, instead of re-transcribing every
    segment's source audio for the whole episode."""
    edl_doc = edl.load(ep.edl_json)
    spans = edl.build_time_map(edl_doc["segments"])
    w0, dur = window
    in_window = []
    for seg in segments:
        out_t = edl.source_to_output(seg["host_in"], spans)
        if out_t is not None and w0 - 1e-6 <= out_t < w0 + dur + 1e-6:
            in_window.append(seg)
    return in_window


def run(ep: Episode, *, window: tuple[float, float] | None = None) -> dict:
    """Check every playback segment's cumulative source position against
    ground truth (source's own transcript). Returns
    ``{"segments": [{"id", "n_matched", "median_offset", "spread"}, ...]}``;
    also written to ``sync_check_report_json``.

    ``window`` (output-time ``(start, length)``) restricts the check to
    segments landing inside it — for checking only what a ``compose
    --preview``/``--range`` render actually produced, without re-transcribing
    the whole episode's source audio. ``None`` (default, and the CLI's
    behaviour) checks every segment.
    """
    from . import transcribe as transcribe_mod  # lazy: keeps this importable without faster_whisper
    from . import ffmpeg

    if not ep.playback_json.exists():
        raise FileNotFoundError(f"No playback map at {ep.playback_json}. Not a reaction episode?")
    if not ep.words_json.exists():
        raise FileNotFoundError(f"Missing {ep.words_json}. Run transcribe first.")

    playback = json.loads(ep.playback_json.read_text(encoding="utf-8"))
    words_doc = json.loads(ep.words_json.read_text(encoding="utf-8"))
    host_words_all = words_doc.get("words", [])
    source_path = Path(playback["source_file"])

    segments_all = playback.get("segments", [])
    segments = _segments_in_window(segments_all, ep, window) if window is not None else segments_all
    if window is not None:
        log.info("sync-check: window %.1f-%.1fs -> %d/%d segment(s) in range",
                 window[0], window[0] + window[1], len(segments), len(segments_all))

    ep.sync_check_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for seg in segments:
        host_in, host_out = seg["host_in"], seg["host_out"]
        source_in, source_out = seg["source_in"], seg["source_out"]
        host_words = [w for w in host_words_all if host_in <= float(w["start"]) < host_out]

        clip = ep.sync_check_dir / f"{seg['id']}_source.wav"
        ffmpeg.run_ffmpeg([
            "-ss", f"{source_in:.3f}", "-to", f"{source_out:.3f}", "-i", str(source_path),
            "-map", "0:a:0", "-ac", "1", "-ar", "16000",
            clip,
        ])
        source_doc = transcribe_mod.transcribe_wav(clip)
        source_words = [{"word": w["word"], "start": float(w["start"]) + source_in}
                        for w in source_doc.get("words", [])]

        offsets = _aligned_offsets(host_words, source_words, host_in, source_in)
        if len(offsets) < MIN_MATCHED_WORDS:
            log.warning("sync-check: %s only %d matched word(s) — too few to trust",
                       seg["id"], len(offsets))
            results.append({"id": seg["id"], "n_matched": len(offsets),
                           "median_offset": None, "spread": None})
            continue
        offsets.sort()
        n = len(offsets)
        median = offsets[n // 2] if n % 2 else (offsets[n // 2 - 1] + offsets[n // 2]) / 2
        spread = offsets[-1] - offsets[0]
        results.append({"id": seg["id"], "n_matched": n,
                       "median_offset": round(median, 3), "spread": round(spread, 3)})

    report = {"segments": results}
    ep.sync_check_report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    log.info("sync-check: %d segment(s) checked", len(results))
    for r in results:
        if r["median_offset"] is None:
            log.info("  %s: %d word(s) matched — too few to trust", r["id"], r["n_matched"])
        else:
            log.info("  %s: median offset %+.3fs (spread %.3fs, %d word(s))",
                     r["id"], r["median_offset"], r["spread"], r["n_matched"])
    return report
