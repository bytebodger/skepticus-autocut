"""Reaction format — cue-leak verification (reaction spec section 2, 6).

The "tary"/"ary" bugs (host_in/host_out and then the cue-drop edges
themselves landing on a raw, early-running Whisper word boundary instead of
confirmed silence) were each found by a human noticing an audible fragment
in one spot-checked transition. With 46 cue transitions in a full episode,
listening to every one doesn't scale. This automates it: after a render,
re-transcribe a short window of the OUTPUT audio around every cue
transition and check whether any word or fragment of the cue phrase
survived the cut. Same verification-harness philosophy as align.py's
lip-sync check clips (spec section 11) — applied to the cut boundary
instead of the playback offset.

A transition is where a cue's own commentary-side edge was cut: for a
start cue ("end my commentary"), that's where the kept PLAYBACK segment
begins in output time; for a stop cue ("begin my commentary"), where the
kept COMMENTARY segment resumes. Both are exactly the boundary between two
adjacent kept EDL segments, so no re-detection of cue text positions is
needed here — only their host_in/host_out, already in playback.json.

Only the COMMENTARY side of each transition is checked, never the
playback/source side: a start cue's risk window is BEFORE it (the
commentary tail the cut was supposed to remove entirely); a stop cue's is
AFTER it (the resumed commentary the cut was supposed to start clean). The
other side is the source video's own arbitrary audio — checking it too
produced a real false positive in testing (a word the source video happened
to say, unrelated to any cue, matched a cue-word fragment) with no
corresponding way for it to ever indicate an actual leak, so it isn't
checked.

Word matching is a heuristic, deliberately generous: a transcribed word
counts as a leak if it or a cue word is a substring of the other (catches
"ary"/"tary" as fragments of "commentary", not just the whole word). False
positives just mean re-listening to a clip that turns out fine — the
failure mode this exists to prevent (an audible fragment nobody checks) is
far worse than a little extra listening.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from . import edl, ffmpeg
from .paths import Episode

log = logging.getLogger("autocut.cuecheck")

CHECK_WINDOW = 1.5    # seconds of output audio to check, on the commentary side only
MIN_FRAGMENT_LEN = 2  # ignore matches shorter than this (avoids single-letter noise)


def _norm(token: str) -> str:
    return re.sub(r"[^a-z]", "", token.lower())


def _cue_words(ep: Episode) -> set[str]:
    from . import align as align_mod  # lazy: keeps this importable without yaml
    cfg = align_mod._reaction_config(ep)
    phrases = [cfg["cue_playback_start"], cfg["cue_playback_stop"], *cfg["cue_playback_start_variants"]]
    return {_norm(w) for p in phrases for w in p.split() if _norm(w)}


def _transitions(playback_segments: list[dict]) -> list[dict]:
    """One entry per cue, each carrying the host-timeline (source) point
    where its own commentary-side edge was cut — a start cue's host_in
    (entering playback) and a stop cue's host_out (resuming commentary)."""
    out: list[dict] = []
    for seg in playback_segments:
        out.append({"id": seg["id"], "kind": "start", "source_t": seg["host_in"]})
        out.append({"id": seg["id"], "kind": "stop", "source_t": seg["host_out"]})
    return out


def _output_position_at_or_after(spans, t: float) -> float | None:
    """The output-time start of the kept span at/after source time ``t``.

    Not just ``edl.source_to_output(t, spans)``: a start cue's host_in is
    itself a kept span's own source_in (a direct hit), but a stop cue's
    host_out is that segment's source *end* — outside the half-open
    ``[src_in, src_out)`` interval ``source_to_output`` checks. Either way,
    the transition is "wherever the next kept span begins", which this
    finds directly regardless of which case it is.
    """
    candidates = [out_start for src_in, _src_out, out_start in spans if src_in >= t - 1e-6]
    return min(candidates) if candidates else None


def _leaks(words: list[dict], cue_words: set[str]) -> list[dict]:
    hits = []
    for w in words:
        norm = _norm(w.get("word", ""))
        if len(norm) < MIN_FRAGMENT_LEN:
            continue
        for cue_w in cue_words:
            if norm in cue_w or cue_w in norm:
                hits.append({"word": w.get("word"), "start": w.get("start"), "matched": cue_w})
                break
    return hits


def _output_duration(output_path: Path) -> float:
    data = ffmpeg.ffprobe_json(["-show_format", str(output_path)])
    return float(data.get("format", {}).get("duration", 0.0))


def run(ep: Episode, output_path: Path | None = None) -> dict:
    """Check every cue transition that falls within ``output_path``'s
    duration (default: the last compose render) for a surviving fragment of
    the cue phrase. Returns
    ``{"checked": int, "skipped": int, "violations": [...]}``; also writes
    the same to ``cuecheck_report_json``.
    """
    from . import transcribe as transcribe_mod  # lazy: keeps this importable without faster_whisper

    output_path = output_path or ep.compose_output
    if not output_path.exists():
        raise FileNotFoundError(f"No rendered output at {output_path}. Run 'autocut compose {ep.episode_id}' first.")
    if not ep.playback_json.exists():
        raise FileNotFoundError(f"No playback map at {ep.playback_json}. Not a reaction episode?")
    if not ep.edl_json.exists():
        raise FileNotFoundError(f"No EDL at {ep.edl_json}. Run 'autocut autoauthor {ep.episode_id}' first.")

    playback = json.loads(ep.playback_json.read_text(encoding="utf-8"))
    edl_doc = edl.load(ep.edl_json)
    spans = edl.build_time_map(edl_doc["segments"])
    cue_words = _cue_words(ep)
    out_dur = _output_duration(output_path)

    checked: list[dict] = []
    skipped = 0
    for t in _transitions(playback.get("segments", [])):
        pos = _output_position_at_or_after(spans, t["source_t"])
        if pos is None or pos >= out_dur - 1e-3:
            skipped += 1
            continue
        checked.append({**t, "output_t": pos})

    if not checked:
        log.warning("cuecheck: no cue transitions fall within %s (%.1fs); nothing verified",
                    output_path.name, out_dur)
        report = {"checked": 0, "skipped": skipped, "violations": []}
        ep.cuecheck_dir.mkdir(parents=True, exist_ok=True)
        ep.cuecheck_report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report

    ep.cuecheck_dir.mkdir(parents=True, exist_ok=True)
    violations = []
    for t in checked:
        if t["kind"] == "start":
            # Risk window is BEFORE entering playback — the commentary tail
            # the cut was supposed to remove entirely.
            ss = max(0.0, t["output_t"] - CHECK_WINDOW)
            dur = t["output_t"] - ss
        else:
            # Risk window is AFTER resuming commentary — never the playback
            # tail before it, which is the source video's own arbitrary
            # audio and can't indicate a cue leak (see module docstring).
            ss = t["output_t"]
            dur = min(CHECK_WINDOW, out_dur - ss)
        if dur <= 0.05:
            continue
        clip = ep.cuecheck_dir / f"{t['id']}_{t['kind']}.wav"
        ffmpeg.run_ffmpeg([
            "-ss", f"{ss:.3f}", "-t", f"{dur:.3f}", "-i", str(output_path),
            "-map", "0:a:0", "-ac", "1", "-ar", "16000",
            clip,
        ])
        doc = transcribe_mod.transcribe_wav(clip)
        words = list(doc.get("words", [])) + [{"word": x["text"]} for x in doc.get("isolated", [])]
        hits = _leaks(words, cue_words)
        if hits:
            violations.append({"id": t["id"], "kind": t["kind"], "output_t": round(t["output_t"], 3),
                               "clip": str(clip), "hits": hits})

    report = {"checked": len(checked), "skipped": skipped, "violations": violations}
    ep.cuecheck_report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if violations:
        log.warning("cuecheck: %d/%d transition(s) have a surviving cue fragment — see %s",
                    len(violations), len(checked), ep.cuecheck_report_json)
        for v in violations:
            words_txt = ", ".join(f"{h['word']!r}~{h['matched']!r}" for h in v["hits"])
            log.warning("  %s (%s cue) @ output %.3fs: %s -> %s",
                       v["id"], v["kind"], v["output_t"], words_txt, v["clip"])
    else:
        log.info("cuecheck: %d transition(s) checked, no surviving fragments (%d skipped, out of window)",
                 len(checked), skipped)
    return report
