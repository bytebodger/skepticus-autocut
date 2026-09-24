"""Reaction format — alignment stage (reaction spec sections 2-5, 12).

**This revision replaces transcript-based playback detection.** Matching the
host's bleed transcript against the source's found playback regions, but
couldn't tell a real play from the host quoting or paraphrasing the source —
which looks identical in text and happens constantly on a reaction channel.
That false-positive class explained both the implausible segment counts and
much of the weak-correlation population from the old approach.

The replacement: the host declares playback boundaries out loud, using the
same explicit-marker-cue principle as retake detection (``autocut.retakes``).

  1. **Cue detection.** Two distinct phrases — one ends commentary and starts
     playback, the other ends playback and resumes commentary — matched as
     complete phrases in the host transcript. Two cues, not one toggle: a
     missed cue then breaks one boundary, not every boundary after it.
  2. **Alternation validation.** Cues must alternate start, stop, start, stop,
     ... starting with 'start' (the recording opens in commentary). Any
     violation means a cue was missed or misheard; fail loudly with every
     violating timestamp rather than guess which one.
  3. **Cumulative source position.** Pausing stops the source clock, so
     segment N's source span resumes exactly where segment N-1's stopped.
     Pure arithmetic, no correlation — cheap and exact if cue timing itself is
     exact.

Only build-order steps 1-3 live here (reaction spec section 12). Refinement by
cross-correlation (step 4), autoauthor constraints, and the later stages are
not touched: the point of stopping here is to see whether the arithmetic
estimate is accurate enough on its own before adding correlation at all.

Output: ``work/<ep>/playback.json`` (spec section 5). ``source_out −
source_in`` equals ``host_out − host_in`` by construction; asserted anyway as
a safety net against a coding error.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import yaml

from . import cache, edl, ffmpeg, retakes
from .paths import Episode

log = logging.getLogger("autocut.align")

# --- cue phrases (reaction spec §2, §9) ---
# Overridable via config/layout.yaml's `reaction:` section. Avoid single words
# that occur naturally in the subject matter — a false hit inverts the audio
# state for everything downstream (spec §2).
DEFAULT_CUE_PLAYBACK_START = "end my commentary"   # ends commentary, starts playback
DEFAULT_CUE_PLAYBACK_STOP = "begin my commentary"  # ends playback, resumes commentary

# Whisper regularly mishears "end" as "and" in the playback-start cue; accepted
# as a variant so those instances still fire. Alternation validation stays the
# safety net — a genuine "and my commentary" in ordinary speech would break
# alternation and fail loudly rather than silently misfire (spec §2).
DEFAULT_CUE_PLAYBACK_START_VARIANTS = ("and my commentary",)


# --------------------------------------------------------------------------- #
# Step 1 — cue detection (spec §2). Pure logic, reuses the retake cue matcher.
# --------------------------------------------------------------------------- #

def _reaction_config(ep: Episode) -> dict:
    """The ``reaction:`` section of config/layout.yaml, defaulting to the
    spec's example phrases when the file or section is absent."""
    cfg_path = ep.root / "config" / "layout.yaml"
    data = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}
    reaction = (data or {}).get("reaction", {}) or {}
    return {
        "cue_playback_start": reaction.get("cue_playback_start", DEFAULT_CUE_PLAYBACK_START),
        "cue_playback_start_variants": reaction.get(
            "cue_playback_start_variants", list(DEFAULT_CUE_PLAYBACK_START_VARIANTS)),
        "cue_playback_stop": reaction.get("cue_playback_stop", DEFAULT_CUE_PLAYBACK_STOP),
    }


def detect_cues(words_doc: dict, *, start_phrase: str, stop_phrase: str,
                start_variants: tuple[str, ...] | list[str] = ()) -> list[dict]:
    """Both spoken cues as complete-phrase matches in the host transcript,
    sorted by time. Each cue is ``{start, end, kind}``, ``kind`` one of:

    - ``"start"`` — the end-commentary cue: commentary ends, playback starts.
    - ``"stop"`` — the start-commentary cue: playback ends, commentary resumes.

    Matched as whole phrases (spec §2 "Avoid 'commentary'" — a single-word
    match would misfire on ordinary subject-matter speech), via the same
    normalised-token phrase matcher retake markers use. ``start_variants``
    are additional phrasings (e.g. mishearings) that also count as the
    playback-start cue.
    """
    words = words_doc.get("words", [])
    start_phrases = [start_phrase, *start_variants]
    start_toks = [[retakes.norm_token(t) for t in p.split()] for p in start_phrases]
    stop_toks = [retakes.norm_token(t) for t in stop_phrase.split()]
    raw: list[tuple[float, float, str]] = []
    for s, e in retakes.phrase_hits(words, start_toks):
        raw.append((s, e, "start"))
    for s, e in retakes.phrase_hits(words, [stop_toks]):
        raw.append((s, e, "stop"))
    raw.sort()
    return [{"start": s, "end": e, "kind": k} for s, e, k in raw]


# --------------------------------------------------------------------------- #
# Step 1 — alternation validation (spec §2).
# --------------------------------------------------------------------------- #

def validate_alternation(cues: list[dict]) -> None:
    """Cues must alternate start, stop, start, stop, ... The recording opens
    in commentary (spec §2), so the first cue must be a 'start' cue.

    Two cues of the same kind in a row (or a leading 'stop') means one was
    missed or misheard. Fails loudly with every violating timestamp rather
    than guessing which cue was the problem (spec §2, §14) — a silent guess
    here is exactly the failure mode explicit cues exist to avoid.
    """
    violations: list[tuple[dict | None, dict]] = []
    expected = "start"
    prev: dict | None = None
    for c in cues:
        if c["kind"] != expected:
            violations.append((prev, c))
        expected = "stop" if c["kind"] == "start" else "start"
        prev = c
    if violations:
        lines = []
        for prev_c, bad_c in violations:
            prev_desc = (f"'{prev_c['kind']}' cue ending {prev_c['end']:.2f}s"
                        if prev_c is not None
                        else "recording start (assumed to open in commentary)")
            lines.append(f"  {prev_desc} -> unexpected '{bad_c['kind']}' cue "
                        f"at {bad_c['start']:.2f}s")
        raise RuntimeError(
            f"align: cue alternation broken — a cue was missed or misheard "
            f"(spec §2). {len(violations)} violation(s):\n" + "\n".join(lines)
        )


# --------------------------------------------------------------------------- #
# Step 2 — cumulative source position (spec §3). Arithmetic only.
# --------------------------------------------------------------------------- #

# Whisper's word timestamps consistently run early in this project (a known
# issue — see autocut.retakes' own silence-fallback logic). Trusting the raw
# cue-word timestamp for host_in leaves the tail of the cue phrase (e.g.
# "-tary") audible at the top of the playback span. Both playback boundaries
# are instead nudged forward onto the nearest confirmed acoustic silence
# around the cue — never backward, since the correction is always "Whisper
# fired early," never late.
CUE_SNAP_MAX_GAP = 2.0  # don't snap across a gap this large; trust the raw cue time instead


def snap_forward_to_silence(t: float, silences: list[dict]) -> tuple[float, bool]:
    """The acoustic silence onset at/after ``t``: the start of the earliest
    confirmed silence interval that hasn't already ended by ``t`` (an
    interval already open at ``t`` counts — the cue timestamp can land inside
    real silence). Returns ``(value, found)`` — ``found`` is False, and
    ``value`` is just ``t`` unchanged, when no such interval is within
    ``CUE_SNAP_MAX_GAP`` (callers should not silently trust an unfound snap —
    see analyze.py's cue-drop fallback+flag handling)."""
    candidates = [iv for iv in silences if float(iv["end"]) >= t]
    if not candidates:
        return t, False
    nearest = min(candidates, key=lambda iv: float(iv["start"]))
    if float(nearest["start"]) - t > CUE_SNAP_MAX_GAP:
        return t, False
    return max(t, float(nearest["start"])), True


def snap_backward_to_silence(t: float, silences: list[dict]) -> tuple[float, bool]:
    """The acoustic silence offset at/before ``t``: the end of the latest
    confirmed silence interval that hasn't already started after ``t`` (an
    interval already open at ``t`` counts). Mirrors
    ``snap_forward_to_silence`` for the opposite direction — never returns
    later than ``t``. Returns ``(value, found)``; see that function for the
    ``found`` contract."""
    candidates = [iv for iv in silences if float(iv["start"]) <= t]
    if not candidates:
        return t, False
    nearest = max(candidates, key=lambda iv: float(iv["end"]))
    if t - float(nearest["end"]) > CUE_SNAP_MAX_GAP:
        return t, False
    return min(t, float(nearest["end"])), True


def snap_forward_to_speech(t: float, silences: list[dict]) -> float:
    """The acoustic speech onset at/after ``t``: the end of the latest
    confirmed silence interval that has already started by ``t`` (mirrors
    ``snap_forward_to_silence``). Falls back to ``t`` if no such interval is
    within ``CUE_SNAP_MAX_GAP``."""
    candidates = [iv for iv in silences if float(iv["start"]) <= t]
    if not candidates:
        return t
    nearest = max(candidates, key=lambda iv: float(iv["end"]))
    if float(nearest["end"]) - t > CUE_SNAP_MAX_GAP:
        return t
    return max(t, float(nearest["end"]))


def segments_from_cues(cues: list[dict], episode_duration: float,
                       silences: list[dict]) -> list[dict]:
    """Raw host-time playback spans from a validated, alternating cue
    sequence. Each 'start' cue opens a segment at the cue phrase's end
    (spec §2: cues are dropped from the output), snapped forward onto the
    confirmed silence right after it; the following 'stop' cue closes it at
    the phrase's start, snapped forward onto the confirmed speech onset right
    before it (both correct for Whisper's early timestamps — see
    ``snap_forward_to_silence``/``snap_forward_to_speech``). A recording
    that ends mid-playback (spec §2: "may end in either state") closes its
    last segment at the episode's own end instead of a cue.
    """
    segments: list[dict] = []
    pending_start: float | None = None
    for c in cues:
        if c["kind"] == "start":
            pending_start, _ = snap_forward_to_silence(c["end"], silences)
        else:
            host_out = snap_forward_to_speech(c["start"], silences)
            segments.append({"host_in": pending_start, "host_out": host_out})
            pending_start = None
    if pending_start is not None:
        segments.append({"host_in": pending_start, "host_out": episode_duration})
    return segments


def assign_source_positions(raw_segments: list[dict]) -> list[dict]:
    """Cumulative source position (spec §3): pausing stops the source clock,
    so segment N resumes exactly where segment N-1 stopped. Pure arithmetic —
    no correlation. This is the estimate whose accuracy build steps 1-3 exist
    to check, before refinement (spec §4, not built here) is added.
    """
    segments: list[dict] = []
    source_pos = 0.0
    for i, seg in enumerate(raw_segments):
        host_in, host_out = seg["host_in"], seg["host_out"]
        duration = host_out - host_in
        source_in = source_pos
        source_out = source_in + duration
        segments.append({
            "id": f"pb{i + 1:03d}",
            "host_in": round(host_in, 3),
            "host_out": round(host_out, 3),
            "source_in": round(source_in, 3),
            "source_out": round(source_out, 3),
            # Not yet corrected by refinement (spec §4, a later increment) —
            # this is the raw cumulative estimate build steps 1-3 are testing.
            "offset_source": "cumulative",
            "refinement_delta": None,
            "seek_detected": False,
        })
        source_pos = source_out
    return segments


def assert_equal_durations(segments: list[dict], *, tol: float = 2e-3) -> None:
    """Assert ``source_out − source_in == host_out − host_in`` for every
    segment (spec §5). True by construction from the cumulative arithmetic;
    kept as a safety net against a future coding error."""
    for s in segments:
        host_len = s["host_out"] - s["host_in"]
        source_len = s["source_out"] - s["source_in"]
        if abs(source_len - host_len) > tol:
            raise RuntimeError(
                f"align: {s['id']} source duration {source_len:.3f}s != host "
                f"duration {host_len:.3f}s (diff {abs(source_len - host_len) * 1000:.1f}ms)"
            )


# --------------------------------------------------------------------------- #
# Orchestration.
# --------------------------------------------------------------------------- #

def _load_host_words(ep: Episode) -> dict:
    if not ep.words_json.exists():
        raise FileNotFoundError(
            f"Missing {ep.words_json.name}. Run probe + transcribe for the host "
            f"recording {ep.episode_id!r} first."
        )
    return json.loads(ep.words_json.read_text(encoding="utf-8"))


def _load_host_silences(ep: Episode) -> list[dict]:
    if not ep.silence_json.exists():
        raise FileNotFoundError(
            f"Missing {ep.silence_json.name}. Run probe + transcribe for the host "
            f"recording {ep.episode_id!r} first."
        )
    data = json.loads(ep.silence_json.read_text(encoding="utf-8"))
    return data.get("silences", [])


def _load_or_build_source_words(ep: Episode, source_path: Path) -> list[dict]:
    """The source file's own full transcript, cached (spec §4's global
    text-locate needs it once per episode, not per segment — see refine.py).
    Transcribing an hour-plus source is genuinely slow; this is why it's
    cached to disk rather than redone on every align run."""
    if ep.source_words_json.exists():
        return json.loads(ep.source_words_json.read_text(encoding="utf-8")).get("words", [])
    from . import transcribe as transcribe_mod  # lazy: keeps align importable without faster_whisper
    log.info("align: no cached source transcript — transcribing %s (this can take a while)", source_path)
    audio_wav = ep.align_dir / "source_audio.wav"
    ep.align_dir.mkdir(parents=True, exist_ok=True)
    ffmpeg.run_ffmpeg([
        "-i", str(source_path), "-map", "0:a:0", "-ac", "1", "-ar", "16000", audio_wav,
    ])
    doc = transcribe_mod.transcribe_wav(audio_wav)
    ep.source_words_json.write_text(json.dumps(doc, indent=2, default=float), encoding="utf-8")
    return doc.get("words", [])


def _load_host_probe(ep: Episode) -> tuple[float, float]:
    """(episode duration, fps) from the host's probe.json."""
    if not ep.probe_json.exists():
        raise FileNotFoundError(
            f"Missing {ep.probe_json.name}. Run 'autocut probe {ep.episode_id}' first."
        )
    data = json.loads(ep.probe_json.read_text(encoding="utf-8"))
    return float(data["source_duration"]), float(data["fps"])


def _probe_source(source_path: Path) -> tuple[float, float]:
    """(fps, duration) of the source file, via ffprobe."""
    data = ffmpeg.ffprobe_json(["-show_streams", "-show_format", str(source_path)])
    streams = data.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), {})
    from fractions import Fraction
    fps = 30.0
    for key in ("r_frame_rate", "avg_frame_rate"):
        val = video.get(key)
        try:
            f = float(Fraction(val)) if val and val != "0/0" else 0.0
            if f > 0:
                fps = f
                break
        except (ValueError, ZeroDivisionError):
            continue
    dur = None
    for src in (video.get("duration"), data.get("format", {}).get("duration")):
        try:
            dur = float(src)
            break
        except (TypeError, ValueError):
            continue
    return fps, (dur if dur is not None else 0.0)


def _resolve_source(ep: Episode, source: str | Path | None) -> Path:
    path = Path(source) if source else ep.source_video
    if not path.exists() and not ffmpeg.is_dry_run():
        raise FileNotFoundError(
            f"No source video at {path}. Pass --source <path>, or drop it at "
            f"{ep.source_video} (inbox/{ep.episode_id}_source.<ext>)."
        )
    return path


def cues_for_episode(ep: Episode, words_doc: dict | None = None) -> list[dict]:
    """Detect and validate this episode's cue sequence (spec §2), using its
    configured (or default) phrases and variants. ``words_doc`` may be passed
    in to avoid re-reading ``words.json`` when the caller already has it.

    Used by both ``run()`` (to build ``playback.json``) and ``autoauthor``
    (spec §6 — it needs the cue phrases' own raw spans, which don't survive
    into ``playback.json``, to cut the cues themselves out of the output).
    """
    if words_doc is None:
        words_doc = _load_host_words(ep)
    cfg = _reaction_config(ep)
    start_phrase, stop_phrase = cfg["cue_playback_start"], cfg["cue_playback_stop"]
    start_variants = cfg["cue_playback_start_variants"]
    cue_list = detect_cues(words_doc, start_phrase=start_phrase, stop_phrase=stop_phrase,
                           start_variants=start_variants)
    n_start = sum(1 for c in cue_list if c["kind"] == "start")
    n_stop = len(cue_list) - n_start
    log.info("align: %d cue(s) detected (%d start, %d stop)", len(cue_list), n_start, n_stop)
    if not cue_list:
        raise RuntimeError(
            f"align: no cues detected. Configured phrases: start={start_phrase!r} "
            f"stop={stop_phrase!r}. Either this episode wasn't recorded with cues "
            f"(reaction spec §13 covers that case) or Whisper didn't transcribe "
            f"the phrases as configured — check {ep.words_json}."
        )
    validate_alternation(cue_list)
    return cue_list


def run(ep: Episode, source: str | Path | None = None, *, force: bool = False) -> dict:
    """Align a reaction episode from spoken cues (reaction spec build steps
    1-4): detect and validate the cue sequence, then cross-correlate the host
    bleed against the source to refine both the playback boundaries (bleed
    onset/offset, not the cue-adjacent silence build steps 1-3 used) and the
    cumulative source position (self-healing — spec §4). Under --dry-run,
    where there's no real audio to correlate against, falls back to the pure
    arithmetic estimate (steps 1-3) instead.
    """
    source_path = _resolve_source(ep, source)
    words_doc = _load_host_words(ep)
    silences = _load_host_silences(ep)
    episode_duration, host_fps = _load_host_probe(ep)
    _src_fps, source_dur = _probe_source(source_path) if not ffmpeg.is_dry_run() else (30.0, None)

    cue_list = cues_for_episode(ep, words_doc)

    raw_segments = segments_from_cues(cue_list, episode_duration, silences)
    for seg in raw_segments:
        seg["host_in"] = edl.snap(seg["host_in"], host_fps)
        seg["host_out"] = edl.snap(seg["host_out"], host_fps)

    if ffmpeg.is_dry_run():
        segments = assign_source_positions(raw_segments)
    else:
        from . import refine as refine_mod  # lazy: keeps align importable without numpy
        host_av = _host_av_source(ep)
        source_words = _load_or_build_source_words(ep, source_path)
        segments = refine_mod.refine_segments(
            host_av, source_path, raw_segments, host_fps, words_doc.get("words", []), source_words)
    assert_equal_durations(segments)

    if source_dur is not None and segments and segments[-1]["source_out"] > source_dur + 0.5:
        raise RuntimeError(
            f"align: cumulative playback ({segments[-1]['source_out']:.1f}s) "
            f"exceeds the source video's duration ({source_dur:.1f}s) by "
            f"{segments[-1]['source_out'] - source_dur:.1f}s. Either the wrong "
            f"source file is configured, or a cue was missed/misheard, "
            f"inflating a segment (spec §2, §14)."
        )

    playback = {
        "version": 1,
        "episode_id": ep.episode_id,
        "source_file": str(source_path).replace("\\", "/"),
        "segments": segments,
    }
    total_play = sum(s["host_out"] - s["host_in"] for s in segments)
    ep.playback_json.write_text(json.dumps(playback, indent=2), encoding="utf-8")
    log.info("align: %d playback segment(s), %.1fs of playback -> %s",
             len(segments), total_play, ep.playback_json)
    return playback


# --------------------------------------------------------------------------- #
# Step 3 — verification harness (spec §11) — the deliverable that matters most.
# --------------------------------------------------------------------------- #

CHECK_SECONDS = 8.0   # long enough to judge lip-sync confidently by eye/ear
CHECK_WIDTH = 1280   # downscale the (4K) source for a quick-to-render check clip


def _host_av_source(ep: Episode) -> Path:
    """The host media to pull bleed audio from for the check clip. The mezzanine
    shares the source/word timeline exactly; fall back to the raw drop."""
    if ep.mezz.exists():
        return ep.mezz
    if ep.raw.exists():
        return ep.raw
    raise FileNotFoundError(
        f"No host media for the check clip (neither {ep.mezz} nor {ep.raw}). "
        f"Run 'autocut probe {ep.episode_id}' first."
    )


def _render_check_clip(source_path: Path, host_av: Path, seg: dict, out: Path,
                       seconds: float) -> None:
    """Render one lip-sync clip: SOURCE video from source_in muxed with HOST mic
    (bleed) audio from host_in. If the offset is right the source speaker's lips
    match the bleed; if it's off, they visibly disagree — which numbers can't show.
    Source video with source audio would always look synced and prove nothing
    (spec §11) — the bleed audio is what's actually being verified.
    """
    ffmpeg.run_ffmpeg([
        "-ss", f"{seg['source_in']:.3f}", "-i", str(source_path),  # 0: source video
        "-ss", f"{seg['host_in']:.3f}", "-i", str(host_av),         # 1: host bleed audio
        "-map", "0:v:0", "-map", "1:a:0",
        "-t", f"{seconds:.3f}",
        "-vf", f"scale={CHECK_WIDTH}:-2",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "160k",
        "-movflags", "+faststart",
        out,
    ])


def _write_check_index(ep: Episode, source_path: Path, segments: list[dict],
                       seconds: float) -> None:
    lines = [
        f"# Lip-sync check — {ep.episode_id}",
        "",
        f"Each clip plays **{seconds:.0f}s of the source video** (from `source_in`) "
        f"with the **host mic audio** (the bleed, from `host_in`) laid over it.",
        "If alignment is right, the source speaker's lips match the bleed audio. "
        "If it's off, they visibly disagree — this is what tells you whether the "
        "cumulative arithmetic estimate is accurate enough on its own (spec §12).",
        "",
        "| clip | host_in | source_in | dur | offset_source | seek_detected |",
        "|------|--------:|----------:|----:|:--------------:|:-------------:|",
    ]
    for s in segments:
        dur = s["host_out"] - s["host_in"]
        lines.append(
            f"| {s['id']}.mp4 | {s['host_in']:.3f} | {s['source_in']:.3f} | "
            f"{dur:.1f}s | {s['offset_source']} | {s['seek_detected']} |"
        )
    lines += ["", f"Source: `{source_path}`", ""]
    (ep.align_check_dir / "index.md").write_text("\n".join(lines), encoding="utf-8")


def render_checks(ep: Episode, *, seconds: float = CHECK_SECONDS, force: bool = False) -> list[Path]:
    """Render the lip-sync verification clips for every playback segment.

    Reads ``playback.json``; writes ``work/<ep>/align/check/pbNNN.mp4`` + an
    index. This is the harness the format is verified by — numbers won't tell you
    whether the sync is right; a few seconds of video will (spec §11).
    """
    if not ep.playback_json.exists():
        raise FileNotFoundError(
            f"No playback map at {ep.playback_json}. Run 'autocut align "
            f"{ep.episode_id}' first."
        )
    playback = json.loads(ep.playback_json.read_text(encoding="utf-8"))
    segments = playback.get("segments", [])
    if not segments:
        log.warning("align-check: playback.json has no segments; nothing to render")
        return []

    source_path = Path(playback["source_file"])
    host_av = _host_av_source(ep)
    ep.align_check_dir.mkdir(parents=True, exist_ok=True)

    stage_dir = ep.align_check_dir
    dry = ffmpeg.is_dry_run()
    # Hashed once, not per segment — these are multi-GB files, and every
    # segment shares the same source/host_av regardless of its own timing.
    source_hash = cache.hash_file(source_path) if (source_path.exists() and not dry) else "dry"
    host_av_hash = cache.hash_file(host_av) if (host_av.exists() and not dry) else "dry"
    outputs: list[Path] = []
    for s in segments:
        out = ep.align_check_dir / f"{s['id']}.mp4"
        seg_hash = cache.hash_inputs({
            "source": source_hash,
            "host_av": host_av_hash,
            "seg": {k: s[k] for k in ("source_in", "host_in")},
            "seconds": seconds, "width": CHECK_WIDTH,
        })
        seg_marker = stage_dir / s["id"]
        if not force and cache.is_current(seg_marker, seg_hash) and out.exists():
            log.info("align-check: %s cache hit", s["id"])
            outputs.append(out)
            continue
        log.info("align-check: %s source_in=%.3f host_in=%.3f -> %s",
                 s["id"], s["source_in"], s["host_in"], out)
        _render_check_clip(source_path, host_av, s, out, seconds)
        cache.mark_done(seg_marker, seg_hash, extra={"stage": "align:check"})
        outputs.append(out)

    _write_check_index(ep, source_path, segments, seconds)
    log.info("align-check: %d clip(s) -> %s", len(outputs), ep.align_check_dir)
    return outputs
