"""Reaction format — alignment stage (reaction spec sections 3-5, 10).

A reaction episode has two inputs: the *host* recording (talking head, with the
source video's audio bleeding into the mic) and the clean *source* video being
reacted to. This stage finds, for each playback segment, its span in the host
timeline and its span in the source timeline. The second is what makes lip sync
work — get it wrong by 200ms and the source speaker's mouth disagrees with the
audio.

Two-stage alignment, the same cheap-then-precise pattern as silence/retake:

  1. **Coarse, by transcript.** The host transcript carries the host's words plus
     the source's words picked up as bleed; the source transcript carries only
     the source's words. A run of matching words localises a playback region and
     its approximate source offset (±200ms). Fuzzy, run-based — bleed
     transcription is imperfect.
  2. **Fine, by audio cross-correlation.** Within each coarse region, correlate a
     band-limited spectral envelope of the host bleed against the source audio to
     lock the lag to <50ms. Playback is real-time 1:1, so a single lag L (host
     time − source time) defines the whole segment; source_in/out follow from the
     host boundaries, which guarantees ``source_out − source_in == host_out −
     host_in`` (asserted).

Output: ``work/<ep>/playback.json`` (spec section 5). A segment whose correlation
peak is too weak to trust stops the pipeline (spec section 10) — bad sync is
worse than no output.

Only the alignment map and the verification harness live here (build-order steps
1-3). The autoauthor constraints, content/audio tracks, and review UI are later
steps and are not touched.
"""

from __future__ import annotations

import json
import logging
import os
import wave
from pathlib import Path
from typing import Any

import numpy as np

from . import cache, edl, ffmpeg, transcribe as transcribe_mod
from .paths import Episode

log = logging.getLogger("autocut.align")

# --- coarse (transcript) matching ---
MIN_RUN = 6          # matched words needed to call a span "playback", not chance
MAX_SKIP = 2         # fuzzy resync budget: bleed drops/adds a word here and there
# --- boundary snapping ---
SNAP_TOL = 0.6       # snap a host boundary to nearby acoustic silence within this
# --- fine (audio) correlation ---
HOP = 0.010          # 10ms envelope hop — the lag resolution
FRAME = 0.025        # 25ms analysis frame
BAND = (300.0, 3500.0)  # Hz — the bleed survives only the speech band (spec §4)
TEMPLATE_S = 6.0     # length of host bleed correlated against the source
SEARCH_S = 3.0       # ± search window around the coarse source estimate
# --- trust ---
# A peak below this can't be trusted to <50ms; fail loud (spec §10) unless the
# operator explicitly accepts it.
MIN_CORRELATION = float(os.environ.get("AUTOCUT_MIN_CORRELATION", "0.5"))


# --------------------------------------------------------------------------- #
# Coarse alignment — pure text, unit-testable without audio.
# --------------------------------------------------------------------------- #

def _norm(token: str) -> str:
    """Normalise a token for matching: lowercase, alphanumerics only."""
    return "".join(c for c in token.lower() if c.isalnum())


def normalize_words(words: list[dict]) -> list[dict]:
    """Project a words.json word list to ``{norm, start, end}`` for matching.

    Non-alphanumeric tokens normalise to ``""`` and never match (they anchor
    nothing), but are kept in place so indices line up with the transcript.
    """
    out = []
    for w in words:
        out.append({"norm": _norm(w.get("word", "")),
                    "start": float(w["start"]), "end": float(w["end"])})
    return out


def _extend_diagonal(host: list[dict], source: list[dict], hi: int, sj: int,
                     max_skip: int) -> tuple[int, int, int]:
    """From an anchor (host[hi] == source[sj]), walk both forward counting matches.

    Tolerates a few mismatches (bleed mis-transcription: a dropped or swapped
    word) by resyncing within a small window before giving up. Returns
    ``(last_host_idx, last_source_idx, match_count)`` for the matched run.
    """
    h, j = hi, sj
    matches = 0
    last_h, last_j = hi, sj
    misses = 0
    while h < len(host) and j < len(source):
        if host[h]["norm"] and host[h]["norm"] == source[j]["norm"]:
            matches += 1
            last_h, last_j = h, j
            misses = 0
            h += 1
            j += 1
            continue
        # Mismatch: try to resync on a nearby word within the skip budget
        # (covers a word dropped from the bleed, or an extra host interjection).
        resynced = False
        for dh in range(0, max_skip + 1):
            for dj in range(0, max_skip + 1):
                if dh == 0 and dj == 0:
                    continue
                hh, jj = h + dh, j + dj
                if (hh < len(host) and jj < len(source)
                        and host[hh]["norm"] and host[hh]["norm"] == source[jj]["norm"]):
                    h, j = hh, jj
                    resynced = True
                    break
            if resynced:
                break
        if not resynced:
            misses += 1
            if misses > max_skip:
                break
            h += 1
            j += 1
    return last_h, last_j, matches


def find_coarse_regions(host: list[dict], source: list[dict], *,
                        min_run: int = MIN_RUN, max_skip: int = MAX_SKIP) -> list[dict]:
    """Localise playback regions by matching runs of host words to source words.

    Each host word is only consumed once, so overlapping candidate runs don't
    double-count. Each region's source offset is found independently — playback
    segments do NOT resume where the previous one stopped (spec §3: don't assume
    contiguity). Returns coarse regions as index spans with a match count.
    """
    src_index: dict[str, list[int]] = {}
    for jj, sw in enumerate(source):
        if sw["norm"]:
            src_index.setdefault(sw["norm"], []).append(jj)

    used = [False] * len(host)
    regions: list[dict] = []
    hi = 0
    while hi < len(host):
        if used[hi] or not host[hi]["norm"]:
            hi += 1
            continue
        best: tuple[int, int, int] | None = None  # (last_h, last_j, matches)
        best_sj = -1
        for sj in src_index.get(host[hi]["norm"], []):
            last_h, last_j, matches = _extend_diagonal(host, source, hi, sj, max_skip)
            if best is None or matches > best[2]:
                best = (last_h, last_j, matches)
                best_sj = sj
        if best is not None and best[2] >= min_run:
            last_h, last_j, matches = best
            regions.append({
                "h_start": hi, "h_end": last_h,
                "s_start": best_sj, "s_end": last_j,
                "matches": matches,
            })
            for k in range(hi, last_h + 1):
                used[k] = True
            hi = last_h + 1
        else:
            hi += 1
    return regions


def region_times(region: dict, host: list[dict], source: list[dict]) -> dict:
    """Turn a coarse index-span region into approximate times + confidence."""
    span = region["h_end"] - region["h_start"] + 1
    return {
        "host_in": host[region["h_start"]]["start"],
        "host_out": host[region["h_end"]]["end"],
        "source_in": source[region["s_start"]]["start"],
        "source_out": source[region["s_end"]]["end"],
        # anchor host time = first matched word (real audio, not a snapped pause),
        # used as the fine-alignment anchor.
        "anchor_host": host[region["h_start"]]["start"],
        "anchor_source": source[region["s_start"]]["start"],
        # density of matches over the host span: 1.0 == every word matched.
        "confidence": round(region["matches"] / span, 3),
        "matches": region["matches"],
    }


# --------------------------------------------------------------------------- #
# Boundary snapping to acoustic silence (spec §4 "Boundaries").
# --------------------------------------------------------------------------- #

def snap_boundary(t: float, silences: list[dict], edge: str, *, tol: float = SNAP_TOL) -> float:
    """Snap a host boundary to a nearby silence edge, if one is within ``tol``.

    ``edge='in'``  → snap playback START to the END of the pause just before it.
    ``edge='out'`` → snap playback END to the START of the pause just after it.
    You pause before and after playback, so the silence is usually there; if it
    isn't within tolerance the raw time is kept.
    """
    best = t
    best_d = tol
    for sil in silences:
        cand = float(sil["end"]) if edge == "in" else float(sil["start"])
        d = abs(cand - t)
        if d <= best_d:
            best_d = d
            best = cand
    return best


# --------------------------------------------------------------------------- #
# Audio helpers — numpy only (no scipy/librosa available).
# --------------------------------------------------------------------------- #

def read_wav_slice(path: Path, t0: float, t1: float) -> tuple[np.ndarray, int]:
    """Read ``[t0, t1)`` seconds of a PCM wav as mono float32 in [-1, 1].

    Seeks rather than loading the whole file, so a long episode's 16k audio is
    read a few seconds at a time.
    """
    with wave.open(str(path), "rb") as w:
        sr = w.getframerate()
        ch = w.getnchannels()
        total = w.getnframes()
        n0 = max(0, int(round(t0 * sr)))
        n1 = min(total, int(round(t1 * sr)))
        if n1 <= n0:
            return np.zeros(0, dtype=np.float32), sr
        w.setpos(n0)
        raw = w.readframes(n1 - n0)
    data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if ch > 1:
        data = data.reshape(-1, ch).mean(axis=1)
    return data, sr


def band_envelope(sig: np.ndarray, sr: int, *, band: tuple[float, float] = BAND,
                  frame: float = FRAME, hop: float = HOP) -> np.ndarray:
    """A band-limited short-time energy envelope, z-normalised.

    Energy in the 300–3.5kHz band per frame, log-compressed and z-scored. The
    acoustic path (speaker → room → mic) destroys phase and the spectral
    extremes, so we correlate this envelope rather than raw samples (spec §4).
    """
    fl = int(round(frame * sr))
    hp = int(round(hop * sr))
    if fl <= 0 or hp <= 0 or len(sig) < fl:
        return np.zeros(0, dtype=np.float32)
    nfr = 1 + (len(sig) - fl) // hp
    idx = np.arange(fl)[None, :] + hp * np.arange(nfr)[:, None]
    frames = sig[idx] * np.hanning(fl)[None, :]
    spec = np.fft.rfft(frames, axis=1)
    freqs = np.fft.rfftfreq(fl, 1.0 / sr)
    keep = (freqs >= band[0]) & (freqs <= band[1])
    energy = np.sqrt((np.abs(spec[:, keep]) ** 2).sum(axis=1) + 1e-12)
    env = np.log(energy + 1e-6)
    std = env.std()
    if std < 1e-9:
        return np.zeros(nfr, dtype=np.float32)  # silence: nothing to lock onto
    return ((env - env.mean()) / std).astype(np.float32)


def slide_correlate(template: np.ndarray, search: np.ndarray) -> tuple[int, float]:
    """Best offset of ``template`` within ``search`` and its normalised peak.

    Returns ``(offset_frames, peak)`` where ``peak`` is a Pearson correlation in
    [-1, 1] at the best offset. ``(0, 0.0)`` if there isn't enough to correlate.
    """
    n, m = len(template), len(search)
    if n == 0 or m < n:
        return 0, 0.0
    t = template - template.mean()
    tn = np.linalg.norm(t)
    if tn < 1e-9:
        return 0, 0.0
    t = t / tn
    best_off, best_peak = 0, -1.0
    for off in range(0, m - n + 1):
        w = search[off:off + n]
        w = w - w.mean()
        wn = np.linalg.norm(w)
        if wn < 1e-9:
            continue
        c = float(np.dot(t, w / wn))
        if c > best_peak:
            best_peak, best_off = c, off
    return best_off, best_peak


def refine_lag(host_wav: Path, source_wav: Path, coarse: dict, *,
               template_s: float = TEMPLATE_S, search_s: float = SEARCH_S) -> tuple[float, float]:
    """Lock the host↔source lag for one region by envelope cross-correlation.

    Correlates a chunk of host bleed starting at the region's first matched word
    against the source audio around the coarse estimate. Returns
    ``(lag, peak)`` where ``lag = host_time − source_time`` for this playback
    segment, and ``peak`` is the correlation strength (spec §4: record it; a weak
    peak is untrustworthy).
    """
    a = coarse["anchor_host"]
    b = coarse["anchor_source"]
    seg_len = coarse["host_out"] - coarse["host_in"]
    tlen = max(0.5, min(template_s, seg_len))

    host_sig, sr = read_wav_slice(host_wav, a, a + tlen)
    src_lo = max(0.0, b - search_s)
    src_sig, sr2 = read_wav_slice(source_wav, src_lo, b + tlen + search_s)
    if sr != sr2:
        raise RuntimeError(f"align: sample-rate mismatch host={sr} source={sr2}")

    he = band_envelope(host_sig, sr)
    se = band_envelope(src_sig, sr)
    off, peak = slide_correlate(he, se)
    # host anchor `a` aligns to this source time:
    source_at_anchor = src_lo + off * HOP
    lag = a - source_at_anchor
    return lag, round(peak, 3)


# --------------------------------------------------------------------------- #
# Playback map assembly + the duration-equality invariant.
# --------------------------------------------------------------------------- #

def _clamp_to_source(host_in: float, host_out: float, source_in: float,
                     source_out: float, source_dur: float) -> tuple[float, float, float, float]:
    """Keep the segment inside the source file while preserving equal durations.

    Any shift/shrink is applied to the host and source ends together, so
    ``source_out − source_in`` stays equal to ``host_out − host_in`` (the
    load-bearing invariant). Normally a no-op; guards a coarse estimate that ran
    off either end of the source.
    """
    if source_in < 0.0:
        delta = -source_in
        host_in += delta
        source_in = 0.0
    if source_dur is not None and source_out > source_dur:
        overflow = source_out - source_dur
        host_out -= overflow
        source_out = source_dur
    return host_in, host_out, source_in, source_out


def build_segments(coarse_regions: list[dict], host_words: list[dict],
                   source_words: list[dict], silences: list[dict],
                   lags: list[tuple[float, float]], *, host_fps: float,
                   source_dur: float | None) -> list[dict]:
    """Assemble playback.json segments from coarse regions + fine lags.

    Host boundaries are snapped to acoustic silence and frames; source boundaries
    follow from the single per-segment lag, which makes the durations equal by
    construction. Raises if the invariant is violated (a coding error).
    """
    segments: list[dict] = []
    for i, region in enumerate(coarse_regions):
        times = region_times(region, host_words, source_words)
        lag, peak = lags[i]

        host_in = snap_boundary(times["host_in"], silences, "in")
        host_out = snap_boundary(times["host_out"], silences, "out")
        host_in = edl.snap(host_in, host_fps)
        host_out = edl.snap(host_out, host_fps)

        # One lag defines the whole segment: source follows the host boundaries.
        source_in = host_in - lag
        source_out = host_out - lag
        host_in, host_out, source_in, source_out = _clamp_to_source(
            host_in, host_out, source_in, source_out,
            source_dur if source_dur is not None else 1e9)

        host_len = host_out - host_in
        source_len = source_out - source_in
        if abs(source_len - host_len) > 1e-6:
            raise RuntimeError(
                f"align: segment {i} durations diverge — host {host_len:.6f}s vs "
                f"source {source_len:.6f}s. This must never happen; it means the "
                f"1:1 playback invariant was broken in assembly."
            )

        segments.append({
            "id": f"pb{i + 1:03d}",
            "host_in": round(host_in, 3),
            "host_out": round(host_out, 3),
            "source_in": round(source_in, 3),
            "source_out": round(source_out, 3),
            "confidence": times["confidence"],
            "correlation_peak": peak,
        })
    return segments


def assert_equal_durations(segments: list[dict], *, tol: float = 2e-3) -> None:
    """Assert ``source_out − source_in == host_out − host_in`` for every segment
    (spec §5). Uses a 2ms tolerance for the 3dp-rounded times in the JSON."""
    for s in segments:
        host_len = s["host_out"] - s["host_in"]
        source_len = s["source_out"] - s["source_in"]
        if abs(source_len - host_len) > tol:
            raise RuntimeError(
                f"align: {s['id']} source duration {source_len:.3f}s != host "
                f"duration {host_len:.3f}s (diff {abs(source_len - host_len)*1000:.1f}ms)"
            )


# --------------------------------------------------------------------------- #
# Source preparation (extract audio + transcribe), cached.
# --------------------------------------------------------------------------- #

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


def _extract_source_speech(ep: Episode, source_path: Path) -> None:
    """Extract 16kHz mono speech audio from the source (same shape as the host's
    speech.wav) — the input to both source transcription and correlation."""
    ep.align_dir.mkdir(parents=True, exist_ok=True)
    ffmpeg.run_ffmpeg([
        "-i", str(source_path),
        "-map", "0:a:0",
        "-ac", "1", "-ar", "16000",
        "-c:a", "pcm_s16le",
        ep.source_speech_wav,
    ])


def prepare_source(ep: Episode, source_path: Path, *, force: bool = False) -> dict:
    """Extract + transcribe the source audio, cached on the source file hash.

    Returns the source words doc. Skips the (heavy) Whisper run when the cache is
    current, mirroring the host transcribe stage.
    """
    dry = ffmpeg.is_dry_run()
    source_hash = cache.hash_file(source_path) if (source_path.exists() and not dry) else "dry"
    input_hash = cache.hash_inputs({
        "source_sha256": source_hash,
        "model": transcribe_mod.MODEL,
        "compute_type": transcribe_mod.COMPUTE_TYPE,
        "beam_size": transcribe_mod.BEAM_SIZE,
        "word_timebase": "source-v2",
    })
    stage_dir = ep.align_dir / "source_transcript"
    if (not force and cache.is_current(stage_dir, input_hash)
            and ep.source_words_json.exists() and ep.source_speech_wav.exists()):
        log.info("align: source transcript cache hit")
        return json.loads(ep.source_words_json.read_text(encoding="utf-8"))

    _extract_source_speech(ep, source_path)
    if dry:
        log.info("align: dry-run — skipping source transcription")
        return {"words": []}

    log.info("align: transcribing source %s", source_path.name)
    words = transcribe_mod.transcribe_wav(ep.source_speech_wav)
    ep.source_words_json.write_text(json.dumps(words, indent=2), encoding="utf-8")
    cache.mark_done(stage_dir, input_hash, extra={"stage": "align:source_transcript"})
    return words


# --------------------------------------------------------------------------- #
# Orchestration.
# --------------------------------------------------------------------------- #

def _load_host_inputs(ep: Episode) -> tuple[list[dict], list[dict], float]:
    """(host words, host silences, host fps). Raises if the host wasn't processed."""
    for path in (ep.words_json, ep.silence_json, ep.probe_json):
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path.name}. Run probe + transcribe for the host "
                f"recording {ep.episode_id!r} first."
            )
    words = json.loads(ep.words_json.read_text(encoding="utf-8")).get("words", [])
    silences = json.loads(ep.silence_json.read_text(encoding="utf-8")).get("silences", [])
    fps = float(json.loads(ep.probe_json.read_text(encoding="utf-8"))["fps"])
    return words, silences, fps


def _resolve_source(ep: Episode, source: str | Path | None) -> Path:
    path = Path(source) if source else ep.source_video
    if not path.exists() and not ffmpeg.is_dry_run():
        raise FileNotFoundError(
            f"No source video at {path}. Pass --source <path>, or drop it at "
            f"{ep.source_video} (inbox/{ep.episode_id}_source.<ext>)."
        )
    return path


def run(ep: Episode, source: str | Path | None = None, *, force: bool = False) -> dict:
    """Align a reaction episode: write ``playback.json`` (spec §5).

    Coarse transcript matching localises the plays; fine cross-correlation locks
    each source offset; the durations are asserted equal; a segment that couldn't
    be aligned confidently stops the pipeline (spec §10).
    """
    source_path = _resolve_source(ep, source)
    host_words_raw, silences, host_fps = _load_host_inputs(ep)
    source_words_doc = prepare_source(ep, source_path, force=force)
    _source_fps, source_dur = _probe_source(source_path) if not ffmpeg.is_dry_run() else (30.0, None)

    host = normalize_words(host_words_raw)
    source = normalize_words(source_words_doc.get("words", []))
    regions = find_coarse_regions(host, source)
    log.info("align: coarse found %d playback region(s) from %d host / %d source words",
             len(regions), len(host), len(source))
    if not regions:
        raise RuntimeError(
            "align: no playback regions found by transcript matching. Either the "
            "source audio didn't bleed into the host mic (headphones? — spec §12), "
            "or the source has too little speech to match. Check "
            f"{ep.source_words_json} and {ep.words_json}."
        )

    # Fine lag per region.
    coarse = [region_times(r, host, source) for r in regions]
    lags: list[tuple[float, float]] = []
    for i, c in enumerate(coarse):
        if ffmpeg.is_dry_run():
            lags.append((c["anchor_host"] - c["anchor_source"], 0.0))
            continue
        lag, peak = refine_lag(ep.speech_wav, ep.source_speech_wav, c)
        log.info("align: region %d lag=%.3fs peak=%.3f (coarse source_in=%.3f)",
                 i + 1, lag, peak, c["source_in"])
        lags.append((lag, peak))

    segments = build_segments(regions, host, source, silences, lags,
                              host_fps=host_fps, source_dur=source_dur)
    assert_equal_durations(segments)

    # Fail loud on weak alignment (spec §10): bad sync looks like a production
    # error, not a missing feature. Escape hatch for a deliberately-accepted run.
    weak = [s for s in segments if s["correlation_peak"] < MIN_CORRELATION]
    if weak and not ffmpeg.is_dry_run():
        ids = ", ".join(f"{s['id']}(peak={s['correlation_peak']:.2f})" for s in weak)
        msg = (
            f"align: {len(weak)} segment(s) below the correlation floor "
            f"{MIN_CORRELATION:.2f}: {ids}. These would render out of sync. "
            f"Longer plays align better (spec §12); or verify by eye with "
            f"'autocut align-check {ep.episode_id}' and, if acceptable, set "
            f"AUTOCUT_ALLOW_LOW_CORRELATION=1."
        )
        if os.environ.get("AUTOCUT_ALLOW_LOW_CORRELATION"):
            log.warning(msg + " (allowed by env)")
        else:
            raise RuntimeError(msg)

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
# Verification harness (spec §10) — the deliverable that matters most.
# --------------------------------------------------------------------------- #

CHECK_SECONDS = 2.0
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
        "If it's off by ~200ms, they visibly disagree.",
        "",
        "| clip | host_in | source_in | dur | confidence | corr peak |",
        "|------|--------:|----------:|----:|-----------:|----------:|",
    ]
    for s in segments:
        dur = s["host_out"] - s["host_in"]
        lines.append(
            f"| {s['id']}.mp4 | {s['host_in']:.3f} | {s['source_in']:.3f} | "
            f"{dur:.1f}s | {s['confidence']:.2f} | {s['correlation_peak']:.2f} |"
        )
    lines += ["", f"Source: `{source_path}`", ""]
    (ep.align_check_dir / "index.md").write_text("\n".join(lines), encoding="utf-8")


def render_checks(ep: Episode, *, seconds: float = CHECK_SECONDS, force: bool = False) -> list[Path]:
    """Render the lip-sync verification clips for every playback segment.

    Reads ``playback.json``; writes ``work/<ep>/align/check/pbNNN.mp4`` + an
    index. This is the harness the format is verified by — numbers won't tell you
    whether the sync is right; two seconds of video will (spec §10).
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
    outputs: list[Path] = []
    for s in segments:
        out = ep.align_check_dir / f"{s['id']}.mp4"
        seg_hash = cache.hash_inputs({
            "source": cache.hash_file(source_path) if (source_path.exists() and not dry) else "dry",
            "host_av": cache.hash_file(host_av) if (host_av.exists() and not dry) else "dry",
            "seg": {k: s[k] for k in ("source_in", "host_in")},
            "seconds": seconds, "width": CHECK_WIDTH,
        })
        seg_marker = stage_dir / s["id"]
        if not force and cache.is_current(seg_marker, seg_hash) and out.exists():
            log.info("align-check: %s cache hit", s["id"])
            outputs.append(out)
            continue
        log.info("align-check: %s source_in=%.3f host_in=%.3f peak=%.2f -> %s",
                 s["id"], s["source_in"], s["host_in"], s["correlation_peak"], out)
        _render_check_clip(source_path, host_av, s, out, seconds)
        cache.mark_done(seg_marker, seg_hash, extra={"stage": "align:check"})
        outputs.append(out)

    _write_check_index(ep, source_path, segments, seconds)
    log.info("align-check: %d clip(s) -> %s", len(outputs), ep.align_check_dir)
    return outputs
