"""Reaction audio track: end-to-end behavioral regression.

Unit tests on the filter-graph *text* (test_audiotrack.py) missed two real
bugs that only showed up in actual rendered audio:

1. Chaining ``afade=t=in`` after ``afade=t=out`` on the host bed silenced the
   ENTIRE track (afade=t=in zeroes everything before its own start_time, not
   just the ramp region — confirmed against real ffmpeg output, contrary to
   what the filter text alone would suggest).
2. Reading the host bed via a naive ``-ss <output window start>`` into the
   raw mezzanine: output time and source time only coincide at the very
   start of the episode. Any EDL drop before the window shifts them apart,
   so the host bed played back the wrong moment of the raw recording — audio
   that decodes and sounds "fine" in isolation, but is the wrong content.

No assertion on graph text catches either: (1) needs the actual rendered
samples, and (2) needs an EDL with a drop before the window, which a
trivial single-keep-segment fixture doesn't exercise.

This renders tiny synthetic episodes through the real ``audiotrack`` module
and correlates the ACTUAL OUTPUT PCM against reference tones. Slow relative
to the rest of the suite (real ffmpeg calls); that cost is the point.
"""

import json

import numpy as np
import pytest

from autocut import audiotrack, ffmpeg
from autocut.paths import resolve

HOST_HZ = 300.0
SOURCE_HZ = 1000.0
CORR_SR = 4000  # downsampled just for correlation — keeps np.correlate cheap


def _make_tone_media(path, segments):
    """segments: list of (hz, duration) concatenated in order."""
    inputs = []
    filters = []
    for i, (hz, dur) in enumerate(segments):
        inputs += ["-f", "lavfi", "-i", f"sine=frequency={hz}:duration={dur}:sample_rate=48000"]
        filters.append(f"[{i}:a]")
    graph = "".join(filters) + f"concat=n={len(segments)}:v=0:a=1[out]" if len(segments) > 1 else None
    if graph:
        ffmpeg.run_ffmpeg([*inputs, "-filter_complex", graph, "-map", "[out]", "-c:a", "pcm_s16le", str(path)])
    else:
        ffmpeg.run_ffmpeg([*inputs, "-c:a", "pcm_s16le", str(path)])


def _decode_pcm(path, ss, t):
    raw = ffmpeg.run_ffmpeg_capture_stdout_bytes([
        "-ss", f"{ss:.3f}", "-t", f"{t:.3f}", "-i", str(path),
        "-map", "0:a:0", "-ac", "1", "-ar", str(CORR_SR), "-f", "s16le", "-",
    ])
    return np.frombuffer(raw, dtype="<i2").astype(np.float64)


def _norm_xcorr_peak(a: np.ndarray, b: np.ndarray) -> float:
    """Peak normalised cross-correlation — 1.0 for identical (up to phase)
    tones, near 0 for unrelated/mismatched-frequency signals."""
    a, b = a - a.mean(), b - b.mean()
    denom = np.sqrt((a ** 2).sum() * (b ** 2).sum())
    if denom < 1e-9:
        return 0.0
    return float(np.max(np.abs(np.correlate(a, b, mode="full"))) / denom)


def _rms(path, ss, t):
    raw = ffmpeg.run_ffmpeg_capture_stdout_bytes([
        "-ss", f"{ss:.3f}", "-t", f"{t:.3f}", "-i", str(path),
        "-map", "0:a", "-f", "s16le", "-ac", "1", "-",
    ])
    arr = np.frombuffer(raw, dtype="<i2").astype(np.float64)
    return float(np.sqrt(np.mean(arr ** 2))) if len(arr) else 0.0


@pytest.fixture
def ep_reaction(tmp_path):
    ep = resolve("ep001", root=tmp_path)
    ep.mezz_dir.mkdir(parents=True, exist_ok=True)
    ep.work.mkdir(parents=True, exist_ok=True)
    source_path = tmp_path / "source.mkv"

    # commentary [0,3) -> playback [3,6) -> commentary [6,9): host mic audible
    # for the first and third spans, source audio for the middle one.
    _make_tone_media(ep.mezz, [(HOST_HZ, 9.0)])
    _make_tone_media(source_path, [(SOURCE_HZ, 9.0)])

    ep.playback_json.write_text(json.dumps({
        "version": 1, "episode_id": "ep001", "source_file": str(source_path).replace("\\", "/"),
        "segments": [{"id": "pb001", "host_in": 3.0, "host_out": 6.0,
                      "source_in": 0.0, "source_out": 3.0,
                      "offset_source": "cumulative", "refinement_delta": None,
                      "seek_detected": False}],
    }), encoding="utf-8")
    ep.edl_json.write_text(json.dumps({
        "version": 1, "fps": 24,
        "segments": [{"id": "s1", "in": 0.0, "out": 9.0, "action": "keep"}],
        "overlays": [],
    }), encoding="utf-8")
    return ep, source_path


def test_reaction_audio_track_correlates_with_the_right_source_per_span(ep_reaction):
    ep, source_path = ep_reaction
    track = audiotrack.render_track(ep, (0.0, 9.0), force=True)
    assert track is not None and track.exists()

    host_ref = _decode_pcm(ep.mezz, 4.0, 0.5)
    source_ref = _decode_pcm(source_path, 1.0, 0.5)

    # Windows sit well clear of the 75ms crossfade at each boundary (3.0, 6.0).
    windows = [
        ("commentary before playback", 1.0, "host"),
        ("playback", 4.0, "source"),
        ("commentary after playback", 7.0, "host"),
    ]
    for label, ss, expected in windows:
        chunk = _decode_pcm(track, ss, 0.5)
        corr_host = _norm_xcorr_peak(chunk, host_ref)
        corr_source = _norm_xcorr_peak(chunk, source_ref)
        dominant = "host" if corr_host > corr_source else "source"
        assert dominant == expected, (
            f"{label} (t={ss}s): expected {expected} audio to dominate, got {dominant} "
            f"(corr_host={corr_host:.3f}, corr_source={corr_source:.3f})"
        )


def test_reaction_audio_track_mutes_host_during_playback(ep_reaction):
    # The host mic must be FULLY muted during playback, not just quieter than
    # the source — the format's explicit "not ducked" requirement (spec §8).
    ep, _source_path = ep_reaction
    track = audiotrack.render_track(ep, (0.0, 9.0), force=True)
    host_ref = _decode_pcm(ep.mezz, 4.0, 0.5)
    chunk = _decode_pcm(track, 4.0, 0.5)
    assert _norm_xcorr_peak(chunk, host_ref) < 0.15


def test_reaction_audio_track_reads_the_correct_source_time_despite_a_leading_drop(tmp_path):
    # Regression guard for the output-time/source-time conflation bug: an EDL
    # drop before the window must shift where the host bed reads from. Host
    # media is 200Hz for its first 5s (the dropped dead air) then 300Hz for
    # the rest (the real content) — reading via a naive, unshifted seek would
    # land in the 200Hz region and this test would catch it.
    ep = resolve("ep001", root=tmp_path)
    ep.mezz_dir.mkdir(parents=True, exist_ok=True)
    ep.work.mkdir(parents=True, exist_ok=True)
    source_path = tmp_path / "source.mkv"

    WRONG_HZ = 200.0
    _make_tone_media(ep.mezz, [(WRONG_HZ, 5.0), (HOST_HZ, 10.0)])  # 15s total
    _make_tone_media(source_path, [(SOURCE_HZ, 3.0)])

    # Source time: dead air [0,5) dropped, then playback [9,12) inside the
    # remaining commentary. Output time: [0,9) with playback at [4,7).
    ep.playback_json.write_text(json.dumps({
        "version": 1, "episode_id": "ep001", "source_file": str(source_path).replace("\\", "/"),
        "segments": [{"id": "pb001", "host_in": 9.0, "host_out": 12.0,
                      "source_in": 0.0, "source_out": 3.0,
                      "offset_source": "cumulative", "refinement_delta": None,
                      "seek_detected": False}],
    }), encoding="utf-8")
    ep.edl_json.write_text(json.dumps({
        "version": 1, "fps": 24,
        "segments": [
            {"id": "s1", "in": 0.0, "out": 5.0, "action": "drop", "reason": "dead_air"},
            {"id": "s2", "in": 5.0, "out": 15.0, "action": "keep"},
        ],
        "overlays": [],
    }), encoding="utf-8")

    track = audiotrack.render_track(ep, (0.0, 10.0), force=True)
    assert track is not None and track.exists()

    correct_ref = _decode_pcm(ep.mezz, 6.0, 0.5)   # 300Hz — the real content
    wrong_ref = _decode_pcm(ep.mezz, 1.0, 0.5)     # 200Hz — the dropped dead air

    chunk = _decode_pcm(track, 1.0, 0.5)  # output commentary, before playback
    corr_correct = _norm_xcorr_peak(chunk, correct_ref)
    corr_wrong = _norm_xcorr_peak(chunk, wrong_ref)
    assert corr_correct > corr_wrong, (
        f"host bed at output t=1.0 correlates more with the DROPPED dead-air tone "
        f"(corr={corr_wrong:.3f}) than the real content (corr={corr_correct:.3f}) — "
        f"it's reading from the wrong source position"
    )


def test_reaction_audio_track_playback_is_audible_late_in_a_long_episode(tmp_path):
    """Regression guard for a third bug in the host-bed fix itself: an
    earlier version seeked the host bed with ``-copyts`` (to keep decoded
    timestamps in absolute source time, needed for exact-duration atrim
    cuts — see audiotrack.py's SEEK_MARGIN docstring). ``-copyts`` is NOT
    scoped to the input it's written before; it's a sticky ffmpeg CLI flag
    that silently applied to the SOURCE playback clips built later in the
    same command too, even though they never asked for it. Those clips are
    positioned on the output timeline by ``adelay`` relative to a PTS that's
    normally rebased to ~0 at the seek point; carrying their true (large)
    absolute source-file PTS instead pushed them far past the render
    window's length, and ffmpeg silently drops audio past that point —
    reads as playback going silent, worse the larger ``source_in`` is.

    ``ep_reaction``'s single playback segment has ``source_in=0.0`` — with a
    near-zero source position, "absolute" and "seek-relative" timestamps are
    (coincidentally) almost the same number, so that fixture could not have
    told the two implementations apart. This one adds a SECOND segment late
    in a long episode with a large ``source_in``, which is exactly where the
    two diverge and the bug was audible.
    """
    ep = resolve("ep001", root=tmp_path)
    ep.mezz_dir.mkdir(parents=True, exist_ok=True)
    ep.work.mkdir(parents=True, exist_ok=True)
    source_path = tmp_path / "source.mkv"

    total = 520.0
    _make_tone_media(ep.mezz, [(HOST_HZ, total)])
    _make_tone_media(source_path, [(SOURCE_HZ, total)])

    ep.playback_json.write_text(json.dumps({
        "version": 1, "episode_id": "ep001", "source_file": str(source_path).replace("\\", "/"),
        "segments": [
            {"id": "pb001", "host_in": 5.0, "host_out": 10.0,
             "source_in": 0.0, "source_out": 5.0,
             "offset_source": "cumulative", "refinement_delta": None, "seek_detected": False},
            # Late in the episode, large source_in — where a leaked -copyts
            # would carry an absolute PTS far past the 520s window length.
            {"id": "pb002", "host_in": 510.0, "host_out": 515.0,
             "source_in": 500.0, "source_out": 505.0,
             "offset_source": "cumulative", "refinement_delta": None, "seek_detected": False},
        ],
    }), encoding="utf-8")
    ep.edl_json.write_text(json.dumps({
        "version": 1, "fps": 24,
        "segments": [{"id": "s1", "in": 0.0, "out": total, "action": "keep"}],
        "overlays": [],
    }), encoding="utf-8")

    track = audiotrack.render_track(ep, (0.0, total), force=True)
    assert track is not None and track.exists()

    commentary_rms = _rms(track, 1.0, 1.0)
    early_playback_rms = _rms(track, 7.5, 1.0)
    late_playback_rms = _rms(track, 512.5, 1.0)

    silence_floor = commentary_rms * 0.1
    assert early_playback_rms > silence_floor, (
        f"early playback (source_in=0.0) is silent (RMS={early_playback_rms:.1f}, "
        f"commentary RMS={commentary_rms:.1f})"
    )
    assert late_playback_rms > silence_floor, (
        f"late playback (source_in=500.0, ~510s into the episode) is silent "
        f"(RMS={late_playback_rms:.1f}, commentary RMS={commentary_rms:.1f}) — "
        f"a source clip landed off the render window instead of at its output position"
    )
