"""Tests for stage-1 frame-rate handling.

The mezzanine must adopt the SOURCE frame rate as an exact rational — a 29.97
source stays 30000/1001 and never rounds to 30 — and VFR sources must normalise
to the average rate. These are pure functions, tested without ffmpeg.
"""

from autocut import probe


def test_cfr_source_uses_r_frame_rate_exactly():
    chosen, fps, is_vfr = probe._choose_mezz_rate("30000/1001", "30000/1001")
    assert chosen == "30000/1001"      # exact rational preserved, not "29.97"
    assert not is_vfr
    assert abs(fps - 30000 / 1001) < 1e-9


def test_integer_rate_preserved():
    chosen, fps, is_vfr = probe._choose_mezz_rate("24/1", "24/1")
    assert chosen == "24/1"
    assert fps == 24.0
    assert not is_vfr


def test_vfr_falls_back_to_avg_rate():
    # Container claims 60fps but the measured average is ~29.97 -> VFR.
    chosen, fps, is_vfr = probe._choose_mezz_rate("60/1", "30000/1001")
    assert is_vfr
    assert chosen == "30000/1001"      # uses avg, exactly
    assert abs(fps - 30000 / 1001) < 1e-9


def test_within_tolerance_is_not_vfr():
    # r and avg identical -> not VFR, keep r_frame_rate.
    chosen, _fps, is_vfr = probe._choose_mezz_rate("30/1", "30/1")
    assert not is_vfr
    assert chosen == "30/1"


def test_missing_avg_uses_r():
    chosen, fps, is_vfr = probe._choose_mezz_rate("25/1", "0/0")
    assert chosen == "25/1"
    assert fps == 25.0
    assert not is_vfr


def test_no_usable_rate_falls_back():
    chosen, fps, is_vfr = probe._choose_mezz_rate(None, None)
    assert chosen == probe.FALLBACK_FPS
    assert fps == 30.0
    assert not is_vfr


def test_first_video_stream_picks_lowest_index():
    streams = [
        {"index": 3, "codec_type": "audio"},
        {"index": 2, "codec_type": "video", "codec_name": "second"},
        {"index": 0, "codec_type": "video", "codec_name": "first"},
    ]
    assert probe._first_video_stream(streams)["codec_name"] == "first"


def test_first_audio_stream_picks_lowest_index():
    streams = [
        {"index": 5, "codec_type": "audio", "codec_name": "a1"},
        {"index": 1, "codec_type": "video"},
        {"index": 4, "codec_type": "audio", "codec_name": "a0"},
    ]
    assert probe._first_audio_stream(streams)["codec_name"] == "a0"
