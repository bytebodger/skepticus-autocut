"""Dry-run must be side-effect-free: it logs encode commands, mutates nothing,
and never runs the heavy transcription. Regression tests for the footguns where
--dry-run corrupted probe.json and ran the real Whisper model."""

import json

import pytest

from autocut import ffmpeg, probe, transcribe
from autocut.paths import resolve


@pytest.fixture
def dry_run():
    ffmpeg.set_dry_run(True)
    try:
        yield
    finally:
        ffmpeg.set_dry_run(False)


def test_dryrun_probe_does_not_clobber_probe_json(dry_run, monkeypatch, tmp_path):
    ep = resolve("ep001", root=tmp_path)
    ep.inbox.mkdir(parents=True, exist_ok=True)
    (ep.inbox / "ep001.mp4").write_bytes(b"not a real mp4, but present")
    ep.work.mkdir(parents=True, exist_ok=True)
    good = {"source_duration": 42.0, "fps": 24.0, "fps_rational": "24/1"}
    ep.probe_json.write_text(json.dumps(good), encoding="utf-8")

    # ffprobe is read-only and runs even in dry-run; stub its result so the test
    # needs no real binary/file.
    monkeypatch.setattr(ffmpeg, "ffprobe_json", lambda args: {
        "streams": [{"codec_type": "video", "index": 0, "r_frame_rate": "24/1",
                     "avg_frame_rate": "24/1", "width": 1920, "height": 1080,
                     "codec_name": "h264", "duration": "42.0"},
                    {"codec_type": "audio", "index": 1, "codec_name": "aac",
                     "sample_rate": "48000", "channels": 2}],
        "format": {"duration": "42.0"},
    })

    probe.run(ep, force=True)

    # The good probe.json is untouched and no cache marker was written.
    assert json.loads(ep.probe_json.read_text(encoding="utf-8")) == good
    assert not (ep.work / ".done").exists()


def test_dryrun_transcribe_skips_model_and_writes_nothing(dry_run, tmp_path, monkeypatch):
    ep = resolve("ep001", root=tmp_path)
    ep.audio_dir.mkdir(parents=True, exist_ok=True)
    ep.speech_wav.write_bytes(b"RIFF....WAVE")  # just needs to exist

    # If dry-run tried to transcribe, it would import faster_whisper — make that
    # explode so the test fails loudly if the guard regresses.
    def boom(_ep):
        raise AssertionError("dry-run must not run the Whisper model")

    monkeypatch.setattr(transcribe, "_transcribe_words", boom)

    result = transcribe.run(ep, force=True)

    assert result == {"words": []}
    assert not ep.words_json.exists()
    assert not ep.silence_json.exists()
    assert not (ep.transcript_dir / ".done").exists()
