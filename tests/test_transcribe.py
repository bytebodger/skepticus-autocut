"""Per-episode silence threshold derivation and the zero-silence loud-failure."""

import json

import pytest

from autocut import ffmpeg, transcribe
from autocut.paths import resolve


def _ep(tmp_path):
    return resolve("ep001", root=tmp_path)


def test_threshold_override_wins(monkeypatch, tmp_path):
    monkeypatch.setenv("AUTOCUT_SILENCE_NOISE_DB", "-42")
    db, source, mean = transcribe._silence_threshold(_ep(tmp_path))
    assert (db, source, mean) == (-42.0, "override", None)


def test_threshold_derived_from_mean(monkeypatch, tmp_path):
    monkeypatch.delenv("AUTOCUT_SILENCE_NOISE_DB", raising=False)
    monkeypatch.setattr(transcribe, "_measure_mean_volume", lambda ep: -27.6)
    db, source, mean = transcribe._silence_threshold(_ep(tmp_path))
    assert source == "derived"
    assert mean == -27.6
    assert db == round(-27.6 - transcribe.SILENCE_MARGIN_DB, 1)


def test_threshold_clamped_for_loud_source(monkeypatch, tmp_path):
    monkeypatch.delenv("AUTOCUT_SILENCE_NOISE_DB", raising=False)
    monkeypatch.setattr(transcribe, "_measure_mean_volume", lambda ep: -3.0)  # very hot
    db, source, _ = transcribe._silence_threshold(_ep(tmp_path))
    lo, hi = transcribe.SILENCE_DB_CLAMP
    assert lo <= db <= hi


def test_threshold_fallback_when_measurement_fails(monkeypatch, tmp_path):
    monkeypatch.delenv("AUTOCUT_SILENCE_NOISE_DB", raising=False)
    monkeypatch.setattr(transcribe, "_measure_mean_volume", lambda ep: None)
    db, source, _ = transcribe._silence_threshold(_ep(tmp_path))
    assert source == "fallback" and db == transcribe.SILENCE_NOISE_DB_FALLBACK


def test_run_refuses_to_write_empty_silence_json(monkeypatch, tmp_path):
    # Zero silences across real speech must raise, not write an empty silence.json.
    ep = _ep(tmp_path)
    ep.audio_dir.mkdir(parents=True, exist_ok=True)
    ep.speech_wav.write_bytes(b"RIFF....WAVE")
    ep.work.mkdir(parents=True, exist_ok=True)
    ep.probe_json.write_text(json.dumps({"source_duration": 120.0}), encoding="utf-8")

    ffmpeg.set_dry_run(False)
    monkeypatch.setattr(transcribe, "_transcribe_words",
                        lambda ep: {"words": [{"i": 0, "start": 0.0, "end": 1.0, "word": "hi", "prob": 1.0}]})
    monkeypatch.setattr(transcribe, "_measure_mean_volume", lambda ep: -27.0)
    monkeypatch.setattr(transcribe, "_detect_silence", lambda ep, db: [])  # nothing found

    with pytest.raises(RuntimeError, match="0 silences"):
        transcribe.run(ep, force=True)
    assert not ep.silence_json.exists()
