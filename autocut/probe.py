"""Stage 1 — probe and normalize.

Reads the source with ffprobe, then builds a constant-frame-rate, all-intra
mezzanine (every frame a keyframe) so later segment cuts are frame-accurate and
fast. Also extracts mono 16kHz audio for Whisper.

VFR detection is not optional: OBS commonly records variable frame rate, and
cutting VFR footage by timestamp produces audio drift that compounds across an
episode. The CFR mezzanine is what removes that risk.
"""

from __future__ import annotations

import json
import logging
from fractions import Fraction

from . import cache, ffmpeg
from .paths import Episode

log = logging.getLogger("autocut.probe")

TARGET_FPS = 30
MEZZ_CRF = 16
AUDIO_RATE = 48000
SPEECH_RATE = 16000


def _rate(value: str | None) -> float | None:
    """Parse an ffprobe rational like '30000/1001' into a float."""
    if not value or value == "0/0":
        return None
    try:
        return float(Fraction(value))
    except (ValueError, ZeroDivisionError):
        return None


def _probe_source(ep: Episode) -> dict:
    data = ffmpeg.ffprobe_json(
        ["-show_streams", "-show_format", str(ep.raw)]
    )
    streams = data.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), {})
    audio = next((s for s in streams if s.get("codec_type") == "audio"), {})
    fmt = data.get("format", {})

    r_fps = _rate(video.get("r_frame_rate"))
    avg_fps = _rate(video.get("avg_frame_rate"))
    # Heuristic VFR detection: if the container's nominal rate (r_frame_rate)
    # and the measured average rate diverge meaningfully, treat as VFR. The
    # mezzanine forces CFR regardless, so this is informational.
    is_vfr = bool(
        r_fps and avg_fps and abs(r_fps - avg_fps) / r_fps > 0.001
    )

    duration = None
    for source in (video.get("duration"), fmt.get("duration")):
        try:
            duration = float(source)
            break
        except (TypeError, ValueError):
            continue

    return {
        "raw_file": str(ep.raw.name),
        "raw_sha256": cache.hash_file(ep.raw),
        "container_fps": r_fps,
        "avg_fps": avg_fps,
        "is_vfr": is_vfr,
        "width": video.get("width"),
        "height": video.get("height"),
        "video_codec": video.get("codec_name"),
        "audio_codec": audio.get("codec_name"),
        "audio_sample_rate": _int(audio.get("sample_rate")),
        "audio_channels": audio.get("channels"),
        "source_duration": duration,
        # The mezzanine is CFR at TARGET_FPS; downstream stages read THIS fps.
        "fps": TARGET_FPS,
    }


def _int(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _build_mezzanine(ep: Episode) -> None:
    ep.mezz_dir.mkdir(parents=True, exist_ok=True)
    ffmpeg.run_ffmpeg([
        "-i", ep.raw,
        "-vsync", "cfr", "-r", TARGET_FPS,
        "-c:v", "libx264", "-preset", "veryfast", "-crf", MEZZ_CRF,
        "-g", "1", "-pix_fmt", "yuv420p",
        "-c:a", "pcm_s16le", "-ar", AUDIO_RATE,
        ep.mezz,
    ])


def _extract_speech_audio(ep: Episode) -> None:
    ep.audio_dir.mkdir(parents=True, exist_ok=True)
    ffmpeg.run_ffmpeg([
        "-i", ep.mezz,
        "-ac", "1", "-ar", SPEECH_RATE,
        "-c:a", "pcm_s16le",
        ep.speech_wav,
    ])


def run(ep: Episode, *, force: bool = False) -> dict:
    """Execute stage 1. Cached on the raw file hash + normalization parameters."""
    if not ep.raw.exists():
        raise FileNotFoundError(
            f"No raw file for episode {ep.episode_id!r}. Expected {ep.raw}."
        )

    raw_hash = cache.hash_file(ep.raw)
    input_hash = cache.hash_inputs({
        "raw_sha256": raw_hash,
        "target_fps": TARGET_FPS,
        "mezz_crf": MEZZ_CRF,
        "audio_rate": AUDIO_RATE,
        "speech_rate": SPEECH_RATE,
    })

    if not force and cache.is_current(ep.work, input_hash) and ep.probe_json.exists():
        log.info("probe: cache hit for %s, skipping", ep.episode_id)
        return json.loads(ep.probe_json.read_text(encoding="utf-8"))

    log.info("probe: %s", ep.raw)
    probe = _probe_source(ep)
    if probe["is_vfr"]:
        log.warning(
            "source is VFR (%.3f vs %.3f fps); mezzanine forces CFR@%d",
            probe["container_fps"] or 0, probe["avg_fps"] or 0, TARGET_FPS,
        )

    _build_mezzanine(ep)
    _extract_speech_audio(ep)

    ep.work.mkdir(parents=True, exist_ok=True)
    ep.probe_json.write_text(json.dumps(probe, indent=2), encoding="utf-8")

    cache.mark_done(ep.work, input_hash, extra={"stage": "probe"})
    return probe
