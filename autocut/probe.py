"""Stage 1 — probe and normalize.

Reads the source with ffprobe, then builds a constant-frame-rate, all-intra
mezzanine (every frame a keyframe) so later segment cuts are frame-accurate and
fast. Also extracts mono 16kHz audio for Whisper.

The mezzanine's frame rate is the *source* rate, kept as an exact rational: a
29.97 source (30000/1001) must stay 30000/1001 and never be rounded to 30, or
timestamps drift by ~0.1% across the episode. Downstream stages read the fps from
probe.json and never guess it.

VFR detection is not optional: OBS commonly records variable frame rate, and
cutting VFR footage by timestamp produces audio drift that compounds across an
episode. When the container's nominal rate (r_frame_rate) and the measured
average (avg_frame_rate) disagree, the source is VFR and we normalise to the
average rate.
"""

from __future__ import annotations

import json
import logging
from fractions import Fraction

from . import cache, ffmpeg
from .paths import Episode

log = logging.getLogger("autocut.probe")

MEZZ_CRF = 16
AUDIO_RATE = 48000
SPEECH_RATE = 16000

# r_frame_rate vs avg_frame_rate divergence above this fraction => VFR.
VFR_TOLERANCE = 0.001
# Used only if the source exposes no usable frame rate at all.
FALLBACK_FPS = "30/1"


def _rate(value: str | None) -> float | None:
    """Parse an ffprobe rational like '30000/1001' into a float (for comparison
    only — the exact rational string is what we hand to ffmpeg)."""
    if not value or value == "0/0":
        return None
    try:
        return float(Fraction(value))
    except (ValueError, ZeroDivisionError):
        return None


def _valid_rate(value: str | None) -> bool:
    return _rate(value) not in (None, 0.0)


def _choose_mezz_rate(r_str: str | None, avg_str: str | None) -> tuple[str, float, bool]:
    """Pick the exact rational rate for the CFR mezzanine.

    Returns (rational_string, float_value, is_vfr). Prefers r_frame_rate; if it
    diverges from avg_frame_rate beyond VFR_TOLERANCE the source is VFR and we
    use avg_frame_rate. The rational string is preserved exactly for ffmpeg's
    -r so fractional rates never round.
    """
    r_f = _rate(r_str)
    avg_f = _rate(avg_str)

    is_vfr = bool(r_f and avg_f and abs(r_f - avg_f) / r_f > VFR_TOLERANCE)

    if is_vfr:
        chosen = avg_str
    elif _valid_rate(r_str):
        chosen = r_str
    elif _valid_rate(avg_str):
        chosen = avg_str
    else:
        chosen = FALLBACK_FPS

    return chosen, float(Fraction(chosen)), is_vfr


def _first_video_stream(streams: list[dict]) -> dict:
    """The v:0 stream, chosen explicitly by lowest index (the source has two
    video streams; never leave selection to ffmpeg's defaults)."""
    videos = sorted(
        (s for s in streams if s.get("codec_type") == "video"),
        key=lambda s: s.get("index", 0),
    )
    return videos[0] if videos else {}


def _first_audio_stream(streams: list[dict]) -> dict:
    audios = sorted(
        (s for s in streams if s.get("codec_type") == "audio"),
        key=lambda s: s.get("index", 0),
    )
    return audios[0] if audios else {}


def _probe_source(ep: Episode, raw_hash: str) -> dict:
    data = ffmpeg.ffprobe_json(["-show_streams", "-show_format", str(ep.raw)])
    streams = data.get("streams", [])
    video = _first_video_stream(streams)
    audio = _first_audio_stream(streams)
    fmt = data.get("format", {})

    r_str = video.get("r_frame_rate")
    avg_str = video.get("avg_frame_rate")
    fps_rational, fps_float, is_vfr = _choose_mezz_rate(r_str, avg_str)

    duration = None
    for source in (video.get("duration"), fmt.get("duration")):
        try:
            duration = float(source)
            break
        except (TypeError, ValueError):
            continue

    return {
        "raw_file": str(ep.raw.name),
        "raw_sha256": raw_hash,
        # Exact source rates, preserved as rational strings.
        "r_frame_rate": r_str,
        "avg_frame_rate": avg_str,
        "is_vfr": is_vfr,
        "width": video.get("width"),
        "height": video.get("height"),
        "video_codec": video.get("codec_name"),
        "audio_codec": audio.get("codec_name"),
        "audio_sample_rate": _int(audio.get("sample_rate")),
        "audio_channels": audio.get("channels"),
        "source_duration": duration,
        # The mezzanine is CFR at the SOURCE rate. Downstream reads fps (float);
        # the mezzanine encode uses fps_rational (exact) for -r.
        "fps": fps_float,
        "fps_rational": fps_rational,
    }


def _int(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _build_mezzanine(ep: Episode, fps_rational: str) -> None:
    ep.mezz_dir.mkdir(parents=True, exist_ok=True)
    ffmpeg.run_ffmpeg([
        "-i", ep.raw,
        # Explicit stream selection: the source has two video streams, so pin
        # v:0. Audio is optional (the '?' suffix) but pinned to a:0 when present.
        "-map", "0:v:0", "-map", "0:a:0?",
        # -fps_mode cfr is the modern replacement for the removed -vsync cfr.
        # -r takes the exact source rational so 29.97 stays 30000/1001.
        "-fps_mode", "cfr", "-r", fps_rational,
        "-c:v", "libx264", "-preset", "veryfast", "-crf", MEZZ_CRF,
        "-g", "1", "-pix_fmt", "yuv420p",
        "-c:a", "pcm_s16le", "-ar", AUDIO_RATE,
        ep.mezz,
    ])


def _extract_speech_audio(ep: Episode) -> None:
    ep.audio_dir.mkdir(parents=True, exist_ok=True)
    ffmpeg.run_ffmpeg([
        "-i", ep.mezz,
        "-map", "0:a:0",
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
    # ffprobe is cheap, so read source metadata up front — the mezzanine rate
    # (and thus the cache key) is derived from it.
    probe = _probe_source(ep, raw_hash)

    input_hash = cache.hash_inputs({
        "raw_sha256": raw_hash,
        "mezz_fps": probe["fps_rational"],
        "mezz_crf": MEZZ_CRF,
        "audio_rate": AUDIO_RATE,
        "speech_rate": SPEECH_RATE,
    })

    if not force and cache.is_current(ep.work, input_hash) and ep.probe_json.exists():
        log.info("probe: cache hit for %s, skipping", ep.episode_id)
        return json.loads(ep.probe_json.read_text(encoding="utf-8"))

    log.info("probe: %s", ep.raw)
    if probe["is_vfr"]:
        log.warning(
            "source is VFR (r=%s vs avg=%s); normalising to avg rate %s for the CFR mezzanine",
            probe["r_frame_rate"], probe["avg_frame_rate"], probe["fps_rational"],
        )
    else:
        log.info("probe: source rate %s (%.4f fps)", probe["fps_rational"], probe["fps"])

    _build_mezzanine(ep, probe["fps_rational"])
    _extract_speech_audio(ep)

    ep.work.mkdir(parents=True, exist_ok=True)
    ep.probe_json.write_text(json.dumps(probe, indent=2), encoding="utf-8")

    cache.mark_done(ep.work, input_hash, extra={"stage": "probe"})
    return probe
