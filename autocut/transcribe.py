"""Stage 2 — transcribe.

Uses faster-whisper (CTranslate2 backend): ~4x faster and lower VRAM than the
reference implementation. large-v3 at float16 uses ~4.7GB, comfortable on 12GB.

Two outputs:
  * words.json  — word-level timestamps for the EDL author and captions.
  * silence.json — an independent silencedetect pass, because Whisper's VAD is
    tuned for speech detection, not editorial pauses.

Determinism (spec section 15): the model revision, compute type, and beam size
are pinned here and recorded into words.json. Don't change them mid-project.
"""

from __future__ import annotations

import json
import logging
import os
import re

from . import cache, ffmpeg
from .paths import Episode

log = logging.getLogger("autocut.transcribe")

# Spec defaults (section 2/15). Overridable by env for machines without a CUDA
# runtime (e.g. CPU int8) or for faster test runs with a smaller model — the
# chosen values are recorded into words.json so the transcript is self-describing.
MODEL = os.environ.get("AUTOCUT_WHISPER_MODEL", "large-v3")
COMPUTE_TYPE = os.environ.get("AUTOCUT_WHISPER_COMPUTE", "float16")
DEVICE = os.environ.get("AUTOCUT_WHISPER_DEVICE", "cuda")
BEAM_SIZE = int(os.environ.get("AUTOCUT_WHISPER_BEAM", "5"))
# Pin the exact model snapshot for byte-reproducible transcripts. Fill in the
# HF revision hash for your downloaded model; None means "latest", which is not
# deterministic across re-downloads.
MODEL_REVISION: str | None = None

SILENCE_MIN_DUR = 0.4
# The silencedetect threshold is derived per-episode from the measured audio
# level, because a hardcoded value can miss a recording's noise floor entirely
# (one mic/room's speech pauses sit where another's speech does). We set it a
# fixed margin below the mean volume: far enough under speech (mean RMS) that it
# never marks real speech as silent, close enough to catch room-tone pauses.
# 12 dB validated across two very different recordings (mean -27.6 and -30.2 dB);
# the loud-failure guard below backstops any recording it still doesn't fit.
SILENCE_MARGIN_DB = float(os.environ.get("AUTOCUT_SILENCE_MARGIN_DB", "12"))
# Sane bounds for the derived threshold, guarding against a bad measurement.
SILENCE_DB_CLAMP = (-55.0, -25.0)
# Used only if the level measurement fails outright.
SILENCE_NOISE_DB_FALLBACK = -35.0
# Below this much audio, "zero silences" is not necessarily anomalous.
SILENCE_SANITY_MIN_DUR = 30.0

_SILENCE_START = re.compile(r"silence_start:\s*([0-9.]+)")
_SILENCE_END = re.compile(r"silence_end:\s*([0-9.]+)\s*\|\s*silence_duration:\s*([0-9.]+)")
_MEAN_VOLUME = re.compile(r"mean_volume:\s*(-?[0-9.]+)\s*dB")


def _measure_mean_volume(ep: Episode) -> float | None:
    """Mean volume of the speech audio in dBFS, via one volumedetect pass.

    Returns None if the value can't be parsed (e.g. ffmpeg output changed)."""
    stderr = ffmpeg.run_ffmpeg_capture_stderr([
        "-i", ep.speech_wav, "-af", "volumedetect", "-f", "null", "-",
    ])
    m = _MEAN_VOLUME.search(stderr)
    return float(m.group(1)) if m else None


def _silence_threshold(ep: Episode) -> tuple[float, str, float | None]:
    """Choose the silencedetect noise threshold for this episode.

    Returns (noise_db, source, mean_volume_db). ``source`` is 'override',
    'derived', or 'fallback'. A manual override wins:
    ``AUTOCUT_SILENCE_NOISE_DB=-42`` forces the threshold; ``AUTOCUT_SILENCE_MARGIN_DB``
    tunes the derived margin below mean volume.
    """
    override = os.environ.get("AUTOCUT_SILENCE_NOISE_DB")
    if override is not None:
        return float(override), "override", None
    mean = _measure_mean_volume(ep)
    if mean is None:
        log.warning("silence: could not measure audio level; using fallback %.0f dB",
                    SILENCE_NOISE_DB_FALLBACK)
        return SILENCE_NOISE_DB_FALLBACK, "fallback", None
    lo, hi = SILENCE_DB_CLAMP
    db = round(max(lo, min(hi, mean - SILENCE_MARGIN_DB)), 1)
    return db, "derived", round(mean, 1)


# Handles returned by os.add_dll_directory remove their directory from the
# search path when garbage-collected. Keep them alive for the process lifetime
# so ctranslate2's load-time dependency resolution can't intermittently break.
_cuda_dll_handles: list[object] = []


def _add_cuda_dll_dirs() -> None:
    """Make the pip-installed CUDA runtime DLLs loadable.

    Two distinct load mechanisms need two distinct fixes (both verified against
    ctranslate2 4.5.0 + nvidia-cudnn-cu12 9.x on Windows 11):

    1. Load-time deps of ctranslate2's .pyd (cublas64_12.dll, cudnn64_9.dll).
       Python 3.8+ resolves an extension module's dependency DLLs through a
       secure search that ignores PATH, so os.add_dll_directory on the
       nvidia-*-cu12 bin dirs is required — PATH alone does nothing here.

    2. cuDNN 9's runtime-loaded sublibraries (cudnn_ops64_9.dll and friends).
       cudnn64_9.dll pulls these in with a plain LoadLibrary when the first
       tensor op runs. That nested load does NOT consult add_dll_directory
       entries and only searches PATH, so the same dirs must go on PATH too.
       Without the PATH prepend you get, at first inference:
         Could not locate cudnn_ops64_9.dll ...
         Invalid handle. Cannot load symbol cudnnCreateTensorDescriptor

    Must run before faster_whisper (hence ctranslate2) is imported.
    """
    if os.name != "nt":
        return
    try:
        import nvidia
    except ImportError:
        return
    from pathlib import Path
    # nvidia is a PEP 420 namespace package (no __file__); use __path__.
    for root in nvidia.__path__:
        for pkg in Path(root).iterdir():
            bin_dir = pkg / "bin"
            if bin_dir.is_dir():
                _cuda_dll_handles.append(os.add_dll_directory(str(bin_dir)))
                os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ["PATH"]


def _transcribe_words(ep: Episode) -> dict:
    # Imported lazily so the rest of the CLI works without CUDA/torch present.
    _add_cuda_dll_dirs()
    from faster_whisper import WhisperModel

    model = WhisperModel(
        MODEL,
        device=DEVICE,
        compute_type=COMPUTE_TYPE,
        revision=MODEL_REVISION,
    )
    segments, info = model.transcribe(
        str(ep.speech_wav),
        word_timestamps=True,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 300},
        beam_size=BEAM_SIZE,
        # Off: leaving it on makes Whisper hallucinate repeated phrases during
        # long silences — exactly the material we're trying to cut.
        condition_on_previous_text=False,
    )

    words = []
    i = 0
    for seg in segments:
        for w in (seg.words or []):
            words.append({
                "i": i,
                "start": round(w.start, 3),
                "end": round(w.end, 3),
                "word": w.word.strip(),
                "prob": round(w.probability, 3),
            })
            i += 1

    return {
        "language": info.language,
        "model": MODEL,
        "model_revision": MODEL_REVISION,
        "compute_type": COMPUTE_TYPE,
        "beam_size": BEAM_SIZE,
        "words": words,
    }


def _speech_duration(ep: Episode, words: dict) -> float:
    """Audio duration in seconds: the source duration from probe.json, or the
    last word's end as a fallback. Used only for the sanity check."""
    if ep.probe_json.exists():
        try:
            d = json.loads(ep.probe_json.read_text(encoding="utf-8")).get("source_duration")
            if d is not None:
                return float(d)
        except (json.JSONDecodeError, OSError, TypeError, ValueError):
            pass
    ws = words.get("words") or []
    return float(ws[-1]["end"]) if ws else 0.0


def _detect_silence(ep: Episode, noise_db: float) -> list[dict]:
    """Silence intervals from an independent silencedetect pass at ``noise_db``."""
    stderr = ffmpeg.run_ffmpeg_capture_stderr([
        "-i", ep.speech_wav,
        "-af", f"silencedetect=noise={noise_db}dB:d={SILENCE_MIN_DUR}",
        "-f", "null", "-",
    ])
    intervals = []
    pending_start: float | None = None
    for line in stderr.splitlines():
        m = _SILENCE_START.search(line)
        if m:
            pending_start = float(m.group(1))
            continue
        m = _SILENCE_END.search(line)
        if m and pending_start is not None:
            end = float(m.group(1))
            dur = float(m.group(2))
            intervals.append({
                "start": round(pending_start, 3),
                "end": round(end, 3),
                "duration": round(dur, 3),
            })
            pending_start = None

    return intervals


def run(ep: Episode, *, force: bool = False) -> dict:
    """Execute stage 2. Cached on the speech audio hash + model parameters."""
    if not ep.speech_wav.exists():
        raise FileNotFoundError(
            f"Missing {ep.speech_wav}. Run 'autocut probe {ep.episode_id}' first."
        )

    input_hash = cache.hash_inputs({
        "speech_sha256": cache.hash_file(ep.speech_wav),
        "model": MODEL,
        "model_revision": MODEL_REVISION,
        "compute_type": COMPUTE_TYPE,
        "beam_size": BEAM_SIZE,
        # The threshold is derived from the (hashed) audio, but the margin/override
        # that shape it must invalidate the cache when changed.
        "silence_margin_db": SILENCE_MARGIN_DB,
        "silence_noise_override": os.environ.get("AUTOCUT_SILENCE_NOISE_DB"),
        "silence_min_dur": SILENCE_MIN_DUR,
    })

    if (
        not force
        and cache.is_current(ep.transcript_dir, input_hash)
        and ep.words_json.exists()
        and ep.silence_json.exists()
    ):
        log.info("transcribe: cache hit for %s, skipping", ep.episode_id)
        return json.loads(ep.words_json.read_text(encoding="utf-8"))

    # Dry-run must not load the (heavy, GPU) Whisper model or write outputs. It
    # also means dry-run never imports faster_whisper, so it works on machines
    # without the CUDA stack installed. Reuse an existing transcript if present.
    if ffmpeg.is_dry_run():
        log.info("transcribe: dry-run - skipping Whisper model run and silence pass")
        if ep.words_json.exists():
            return json.loads(ep.words_json.read_text(encoding="utf-8"))
        return {"words": []}

    ep.transcript_dir.mkdir(parents=True, exist_ok=True)

    log.info("transcribe: running %s on %s", MODEL, ep.speech_wav)
    words = _transcribe_words(ep)
    ep.words_json.write_text(json.dumps(words, indent=2), encoding="utf-8")

    log.info("transcribe: silence pass")
    noise_db, source, mean_db = _silence_threshold(ep)
    intervals = _detect_silence(ep, noise_db)
    log.info("transcribe: silence threshold %.1f dB (%s%s) -> %d silences",
             noise_db, source,
             f", mean {mean_db} dB" if mean_db is not None else "", len(intervals))

    # Fail loud: zero silences across real speech is a broken result, not a valid
    # one. Refuse to write an empty silence.json that would look like success and
    # then starve autoauthor (spec: no silent-success failure modes).
    speech_dur = _speech_duration(ep, words)
    if not intervals and speech_dur > SILENCE_SANITY_MIN_DUR:
        raise RuntimeError(
            f"silencedetect found 0 silences in {speech_dur:.0f}s of audio at "
            f"{noise_db} dB ({source}): implausible for speech. The threshold likely "
            f"does not match this recording's noise floor. Override with "
            f"AUTOCUT_SILENCE_NOISE_DB=<dB> (try a few dB lower) or tune "
            f"AUTOCUT_SILENCE_MARGIN_DB, then re-run: autocut transcribe {ep.episode_id} --force"
        )

    silence = {
        "noise_db": noise_db,
        "noise_db_source": source,
        "mean_volume_db": mean_db,
        "min_duration": SILENCE_MIN_DUR,
        "silences": intervals,
    }
    ep.silence_json.write_text(json.dumps(silence, indent=2), encoding="utf-8")

    cache.mark_done(ep.transcript_dir, input_hash, extra={"stage": "transcribe"})
    return words
