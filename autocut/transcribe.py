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

SILENCE_NOISE_DB = -35
SILENCE_MIN_DUR = 0.4

_SILENCE_START = re.compile(r"silence_start:\s*([0-9.]+)")
_SILENCE_END = re.compile(r"silence_end:\s*([0-9.]+)\s*\|\s*silence_duration:\s*([0-9.]+)")


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


def _detect_silence(ep: Episode) -> dict:
    stderr = ffmpeg.run_ffmpeg_capture_stderr([
        "-i", ep.speech_wav,
        "-af", f"silencedetect=noise={SILENCE_NOISE_DB}dB:d={SILENCE_MIN_DUR}",
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

    return {
        "noise_db": SILENCE_NOISE_DB,
        "min_duration": SILENCE_MIN_DUR,
        "silences": intervals,
    }


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
        "silence_noise_db": SILENCE_NOISE_DB,
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
    silence = _detect_silence(ep)
    ep.silence_json.write_text(json.dumps(silence, indent=2), encoding="utf-8")

    cache.mark_done(ep.transcript_dir, input_hash, extra={"stage": "transcribe"})
    return words
