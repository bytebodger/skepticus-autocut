"""Phase-2 compositor — audio chain (spec section 7).

highpass -> gentle broadband denoise -> compressor -> two-pass loudnorm to
YouTube's -14 LUFS. Every parameter comes from the layout ``audio`` config.

loudnorm is two-pass: an analysis pass measures the (already highpass/denoise/
compressed) signal and prints JSON stats, which feed the final pass's
``measured_*`` inputs so the normalisation is accurate rather than the looser
single-pass estimate. Both passes apply the same pre-filters, since loudnorm
must measure exactly what it will later normalise.
"""

from __future__ import annotations

import json
import logging
import re

from . import ffmpeg
from .paths import Episode

log = logging.getLogger("autocut.audio")


def _pre_loudnorm(cfg: dict) -> list[str]:
    """The filters ahead of loudnorm, in order, from config."""
    parts: list[str] = []
    hp = cfg.get("highpass")
    if hp:
        parts.append(f"highpass=f={hp}")
    afftdn = cfg.get("afftdn") or {}
    if afftdn.get("nf") is not None:
        parts.append(f"afftdn=nf={afftdn['nf']}")
    comp = cfg.get("compressor")
    if comp:
        parts.append("acompressor=" + ":".join([
            f"threshold={comp.get('threshold', -18)}dB",
            f"ratio={comp.get('ratio', 3)}",
            f"attack={comp.get('attack', 5)}",
            f"release={comp.get('release', 120)}",
        ]))
    return parts


def _targets(cfg: dict) -> tuple:
    ln = cfg.get("loudnorm") or {}
    return ln.get("I", -14), ln.get("TP", -1.5), ln.get("LRA", 11)


def analysis_af(cfg: dict) -> str:
    """Filter chain for the loudnorm analysis pass (prints JSON stats)."""
    I, TP, LRA = _targets(cfg)
    return ",".join(_pre_loudnorm(cfg) + [f"loudnorm=I={I}:TP={TP}:LRA={LRA}:print_format=json"])


def final_af(cfg: dict, measured: dict | None) -> str:
    """Filter chain for the final pass. With ``measured`` it is a linear,
    accurate second pass; without it, a single-pass fallback."""
    I, TP, LRA = _targets(cfg)
    ln = f"loudnorm=I={I}:TP={TP}:LRA={LRA}"
    if measured is not None:
        ln += (
            f":measured_I={measured['input_i']}:measured_TP={measured['input_tp']}"
            f":measured_LRA={measured['input_lra']}:measured_thresh={measured['input_thresh']}"
            f":offset={measured['target_offset']}:linear=true"
        )
    return ",".join(_pre_loudnorm(cfg) + [ln])


_JSON_RE = re.compile(r"\{[^{}]*\"input_i\"[^{}]*\}", re.S)
_MEASURED_KEYS = ("input_i", "input_tp", "input_lra", "input_thresh", "target_offset")


def analyze(ep: Episode, source: str, window: tuple, cfg: dict) -> dict | None:
    """Run the loudnorm analysis pass over the output window and return the
    measured stats, or None if they can't be parsed / are degenerate (silence),
    in which case the caller should fall back to single-pass."""
    w0, length = window
    stderr = ffmpeg.run_ffmpeg_capture_stderr([
        "-ss", f"{w0:.3f}", "-t", f"{length:.3f}", "-i", source,
        "-map", "0:a:0",
        "-af", analysis_af(cfg),
        "-f", "null", "-",
    ])
    m = _JSON_RE.search(stderr)
    if not m:
        log.warning("audio: could not parse loudnorm analysis; using single-pass")
        return None
    try:
        measured = json.loads(m.group(0))
        # Degenerate measurements (e.g. -inf on silence) can't drive a linear pass.
        for k in _MEASURED_KEYS:
            float(measured[k])
    except (json.JSONDecodeError, KeyError, ValueError):
        log.warning("audio: loudnorm analysis unusable; using single-pass")
        return None
    log.info("audio: measured input I=%s LUFS, TP=%s dB (target %s)",
             measured.get("input_i"), measured.get("input_tp"), _targets(cfg)[0])
    return {k: measured[k] for k in _MEASURED_KEYS}
