"""Reaction format — playback refinement (reaction spec section 4).

Three problems with the pure-arithmetic estimate (align.py build steps
1-3), all confirmed against a real episode:

1. **Boundary placement.** host_in/host_out were snapped to confirmed
   silence *around* the cue, not to where the source's bleed actually
   starts/stops in the host recording. The gap between them — click
   latency after the "end my commentary" cue, the pause before "begin my
   commentary" — gets counted as playback duration that never happened,
   inflating every segment's source_out by however long that gap was.
   Fixed by cross-correlation (``_detect_onset``/``_detect_offset``).
2. **Residual drift.** Even with correct boundaries, host and source clocks
   aren't perfectly locked — a real episode showed residual drift *within*
   a single segment, on the order of ~1% of its length. Fixed by sampling
   several windows across the segment and taking their consensus
   (``_mid_segment_consensus``).
3. **A chain that isn't actually self-healing.** The first version of this
   module anchored each segment's cross-correlation search to the
   *previous* segment's corrected position, searching only a narrow window
   (±3s) around it. That's fine when the estimate is close — but once one
   segment's error exceeds the search window (confirmed: a genuinely
   correct match scored 0.40, just under the confidence floor, on a real
   episode — not a case of "the answer wasn't there", the window was
   centred wrong), the correlation can't find the true position, and every
   later segment inherits the same error with no way to recover: spec
   section 4's "large disagreement means a seek" can't fire when the
   window is too narrow to ever see the disagreement.

   Fixed by decoupling location from the chain entirely. Per segment:
   (a) **global locate** — text-align the host's own bleed-word transcript
   (already in words.json) against the source's full transcript
   (``source_words_json``, transcribed once up front) to find the segment's
   true source position directly from content, independent of every other
   segment. This is exactly what the original (pre-cue) reaction design
   tried and couldn't do — quoting or paraphrasing the source reads
   identically to actually playing it in an unconstrained transcript search
   — except now cue detection has already confirmed *this specific host
   span is really playback*, so a text match inside it can be trusted.
   (b) **local refine** — cross-correlate in a narrow window around that
   independent estimate, exactly as before, for sub-word precision.
   (c) the old cumulative-arithmetic chain is kept only as a **sanity
   check**: when it disagrees with the text-located position by more than
   a few seconds, the segment is flagged (``seek_detected``) rather than
   the disagreement being silently absorbed.
"""

from __future__ import annotations

import difflib
import logging
import re
from pathlib import Path

import numpy as np

from . import ffmpeg

log = logging.getLogger("autocut.refine")

MIN_LOCATE_WORDS = 8   # matched words required to trust a global text-locate

SR = 8000            # working sample rate for correlation (Nyquist 4kHz > the 3.5kHz band)
BAND = (300.0, 3500.0)   # bleed has been through speakers, a room, and a mic (spec §4)
ENV_SR = 200          # envelope sample rate — coarse and cheap to correlate

REF_LEN = 2.5          # seconds of reference audio used to locate a boundary
ONSET_SEARCH_BACK = 0.5
ONSET_SEARCH_FWD = 3.5
OFFSET_SEARCH_BACK = 4.5
OFFSET_SEARCH_FWD = 0.5
EDGE_STEP = 0.1         # scan granularity when hunting for the offset's correlation drop

MID_SAMPLES = 3         # windows sampled across a segment for residual-drift consensus
MID_SEARCH_SLACK = 2.0  # seconds either side of the arithmetic estimate

MIN_STRENGTH = 0.45     # normalised-correlation floor to trust a match (calibrated against reaction003: true matches ~0.78-0.99, unrelated content ~0.23-0.27)
SEEK_FLAG_SECONDS = 2.5  # a correction bigger than this from the naive estimate is a likely seek


# --------------------------------------------------------------------------- #
# Signal primitives (numpy-only).
# --------------------------------------------------------------------------- #

def _read_pcm(path: Path, start: float, dur: float, sr: int = SR) -> np.ndarray:
    """Mono float32 PCM for ``[start, start+dur)``. Clamps ``start`` at 0."""
    start = max(0.0, start)
    if dur <= 0:
        return np.zeros(0, dtype=np.float32)
    raw = ffmpeg.run_ffmpeg_capture_stdout_bytes([
        "-ss", f"{start:.3f}", "-t", f"{dur:.3f}", "-i", str(path),
        "-map", "0:a:0", "-ac", "1", "-ar", str(sr), "-f", "f32le", "-",
    ])
    return np.frombuffer(raw, dtype="<f4")


def _bandpass_kernel(sr: int, lo: float, hi: float, numtaps: int = 127) -> np.ndarray:
    """Windowed-sinc FIR bandpass kernel (numpy-only — no scipy.signal)."""
    n = np.arange(numtaps) - (numtaps - 1) / 2.0

    def _lowpass(cutoff: float) -> np.ndarray:
        return 2 * cutoff / sr * np.sinc(2 * cutoff / sr * n)

    kernel = _lowpass(hi) - _lowpass(lo)
    kernel *= np.hamming(numtaps)
    return kernel


def _bandlimit(x: np.ndarray, sr: int, lo: float, hi: float) -> np.ndarray:
    """Band-limit via a localised FIR convolution, not a whole-buffer FFT
    brick-wall filter: the latter's ideal (hard-cutoff) frequency response
    has a slowly-decaying sinc impulse response, and because the FFT treats
    the buffer as one periodic block, a loud transient anywhere in a
    multi-second read can ring across the ENTIRE buffer — including
    genuinely silent regions seconds away, corrupting envelope correlation
    right where boundary detection depends on silence reading as silence
    (confirmed: a synthetic all-zero region measured a false ~0.87
    correlation strength before this fix). A short FIR kernel's influence is
    bounded to its own tap span (~2.6ms here), so it can't do that."""
    n = len(x)
    if n == 0:
        return x
    kernel = _bandpass_kernel(sr, lo, hi)
    return np.convolve(x, kernel, mode="same")


def envelope(x: np.ndarray, sr: int = SR, env_sr: int = ENV_SR) -> np.ndarray:
    """Band-limited rectified-and-smoothed amplitude envelope, downsampled to
    ``env_sr`` — robust to the phase/timbre changes bleed picks up going
    through speakers, a room, and a mic, unlike raw-sample correlation."""
    if len(x) == 0:
        return np.zeros(0, dtype=np.float64)
    bl = _bandlimit(x.astype(np.float64), sr, *BAND)
    rectified = np.abs(bl)
    win = max(1, int(round(sr / env_sr)))
    kernel = np.ones(win) / win
    smoothed = np.convolve(rectified, kernel, mode="same")
    return smoothed[::win]


def _best_lag(query_env: np.ndarray, ref_env: np.ndarray, env_sr: int = ENV_SR) -> tuple[float, float]:
    """Where ``ref_env`` best matches inside ``query_env``: ``(lag_seconds,
    strength)`` such that ``query_env[lag : lag+len(ref_env)]`` is the best
    match, and ``strength`` is a normalised correlation (~1.0 = a strong
    match, ~0.0 = none). ``(0.0, 0.0)`` if either is degenerate."""
    if len(query_env) < len(ref_env) or len(ref_env) == 0:
        return 0.0, 0.0
    q, r = query_env - query_env.mean(), ref_env - ref_env.mean()
    r_norm = np.linalg.norm(r)
    if r_norm < 1e-9:
        return 0.0, 0.0
    corr = np.correlate(q, r, mode="valid")  # length len(q)-len(r)+1; index i == lag i
    # Per-position normalisation (query energy varies a lot across a scan —
    # a flat r_norm-only normalisation would bias toward the loudest offset).
    q_sq = q ** 2
    window_energy = np.convolve(q_sq, np.ones(len(r)), mode="valid")
    denom = np.sqrt(window_energy) * r_norm
    denom[denom < 1e-9] = np.inf
    corr_norm = corr / denom
    best_idx = int(np.argmax(corr_norm))
    return best_idx / env_sr, float(corr_norm[best_idx])


def _match(host_av: Path, source_path: Path, host_t: float, source_t: float,
          *, ref_len: float = REF_LEN, search: float = MID_SEARCH_SLACK) -> tuple[float, float]:
    """Correlate a ``ref_len``-second host clip at ``host_t`` against a
    ``search``-padded source window centred on ``source_t``. Returns
    ``(matched_source_time, strength)`` — the source time the host clip best
    matches, and how strongly."""
    host_env = envelope(_read_pcm(host_av, host_t, ref_len))
    src_start = source_t - search
    source_env = envelope(_read_pcm(source_path, src_start, ref_len + 2 * search))
    lag, strength = _best_lag(source_env, host_env)
    return src_start + lag, strength


# --------------------------------------------------------------------------- #
# Boundary detection — bleed onset/offset, not the cue-adjacent silence.
# --------------------------------------------------------------------------- #

def _detect_onset(host_av: Path, source_path: Path, approx_host_in: float,
                  source_in_base: float) -> tuple[float, bool]:
    """Where bleed actually begins: search host time in
    ``[approx_host_in - ONSET_SEARCH_BACK, approx_host_in + ONSET_SEARCH_FWD]``
    for where a clip of ``source[source_in_base : +REF_LEN]`` first matches.
    Returns ``(host_in_true, found)``."""
    search_start = approx_host_in - ONSET_SEARCH_BACK
    search_len = ONSET_SEARCH_BACK + ONSET_SEARCH_FWD + REF_LEN
    host_env = envelope(_read_pcm(host_av, search_start, search_len))
    ref_env = envelope(_read_pcm(source_path, source_in_base, REF_LEN))
    lag, strength = _best_lag(host_env, ref_env)
    if strength < MIN_STRENGTH:
        return approx_host_in, False
    return search_start + lag, True


def _detect_offset(host_av: Path, source_path: Path, host_in_true: float,
                   approx_host_out: float, source_in_base: float) -> tuple[float, bool]:
    """Where bleed actually stops. The boundary is a correlation *drop*, not
    a peak, so it needs a scan rather than a single best-lag search: step
    forward from just after ``host_in_true`` to
    ``approx_host_out + OFFSET_SEARCH_FWD``, tracking the correlation
    strength of a short host clip against wherever the running arithmetic
    mapping predicts it in source time. Stops at the first SUSTAINED drop
    (``LOW_RUN_STOP`` consecutive low-strength steps) rather than scanning
    the whole range and taking the last high point seen anywhere in it — a
    later spurious blip (a host clip's tail overlapping into the next real
    speech, matching *something* in the source purely on having energy
    present, unrelated to the actual content) would otherwise re-extend the
    boundary past a genuine drop; confirmed on synthetic audio, where a
    momentary post-silence match pushed the detected offset 1.6s past the
    true one before this fix. Returns ``(host_out_true, found)`` — the end of
    the clip at the start of the confirmed drop. Two ffmpeg reads total (the
    full scan range for each side), not one per step — the per-step
    correlation is done on the already-loaded envelopes.
    """
    ref_len = 0.5   # short — a longer clip smears the edge (correlates "good enough" even
                     # once most of it has fallen into dead air, overshooting the true edge)
    t0 = max(host_in_true + 1.0, approx_host_out - OFFSET_SEARCH_BACK)
    t_end = approx_host_out + OFFSET_SEARCH_FWD
    if t0 >= t_end:
        return approx_host_out, False

    host_env = envelope(_read_pcm(host_av, t0, t_end - t0 + ref_len))
    src_lo = source_in_base + (t0 - host_in_true) - MID_SEARCH_SLACK
    src_hi = source_in_base + (t_end - host_in_true) + ref_len + MID_SEARCH_SLACK
    source_env = envelope(_read_pcm(source_path, src_lo, src_hi - src_lo))

    ref_len_env = int(round(ref_len * ENV_SR))
    n_steps = int(round((t_end - t0) / EDGE_STEP))
    LOW_RUN_STOP = 3  # consecutive low-strength steps (0.3s) to confirm a sustained drop
    last_good: float | None = None
    low_run = 0
    for i in range(n_steps):
        t = t0 + i * EDGE_STEP
        host_idx = int(round((t - t0) * ENV_SR))
        clip = host_env[host_idx:host_idx + ref_len_env]
        expected_source = source_in_base + (t - host_in_true)
        lo_idx = max(0, int(round((expected_source - MID_SEARCH_SLACK - src_lo) * ENV_SR)))
        hi_idx = int(round((expected_source + MID_SEARCH_SLACK + ref_len - src_lo) * ENV_SR))
        window = source_env[lo_idx:hi_idx]
        _, strength = _best_lag(window, clip)
        if strength >= MIN_STRENGTH:
            last_good = t  # the clip's own START, not its end — a longer clip
                            # extending past the edge would still often read as
                            # "good enough" and overshoot the true boundary
            low_run = 0
        else:
            low_run += 1
            if last_good is not None and low_run >= LOW_RUN_STOP:
                break
    if last_good is None:
        return approx_host_out, False
    return min(last_good, t_end), True


def _mid_segment_consensus(host_av: Path, source_path: Path, host_in_true: float,
                           host_out_true: float, source_in_base: float) -> tuple[float, float, bool]:
    """Sample ``MID_SAMPLES`` windows across the segment, each correlated
    against where the running arithmetic mapping predicts it in source time.
    Returns ``(median_residual_seconds, spread_seconds, all_matched)`` — the
    residual is the small per-segment correction spec §4 exists for even
    after the boundaries themselves are right; the spread flags disagreement
    (windows disagreeing is a signal in itself — spec §4)."""
    duration = host_out_true - host_in_true
    if duration <= 1.0:
        return 0.0, 0.0, True
    residuals = []
    matched = True
    for frac in np.linspace(0.2, 0.8, MID_SAMPLES):
        t = host_in_true + frac * duration
        expected_source = source_in_base + (t - host_in_true)
        matched_source, strength = _match(host_av, source_path, t, expected_source)
        if strength < MIN_STRENGTH:
            matched = False
            continue
        residuals.append(matched_source - expected_source)
    # NOTE: tried requiring >=2 of MID_SAMPLES matched samples here (a single
    # match is one data point, not spec §4's "consensus"). On the real
    # episode it made things WORSE overall: rejecting a lone-but-correct
    # match for a segment with genuinely quiet/hard-to-correlate bleed fell
    # back to the old cue-adjacent-silence boundary for THAT segment,
    # reintroducing its duration-inflation bug, which then poisoned every
    # segment after it (source_in_base cascades). Accepting a single
    # confident match is the better trade-off empirically, even though it's
    # not literally spec's "several windows" consensus — see sync_check.py
    # for measuring which segments this actually leaves off.
    if not residuals:
        return 0.0, 0.0, False
    arr = np.array(residuals)
    return float(np.median(arr)), float(arr.max() - arr.min()), matched


# --------------------------------------------------------------------------- #
# Global locate — text alignment, independent of every other segment.
# --------------------------------------------------------------------------- #

def _norm_word(token: str) -> str:
    return re.sub(r"[^a-z]", "", token.lower())


def _global_locate(host_words_seg: list[dict], source_words: list[dict],
                   host_in: float) -> tuple[float | None, int]:
    """The source position that aligns with ``host_in``, found by text-
    matching this segment's own bleed words against the FULL source
    transcript — no dependency on any other segment's result. Cue detection
    has already confirmed this host span really is playback (not quoting),
    so a text match here can be trusted the way an unconstrained transcript
    search couldn't be (see module docstring).

    For every matched word pair, the source position implied by anchoring at
    ``host_in`` is ``source_word.start - (host_word.start - host_in)``; the
    median across all matches is the estimate. Returns ``(source_in, n)`` —
    ``(None, n)`` if fewer than ``MIN_LOCATE_WORDS`` words matched.
    """
    if len(host_words_seg) < MIN_LOCATE_WORDS:
        return None, 0
    host_tokens = [_norm_word(w["word"]) for w in host_words_seg]
    source_tokens = [_norm_word(w["word"]) for w in source_words]
    # autojunk=True here (unlike sync_check's per-segment comparison): the
    # source transcript is the whole episode, so very common short words
    # genuinely are "junk" for alignment purposes at this scale, and
    # disabling the heuristic would be a real performance cliff.
    matcher = difflib.SequenceMatcher(a=host_tokens, b=source_tokens, autojunk=True)
    implied: list[float] = []
    for block in matcher.get_matching_blocks():
        for k in range(block.size):
            hw = host_words_seg[block.a + k]
            sw = source_words[block.b + k]
            implied.append(float(sw["start"]) - (float(hw["start"]) - host_in))
    if len(implied) < MIN_LOCATE_WORDS:
        return None, len(implied)
    implied.sort()
    n = len(implied)
    median = implied[n // 2] if n % 2 else (implied[n // 2 - 1] + implied[n // 2]) / 2
    return median, len(implied)


# --------------------------------------------------------------------------- #
# Orchestration.
# --------------------------------------------------------------------------- #

def refine_segments(host_av: Path, source_path: Path, raw_segments: list[dict], host_fps: float,
                    host_words: list[dict], source_words: list[dict]) -> list[dict]:
    """Playback refinement (reaction spec section 4) — see module docstring
    point 3 for why each segment is located independently (global text
    locate) before any cross-correlation (local refine), rather than
    chaining each segment's search window from the previous segment's
    result. ``host_words`` is the full episode's words.json; ``source_words``
    is the source's own full transcript (``source_words_json``).
    """
    from . import edl  # avoids align.py needing this module (and numpy) just to import

    segments: list[dict] = []
    chain_pos = 0.0  # cumulative arithmetic position — kept only as a sanity check now
    for i, raw in enumerate(raw_segments):
        approx_host_in, approx_host_out = raw["host_in"], raw["host_out"]

        # 1. Global locate: text-align this segment's own bleed words against
        # the FULL source transcript. Independent of every other segment —
        # nothing here depends on chain_pos or any prior segment's result.
        host_words_seg = [w for w in host_words
                          if approx_host_in - 1.0 <= float(w["start"]) < approx_host_out + 1.0]
        located, n_matched = _global_locate(host_words_seg, source_words, approx_host_in)
        located_ok = located is not None and located >= 0.0
        anchor = located if located_ok else chain_pos

        # 2. Local refine: cross-correlate in a narrow window around the
        # located (or, only on locate failure, chain) anchor.
        host_in_true, onset_found = _detect_onset(host_av, source_path, approx_host_in, anchor)
        if not onset_found:
            host_in_true = approx_host_in
        host_out_true, offset_found = _detect_offset(
            host_av, source_path, host_in_true, approx_host_out, anchor)
        if not offset_found or host_out_true <= host_in_true + 0.5:
            host_out_true = approx_host_out

        host_in_true = edl.snap(host_in_true, host_fps)
        host_out_true = edl.snap(host_out_true, host_fps)
        duration = host_out_true - host_in_true

        median_residual, spread, mid_matched = _mid_segment_consensus(
            host_av, source_path, host_in_true, host_out_true, anchor)

        source_in = max(0.0, anchor + median_residual)
        source_out = source_in + duration

        # 3. Sanity check: the old cumulative-arithmetic chain, compared
        # against the independently text-located position. Disagreement
        # beyond a few seconds is exactly spec §4's "the source was moved" —
        # and unlike the old chained-search design, this can actually fire,
        # since the chain no longer determines where refinement even looks.
        chain_disagreement = abs(source_in - chain_pos)

        notes = []
        if not located_ok:
            notes.append(f"global locate failed ({n_matched} word(s) matched) — fell back to the arithmetic chain")
        if not onset_found:
            notes.append("onset not confidently located (kept the silence-snapped boundary)")
        if not offset_found:
            notes.append("offset not confidently located (kept the silence-snapped boundary)")
        if not mid_matched:
            notes.append("mid-segment consensus incomplete — no residual correction applied")

        seek_detected = (not located_ok or chain_disagreement > SEEK_FLAG_SECONDS
                         or spread > SEEK_FLAG_SECONDS or not mid_matched)
        if notes:
            log.warning("refine: pb%03d — %s", i + 1, "; ".join(notes))
        if located_ok and chain_disagreement > SEEK_FLAG_SECONDS:
            log.warning("refine: pb%03d — text-located position disagrees with the arithmetic "
                       "chain by %.2fs — flagged as a likely seek", i + 1, chain_disagreement)

        segments.append({
            "id": f"pb{i + 1:03d}",
            "host_in": round(host_in_true, 3),
            "host_out": round(host_out_true, 3),
            "source_in": round(source_in, 3),
            "source_out": round(source_out, 3),
            "offset_source": "text_located" if located_ok else "chain_fallback",
            "refinement_delta": round(median_residual, 3),
            "seek_detected": seek_detected,
        })
        chain_pos = source_out
    return segments
