"""Playback refinement (reaction spec section 4): synthetic end-to-end check.

Builds a tiny synthetic episode where the "true" facts are known exactly —
band-limited noise as the source (real speech-like envelope variation, unlike
a pure tone), a host recording with real dead air (click latency, pre-cue
pause) injected around a copy of that noise as the "bleed" — and asserts
refine_segments recovers the true boundaries and source position despite the
inflated cue-adjacent silence a naive arithmetic estimate would have used.
"""

import numpy as np
import pytest

from autocut import ffmpeg, refine


# --------------------------------------------------------------------------- #
# Global locate: text alignment (pure logic, no audio).
# --------------------------------------------------------------------------- #

def _w(word, start):
    return {"word": word, "start": start}


def test_global_locate_finds_the_true_offset_independent_of_any_anchor():
    # Source transcript spans a long episode; the segment's bleed words
    # match a chunk far from any "naive" position — global locate must find
    # it purely from content, with no notion of where it "should" be.
    source_words = ([_w("filler", t) for t in range(0, 500, 1)]
                    + [_w(w, 500.0 + i * 0.3) for i, w in
                       enumerate("the quick brown fox jumps over the lazy dog again and again".split())])
    host_words_seg = [_w(w, 10.0 + i * 0.3) for i, w in
                      enumerate("the quick brown fox jumps over the lazy dog".split())]
    source_in, n = refine._global_locate(host_words_seg, source_words, host_in=10.0)
    assert n >= refine.MIN_LOCATE_WORDS
    assert source_in == pytest.approx(500.0, abs=0.05)


def test_global_locate_returns_none_below_minimum_matched_words():
    host_words_seg = [_w("hello", 10.0), _w("world", 10.5)]
    source_words = [_w("hello", 5.0), _w("world", 5.5)]
    source_in, n = refine._global_locate(host_words_seg, source_words, host_in=10.0)
    assert source_in is None
    assert n < refine.MIN_LOCATE_WORDS


def test_global_locate_ignores_a_coincidental_short_match_elsewhere():
    # A short, common phrase appears twice; the LONG, distinctive match
    # should dominate the median rather than a coincidental short one.
    source_words = ([_w(w, 100.0 + i * 0.3) for i, w in enumerate("okay so anyway".split())]
                    + [_w("filler", t) for t in range(200, 400)]
                    + [_w(w, 500.0 + i * 0.3) for i, w in
                       enumerate("okay so anyway lets discuss the specific detailed argument here".split())])
    host_words_seg = [_w(w, 10.0 + i * 0.3) for i, w in
                      enumerate("okay so anyway lets discuss the specific detailed argument here".split())]
    source_in, n = refine._global_locate(host_words_seg, source_words, host_in=10.0)
    assert source_in == pytest.approx(500.0, abs=0.5)


def _write_wav_from_samples(path, samples, sr=48000):
    import subprocess
    cmd = [ffmpeg.ffmpeg_path(), "-hide_banner", "-y", "-f", "f32le", "-ar", str(sr),
          "-ac", "1", "-i", "-", "-c:a", "pcm_s16le", str(path)]
    proc = subprocess.run(cmd, input=samples.astype("<f4").tobytes(),
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.returncode == 0, proc.stderr.decode(errors="replace")


def _noise(duration, seed, sr=48000, band_lo=300, band_hi=3500):
    rng = np.random.default_rng(seed)
    n = int(duration * sr)
    x = rng.normal(0, 1.0, n).astype(np.float64)
    t = np.arange(n) / sr
    mod = 0.5 + 0.5 * np.sin(2 * np.pi * 0.7 * t + seed)
    x *= mod
    spectrum = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)
    spectrum[(freqs < band_lo) | (freqs > band_hi)] = 0.0
    x = np.fft.irfft(spectrum, n)
    x /= np.max(np.abs(x)) + 1e-9
    return x * 0.8


def _silence(duration, sr=48000):
    return np.zeros(int(duration * sr), dtype=np.float64)


def _tone(duration, hz, sr=48000):
    t = np.arange(int(duration * sr)) / sr
    return 0.6 * np.sin(2 * np.pi * hz * t)


@pytest.fixture
def synthetic_episode(tmp_path):
    sr = 48000
    source_samples = _noise(20.0, seed=1, sr=sr)
    source_path = tmp_path / "source.wav"
    _write_wav_from_samples(source_path, source_samples, sr)

    # True playback: source[0.0:10.0) (10s) — segment 1 must start at source
    # 0.0 (spec §3: "source_in(1) = 0", always, unconditionally), embedded in
    # the host with real dead air the current cue-adjacent-silence snap would
    # (and does, on the real episode) count as playback duration: 0.9s click
    # latency after the cue, 1.6s pause before the next cue.
    click_latency = 0.9
    pre_cue_pause = 1.6
    bleed = source_samples[0:int(10.0 * sr)] * 0.5  # attenuated, as bleed would be

    host_parts = [
        _tone(3.0, 300.0, sr),          # commentary before
        _silence(click_latency, sr),     # click latency (silence, not bleed)
        bleed,                            # true playback, 10s
        _silence(pre_cue_pause, sr),     # pause before the stop cue
        _tone(3.0, 300.0, sr),          # commentary after
    ]
    host_samples = np.concatenate(host_parts)
    host_path = tmp_path / "host.wav"
    _write_wav_from_samples(host_path, host_samples, sr)

    true_host_in = 3.0 + click_latency       # 3.9
    true_host_out = true_host_in + 10.0      # 13.9
    return host_path, source_path, true_host_in, true_host_out


def test_refine_recovers_true_boundaries_despite_inflated_naive_estimate(synthetic_episode):
    host_path, source_path, true_host_in, true_host_out = synthetic_episode
    # The naive (cue-adjacent-silence) estimate the current build-steps-1-3
    # arithmetic would produce: dead air on both sides counted as playback,
    # exactly the confirmed bug (host_in too early, host_out too late).
    naive_host_in = 3.0
    naive_host_out = true_host_out + 1.6

    raw_segments = [{"host_in": naive_host_in, "host_out": naive_host_out}]
    # No word transcripts in this synthetic fixture -> global locate can't
    # run, falls back to the arithmetic chain (0.0 for segment 1, which
    # happens to be correct here) -> local refine (cross-correlation) still
    # does the real work this test exists to check.
    segments = refine.refine_segments(host_path, source_path, raw_segments, host_fps=30.0,
                                      host_words=[], source_words=[])

    assert len(segments) == 1
    seg = segments[0]
    assert seg["host_in"] == pytest.approx(true_host_in, abs=0.15)
    assert seg["host_out"] == pytest.approx(true_host_out, abs=0.15)
    assert seg["source_in"] == pytest.approx(0.0, abs=0.15)
    assert seg["source_out"] == pytest.approx(10.0, abs=0.3)
    # The naive estimate was inflated by click_latency + pre_cue_pause (~2.5s);
    # refinement must have actually corrected the boundaries, not passed them through.
    assert abs(seg["host_out"] - seg["host_in"] - (naive_host_out - naive_host_in)) > 1.0
