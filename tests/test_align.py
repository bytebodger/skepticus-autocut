"""Reaction alignment: coarse transcript matching, boundary snapping, the fine
cross-correlation, and the equal-duration invariant.

Pure logic + a synthetic audio round-trip. The end-to-end run (Whisper + ffmpeg
against a real fixture) and the eyeball lip-sync check are done separately."""

import wave

import numpy as np
import pytest

from autocut import align


# --------------------------------------------------------------------------- #
# Coarse transcript matching
# --------------------------------------------------------------------------- #

def _words(tokens, t0=0.0, step=0.5):
    """A words.json-shaped list from tokens, evenly spaced."""
    out = []
    for k, tok in enumerate(tokens):
        s = t0 + k * step
        out.append({"i": k, "word": tok, "start": round(s, 3), "end": round(s + step, 3)})
    return out


PHRASE = "the argument here is simple and clearly wrong about everything today".split()


def test_normalize_strips_punctuation_and_case():
    n = align.normalize_words([{"word": "The,", "start": 0, "end": 1},
                               {"word": "—", "start": 1, "end": 2}])
    assert n[0]["norm"] == "the"
    assert n[1]["norm"] == ""          # punctuation-only normalises to empty


def test_coarse_finds_embedded_playback_run():
    source = align.normalize_words(_words(PHRASE))
    # Host: some talk, then the source phrase as bleed, then more talk.
    host_tokens = "so let us watch".split() + PHRASE + "wow that was wild".split()
    host = align.normalize_words(_words(host_tokens))

    regions = align.find_coarse_regions(host, source)
    assert len(regions) == 1
    r = regions[0]
    # The matched host span is the embedded phrase (offset by the 4 lead words).
    assert r["h_start"] == 4
    assert r["h_end"] == 4 + len(PHRASE) - 1
    assert r["s_start"] == 0
    t = align.region_times(r, host, source)
    assert t["confidence"] == 1.0      # every word matched


def test_coarse_tolerates_a_garbled_bleed_word():
    source = align.normalize_words(_words(PHRASE))
    garbled = PHRASE.copy()
    garbled[5] = "clumsy"              # one bleed word mis-transcribed
    host = align.normalize_words(_words("intro here".split() + garbled + ["done"]))

    regions = align.find_coarse_regions(host, source)
    assert len(regions) == 1
    r = regions[0]
    assert r["matches"] >= align.MIN_RUN
    # The run still spans the whole phrase despite the single miss.
    assert r["h_end"] - r["h_start"] + 1 == len(PHRASE)


def test_coarse_handles_non_contiguous_plays():
    # Source has two phrases; host plays the SECOND first, then the FIRST — offsets
    # must be found independently (spec: don't assume contiguity).
    p1 = "alpha bravo charlie delta echo foxtrot".split()
    p2 = "one two three four five six seven".split()
    source = align.normalize_words(_words(p1 + p2))       # p2 starts at index 6
    host = align.normalize_words(
        _words("hi".split() + p2 + "then".split() + p1 + ["bye"]))

    regions = align.find_coarse_regions(host, source)
    assert len(regions) == 2
    # First host region matches p2 (source index 6), second matches p1 (index 0).
    first, second = regions
    assert first["s_start"] == 6
    assert second["s_start"] == 0


def test_coarse_ignores_short_incidental_matches():
    # A couple of common words in the host that also appear in source must not be
    # mistaken for playback (needs a run of MIN_RUN).
    source = align.normalize_words(_words(PHRASE))
    host = align.normalize_words(_words("the here is fine".split()))  # scattered, < MIN_RUN
    assert align.find_coarse_regions(host, source) == []


# --------------------------------------------------------------------------- #
# Boundary snapping
# --------------------------------------------------------------------------- #

def test_snap_in_pulls_to_preceding_silence_end():
    silences = [{"start": 44.0, "end": 47.05}]
    # playback start near the silence end -> snaps to it
    assert align.snap_boundary(47.2, silences, "in") == 47.05


def test_snap_out_pulls_to_following_silence_start():
    silences = [{"start": 112.6, "end": 114.0}]
    assert align.snap_boundary(112.8, silences, "out") == 112.6


def test_snap_leaves_boundary_alone_when_no_silence_within_tol():
    silences = [{"start": 10.0, "end": 11.0}]
    assert align.snap_boundary(47.2, silences, "in") == 47.2


# --------------------------------------------------------------------------- #
# Equal-duration invariant
# --------------------------------------------------------------------------- #

def test_build_segments_enforces_equal_durations():
    source = align.normalize_words(_words(PHRASE))
    host = align.normalize_words(_words("lead in now".split() + PHRASE + ["end"]))
    regions = align.find_coarse_regions(host, source)
    # A constant lag for the one region.
    lag = host[3]["start"] - source[0]["start"]
    segs = align.build_segments(regions, host, source, silences=[], lags=[(lag, 0.9)],
                                host_fps=30.0, source_dur=1000.0)
    assert len(segs) == 1
    s = segs[0]
    assert set(s) == {"id", "host_in", "host_out", "source_in", "source_out",
                      "confidence", "correlation_peak"}
    assert s["id"] == "pb001"
    assert (s["source_out"] - s["source_in"]) == pytest.approx(
        s["host_out"] - s["host_in"], abs=1e-6)
    align.assert_equal_durations(segs)   # does not raise


def test_assert_equal_durations_rejects_divergent_segment():
    bad = [{"id": "pb001", "host_in": 10.0, "host_out": 20.0,
            "source_in": 0.0, "source_out": 9.0,   # 9s vs 10s
            "confidence": 0.9, "correlation_peak": 0.8}]
    with pytest.raises(RuntimeError, match="source duration"):
        align.assert_equal_durations(bad)


def test_clamp_preserves_equal_durations_off_source_head():
    # A coarse estimate that put source_in negative: clamp shifts host too, so the
    # durations stay equal.
    hi, ho, si, so = align._clamp_to_source(47.0, 60.0, -1.5, 11.5, source_dur=1000.0)
    assert si == 0.0
    assert (so - si) == pytest.approx(ho - hi, abs=1e-9)


# --------------------------------------------------------------------------- #
# Fine alignment — synthetic audio round-trip through real wav I/O.
# --------------------------------------------------------------------------- #

SR = 16000


def _write_wav(path, sig):
    pcm = np.clip(sig, -1.0, 1.0)
    pcm = (pcm * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


def _structured_signal(seconds, seed=0):
    """Band-limited carrier with a slow, random amplitude envelope, so the energy
    envelope has structure to lock onto (stationary noise would not)."""
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    t = np.arange(n) / SR
    carrier = sum(np.sin(2 * np.pi * f * t) for f in (500, 1200, 2600))
    # slow envelope: smoothed noise at ~5 Hz
    env_raw = rng.standard_normal(int(seconds * 5) + 2)
    env = np.interp(t, np.linspace(0, seconds, len(env_raw)), env_raw)
    env = 0.5 + 0.5 * (env - env.min()) / (np.ptp(env) + 1e-9)
    return (carrier / 3.0) * env


def test_read_wav_slice_returns_requested_span(tmp_path):
    sig = _structured_signal(5.0)
    p = tmp_path / "s.wav"
    _write_wav(p, sig)
    data, sr = align.read_wav_slice(p, 1.0, 2.0)
    assert sr == SR
    assert abs(len(data) - SR) <= 1     # ~1 second of samples


def test_refine_lag_recovers_a_known_delay(tmp_path):
    lag = 1.3                            # host lags source by 1.3s
    src = _structured_signal(14.0, seed=1)
    source_path = tmp_path / "source.wav"
    _write_wav(source_path, src)

    # Host = source delayed by `lag`, degraded (band-ish noise added) as real
    # bleed would be. host[t] == source[t - lag].
    host = np.zeros(int(16.0 * SR), dtype=np.float64)
    start = int(lag * SR)
    host[start:start + len(src)] = src
    rng = np.random.default_rng(2)
    host += 0.05 * rng.standard_normal(len(host))
    host_path = tmp_path / "host.wav"
    _write_wav(host_path, host)

    # Coarse gave the anchor host time and a source estimate 0.4s off the truth.
    anchor_host = 5.0                    # true source at anchor = 5.0 - 1.3 = 3.7
    coarse = {"anchor_host": anchor_host, "anchor_source": 3.3,
              "host_in": 4.0, "host_out": 12.0}
    recovered, peak = align.refine_lag(host_path, source_path, coarse)
    assert recovered == pytest.approx(lag, abs=0.05)   # target < 50ms (spec §4)
    assert peak > 0.5
