"""Audio chain construction (spec section 7). The two-pass loudnorm is verified
end-to-end against real footage separately; these cover the config-driven filter
strings and the analysis/apply split."""

from autocut import audio

CFG = {
    "highpass": 80,
    "afftdn": {"nf": -25},
    "compressor": {"threshold": -18, "ratio": 3, "attack": 5, "release": 120},
    "loudnorm": {"I": -14, "TP": -1.5, "LRA": 11},
}


def test_pre_loudnorm_chain_from_config():
    assert audio._pre_loudnorm(CFG) == [
        "highpass=f=80",
        "afftdn=nf=-25",
        "acompressor=threshold=-18dB:ratio=3:attack=5:release=120",
    ]


def test_pre_loudnorm_omits_absent_filters():
    assert audio._pre_loudnorm({"highpass": 80, "loudnorm": {}}) == ["highpass=f=80"]


def test_analysis_pass_prints_json():
    af = audio.analysis_af(CFG)
    assert af.startswith("highpass=f=80,afftdn=nf=-25,acompressor=")
    assert af.endswith("loudnorm=I=-14:TP=-1.5:LRA=11:print_format=json")


def test_final_pass_two_pass_uses_measured():
    measured = {"input_i": "-16.5", "input_tp": "-3.2", "input_lra": "7.3",
                "input_thresh": "-27.6", "target_offset": "0.3"}
    af = audio.final_af(CFG, measured)
    assert "loudnorm=I=-14:TP=-1.5:LRA=11" in af
    assert "measured_I=-16.5" in af and "measured_TP=-3.2" in af
    assert "measured_thresh=-27.6" in af and "offset=0.3" in af
    assert "linear=true" in af
    # pre-filters still lead the chain
    assert af.startswith("highpass=f=80,afftdn=nf=-25,acompressor=")


def test_final_pass_single_pass_fallback():
    af = audio.final_af(CFG, None)
    assert "loudnorm=I=-14:TP=-1.5:LRA=11" in af
    assert "measured_" not in af and "linear=true" not in af
