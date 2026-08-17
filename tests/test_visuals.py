"""Illustration generation (visuals spec sections 4, 8, 9).

The image backend is swappable and the real one is hosted; these tests run the
deterministic `placeholder` backend end to end and cover the parts that must be
correct regardless of backend: prompt assembly, the always-on negative,
deterministic seeds, per-candidate caching, resume, and the recorded manifest.
"""

import json
import struct

import pytest

from autocut import visuals
from autocut.paths import Episode


STYLE = {
    "version": 3,
    "style_fragment": "flat vector illustration, limited palette",
    "negative_fragment": "photograph, photorealistic, text",
    "variants": {"engraving": "pen-and-ink engraving"},
    "params": {"model": "gemini-2.5-flash-image"},
}


# --- prompt assembly ------------------------------------------------------- #

def test_prompt_is_style_plus_variant_plus_concept():
    shot = {"concept": "a stone temple at dusk", "variant": "engraving"}
    pos, neg = visuals.assemble_prompt(STYLE, shot)
    assert pos == "flat vector illustration, limited palette pen-and-ink engraving a stone temple at dusk"
    assert neg == "photograph, photorealistic, text"


def test_negative_is_always_present_even_without_variant():
    pos, neg = visuals.assemble_prompt(STYLE, {"concept": "a river"})
    assert "engraving" not in pos and pos.endswith("a river")
    assert neg  # anti-photorealism terms are load-bearing


def test_unknown_variant_is_ignored_not_injected():
    pos, _ = visuals.assemble_prompt(STYLE, {"concept": "a city", "variant": "nope"})
    assert pos == "flat vector illustration, limited palette a city"


# --- deterministic seeds --------------------------------------------------- #

def test_seed_is_deterministic_and_per_candidate():
    assert visuals._seed("sh001", 1) == visuals._seed("sh001", 1)
    assert visuals._seed("sh001", 1) != visuals._seed("sh001", 2)
    assert visuals._seed("sh001", 1) != visuals._seed("sh002", 1)


# --- PNG encoder produces a valid file ------------------------------------- #

def test_placeholder_writes_a_valid_png():
    backend = visuals.PlaceholderBackend(size=(64, 48))
    data = backend.generate(prompt="x", negative="y", seed=7, references=[])
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    w, h = struct.unpack(">II", data[16:24])  # IHDR width/height
    assert (w, h) == (64, 48)


# --- end to end with the placeholder backend ------------------------------- #

def _episode(tmp_path):
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "layout.yaml").write_text(
        "visuals:\n  backend: placeholder\n  candidates: 3\n  size: [48, 32]\n",
        encoding="utf-8",
    )
    style_dir = tmp_path / "styles" / "default"
    style_dir.mkdir(parents=True)
    (style_dir / "style.yaml").write_text(json.dumps(STYLE), encoding="utf-8")

    ep = Episode(episode_id="ep", root=tmp_path)
    ep.work.mkdir(parents=True)
    shotlist = {"episode_id": "ep", "style": "default", "shots": [
        {"id": "sh001", "kind": "illustration", "concept": "a temple", "variant": "engraving"},
        {"id": "sh002", "kind": "illustration", "concept": "a river", "variant": ""},
        {"id": "sh003", "kind": "infographic", "composition": "bar_chart", "props": {"a": 1}},
    ]}
    ep.shotlist_json.write_text(json.dumps(shotlist), encoding="utf-8")
    return ep


def test_run_generates_candidates_only_for_illustrations(tmp_path):
    ep = _episode(tmp_path)
    manifest = visuals.run(ep)

    # infographic shot is skipped; 2 illustrations x 3 candidates.
    assert [s["id"] for s in manifest["shots"]] == ["sh001", "sh002"]
    pngs = sorted(p.name for p in ep.visuals_dir.glob("*.png"))
    assert pngs == ["sh001_1.png", "sh001_2.png", "sh001_3.png",
                    "sh002_1.png", "sh002_2.png", "sh002_3.png"]


def test_run_records_prompt_seed_model_and_style(tmp_path):
    ep = _episode(tmp_path)
    manifest = visuals.run(ep)
    rec = manifest["shots"][0]
    assert rec["style"] == "default" and rec["style_version"] == 3
    assert rec["model"] == "placeholder-v1"
    assert "flat vector illustration" in rec["prompt"]
    assert rec["negative"] == "photograph, photorealistic, text"
    # every candidate carries its own seed + cache key
    seeds = [c["seed"] for c in rec["candidates"]]
    assert len(set(seeds)) == 3 and all(c["key"] for c in rec["candidates"])


def _count_generate(monkeypatch):
    """Count backend.generate calls so cache/resume behaviour is observable
    without relying on filesystem mtime resolution."""
    calls = []
    orig = visuals.PlaceholderBackend.generate

    def counting(self, **kw):
        calls.append(kw["seed"])
        return orig(self, **kw)

    monkeypatch.setattr(visuals.PlaceholderBackend, "generate", counting)
    return calls


def test_run_is_resumable_and_cached(tmp_path, monkeypatch):
    ep = _episode(tmp_path)
    calls = _count_generate(monkeypatch)
    visuals.run(ep)
    assert len(calls) == 6  # 2 illustrations x 3 candidates

    calls.clear()
    visuals.run(ep)  # everything current -> regenerate nothing
    assert calls == []


def test_style_version_change_invalidates_cache(tmp_path, monkeypatch):
    ep = _episode(tmp_path)
    calls = _count_generate(monkeypatch)
    visuals.run(ep)
    calls.clear()

    bumped = dict(STYLE, version=4)
    (ep.styles_dir / "default" / "style.yaml").write_text(json.dumps(bumped), encoding="utf-8")
    visuals.run(ep)
    assert len(calls) == 6  # version is part of the cache key -> full regen


def test_missing_shotlist_is_a_clean_error(tmp_path):
    ep = _episode(tmp_path)
    ep.shotlist_json.unlink()
    with pytest.raises(FileNotFoundError):
        visuals.run(ep)
