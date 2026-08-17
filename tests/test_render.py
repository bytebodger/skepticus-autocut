"""Render stage (visuals spec sections 3, 9).

The heavy parts — headless-Chrome capture and ffmpeg encoding — are exercised by
running the stage for real; these tests cover the deterministic orchestration
with those two mocked out: which shots are renderable, that unbuilt compositions
are skipped (not errored), the content.json shape, and that the cache key moves
with composition source / props / style version / duration.
"""

import json

import pytest

from autocut import render
from autocut.paths import Episode


def _episode(tmp_path, shots, *, built=("pull_quote",)):
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "layout.yaml").write_text(
        "canvas: {width: 3840, height: 2160, fps: from_source}\n"
        "background: {image: bg.png}\n"
        "speaker: {rect: [2640, 240, 1080, 1680], key: {color: '0x00b140'}}\n"
        "content: {rect: [160, 240, 2400, 1680]}\n", encoding="utf-8")
    (tmp_path / "bg.png").write_bytes(b"PNGSTUB")

    style = tmp_path / "styles" / "default"
    (style / "fonts").mkdir(parents=True)
    (style / "tokens.css").write_text(":root{}", encoding="utf-8")
    (style / "style.yaml").write_text("version: 2\n", encoding="utf-8")
    (style / "fonts" / "Montserrat[wght].ttf").write_bytes(b"FONTSTUB")

    for name in built:
        d = tmp_path / "compositions" / name
        d.mkdir(parents=True)
        (d / "index.html").write_text(
            '<html><head><meta name="hyperframes-surface" content="content-card">'
            "</head><body></body></html>", encoding="utf-8")

    ep = Episode(episode_id="ep", root=tmp_path)
    ep.work.mkdir(parents=True)
    ep.shotlist_json.write_text(json.dumps(
        {"episode_id": "ep", "style": "default", "shots": shots}), encoding="utf-8")
    return ep


def _shot(sid, kind, src, dur=20.0, props=None):
    s = {"id": sid, "kind": kind, "source_time": src, "duration": dur}
    if props is not None:
        s["props"] = props
    return s


@pytest.fixture
def no_browser(monkeypatch):
    """Stub the Chrome capture + ffmpeg so run() exercises only orchestration.
    Each 'rendered' shot gets a stub .mov written so content.json includes it."""
    async def fake_worker(shots, jobs, w, h, fps, tokens, font, results):
        for shot in shots:
            job = jobs[shot["id"]]
            job["out"].write_bytes(b"MOV")            # stand-in artifact
            render.cache.mark_done(job["stage_dir"], job["key"], extra={"shot": shot["id"]})
            results[shot["id"]] = True
    monkeypatch.setattr(render, "_render_worker", fake_worker)


def test_renderable_shots_produce_content_items(tmp_path, no_browser):
    ep = _episode(tmp_path, [
        _shot("sh001", "pull_quote", 10.0, props={"quote": "q"}),
        _shot("sh002", "pull_quote", 30.0, props={"quote": "q2"}),
    ], built=("pull_quote",))
    content = render.run(ep)
    assert [it["shot_id"] for it in content["items"]] == ["sh001", "sh002"]
    assert content["items"][0] == {"shot_id": "sh001", "composition": "pull_quote",
                                   "file": "sh001.mov", "source_time": 10.0, "duration": 20.0}
    assert content["style_version"] == 2
    assert (ep.visuals_dir / "sh001.mov").exists()
    assert ep.visuals_content_json.exists()


def test_unbuilt_composition_is_skipped_not_errored(tmp_path, no_browser, caplog):
    ep = _episode(tmp_path, [
        _shot("sh001", "pull_quote", 10.0, props={"quote": "q"}),
        _shot("sh002", "map", 20.0, props={"region": "x"}),        # composition not built
        _shot("sh003", "chart", 30.0, props={"series": []}),       # composition not built
    ], built=("pull_quote",))
    content = render.run(ep)                                        # must not raise
    assert [it["shot_id"] for it in content["items"]] == ["sh001"]
    assert not (ep.visuals_dir / "sh002.mov").exists()


def test_composition_without_surface_marker_is_skipped(tmp_path, no_browser):
    # A dir that exists but lacks the content-card marker (e.g. a Phase-1 overlay
    # sharing the name title_card) must be treated as not-built for this stage.
    ep = _episode(tmp_path, [
        _shot("sh001", "title_card", 10.0, props={"title": "t"}),
        _shot("sh002", "pull_quote", 20.0, props={"quote": "q"}),
    ], built=("pull_quote",))
    overlay = ep.compositions_dir / "title_card"
    overlay.mkdir(parents=True)
    (overlay / "index.html").write_text("<html><body>phase-1 overlay</body></html>", encoding="utf-8")
    content = render.run(ep)
    assert [it["shot_id"] for it in content["items"]] == ["sh002"]
    assert not (ep.visuals_dir / "sh001.mov").exists()


def test_non_composition_kinds_are_ignored(tmp_path, no_browser):
    ep = _episode(tmp_path, [
        _shot("sh001", "generated_image", 10.0),   # handled by the generate stage, not here
        _shot("sh002", "none", 20.0),
        _shot("sh003", "pull_quote", 30.0, props={"quote": "q"}),
    ], built=("pull_quote",))
    content = render.run(ep)
    assert [it["shot_id"] for it in content["items"]] == ["sh003"]


def test_content_items_sorted_by_source_time(tmp_path, no_browser):
    ep = _episode(tmp_path, [
        _shot("sh001", "pull_quote", 90.0, props={"quote": "late"}),
        _shot("sh002", "pull_quote", 12.0, props={"quote": "early"}),
    ], built=("pull_quote",))
    content = render.run(ep)
    assert [it["source_time"] for it in content["items"]] == [12.0, 90.0]


def test_cache_key_tracks_props_duration_and_style_version(tmp_path):
    ep = _episode(tmp_path, [_shot("sh001", "pull_quote", 10.0, props={"quote": "q"})])
    comp_dir = ep.compositions_dir / "pull_quote"
    tokens = ep.styles_dir / "default" / "tokens.css"
    font = ep.styles_dir / "default" / "fonts" / "Montserrat[wght].ttf"
    size = (2400, 1680)

    shot = _shot("sh001", "pull_quote", 10.0, props={"quote": "q"})
    base = render._shot_key(comp_dir, shot, 1, render.DEFAULT_FPS, tokens, font, size)
    same = render._shot_key(comp_dir, dict(shot), 1, render.DEFAULT_FPS, tokens, font, size)
    diff_props = render._shot_key(comp_dir, _shot("sh001", "pull_quote", 10.0, props={"quote": "Q!"}),
                                  1, render.DEFAULT_FPS, tokens, font, size)
    diff_dur = render._shot_key(comp_dir, _shot("sh001", "pull_quote", 10.0, dur=25.0, props={"quote": "q"}),
                                1, render.DEFAULT_FPS, tokens, font, size)
    diff_ver = render._shot_key(comp_dir, shot, 2, render.DEFAULT_FPS, tokens, font, size)
    assert base == same
    assert len({base, diff_props, diff_dur, diff_ver}) == 4


def test_fps_float_parses_rational_and_int():
    assert abs(render._fps_float("24000/1001") - 23.976023) < 1e-4
    assert render._fps_float("30") == 30.0
