"""Phase-3 visuals — illustration generation (visuals spec sections 4, 8, 9).

Reads ``shotlist.json`` and generates candidate illustrations for each shot whose
``kind`` is ``illustration`` (infographics are rendered by HyperFrames, not
here). For each shot it produces N candidates at distinct seeds and writes them
to ``work/<ep>/visuals/<shot_id>_<n>.png``.

The prompt assembles from the selected style's fragments plus the shot's concept
(spec section 8): ``{style_fragment} {variant_fragment} {concept}``. The negative
prompt always carries the anti-photorealism terms — enforced mechanically here
rather than trusting the positive prompt.

The generation engine sits behind the ``ImageBackend`` interface so backends
swap by config. The default is a hosted, reference-image-conditioned backend
(Gemini): styles are swappable per episode by dropping approved reference images
into ``styles/<name>/references/`` — no retraining. A deterministic
``placeholder`` backend runs offline for wiring/iteration without credentials.

The job is resumable (spec section 9 / the 30-minute convention): every candidate
is cached on prompt + seed + model + style version and checkpointed into
``manifest.json`` as it lands, so a re-run skips finished candidates and only
generates what is missing or stale.
"""

from __future__ import annotations

import hashlib
import json
import logging
import struct
import zlib
from pathlib import Path

import numpy as np
import yaml

from . import cache
from .paths import Episode

log = logging.getLogger("autocut.visuals")

DEFAULT_CANDIDATES = 4
DEFAULT_SIZE = (1536, 1024)
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


# --------------------------------------------------------------------------- #
# Style + config loading
# --------------------------------------------------------------------------- #

def _config(ep: Episode) -> dict:
    cfg_path = ep.root / "config" / "layout.yaml"
    data = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}
    return data or {}


def _load_style(ep: Episode, name: str) -> dict:
    path = ep.styles_dir / name / "style.yaml"
    if not path.exists():
        raise FileNotFoundError(
            f"No style.yaml at {path}. The style bible (Phase 3A) provides the "
            f"prompt fragments; see styles/default/ for the shape."
        )
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _references(ep: Episode, name: str) -> list[Path]:
    ref_dir = ep.styles_dir / name / "references"
    if not ref_dir.is_dir():
        return []
    return sorted(p for p in ref_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def assemble_prompt(style: dict, shot: dict) -> tuple[str, str]:
    """(positive, negative) for a shot (spec section 8). Only the concept is
    per-shot; the style fragment holds consistency across episodes."""
    parts = [style.get("style_fragment", "").strip()]
    variant = shot.get("variant", "").strip()
    if variant:
        frag = (style.get("variants") or {}).get(variant, "").strip()
        if frag:
            parts.append(frag)
    parts.append(shot.get("concept", "").strip())
    positive = " ".join(p for p in parts if p)
    negative = style.get("negative_fragment", "").strip()
    return positive, negative


def _seed(shot_id: str, n: int) -> int:
    """Deterministic per-candidate seed (reproducible across runs)."""
    digest = hashlib.sha256(f"{shot_id}:{n}".encode()).digest()
    return int.from_bytes(digest[:4], "big")


# --------------------------------------------------------------------------- #
# Backends
# --------------------------------------------------------------------------- #

class PlaceholderBackend:
    """Deterministic, offline. A seeded flat-palette composition so the pipeline,
    caching, resume, and candidate variety are exercisable without an image API.
    Swapping to a real backend is a config change."""

    def __init__(self, size, **_):
        self.size = size
        self.model = "placeholder-v1"

    def generate(self, *, prompt, negative, seed, references, **_) -> bytes:
        w, h = self.size
        rng = np.random.default_rng(seed)
        palette = rng.integers(30, 225, size=(rng.integers(4, 7), 3), dtype=np.uint16).astype(np.uint8)
        img = np.empty((h, w, 3), np.uint8)
        img[:] = palette[0]
        yy, xx = np.mgrid[0:h, 0:w]
        for colour in palette[1:]:
            cx, cy = int(rng.integers(0, w)), int(rng.integers(0, h))
            r = int(rng.integers(min(w, h) // 6, min(w, h) // 2))
            img[((xx - cx) ** 2 + (yy - cy) ** 2) < r * r] = colour
        return _encode_png(img)


class GeminiBackend:
    """Hosted, reference-image-conditioned (visuals spec section 4). Holds the
    style from reference images with no training, so styles swap per episode.
    Requires the ``google-genai`` SDK and GEMINI_API_KEY."""

    def __init__(self, size, model="gemini-2.5-flash-image", **_):
        try:
            from google import genai
        except ImportError as e:
            raise RuntimeError(
                "gemini backend needs google-genai: pip install -e '.[gemini]' "
                "(or set visuals.backend: placeholder)."
            ) from e
        self._genai = genai
        self._client = genai.Client()  # resolves GEMINI_API_KEY / GOOGLE_API_KEY
        self.size = size
        self.model = model

    def generate(self, *, prompt, negative, seed, references, **_) -> bytes:
        from google.genai import types
        w, h = self.size
        text = f"{prompt}\nComposition: {w}x{h} landscape."
        if negative:
            text += f"\nDo not include: {negative}."
        contents = [text]
        for ref in references:  # style-conditioning reference images
            contents.append(types.Part.from_bytes(data=ref.read_bytes(), mime_type="image/png"))
        resp = self._client.models.generate_content(
            model=self.model,
            contents=contents,
            config=types.GenerateContentConfig(response_modalities=["IMAGE"], seed=seed),
        )
        for part in resp.candidates[0].content.parts:
            blob = getattr(part, "inline_data", None)
            if blob and blob.data:
                return blob.data
        raise RuntimeError("gemini backend returned no image data")


_BACKENDS = {"placeholder": PlaceholderBackend, "gemini": GeminiBackend}


def _make_backend(name: str, *, size, model) -> object:
    if name not in _BACKENDS:
        raise ValueError(f"unknown visuals backend {name!r}; choices: {sorted(_BACKENDS)}")
    return _BACKENDS[name](size=size, model=model)


# --------------------------------------------------------------------------- #
# Minimal RGB PNG writer (no Pillow dependency for the placeholder path)
# --------------------------------------------------------------------------- #

def _encode_png(arr: np.ndarray) -> bytes:
    h, w, _ = arr.shape

    def chunk(typ: bytes, data: bytes) -> bytes:
        body = typ + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)

    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)  # 8-bit RGB
    raw = bytearray()
    for row in arr:
        raw.append(0)  # filter type: none
        raw.extend(row.tobytes())
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr)
            + chunk(b"IDAT", zlib.compress(bytes(raw), 6)) + chunk(b"IEND", b""))


# --------------------------------------------------------------------------- #
# Stage
# --------------------------------------------------------------------------- #

def _candidate_key(prompt, negative, seed, model, style_version, size) -> str:
    return cache.hash_inputs({
        "prompt": prompt, "negative": negative, "seed": seed,
        "model": model, "style_version": style_version, "size": list(size),
    })


def run(ep: Episode, *, force: bool = False) -> dict:
    """Generate illustration candidates. Resumable + cached per candidate."""
    if not ep.shotlist_json.exists():
        raise FileNotFoundError(
            f"No shot list at {ep.shotlist_json}. Run 'autocut shotlist {ep.episode_id}' first."
        )
    shotlist = json.loads(ep.shotlist_json.read_text(encoding="utf-8"))
    style_name = shotlist.get("style", "default")
    style = _load_style(ep, style_name)
    style_version = style.get("version", 1)
    references = _references(ep, style_name)

    cfg = (_config(ep).get("visuals") or {})
    backend_name = cfg.get("backend", "gemini")
    candidates = int(cfg.get("candidates", DEFAULT_CANDIDATES))
    size = tuple(cfg.get("size", DEFAULT_SIZE))
    import os
    backend_name = os.environ.get("AUTOCUT_VISUALS_BACKEND", backend_name)

    shots = [s for s in shotlist["shots"] if s.get("kind") == "illustration"]
    log.info("visuals: %d illustration shot(s) x %d candidate(s), backend=%s, style=%s (v%s), %d reference image(s)",
             len(shots), candidates, backend_name, style_name, style_version, len(references))

    ep.visuals_dir.mkdir(parents=True, exist_ok=True)
    prior = {}
    if ep.visuals_manifest.exists() and not force:
        try:
            prior = {s["id"]: s for s in json.loads(ep.visuals_manifest.read_text(encoding="utf-8")).get("shots", [])}
        except (json.JSONDecodeError, KeyError):
            prior = {}

    backend = _make_backend(backend_name, size=size, model=style.get("params", {}).get("model"))
    manifest = {
        "episode_id": ep.episode_id, "style": style_name, "style_version": style_version,
        "backend": backend_name, "model": backend.model, "candidates_per_shot": candidates,
        "shots": [],
    }

    generated = skipped = 0
    for shot in shots:
        sid = shot["id"]
        prompt, negative = assemble_prompt(style, shot)
        prior_cands = {c["n"]: c for c in prior.get(sid, {}).get("candidates", [])}
        rec = {"id": sid, "concept": shot.get("concept", ""), "variant": shot.get("variant", ""),
               "prompt": prompt, "negative": negative, "model": backend.model,
               "style": style_name, "style_version": style_version, "candidates": []}
        for n in range(1, candidates + 1):
            seed = _seed(sid, n)
            key = _candidate_key(prompt, negative, seed, backend.model, style_version, size)
            fname = f"{sid}_{n}.png"
            out = ep.visuals_dir / fname
            was = prior_cands.get(n)
            if not force and out.exists() and was and was.get("key") == key:
                skipped += 1
            else:
                img = backend.generate(prompt=prompt, negative=negative, seed=seed, references=references)
                out.write_bytes(img)
                generated += 1
            rec["candidates"].append({"n": n, "seed": seed, "file": fname, "key": key})
        manifest["shots"].append(rec)
        # Checkpoint after each shot so a crash resumes without re-work.
        ep.visuals_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    log.info("visuals: %d generated, %d resumed/cached -> %s", generated, skipped, ep.visuals_dir)
    return manifest
