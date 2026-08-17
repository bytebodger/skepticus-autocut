"""Phase-3 visuals — render stage (visuals spec sections 3, 9).

Reads ``shotlist.json`` and renders each shot whose ``kind`` is a built
HyperFrames composition to ``work/<ep>/visuals/<shot_id>.mov`` (ProRes 4444 with
alpha), then emits ``content.json`` for the compositor to consume.

Rendering is deterministic headless-Chrome frame capture — the engine the spec
calls HyperFrames — driven directly over the DevTools protocol so we control it
end to end:

* The active style's ``tokens.css`` and the **bundled** Montserrat variable font
  are assembled into the render context, and the font is pinned with an
  ``@font-face`` (``font-weight: 100 900``). Nothing is read from system-installed
  fonts, so 800/900 no longer synth-bold off whatever weight happens to be
  installed — the render is identical on any machine.
* ``window.DURATION`` is injected from the shot's duration so each composition
  paces its animation to fit (spec section 3).
* Each frame is produced by seeking the composition's animations to that frame's
  timestamp (Web Animations ``currentTime``) and screenshotting — a fixed input
  gives a fixed frame. Only the *active* portion of the animation is captured;
  once every animation has finished the last frame is held for the remainder, so
  a 25s card with an 8s reveal captures ~8s of frames, not 25s.

**Alpha.** Each card is encoded to ProRes 4444 with a real alpha channel
(``yuva444p10le``) — the same alpha-carrying format the speaker layer already
uses in ``compose.py``. The card keeps its own transparency (its panel is
semi-opaque, its margins fully transparent), so the compositor composes it over
whatever background is behind the content window. Cards are therefore
*background-independent*: nothing is baked in, which keeps the door open to
background motion, content-rect changes, and the layered parallax the spec wants
for vector scenes. (WebM alpha was a dead end here — this ffmpeg build's VP9/VP8
drop the alpha plane — but ProRes alpha is proven in this codebase already.)

Caching (spec section 9): each card is keyed on composition source + props +
style version (plus duration, fps, tokens, font, render size, and output codec),
so a re-run only re-renders what changed. Shots whose composition isn't built yet
are skipped with a warning, never an error — most of the twelve are still to come.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import shutil
import socket
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path

import yaml

from . import cache, ffmpeg, layout as layout_mod
from .paths import Episode
from .shotlist import COMPOSITIONS

log = logging.getLogger("autocut.render")

DEFAULT_FPS = "24000/1001"
DEFAULT_WORKERS = 3          # parallel Chrome contexts; capture is the bottleneck
SETTLE_FRAMES = 2            # extra frames past the animation end before holding
HOLD_BUFFER = 2.0           # seconds of held frame baked into the .mov; the
                            # compositor's hold extends the rest of the span
COMPOSITION_SET = frozenset(COMPOSITIONS)
# A Phase-3 content-window card declares this marker in its <head>. It separates
# the real content compositions from the Phase-1 overlay compositions that share
# the compositions/ directory and can collide on a taxonomy name (e.g. title_card).
SURFACE_MARKER = "hyperframes-surface"


def _is_content_composition(comp_dir: Path) -> bool:
    index = comp_dir / "index.html"
    if not index.exists():
        return False
    return SURFACE_MARKER in index.read_text(encoding="utf-8")


def _fps_float(fps: str) -> float:
    """Parse an ffmpeg fps string (``24000/1001`` or ``30``) to a float."""
    fps = str(fps).strip()
    if "/" in fps:
        num, den = fps.split("/", 1)
        return float(num) / float(den)
    return float(fps)


# --------------------------------------------------------------------------- #
# Chrome discovery
# --------------------------------------------------------------------------- #

def _chrome_binary() -> str:
    override = os.environ.get("AUTOCUT_CHROME")
    if override:
        return override
    candidates = [
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    ]
    for c in candidates:
        if Path(c).exists():
            return c
    found = shutil.which("chrome") or shutil.which("msedge")
    if found:
        return found
    raise RuntimeError(
        "No Chrome/Edge found for HyperFrames rendering. Install Chrome or set "
        "AUTOCUT_CHROME to a Chromium-based browser's executable."
    )


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


# --------------------------------------------------------------------------- #
# Render context assembly
# --------------------------------------------------------------------------- #

FONT_FACE = ('<style>@font-face{{font-family:"Montserrat";'
             'src:url("fonts/{fontfile}") format("truetype");'
             'font-weight:100 900;font-style:normal;font-display:block;}}</style>')


def _fonts(style_dir: Path) -> Path | None:
    """The bundled variable font for this style, if present."""
    fonts = sorted(style_dir.glob("fonts/*.ttf")) + sorted(style_dir.glob("fonts/*.otf"))
    # Prefer a variable (all-weights) file; any single file works.
    for f in fonts:
        if "wght" in f.name.lower() or "variable" in f.name.lower():
            return f
    return fonts[0] if fonts else None


def _build_context(dst: Path, comp_dir: Path, tokens: Path, font: Path | None,
                   props: dict, duration: float) -> str:
    """Assemble a self-contained render dir and return its index.html file URI."""
    for f in comp_dir.iterdir():                    # composition assets (schema, etc.)
        if f.is_file():
            shutil.copy(f, dst / f.name)
    shutil.copy(tokens, dst / "tokens.css")

    html = (comp_dir / "index.html").read_text(encoding="utf-8")
    head_inject = ""
    if font is not None:
        (dst / "fonts").mkdir(exist_ok=True)
        shutil.copy(font, dst / "fonts" / font.name)
        head_inject = FONT_FACE.format(fontfile=font.name)
    html = html.replace("</head>", head_inject + "</head>", 1)
    html = html.replace(
        "<body>",
        "<body>\n<script>window.PROPS=%s;window.DURATION=%s;</script>"
        % (json.dumps(props), duration), 1)
    (dst / "index.html").write_text(html, encoding="utf-8")
    return (dst / "index.html").as_uri()


# --------------------------------------------------------------------------- #
# DevTools protocol driver
# --------------------------------------------------------------------------- #

# Wait for fonts + the composition's own build script, then report how many
# animations exist and when the last one ends (ms) so we capture only the active
# portion and hold the rest.
_PREPARE_JS = (
    "(async()=>{await document.fonts.ready;"
    "for(let i=0;i<180;i++){if(document.getAnimations().length>0)break;"
    "await new Promise(r=>requestAnimationFrame(r));}"
    "const A=document.getAnimations();let end=0;"
    "A.forEach(a=>{const t=a.effect.getComputedTiming();"
    "const e=(t.endTime!=null?t.endTime:((t.delay||0)+(t.activeDuration||0)));"
    "if(e>end)end=e;});return JSON.stringify({n:A.length,end:end});})()"
)


class _CDP:
    """Minimal DevTools protocol client over one page target's websocket."""

    def __init__(self, ws):
        self.ws = ws
        self._id = 0
        self._pending: dict[int, asyncio.Future] = {}
        self._waits: dict[str, asyncio.Future] = {}

    async def reader(self) -> None:
        async for raw in self.ws:
            m = json.loads(raw)
            if "id" in m and m["id"] in self._pending:
                fut = self._pending.pop(m["id"])
                if not fut.done():
                    fut.set_exception(RuntimeError(str(m["error"]))) if "error" in m \
                        else fut.set_result(m.get("result", {}))
            elif "method" in m and m["method"] in self._waits:
                fut = self._waits.pop(m["method"])
                if not fut.done():
                    fut.set_result(m.get("params", {}))

    async def send(self, method: str, params: dict | None = None) -> dict:
        self._id += 1
        mid = self._id
        fut = asyncio.get_event_loop().create_future()
        self._pending[mid] = fut
        await self.ws.send(json.dumps({"id": mid, "method": method, "params": params or {}}))
        return await fut

    def wait_event(self, method: str) -> asyncio.Future:
        fut = asyncio.get_event_loop().create_future()
        self._waits[method] = fut
        return fut

    async def evaluate(self, expression: str, *, await_promise: bool = False) -> dict:
        return await self.send("Runtime.evaluate", {
            "expression": expression, "awaitPromise": await_promise, "returnByValue": True})


class _Browser:
    """A headless Chrome process + a CDP connection to its page target."""

    def __init__(self, w: int, h: int):
        self.w, self.h = w, h
        self._proc = None
        self._profile = None
        self._ws = None
        self._reader = None
        self.cdp: _CDP | None = None

    async def __aenter__(self):
        import websockets
        self._profile = Path(tempfile.mkdtemp(prefix="hf_prof_"))
        port = _free_port()
        self._proc = subprocess.Popen(
            [_chrome_binary(), "--headless=new", "--disable-gpu", "--hide-scrollbars",
             "--force-device-scale-factor=1", "--no-first-run", "--no-default-browser-check",
             "--disable-extensions", "--mute-audio", "--remote-allow-origins=*",
             "--user-data-dir=%s" % self._profile, "--remote-debugging-port=%d" % port,
             "--window-size=%d,%d" % (self.w, self.h), "about:blank"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        ws_url = None
        for _ in range(150):
            try:
                targets = json.load(urllib.request.urlopen("http://127.0.0.1:%d/json" % port, timeout=1))
                pages = [t for t in targets if t.get("type") == "page" and t.get("webSocketDebuggerUrl")]
                if pages:
                    ws_url = pages[0]["webSocketDebuggerUrl"]
                    break
            except Exception:
                await asyncio.sleep(0.1)
        if not ws_url:
            raise RuntimeError("Chrome DevTools endpoint did not come up")
        self._ws = await websockets.connect(ws_url, max_size=None, ping_interval=None)
        self.cdp = _CDP(self._ws)
        self._reader = asyncio.create_task(self.cdp.reader())
        await self.cdp.send("Page.enable")
        await self.cdp.send("Runtime.enable")
        await self.cdp.send("Emulation.setDeviceMetricsOverride",
                            {"width": self.w, "height": self.h, "deviceScaleFactor": 1, "mobile": False})
        await self.cdp.send("Emulation.setDefaultBackgroundColorOverride",
                            {"color": {"r": 0, "g": 0, "b": 0, "a": 0}})
        return self

    async def __aexit__(self, *exc):
        if self._reader:
            self._reader.cancel()
        if self._ws:
            await self._ws.close()
        if self._proc:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        if self._profile:
            shutil.rmtree(self._profile, ignore_errors=True)

    async def capture_frames(self, url: str, fps: float, duration: float, frames_dir: Path) -> int:
        """Navigate, seek per frame, screenshot the active portion. Returns the
        number of frames captured (the tail is held by the encoder)."""
        cdp = self.cdp
        loaded = cdp.wait_event("Page.loadEventFired")
        await cdp.send("Page.navigate", {"url": url})
        await asyncio.wait_for(loaded, 30)
        info = json.loads((await cdp.evaluate(_PREPARE_JS, await_promise=True))["result"]["value"])
        anim_end = float(info.get("end") or 0.0) / 1000.0
        total = max(1, round(duration * fps))
        active = min(total, max(1, int(anim_end * fps) + 1 + SETTLE_FRAMES))
        for i in range(active):
            t_ms = min(i / fps, duration) * 1000.0
            await cdp.evaluate(
                "document.getAnimations().forEach(a=>{try{a.currentTime=%f;a.pause();}"
                "catch(e){}});0" % t_ms)
            shot = await cdp.send("Page.captureScreenshot", {
                "format": "png", "optimizeForSpeed": True,
                "clip": {"x": 0, "y": 0, "width": self.w, "height": self.h, "scale": 1},
                "captureBeyondViewport": True})
            (frames_dir / ("frame_%05d.png" % i)).write_bytes(base64.b64decode(shot["data"]))
        return active


# --------------------------------------------------------------------------- #
# Encode: hold the animation's static tail, emit a ProRes 4444 alpha .mov
# --------------------------------------------------------------------------- #

def _encode(frames_dir: Path, n_frames: int, fps: str, duration: float,
            out_mov: Path) -> float:
    """Encode the captured reveal plus a short held tail to a ProRes 4444 alpha
    .mov (yuva444p10le) — the alpha format the speaker layer uses in compose.py.

    The file length is bounded to the reveal plus HOLD_BUFFER, NOT the full (now
    contiguous, often 60s+) shot duration: ProRes is intra-frame, so a full-length
    held card would be ~19 MB/s of duplicate frames. The compositor's `hold` gap
    behaviour clone-extends the card's final frame to fill its span, so on screen
    the card still runs its full duration — the held tail just isn't baked into a
    multi-GB intermediate. Returns the encoded length. Nothing is baked in, so the
    card composes over any background."""
    active_len = n_frames / _fps_float(fps)
    mov_len = min(duration, active_len + HOLD_BUFFER)
    tail = max(0.0, mov_len - active_len)
    graph = f"[0:v]tpad=stop_mode=clone:stop_duration={tail:.3f},format=yuva444p10le[v]"
    ffmpeg.run_ffmpeg([
        "-framerate", fps, "-i", str(frames_dir / "frame_%05d.png"),
        "-filter_complex", graph, "-map", "[v]",
        "-t", f"{mov_len:.3f}", "-r", fps,
        "-c:v", "prores_ks", "-profile:v", "4444", "-pix_fmt", "yuva444p10le",
        out_mov,
    ])
    return mov_len


# --------------------------------------------------------------------------- #
# Stage
# --------------------------------------------------------------------------- #

def _style_meta(ep: Episode, style: str) -> tuple[Path, int, Path, Path | None]:
    style_dir = ep.styles_dir / style
    tokens = style_dir / "tokens.css"
    if not tokens.exists():
        raise FileNotFoundError(f"style {style!r} has no tokens.css at {tokens}")
    syaml = style_dir / "style.yaml"
    version = 1
    if syaml.exists():
        version = int((yaml.safe_load(syaml.read_text(encoding="utf-8")) or {}).get("version", 1))
    return style_dir, version, tokens, _fonts(style_dir)


def _comp_hash(comp_dir: Path) -> str:
    parts = {f.relative_to(comp_dir).as_posix(): cache.hash_file(f)
             for f in sorted(comp_dir.rglob("*")) if f.is_file()}
    return cache.hash_inputs(parts)


def _shot_key(comp_dir: Path, shot: dict, style_version: int, fps: str,
              tokens: Path, font: Path | None, size: tuple[int, int]) -> str:
    return cache.hash_inputs({
        "composition": _comp_hash(comp_dir),
        "props": shot.get("props", {}),
        "duration": round(float(shot["duration"]), 3),
        "style_version": style_version,
        "fps": fps,
        "tokens": cache.hash_file(tokens),
        "font": cache.hash_file(font) if font else "none",
        "size": list(size),
        "codec": "prores4444-yuva444p10le",   # re-render if the output format changes
    })


async def _render_worker(shots: list[dict], jobs: dict, w: int, h: int, fps: str,
                         tokens: Path, font: Path, results: dict) -> None:
    fps_f = _fps_float(fps)
    async with _Browser(w, h) as browser:
        for shot in shots:
            job = jobs[shot["id"]]
            tmp = Path(tempfile.mkdtemp(prefix="hf_ctx_"))
            frames_dir = tmp / "frames"
            frames_dir.mkdir()
            try:
                url = _build_context(tmp, job["comp_dir"], tokens, font,
                                     shot.get("props", {}), float(shot["duration"]))
                n = await browser.capture_frames(url, fps_f, float(shot["duration"]), frames_dir)
                mov_len = _encode(frames_dir, n, fps, float(shot["duration"]), job["out"])
                cache.mark_done(job["stage_dir"], job["key"], extra={"stage": "render", "shot": shot["id"]})
                results[shot["id"]] = True
                log.info("render: %s (%s) %d frames, %.1fs mov (shot %.1fs) -> %s",
                         shot["id"], shot["kind"], n, mov_len, float(shot["duration"]), job["out"].name)
            finally:
                shutil.rmtree(tmp, ignore_errors=True)


def run(ep: Episode, *, force: bool = False, workers: int | None = None) -> dict:
    """Render built compositions from shotlist.json and emit content.json."""
    if not ep.shotlist_json.exists():
        raise FileNotFoundError(
            f"No shot list at {ep.shotlist_json}. Run 'autocut shotlist {ep.episode_id}' first.")
    shotlist = json.loads(ep.shotlist_json.read_text(encoding="utf-8"))
    style = shotlist.get("style", "default")
    style_dir, style_version, tokens, font = _style_meta(ep, style)

    layout = layout_mod.load(ep.root)
    _, _, cw, ch = layout_mod.rect(layout, "content")
    fps = str((layout.get("visuals") or {}).get("render_fps") or DEFAULT_FPS)
    workers = workers or int((layout.get("visuals") or {}).get("render_workers") or DEFAULT_WORKERS)

    ep.visuals_dir.mkdir(parents=True, exist_ok=True)
    if font is None:
        log.warning("render: style %r bundles no font; text will fall back to a system font "
                    "(determinism leak). Add styles/%s/fonts/<Montserrat>.ttf.", style, style)

    # Partition shots: renderable (built composition), unbuilt (warn), non-composition.
    todo: list[dict] = []
    jobs: dict[str, dict] = {}
    unbuilt: dict[str, int] = {}
    items: list[dict] = []
    for shot in shotlist["shots"]:
        kind = shot["kind"]
        if kind not in COMPOSITION_SET:
            continue                                # generated_image/none: not this stage
        comp_dir = ep.compositions_dir / kind
        if not _is_content_composition(comp_dir):
            # Not built as a Phase-3 content card (missing, or a Phase-1 overlay
            # that merely shares the name). Skip with a warning, never an error.
            unbuilt[kind] = unbuilt.get(kind, 0) + 1
            continue
        out = ep.visuals_dir / f"{shot['id']}.mov"
        stage_dir = ep.visuals_dir / ".cache" / shot["id"]
        key = _shot_key(comp_dir, shot, style_version, fps, tokens, font, (cw, ch))
        jobs[shot["id"]] = {"comp_dir": comp_dir, "out": out, "stage_dir": stage_dir, "key": key}
        items.append({"shot_id": shot["id"], "composition": kind, "file": out.name,
                      "source_time": round(float(shot["source_time"]), 3),
                      "duration": round(float(shot["duration"]), 3)})
        if force or not (cache.is_current(stage_dir, key) and out.exists()):
            todo.append(shot)

    for kind, count in sorted(unbuilt.items()):
        log.warning("render: composition %r not built yet — skipping %d shot(s)", kind, count)
    log.info("render: %d renderable shot(s), %d to (re)render, %d cached; %d unbuilt kind(s) skipped",
             len(items), len(todo), len(items) - len(todo), len(unbuilt))

    if todo:
        chunks: list[list[dict]] = [[] for _ in range(min(workers, len(todo)))]
        for i, shot in enumerate(todo):
            chunks[i % len(chunks)].append(shot)
        results: dict[str, bool] = {}

        async def _main():
            await asyncio.gather(*[
                _render_worker(chunk, jobs, cw, ch, fps, tokens, font, results)
                for chunk in chunks if chunk])
        asyncio.run(_main())

    # content.json — only items whose webm actually exists (skips/failures excluded).
    items = [it for it in items if (ep.visuals_dir / it["file"]).exists()]
    items.sort(key=lambda it: it["source_time"])
    content = {"version": 1, "episode_id": ep.episode_id, "style": style,
               "style_version": style_version, "fps": fps, "items": items}
    ep.visuals_content_json.write_text(json.dumps(content, indent=2), encoding="utf-8")
    log.info("render: wrote %s with %d item(s)", ep.visuals_content_json, len(items))
    return content
