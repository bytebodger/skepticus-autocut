"""Review gate — ``python -m autocut review <ep>``.

A localhost FastAPI app that shows every proposed drop with its transcript text
and surrounding context, plus audio scrub for anything under 0.8 confidence.
Vetoing a drop writes it back to edl.json as ``action: keep`` + ``override:
true``. Re-running stage 3 preserves those overrides (see analyze.apply_overrides
and edl.collect_overrides).

Ten minutes here beats three hours in Premiere.
"""

from __future__ import annotations

import html
import json
import logging
from pathlib import Path

from . import edl, ffmpeg
from .paths import Episode, resolve

log = logging.getLogger("autocut.review")


def _load(ep: Episode):
    edl_doc = edl.load(ep.edl_json)
    words = []
    if ep.words_json.exists():
        words = json.loads(ep.words_json.read_text(encoding="utf-8")).get("words", [])
    return edl_doc, words


def _text_between(words, start, end):
    return " ".join(w["word"] for w in words if start <= w["start"] < end).strip()


def _context(words, start, end, window=2.0):
    before = " ".join(w["word"] for w in words if start - window <= w["start"] < start).strip()
    after = " ".join(w["word"] for w in words if end <= w["start"] < end + window).strip()
    return before, after


def _reviewable(edl_doc):
    """Segments that are drops or have been overridden — the review surface."""
    return [s for s in edl_doc["segments"]
            if s.get("action") == "drop" or s.get("override")]


def _render_page(ep: Episode, edl_doc, words) -> str:
    rows = []
    for s in _reviewable(edl_doc):
        dur = s["out"] - s["in"]
        conf = s.get("confidence", 1.0)
        text = html.escape(_text_between(words, s["in"], s["out"]) or "—")
        before, after = _context(words, s["in"], s["out"])
        checked = "checked" if s.get("override") else ""
        low = conf < 0.8
        audio = (
            f'<audio controls preload="none" src="/api/audio/{s["id"]}"></audio>'
            if low else "<span class=muted>—</span>"
        )
        rows.append(f"""
          <tr class="{'low' if low else ''}">
            <td><input type="checkbox" name="veto" value="{s['id']}" {checked}></td>
            <td>{s['id']}</td>
            <td>{s['in']:.2f}–{s['out']:.2f}<br><span class=muted>{dur:.2f}s</span></td>
            <td>{html.escape(str(s.get('reason', '')))}</td>
            <td>{conf:.2f}</td>
            <td><span class=ctx>{html.escape(before)}</span>
                <b>{text}</b>
                <span class=ctx>{html.escape(after)}</span></td>
            <td>{audio}</td>
          </tr>""")

    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Review — {ep.episode_id}</title>
<style>
  body {{ font: 15px/1.5 system-ui, sans-serif; margin: 2rem; max-width: 1100px; }}
  table {{ border-collapse: collapse; width: 100%; }}
  th, td {{ border-bottom: 1px solid #ddd; padding: .5rem; text-align: left; vertical-align: top; }}
  tr.low {{ background: #fff7e6; }}
  .muted {{ color: #999; }}
  .ctx {{ color: #888; }}
  b {{ color: #b00; }}
  button {{ font-size: 1rem; padding: .6rem 1.2rem; margin-top: 1rem; cursor: pointer; }}
  .head {{ display: flex; justify-content: space-between; align-items: baseline; }}
</style></head><body>
<div class=head>
  <h1>Review — {ep.episode_id}</h1>
  <span class=muted>{len(_reviewable(edl_doc))} proposed drops · check = keep (veto the cut)</span>
</div>
<form method="post" action="/api/veto">
<table>
  <tr><th>Keep</th><th>ID</th><th>Time</th><th>Reason</th><th>Conf</th>
      <th>Transcript (context · <b>removed</b> · context)</th><th>Scrub</th></tr>
  {''.join(rows) or '<tr><td colspan=7>No drops proposed.</td></tr>'}
</table>
<button type="submit">Save vetoes to edl.json</button>
</form>
</body></html>"""


def create_app(ep: Episode):
    from fastapi import FastAPI, Form
    from fastapi.responses import HTMLResponse, RedirectResponse, FileResponse, PlainTextResponse

    app = FastAPI(title=f"autocut review — {ep.episode_id}")

    @app.get("/", response_class=HTMLResponse)
    def index():
        edl_doc, words = _load(ep)
        return _render_page(ep, edl_doc, words)

    @app.post("/api/veto")
    def veto(veto: list[str] = Form(default=[])):
        vetoed = set(veto)
        edl_doc, _ = _load(ep)
        changed = 0
        for s in _reviewable(edl_doc):
            if s["id"] in vetoed:
                if s.get("action") != "keep" or not s.get("override"):
                    changed += 1
                s["action"] = "keep"
                s["override"] = True
            else:
                # Unchecked: this drop stands. Clear any prior override.
                if s.get("override"):
                    changed += 1
                s["action"] = "drop"
                s.pop("override", None)
        edl.save(edl_doc, ep.edl_json)
        log.info("review: saved %d change(s), %d veto(es)", changed, len(vetoed))
        return RedirectResponse("/", status_code=303)

    @app.get("/api/audio/{seg_id}")
    def audio(seg_id: str):
        edl_doc, _ = _load(ep)
        seg = next((s for s in edl_doc["segments"] if s["id"] == seg_id), None)
        if seg is None:
            return PlainTextResponse("unknown segment", status_code=404)
        clip_dir = ep.work / "review"
        clip_dir.mkdir(parents=True, exist_ok=True)
        clip = clip_dir / f"{seg_id}.wav"
        # Extract a padded clip so you can hear the surrounding context.
        pad = 1.0
        start = max(0.0, seg["in"] - pad)
        end = seg["out"] + pad
        ffmpeg.run_ffmpeg([
            "-ss", f"{start:.3f}", "-to", f"{end:.3f}", "-i", ep.mezz,
            "-ac", "1", "-ar", "44100", clip,
        ])
        return FileResponse(clip, media_type="audio/wav")

    return app


def serve(episode_id: str, host: str = "127.0.0.1", port: int = 8765, root: Path | None = None) -> None:
    import uvicorn

    ep = resolve(episode_id, root)
    if not ep.edl_json.exists():
        raise FileNotFoundError(f"No EDL at {ep.edl_json}. Author it (stage 3) first.")
    app = create_app(ep)
    log.info("review: serving %s at http://%s:%d", episode_id, host, port)
    print(f"\n  Review gate for {episode_id}: http://{host}:{port}\n  Ctrl-C to stop.\n")
    uvicorn.run(app, host=host, port=port, log_level="warning")
