"""Command-line entry point: ``python -m autocut <stage> <episode>``.

Each stage is a pure function of its inputs and is independently cached, so any
command can be re-run cheaply. Claude Code drives this CLI; it never calls ffmpeg
directly.
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import sys
from pathlib import Path

from . import (
    analyze,
    captions as captions_mod,
    composite as composite_mod,
    cut as cut_mod,
    ffmpeg,
    grade as grade_mod,
    overlays as overlays_mod,
    probe as probe_mod,
    qc as qc_mod,
    transcribe as transcribe_mod,
)
from .edl import ValidationError
from .paths import resolve

log = logging.getLogger("autocut")

# Third-party runtime deps the pipeline needs. They live in the project
# virtualenv; the usual way to trip over a missing one is invoking a system
# Python (e.g. `python -m autocut`) instead of `.venv\Scripts\python.exe`.
_RUNTIME_DEPS = {"faster_whisper", "ctranslate2", "fastapi", "uvicorn", "jinja2", "requests", "yaml"}
# Stages that import the (heavy, lazily-loaded) transcription stack. We check
# these up front so a wrong interpreter fails in the first second instead of
# after a multi-minute probe + mezzanine build.
_STAGE_DEPS = {"transcribe": ("faster_whisper", "ctranslate2"),
               "all": ("faster_whisper", "ctranslate2")}


def _dependency_error(missing: list[str]) -> str:
    named = f" required package(s): {', '.join(missing)}" if missing else " a required package"
    return (
        f"error: this Python interpreter is missing{named}.\n"
        f"  interpreter: {sys.executable}\n"
        f"  version:     Python {sys.version.split()[0]} (this project requires >=3.12,<3.13)\n"
        "  The pipeline's dependencies live in the project virtualenv. Run it with:\n"
        "    .venv\\Scripts\\python.exe -m autocut ...\n"
        "  or activate the venv first:  .venv\\Scripts\\Activate.ps1"
    )


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )


def _cmd_probe(ep, args):
    probe_mod.run(ep, force=args.force)


def _cmd_transcribe(ep, args):
    transcribe_mod.run(ep, force=args.force)


def _cmd_autoauthor(ep, args):
    analyze.autoauthor(ep)


def _cmd_validate(ep, args) -> int:
    errors = analyze.validate(ep)
    if errors:
        print(f"EDL INVALID ({len(errors)} error(s)):", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        return 1
    print("EDL valid.")
    return 0


def _cmd_cut(ep, args):
    cut_mod.run(ep, force=args.force)


def _cmd_grade(ep, args):
    grade_mod.run(ep, force=args.force)


def _cmd_overlays(ep, args):
    overlays_mod.run(ep, force=args.force)


def _cmd_captions(ep, args):
    captions_mod.run(ep, force=args.force)


def _cmd_composite(ep, args):
    composite_mod.run(ep, force=args.force)


def _cmd_qc(ep, args):
    qc_mod.run(ep)


def _cmd_compose(ep, args):
    """Phase-2 compositor: produced show frame (background, speaker, content, captions, audio)."""
    from . import compose as compose_mod  # lazy: keeps cli importable without pyyaml
    compose_mod.run(ep, force=args.force, preview=args.preview, render_range=args.render_range)


def _cmd_align(ep, args):
    """Reaction format: align host<->source and write playback.json (spec steps 1-2)."""
    from . import align as align_mod  # lazy: keeps cli importable without numpy
    align_mod.run(ep, source=args.source, force=args.force)


def _cmd_align_check(ep, args):
    """Reaction format: render 2s lip-sync verification clips per playback segment."""
    from . import align as align_mod  # lazy: keeps cli importable without numpy
    align_mod.render_checks(ep, seconds=args.seconds, force=args.force)


def _cmd_shotlist(ep, args):
    """Phase-3 visuals: author shotlist.json from the transcript (words.json)."""
    from . import shotlist as shotlist_mod  # lazy: keeps cli importable without pyyaml
    shotlist_mod.run(ep, force=args.force)


def _cmd_visuals(ep, args):
    """Phase-3 visuals: generate illustration candidates from shotlist.json."""
    from . import visuals as visuals_mod  # lazy: keeps cli importable without numpy/genai
    visuals_mod.run(ep, force=args.force)


def _cmd_visuals_render(ep, args):
    """Phase-3 visuals: render built HyperFrames compositions to webm + content.json."""
    from . import render as render_mod  # lazy: keeps cli importable without websockets
    render_mod.run(ep, force=args.force, workers=args.workers)


def _cmd_render(ep, args):
    """Everything after review: cut -> grade -> captions -> overlays -> composite -> qc."""
    cut_mod.run(ep, force=args.force)
    grade_mod.run(ep, force=args.force)
    captions_mod.run(ep, force=args.force)
    overlays_mod.run(ep, force=args.force)
    composite_mod.run(ep, force=args.force)
    qc_mod.run(ep)


def _cmd_all(ep, args) -> int:
    """The pre-review pipeline: probe -> transcribe -> baseline EDL -> validate.
    Stops before rendering, as the pipeline design intends."""
    probe_mod.run(ep, force=args.force)
    transcribe_mod.run(ep, force=args.force)
    analyze.autoauthor(ep)
    rc = _cmd_validate(ep, args)
    if rc == 0:
        print(f"\nReady for review: python -m autocut review {ep.episode_id}")
    return rc


def _cmd_review(ep, args):
    from . import review
    review.serve(ep.episode_id, host=args.host, port=args.port, root=ep.root)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="autocut", description="Skepticus Autocut pipeline.")
    p.add_argument("--dry-run", action="store_true", help="log ffmpeg commands without running them")
    p.add_argument("--force", action="store_true", help="ignore stage caches and rebuild")
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument("--root", type=Path, default=None, help="repo root (default: auto-detected)")

    sub = p.add_subparsers(dest="stage", required=True)

    simple = {
        "probe": _cmd_probe,
        "transcribe": _cmd_transcribe,
        "autoauthor": _cmd_autoauthor,
        "validate": _cmd_validate,
        "cut": _cmd_cut,
        "grade": _cmd_grade,
        "overlays": _cmd_overlays,
        "captions": _cmd_captions,
        "composite": _cmd_composite,
        "qc": _cmd_qc,
        "shotlist": _cmd_shotlist,
        "visuals": _cmd_visuals,
        "render": _cmd_render,
        "all": _cmd_all,
    }
    for name, fn in simple.items():
        sp = sub.add_parser(name, help=fn.__doc__ or name)
        sp.add_argument("episode")
        sp.set_defaults(func=fn)

    # visuals-render renders HyperFrames compositions; --workers tunes parallel
    # Chrome contexts (capture is the bottleneck).
    vr = sub.add_parser("visuals-render", help=_cmd_visuals_render.__doc__ or "visuals-render")
    vr.add_argument("episode")
    vr.add_argument("--workers", type=int, default=None,
                    help="parallel Chrome render contexts (default: config or 3)")
    vr.set_defaults(func=_cmd_visuals_render)

    # compose takes preview/range options for iterating without a full render.
    cp = sub.add_parser("compose", help=_cmd_compose.__doc__ or "compose")
    cp.add_argument("episode")
    cp.add_argument("--preview", action="store_true",
                    help="fast 1080p render of a short window for iteration")
    cp.add_argument("--range", dest="render_range", metavar="START:END",
                    help="render only the output window START:END (seconds)")
    cp.set_defaults(func=_cmd_compose)

    # Reaction format: align (playback.json) and align-check (lip-sync clips).
    al = sub.add_parser("align", help=_cmd_align.__doc__ or "align")
    al.add_argument("episode")
    al.add_argument("--source", default=None,
                    help="source video path (default: inbox/<episode>_source.<ext>)")
    al.set_defaults(func=_cmd_align)

    ac = sub.add_parser("align-check", help=_cmd_align_check.__doc__ or "align-check")
    ac.add_argument("episode")
    ac.add_argument("--seconds", type=float, default=2.0,
                    help="length of each verification clip (default: 2)")
    ac.set_defaults(func=_cmd_align_check)

    rp = sub.add_parser("review", help="start the review gate on localhost")
    rp.add_argument("episode")
    rp.add_argument("--host", default="127.0.0.1")
    rp.add_argument("--port", type=int, default=8765)
    rp.set_defaults(func=_cmd_review)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _setup_logging(args.verbose)
    ffmpeg.set_dry_run(args.dry_run)

    # Fail fast on a wrong interpreter, before any expensive stage runs.
    # find_spec probes availability without importing the heavy modules. Skipped
    # under --dry-run, which never imports the transcription stack.
    if not args.dry_run:
        missing = [d for d in _STAGE_DEPS.get(args.stage, ())
                   if importlib.util.find_spec(d) is None]
        if missing:
            print(_dependency_error(missing), file=sys.stderr)
            return 1

    ep = resolve(args.episode, args.root)
    try:
        result = args.func(ep, args)
        return result if isinstance(result, int) else 0
    except ValidationError as e:
        print(str(e), file=sys.stderr)
        return 1
    except (FileNotFoundError, RuntimeError) as e:
        # RuntimeError covers ffmpeg failures (FFmpegError/FFmpegNotFound subclass
        # it) and the pipeline's loud plausibility guards (empty silence.json,
        # implausibly sparse EDL). Print cleanly instead of a traceback.
        print(f"error: {e}", file=sys.stderr)
        return 1
    except ModuleNotFoundError as e:
        # A stage that lazily imports a known runtime dep (e.g. the review
        # server's fastapi) hit a missing package — almost always the wrong
        # interpreter. Give the actionable hint; let real bugs surface as-is.
        if (e.name or "").split(".")[0] in _RUNTIME_DEPS:
            print(_dependency_error([e.name]), file=sys.stderr)
            return 1
        raise


if __name__ == "__main__":
    raise SystemExit(main())
