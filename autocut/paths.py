"""Filesystem layout for the pipeline.

Everything is derived from the repo root and an episode id. Stages read and
write under ``work/<episode_id>/<stage>/``; nothing outside ``work/`` (except
``out/``) is written by the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


def repo_root() -> Path:
    """Repo root = parent of the ``autocut`` package directory."""
    return Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Episode:
    """Resolved paths for a single episode.

    An episode id is the raw file's stem, e.g. ``ep042`` for ``inbox/ep042.mp4``.
    """

    episode_id: str
    root: Path

    # --- source ---
    @property
    def inbox(self) -> Path:
        return self.root / "inbox"

    @property
    def raw(self) -> Path:
        """The raw drop. Read-only, never modified by the pipeline."""
        # Accept common extensions; prefer .mp4.
        for ext in (".mp4", ".mkv", ".mov", ".MP4", ".MKV", ".MOV"):
            candidate = self.inbox / f"{self.episode_id}{ext}"
            if candidate.exists():
                return candidate
        return self.inbox / f"{self.episode_id}.mp4"

    # --- work tree ---
    @property
    def work(self) -> Path:
        return self.root / "work" / self.episode_id

    @property
    def mezz_dir(self) -> Path:
        return self.work / "mezz"

    @property
    def mezz(self) -> Path:
        return self.mezz_dir / "mezz.mkv"

    @property
    def audio_dir(self) -> Path:
        return self.work / "audio"

    @property
    def speech_wav(self) -> Path:
        return self.audio_dir / "speech.wav"

    @property
    def probe_json(self) -> Path:
        return self.work / "probe.json"

    @property
    def transcript_dir(self) -> Path:
        return self.work / "transcript"

    @property
    def words_json(self) -> Path:
        return self.transcript_dir / "words.json"

    @property
    def silence_json(self) -> Path:
        return self.transcript_dir / "silence.json"

    @property
    def edl_json(self) -> Path:
        return self.work / "edl.json"

    @property
    def segments_dir(self) -> Path:
        return self.work / "segments"

    @property
    def concat_list(self) -> Path:
        return self.segments_dir / "list.txt"

    @property
    def cut(self) -> Path:
        return self.work / "cut.mkv"

    @property
    def graded(self) -> Path:
        return self.work / "graded.mkv"

    @property
    def overlays_dir(self) -> Path:
        return self.work / "overlays"

    @property
    def captions_ass(self) -> Path:
        return self.work / "captions.ass"

    @property
    def filter_script(self) -> Path:
        return self.work / "filter.txt"

    # --- phase 2 compositor (compositor spec) ---
    @property
    def compose_dir(self) -> Path:
        return self.work / "compose"

    @property
    def speaker_layer(self) -> Path:
        """Keyed speaker layer — its own cached artifact so key params can be
        tuned without re-running the composite. ProRes 4444 keeps the alpha."""
        return self.compose_dir / "speaker.mov"

    @property
    def speaker_filter_script(self) -> Path:
        return self.compose_dir / "speaker.filter"

    @property
    def compose_filter_script(self) -> Path:
        return self.compose_dir / "composite.filter"

    @property
    def compose_output(self) -> Path:
        """Composited show frame (full duration, or a --preview/--range window)."""
        return self.compose_dir / "composite.mp4"

    # --- reaction format (reaction spec) ---
    @property
    def source_video(self) -> Path:
        """The clean source video being reacted to. Convention: a sibling of the
        host drop named ``<episode_id>_source.<ext>`` (reaction spec section 5/9).
        A ``--source`` CLI override wins; this is the default lookup."""
        for ext in (".mp4", ".mkv", ".mov", ".webm", ".MP4", ".MKV", ".MOV"):
            candidate = self.inbox / f"{self.episode_id}_source{ext}"
            if candidate.exists():
                return candidate
        return self.inbox / f"{self.episode_id}_source.mp4"

    @property
    def align_dir(self) -> Path:
        """Work area for the reaction alignment stage."""
        return self.work / "align"

    @property
    def source_speech_wav(self) -> Path:
        """16kHz mono speech audio extracted from the source, for transcription
        and cross-correlation (mirrors the host's speech.wav)."""
        return self.align_dir / "source_speech.wav"

    @property
    def source_words_json(self) -> Path:
        """Word-level transcript of the source file (its own words, no bleed)."""
        return self.align_dir / "source_words.json"

    @property
    def playback_json(self) -> Path:
        """The playback segment map: host<->source spans per play (reaction spec
        section 5). Authored by the align stage, consumed downstream."""
        return self.work / "playback.json"

    @property
    def align_check_dir(self) -> Path:
        """Rendered 2s lip-sync verification clips, one per playback segment."""
        return self.align_dir / "check"

    @property
    def content_dir(self) -> Path:
        """Drop folder for b-roll images/clips + content.json (spec section 4)."""
        return self.inbox / f"{self.episode_id}_content"

    @property
    def content_json(self) -> Path:
        return self.content_dir / "content.json"

    @property
    def content_track(self) -> Path:
        """Content-rect-sized video layer, full output duration. Own cached step."""
        return self.compose_dir / "content_track.mov"

    @property
    def content_filter_script(self) -> Path:
        return self.compose_dir / "content_track.filter"

    # --- phase 3 visuals ---
    @property
    def shotlist_json(self) -> Path:
        """Authored shot list (words.json -> shotlist.json), spec section 5."""
        return self.work / "shotlist.json"

    @property
    def visuals_dir(self) -> Path:
        """Generated illustration candidates + manifest (spec section 9)."""
        return self.work / "visuals"

    @property
    def visuals_manifest(self) -> Path:
        """Resumable checkpoint + per-shot generation record."""
        return self.visuals_dir / "manifest.json"

    @property
    def visuals_content_json(self) -> Path:
        """content.json emitted by the render stage for the compositor to consume
        (visuals spec section 9). Items reference the rendered <shot_id>.webm."""
        return self.visuals_dir / "content.json"

    # --- output ---
    @property
    def out_dir(self) -> Path:
        return self.root / "out"

    @property
    def output_mp4(self) -> Path:
        return self.out_dir / f"{self.episode_id}.mp4"

    @property
    def report_md(self) -> Path:
        return self.out_dir / f"{self.episode_id}_report.md"

    @property
    def contact_sheet(self) -> Path:
        return self.out_dir / f"{self.episode_id}_contact.png"

    # --- shared assets ---
    @property
    def luts_dir(self) -> Path:
        return self.root / "luts"

    @property
    def styles_dir(self) -> Path:
        return self.root / "styles"

    @property
    def compositions_dir(self) -> Path:
        return self.root / "compositions"


def resolve(episode_id: str, root: Path | None = None) -> Episode:
    return Episode(episode_id=episode_id, root=root or repo_root())
