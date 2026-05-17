"""Stitch per-scene assets into the final video.

Enhancements (v1.1):
  - Per-scene .mp4 clips saved alongside final.mp4 (for the Interactive player).
  - 2-second branded title card prepended.
  - 2-second branded outro card appended.
  - Optional subtitle burn-in (full dialogue at bottom third) when enable_subtitles is True.
  - On-screen headline (on_screen_text) rendered at the upper third, separate from the subtitle band.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

from moviepy import (
    AudioFileClip,
    ColorClip,
    CompositeAudioClip,
    CompositeVideoClip,
    ImageClip,
    TextClip,
    VideoFileClip,
    concatenate_videoclips,
    vfx,
)

from . import config
from .director import Scene
from .visuals import VisualResult

logger = logging.getLogger(__name__)


# Visual layout constants (1920x1080 reference).
HEADLINE_Y = 110                     # upper-third on-screen_text
HEADLINE_HEIGHT = 160
SUBTITLE_Y = config.VIDEO_HEIGHT - 180  # bottom-third subtitle band
SUBTITLE_HEIGHT = 130
SUBTITLE_FONT_SIZE = 44
HEADLINE_FONT_SIZE = 64
TITLE_CARD_DURATION = 2.0
OUTRO_CARD_DURATION = 2.0


@dataclass
class SceneAssets:
    scene: Scene
    visual: VisualResult
    audio_path: Path
    audio_duration: float


def assemble_video(
    scene_assets: list[SceneAssets],
    output_path: Path,
    title: str | None = None,
    tagline: str | None = None,
    outro_line: str | None = None,
) -> Path:
    """Assemble the final video: title card + scenes + outro card.

    Side effect: writes a `scene_<index>_clip.mp4` per scene to the same directory
    as `output_path`, to support the Interactive player.
    """
    run_dir = output_path.parent
    clips: list = []

    # Title card
    if title is not None:
        title_clip = _build_title_card(title, tagline)
        title_path = run_dir / "title_card.mp4"
        _save_clip(title_clip, title_path)
        clips.append(title_clip)

    # Per-scene clips
    scene_clips = []
    for sa in scene_assets:
        scene_clip = _build_scene_clip(sa)
        per_scene_path = run_dir / f"scene_{sa.scene.scene_index}_clip.mp4"
        _save_clip(scene_clip, per_scene_path)
        scene_clips.append(scene_clip)
    clips.extend(scene_clips)

    # Outro card
    if outro_line is not None:
        outro_clip = _build_outro_card(outro_line)
        outro_path = run_dir / "outro_card.mp4"
        _save_clip(outro_clip, outro_path)
        clips.append(outro_clip)

    final = concatenate_videoclips(clips, method="compose")
    _save_clip(final, output_path)
    final.close()
    return output_path


def _save_clip(clip, path: Path) -> None:
    """Write a clip to disk with the project's standard codec settings."""
    clip.write_videofile(
        str(path),
        fps=config.VIDEO_FPS,
        codec="libx264",
        audio_codec="aac",
        logger=None,
        threads=4,
    )


# -------- Scene builder --------

SLOW_FLOOR = 0.6  # Cap on slow-mo before we accept a freeze-frame tail.


def _build_scene_clip(sa: SceneAssets):
    """Build one scene clip with audio-authoritative duration.

    The TTS narration is the source of truth for scene length. Veo clips that
    come up short are slowed (down to SLOW_FLOOR = 0.6x) to fit the narration;
    if they're still short after capping, the last frame freezes briefly to
    fill — much rarer than before. Veo clips that are longer than the
    narration are trimmed at audio end. Imagen scenes are static and accept
    any duration via Ken Burns motion.
    """
    target_duration = sa.audio_duration + 0.2  # short breathing tail

    veo_ambient = None
    if sa.visual.kind == "image":
        base = _apply_motion(
            ImageClip(str(sa.visual.path))
            .resized((config.VIDEO_WIDTH, config.VIDEO_HEIGHT))
            .with_duration(target_duration),
            motion_style=getattr(sa.scene, "motion_style", "gentle_drift"),
            duration=target_duration,
        )
    else:
        video = VideoFileClip(str(sa.visual.path)).resized(
            (config.VIDEO_WIDTH, config.VIDEO_HEIGHT)
        )
        veo_ambient = video.audio  # keep original ambient before any time-stretch
        base = _fit_video_to_audio(video, target_duration)

    layers = [base]

    if sa.scene.on_screen_text:
        layers.extend(_headline_overlay(sa.scene.on_screen_text, target_duration))

    if sa.scene.enable_subtitles:
        layers.extend(_subtitle_overlay(sa.scene.audio_dialogue, target_duration))

    composite = CompositeVideoClip(layers, size=(config.VIDEO_WIDTH, config.VIDEO_HEIGHT))
    composite = composite.with_duration(target_duration)
    composite = composite.with_audio(_build_scene_audio(sa.audio_path, veo_ambient, target_duration))
    return composite


def _fit_video_to_audio(video, target_duration: float):
    """Stretch or trim a Veo clip to match the narration's runtime.

    - If narration <= video: trim video at audio end (preserves full action).
    - If narration > video: slow video by the needed factor, capped at
      SLOW_FLOOR. If even SLOW_FLOOR isn't enough, the last frame freezes
      to fill the short remaining gap (much smaller than before).
    """
    if target_duration <= video.duration:
        return video.subclipped(0, target_duration)

    needed_factor = video.duration / target_duration
    if needed_factor >= SLOW_FLOOR:
        # Slow the clip proportionally so its new duration matches the audio.
        try:
            return video.with_speed_scaled(needed_factor)
        except AttributeError:
            logger.warning("with_speed_scaled unavailable; using freeze-frame fallback.")
            return video.with_duration(target_duration)

    # Mismatch too large — cap the slow-down at SLOW_FLOOR and freeze the
    # remaining tail. The freeze is now bounded to at most a couple of seconds.
    try:
        slowed = video.with_speed_scaled(SLOW_FLOOR)
    except AttributeError:
        slowed = video
    return slowed.with_duration(target_duration)


def _build_scene_audio(tts_path: Path, veo_ambient, duration: float):
    """Mix the TTS narration with Veo's native ambient sound (if present).

    For Imagen scenes there is no ambient; the narration plays clean.
    For Veo scenes the ambient is ducked to ~35% and layered under the
    narration so footsteps / wind / environmental sound feel present and
    grounded without competing with the voice.
    """
    tts = AudioFileClip(str(tts_path))
    if veo_ambient is None:
        return tts
    try:
        ducked = veo_ambient.with_volume_scaled(0.35)
    except AttributeError:
        # Fall back gracefully if a future moviepy release drops the helper —
        # keep the narration clean rather than crash.
        logger.warning("AudioClip.with_volume_scaled unavailable; skipping ambient mix.")
        return tts
    if ducked.duration > duration:
        ducked = ducked.subclipped(0, duration)
    return CompositeAudioClip([ducked, tts])


def _apply_motion(clip, motion_style: str, duration: float):
    """Apply per-scene motion to an ImageClip for cinematic feel.

    motion_style controls Imagen frame motion (Veo clips have their own motion):
      - zoom_in:      1.00x -> 1.18x, pulls viewer in (reveals, intimate moments)
      - zoom_out:     1.18x -> 1.00x, pulls back (establishing, sense of scale)
      - gentle_drift: 1.00x -> 1.08x, safe default for dialogue-heavy scenes
    """
    if motion_style == "zoom_in":
        return clip.with_effects([
            vfx.Resize(lambda t, d=duration: 1.0 + 0.18 * t / d)
        ])
    if motion_style == "zoom_out":
        return clip.with_effects([
            vfx.Resize(lambda t, d=duration: 1.18 - 0.18 * t / d)
        ])
    # gentle_drift (default)
    return clip.with_effects([
        vfx.Resize(lambda t, d=duration: 1.0 + 0.08 * t / d)
    ])


def _headline_overlay(text: str, duration: float) -> list:
    """Upper-third headline — clean text with strong stroke, no background box."""
    headline = (
        TextClip(
            text=text,
            font=config.CAPTION_FONT_PATH,
            font_size=HEADLINE_FONT_SIZE,
            color="white",
            stroke_color="black",
            stroke_width=4,
            method="caption",
            size=(int(config.VIDEO_WIDTH * 0.85), HEADLINE_HEIGHT - 20),
        )
        .with_duration(duration)
        .with_position(("center", HEADLINE_Y))
    )
    return [headline]


def _split_into_sentences(text: str) -> list[str]:
    """Split dialogue on terminal punctuation, keeping the punctuation with each chunk.

    Examples:
      "Hey there! Ever had a pizza?"  -> ["Hey there!", "Ever had a pizza?"]
      "One sentence with no end"      -> ["One sentence with no end"]
      "A. B. C."                       -> ["A.", "B.", "C."]
    """
    text = text.strip()
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def _subtitle_overlay(dialogue: str, duration: float) -> list:
    """Bottom-third subtitles — sentence-by-sentence timed, no background box.

    The spoken dialogue is split on terminal punctuation; each sentence is
    rendered as its own TextClip with start/duration mathematically divided
    across the scene's runtime. Text uses a thick black stroke for readability
    over any background (light frame, dark frame, busy frame) without needing
    an opaque rectangle behind it.
    """
    sentences = _split_into_sentences(dialogue)
    if not sentences:
        return []

    layers: list = []
    per_sentence = duration / len(sentences)
    for i, sentence in enumerate(sentences):
        start = i * per_sentence
        chunk_duration = (duration - start) if i == len(sentences) - 1 else per_sentence
        subtitle = (
            TextClip(
                text=sentence,
                font=config.CAPTION_FONT_PATH,
                font_size=SUBTITLE_FONT_SIZE,
                color="white",
                stroke_color="black",
                stroke_width=3,
                method="caption",
                size=(int(config.VIDEO_WIDTH * 0.92), SUBTITLE_HEIGHT - 20),
            )
            .with_start(start)
            .with_duration(chunk_duration)
            .with_position(("center", SUBTITLE_Y))
        )
        layers.append(subtitle)

    return layers


# -------- Title / outro cards --------

def _build_title_card(title: str, tagline: str | None):
    bg = ColorClip(
        size=(config.VIDEO_WIDTH, config.VIDEO_HEIGHT),
        color=(15, 20, 40),
    ).with_duration(TITLE_CARD_DURATION)

    title_clip = (
        TextClip(
            text=title,
            font=config.CAPTION_FONT_PATH,
            font_size=110,
            color="white",
            method="caption",
            size=(int(config.VIDEO_WIDTH * 0.85), 280),
        )
        .with_duration(TITLE_CARD_DURATION)
        .with_position(("center", config.VIDEO_HEIGHT // 2 - 200))
    )

    layers = [bg, title_clip]

    if tagline:
        tagline_clip = (
            TextClip(
                text=tagline,
                font=config.CAPTION_FONT_PATH,
                font_size=52,
                color="#cccccc",
                method="caption",
                size=(int(config.VIDEO_WIDTH * 0.75), 200),
            )
            .with_duration(TITLE_CARD_DURATION)
            .with_position(("center", config.VIDEO_HEIGHT // 2 + 80))
        )
        layers.append(tagline_clip)

    # Title card has no audio, but ensure it still concatenates cleanly: 1-second silence.
    return CompositeVideoClip(layers, size=(config.VIDEO_WIDTH, config.VIDEO_HEIGHT))


def _build_outro_card(line: str):
    bg = ColorClip(
        size=(config.VIDEO_WIDTH, config.VIDEO_HEIGHT),
        color=(15, 20, 40),
    ).with_duration(OUTRO_CARD_DURATION)

    line_clip = (
        TextClip(
            text=line,
            font=config.CAPTION_FONT_PATH,
            font_size=60,
            color="white",
            method="caption",
            size=(int(config.VIDEO_WIDTH * 0.8), 220),
        )
        .with_duration(OUTRO_CARD_DURATION)
        .with_position(("center", config.VIDEO_HEIGHT // 2 - 60))
    )

    return CompositeVideoClip([bg, line_clip], size=(config.VIDEO_WIDTH, config.VIDEO_HEIGHT))
