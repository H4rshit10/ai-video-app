"""Stitch per-scene assets into a final .mp4 with motion + burned-in captions."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from moviepy import (
    AudioFileClip,
    ColorClip,
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


@dataclass
class SceneAssets:
    scene: Scene
    visual: VisualResult
    audio_path: Path
    audio_duration: float


def assemble_video(scene_assets: list[SceneAssets], output_path: Path) -> Path:
    clips = [_build_scene_clip(sa) for sa in scene_assets]
    final = concatenate_videoclips(clips, method="compose")
    final.write_videofile(
        str(output_path),
        fps=config.VIDEO_FPS,
        codec="libx264",
        audio_codec="aac",
        logger=None,
        threads=4,
    )
    final.close()
    return output_path


def _build_scene_clip(sa: SceneAssets):
    duration = max(sa.scene.duration_seconds, sa.audio_duration) + 0.4

    if sa.visual.kind == "image":
        base = (
            ImageClip(str(sa.visual.path))
            .resized((config.VIDEO_WIDTH, config.VIDEO_HEIGHT))
            .with_duration(duration)
            .with_effects([vfx.Resize(lambda t, d=duration: 1.0 + 0.14 * t / d)])
        )
    else:
        clip = VideoFileClip(str(sa.visual.path)).resized(
            (config.VIDEO_WIDTH, config.VIDEO_HEIGHT)
        )
        if clip.duration < duration:
            base = clip.with_duration(duration)
        else:
            base = clip.subclipped(0, duration)

    layers = [base]

    if sa.scene.on_screen_text:
        caption_bg = (
            ColorClip(size=(config.VIDEO_WIDTH, 180), color=(0, 0, 0))
            .with_opacity(0.55)
            .with_duration(duration)
            .with_position(("center", config.VIDEO_HEIGHT - 220))
        )
        caption = (
            TextClip(
                text=sa.scene.on_screen_text,
                font=config.CAPTION_FONT_PATH,
                font_size=64,
                color="white",
                stroke_color="black",
                stroke_width=2,
                method="caption",
                size=(int(config.VIDEO_WIDTH * 0.85), 160),
            )
            .with_duration(duration)
            .with_position(("center", config.VIDEO_HEIGHT - 210))
        )
        layers.extend([caption_bg, caption])

    composite = CompositeVideoClip(layers, size=(config.VIDEO_WIDTH, config.VIDEO_HEIGHT))
    composite = composite.with_duration(duration)
    composite = composite.with_audio(AudioFileClip(str(sa.audio_path)))
    return composite
