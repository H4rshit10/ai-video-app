"""Orchestrator: Director -> per-scene assets -> assembly. Tracks cost.

v1.1 changes:
  - Voice auto-selected from Director's voice_recommendation (Chirp 3 HD), with optional UI override
  - Per-scene MP4 clips saved alongside final.mp4
  - Title/outro cards generated from the Director's title + tagline
  - quiz.json saved separately for easy frontend access
  - voice_used reflects any silent Chirp -> Neural2 fallbacks
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

from . import assemble, config, director, visuals, voice
from .assemble import SceneAssets

logger = logging.getLogger(__name__)


@dataclass
class CostBreakdown:
    director_usd: float = 0.0
    imagen_usd: float = 0.0
    veo_usd: float = 0.0
    tts_usd: float = 0.0

    @property
    def total_usd(self) -> float:
        return self.director_usd + self.imagen_usd + self.veo_usd + self.tts_usd


@dataclass
class RunResult:
    run_id: str
    final_video: Path
    plan: director.VideoPlan
    cost: CostBreakdown
    elapsed_seconds: float
    veo_attempted: bool
    veo_fallback_used: bool
    voice_used: str
    voice_fallback_used: bool
    per_scene_clips: list[Path] = field(default_factory=list)


ProgressCb = Callable[[str], None] | None


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return slug[:40] or "video"


def _audience_speaking_rate(audience: str) -> float:
    a = audience.lower()
    kid_markers = ("year-old", "year old", "kid", "child", "first-time", "first time", "beginner")
    return config.TTS_SPEAKING_RATE_KIDS if any(t in a for t in kid_markers) else config.TTS_SPEAKING_RATE_ADULT


def _emit(progress: ProgressCb, msg: str) -> None:
    logger.info(msg)
    if progress:
        progress(msg)


def generate_video(
    topic: str,
    audience: str,
    content_type: str | None = None,
    voice_name: str | None = None,
    allow_veo: bool = False,
    progress: ProgressCb = None,
) -> RunResult:
    """Generate one video end-to-end.

    voice_name: explicit override. If None, uses Director's voice_recommendation.
    """
    started = time.time()
    run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{_slugify(topic)}"
    run_dir = config.OUTPUT_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    cost = CostBreakdown()
    veo_fallback_used = False

    # 1) Director
    _emit(progress, f"Planning '{topic}' for {audience}...")
    plan = director.plan_video(topic, audience, content_type)

    # Veo policy: when disallowed, force every scene to imagen for consistency + cost predictability.
    if not allow_veo:
        plan = plan.model_copy(update={
            "scenes": [s.model_copy(update={"generator": "imagen"}) for s in plan.scenes]
        })

    veo_attempted = any(s.generator == "veo" for s in plan.scenes)

    (run_dir / "plan.json").write_text(plan.model_dump_json(indent=2), encoding="utf-8")
    if plan.end_quiz is not None:
        (run_dir / "quiz.json").write_text(plan.end_quiz.model_dump_json(indent=2), encoding="utf-8")
    # Collect any per-scene checkpoints (Interactive mode reads this).
    checkpoints = {
        str(s.scene_index): s.checkpoint.model_dump()
        for s in plan.scenes if s.checkpoint is not None
    }
    if checkpoints:
        import json
        (run_dir / "checkpoints.json").write_text(json.dumps(checkpoints, indent=2), encoding="utf-8")

    cost.director_usd = (
        1500 * config.COST_PER_GEMINI_INPUT_TOKEN_USD
        + 1200 * config.COST_PER_GEMINI_OUTPUT_TOKEN_USD
    )

    # 2) Voice selection: explicit override > Director recommendation > content-type default.
    chosen_voice = voice_name or plan.voice_recommendation or voice.voice_for_content_type(plan.content_type)
    speaking_rate = _audience_speaking_rate(audience)
    voice_fallback_used = False
    voice_used = chosen_voice

    # 3) Per-scene assets (sequential — clean progress updates).
    assets: list[SceneAssets] = []
    n = len(plan.scenes)
    for scene in plan.scenes:
        idx = scene.scene_index

        _emit(progress, f"Scene {idx + 1}/{n}: generating visual ({scene.generator})...")
        visual = visuals.generate_visual(scene, run_dir)
        if scene.generator == "veo" and visual.fallback_used:
            veo_fallback_used = True
        if visual.kind == "image":
            cost.imagen_usd += config.COST_PER_IMAGEN_USD
        else:
            cost.veo_usd += scene.duration_seconds * config.COST_PER_VEO_SECOND_USD

        _emit(progress, f"Scene {idx + 1}/{n}: synthesizing voice ({chosen_voice})...")
        audio_path, audio_duration, actual_voice = voice.synthesize_scene_audio(
            scene, run_dir, voice_name=chosen_voice, speaking_rate=speaking_rate,
        )
        if actual_voice != chosen_voice:
            voice_fallback_used = True
            voice_used = actual_voice
            chosen_voice = actual_voice  # stick with the fallback for remaining scenes
        cost.tts_usd += len(scene.audio_dialogue) * config.COST_PER_TTS_CHAR_USD

        assets.append(SceneAssets(
            scene=scene, visual=visual,
            audio_path=audio_path, audio_duration=audio_duration,
        ))

    # 4) Assemble: title -> scenes (with per-scene clips saved) -> outro.
    _emit(progress, "Stitching final video with title and outro cards...")
    final_path = assemble.assemble_video(
        scene_assets=assets,
        output_path=run_dir / "final.mp4",
        title=plan.title,
        tagline=plan.tagline,
        outro_line="Built on the Google AI stack",
    )

    # Collect per-scene clip paths for Interactive mode.
    per_scene_clips = [
        run_dir / f"scene_{sa.scene.scene_index}_clip.mp4" for sa in assets
    ]

    elapsed = time.time() - started
    return RunResult(
        run_id=run_id,
        final_video=final_path,
        plan=plan,
        cost=cost,
        elapsed_seconds=elapsed,
        veo_attempted=veo_attempted,
        veo_fallback_used=veo_fallback_used,
        voice_used=voice_used,
        voice_fallback_used=voice_fallback_used,
        per_scene_clips=per_scene_clips,
    )
