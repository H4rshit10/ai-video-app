"""Orchestrator: Director -> per-scene assets -> assembly. Tracks cost."""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
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
    voice_name: str = config.TTS_VOICE_DEFAULT,
    allow_veo: bool = False,
    progress: ProgressCb = None,
) -> RunResult:
    started = time.time()
    run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{_slugify(topic)}"
    run_dir = config.OUTPUT_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    cost = CostBreakdown()
    veo_fallback_used = False

    # 1) Director
    _emit(progress, f"Planning '{topic}' for {audience}...")
    plan = director.plan_video(topic, audience, content_type)

    # Veo policy: when allow_veo is False, override every scene to use Imagen.
    # This keeps cost predictable (~$0.20) and visuals consistent across scenes.
    if not allow_veo:
        plan = plan.model_copy(update={
            "scenes": [s.model_copy(update={"generator": "imagen"}) for s in plan.scenes]
        })

    veo_attempted = any(s.generator == "veo" for s in plan.scenes)
    (run_dir / "plan.json").write_text(plan.model_dump_json(indent=2), encoding="utf-8")
    cost.director_usd = (
        1500 * config.COST_PER_GEMINI_INPUT_TOKEN_USD
        + 1000 * config.COST_PER_GEMINI_OUTPUT_TOKEN_USD
    )

    # 2) Per-scene assets (sequential — gives clean progress updates)
    speaking_rate = _audience_speaking_rate(audience)
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

        _emit(progress, f"Scene {idx + 1}/{n}: synthesizing voice...")
        audio_path, audio_duration = voice.synthesize_scene_audio(
            scene, run_dir, voice_name=voice_name, speaking_rate=speaking_rate
        )
        cost.tts_usd += len(scene.audio_dialogue) * config.COST_PER_TTS_CHAR_USD

        assets.append(SceneAssets(
            scene=scene, visual=visual,
            audio_path=audio_path, audio_duration=audio_duration,
        ))

    # 3) Assemble
    _emit(progress, "Stitching final video...")
    final_path = assemble.assemble_video(assets, run_dir / "final.mp4")

    elapsed = time.time() - started
    return RunResult(
        run_id=run_id,
        final_video=final_path,
        plan=plan,
        cost=cost,
        elapsed_seconds=elapsed,
        veo_attempted=veo_attempted,
        veo_fallback_used=veo_fallback_used,
    )
