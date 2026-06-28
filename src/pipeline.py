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

from . import assemble, config, director, dispatcher, visuals, voice
from .assemble import SceneAssets
from .director import VisualManifest, render_manifest_md
from .dispatcher import CampaignTask, DispatchedJob

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
class DispatchedAssetResult:
    """Output of one dispatched campaign-mode pass.

    The Visual Print Shop node runs ``generate_dispatched_asset()`` per task
    caught by the dispatcher. Each pass produces a synchronised pair of
    artifacts in the run directory:

      - ``hero_asset.png`` — the rendered visual
      - ``manifest.md`` — the professional spec document defending it

    All paths are absolute. ``audit_log`` carries the dispatcher's per-task
    trail forward so the UI can surface it without re-running dispatch.
    """

    run_id: str
    campaign: str
    task: CampaignTask
    asset_path: Path
    manifest_path: Path
    manifest: VisualManifest
    audit_log: list[str]
    cost: "CostBreakdown"
    elapsed_seconds: float
    image_generator_used: str
    fallback_used: bool = False


@dataclass
class RunResult:
    run_id: str
    final_video: Path | None              # None when render_video=False (assets-only mode)
    plan: director.VideoPlan
    cost: CostBreakdown
    elapsed_seconds: float
    veo_attempted: bool
    veo_fallback_used: bool
    voice_used: str
    voice_fallback_used: bool
    per_scene_clips: list[Path] = field(default_factory=list)
    scene_images: list[Path] = field(default_factory=list)   # paths to raw per-scene PNGs
    rendered: bool = True                  # whether the full video (audio + assembly) was produced


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
    image_generator: str = "imagen",
    domain: str | None = None,
    render_video: bool = True,
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
    plan = director.plan_video(topic, audience, content_type, domain=domain)

    # Veo policy:
    # 1. If allow_veo=False -> force every scene to imagen (cost-safe default).
    # 2. If allow_veo=True and content_type in {story, general} but the Director
    #    didn't pick any "veo" scenes -> force the middle scenes (everything
    #    except first and last) to "veo". This is the guardrail: a story video
    #    with zero Veo scenes is the bug we just shipped through. Surfacing a
    #    warning isn't enough; users expect Veo to actually fire when they
    #    enable it.
    if not allow_veo:
        plan = plan.model_copy(update={
            "scenes": [s.model_copy(update={"generator": "imagen"}) for s in plan.scenes]
        })
    elif (
        plan.content_type in ("story", "general")
        and not any(s.generator == "veo" for s in plan.scenes)
        and len(plan.scenes) >= 3
    ):
        logger.info("allow_veo=True but Director picked no veo scenes; forcing middle scenes to veo.")
        last_idx = len(plan.scenes) - 1
        forced = [
            s.model_copy(update={"generator": "veo"})
            if 0 < s.scene_index < last_idx
            else s
            for s in plan.scenes
        ]
        plan = plan.model_copy(update={"scenes": forced})

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

        gen_label = scene.generator if scene.generator == "veo" else image_generator
        _emit(progress, f"Scene {idx + 1}/{n}: generating visual ({gen_label})...")
        visual = visuals.generate_visual(
            scene,
            run_dir,
            content_type=plan.content_type,
            image_generator=image_generator,
        )
        if scene.generator == "veo" and visual.fallback_used:
            veo_fallback_used = True
        if visual.kind == "image":
            cost.imagen_usd += config.COST_PER_IMAGEN_USD
        else:
            cost.veo_usd += scene.duration_seconds * config.COST_PER_VEO_SECOND_USD

        # Audio + assembly only run when render_video=True. Assets-only mode
        # returns the image grid for instant workspace display (saves $/time).
        if render_video:
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
        else:
            # Lightweight asset-only mode: track the image path for the workspace.
            assets.append(SceneAssets(
                scene=scene, visual=visual,
                audio_path=Path(),
                audio_duration=0.0,
            ))

    # Collect raw image paths for asset-only workspaces (interior grid, brand kit, slide deck).
    scene_images = [sa.visual.path for sa in assets if sa.visual.kind == "image"]

    final_path: Path | None
    per_scene_clips: list[Path]
    if render_video:
        # Full video assembly: title -> scenes -> outro.
        _emit(progress, "Stitching final video with title and outro cards...")
        final_path = assemble.assemble_video(
            scene_assets=assets,
            output_path=run_dir / "final.mp4",
            title=plan.title,
            tagline=plan.tagline,
            outro_line="Built on the Google AI stack",
        )
        per_scene_clips = [
            run_dir / f"scene_{sa.scene.scene_index}_clip.mp4" for sa in assets
        ]
    else:
        _emit(progress, "Asset-only mode — skipped audio + video assembly.")
        final_path = None
        per_scene_clips = []

    elapsed = time.time() - started
    result = RunResult(
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
        scene_images=scene_images,
        rendered=render_video,
    )

    # Optional persistence — when ENABLE_POSTGRES is true, persist run metadata.
    # Failure here is logged but never aborts the user-facing run. Storage is a
    # side-channel; the video is already on disk.
    if config.ENABLE_POSTGRES:
        try:
            from . import storage
            stored = storage.stored_run_from_result(result)
            # Preserve the audience the caller passed in; RunResult doesn't carry it natively.
            stored.audience = audience
            row_id = storage.save_run(stored)
            _emit(progress, f"Run persisted to Postgres (row id={row_id}).")
        except Exception as e:
            logger.warning("Postgres save_run failed (run still completed locally): %s", e)

    return result


# =============================================================================
# Campaign-mode dispatched asset pipeline
# -----------------------------------------------------------------------------
# Called from the Streamlit UI when the user picks a campaign profile and runs
# a dispatch pass. Self-contained — does NOT use the multi-scene Director or
# the video assembly path. Returns one (image, manifest) pair per task.
# =============================================================================


def generate_dispatched_asset(
    job: DispatchedJob,
    task: CampaignTask,
    image_generator: str = "imagen",
    progress: ProgressCb = None,
) -> DispatchedAssetResult:
    """Execute one dispatched campaign task end-to-end.

    Flow:
      1. The Manifest Director reads the task action + role.md persona +
         rules.md guardrails, returns a structured VisualManifest.
      2. The manifest's enhanced ``visual_prompt`` is fed to the image
         generator (Imagen 4 / Nano Banana / FLUX). Failures of the chosen
         generator silently fall back to Imagen 4 per the existing
         visuals.generate_visual contract.
      3. Both artifacts (PNG + manifest.md) are written to a unique run
         directory under ``OUTPUT_DIR/run_<id>/``.

    Args:
        job: the DispatchedJob returned by ``dispatcher.dispatch_campaign``
            — carries the campaign slug, role text, rules text, and the
            full audit log we want to surface in the UI.
        task: the specific CampaignTask to render (from ``job.visual_tasks``).
        image_generator: 'imagen' (default) | 'nano_banana' | 'flux'.
        progress: optional callback for live UI updates.

    Returns:
        A DispatchedAssetResult carrying both file paths, the manifest
        object, the full audit log, cost, and elapsed time.
    """
    started = time.time()
    run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{_slugify(job.campaign)}_t{task.index}"
    run_dir = config.OUTPUT_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    cost = CostBreakdown()

    # 1) Manifest Director — produces the spec + enhanced visual prompt.
    _emit(progress, f"Manifest Director: planning Task {task.index} — {task.title}")
    asset_filename = "hero_asset.png"
    manifest = director.plan_visual_manifest(
        campaign=job.campaign,
        task_title=task.title,
        action_prompt=task.action,
        role_text=job.role,
        rules_text=job.rules,
        asset_filename=asset_filename,
    )
    cost.director_usd = (
        2000 * config.COST_PER_GEMINI_INPUT_TOKEN_USD
        + 1500 * config.COST_PER_GEMINI_OUTPUT_TOKEN_USD
    )

    # 2) Persist the manifest.md immediately — even if the image render
    # fails afterwards, the spec is on disk and recoverable.
    manifest_path = run_dir / "manifest.md"
    manifest_path.write_text(render_manifest_md(manifest), encoding="utf-8")
    (run_dir / "manifest.json").write_text(
        manifest.model_dump_json(indent=2), encoding="utf-8"
    )
    _emit(progress, f"Manifest written to {manifest_path.name}.")

    # 3) Image generation. Reuse visuals.generate_visual by constructing a
    # synthetic single-scene plan — same fallback chain, same cost tracking,
    # zero divergence from the proven path.
    scene = director.Scene(
        scene_index=0,
        visual_prompt=manifest.visual_prompt,
        generator="imagen",  # campaign-mode is image-only; Veo is for video flows
        audio_dialogue="(campaign-mode: no audio dialogue)",
        on_screen_text=None,
        duration_seconds=5.0,
        motion_style="gentle_drift",
        enable_subtitles=False,
    )

    _emit(progress, f"Rendering hero asset via {image_generator}...")
    visual = visuals.generate_visual(
        scene,
        run_dir,
        content_type="general",
        image_generator=image_generator,
    )

    # The visual is named scene_0.png by the existing convention; rename to
    # the campaign's expected asset filename so downstream tools can find it.
    final_asset = run_dir / asset_filename
    if visual.path != final_asset:
        visual.path.rename(final_asset)

    cost.imagen_usd = config.COST_PER_IMAGEN_USD
    fallback_used = visual.fallback_used

    elapsed = time.time() - started

    return DispatchedAssetResult(
        run_id=run_id,
        campaign=job.campaign,
        task=task,
        asset_path=final_asset,
        manifest_path=manifest_path,
        manifest=manifest,
        audit_log=list(job.audit_log),
        cost=cost,
        elapsed_seconds=elapsed,
        image_generator_used=image_generator,
        fallback_used=fallback_used,
    )
