"""Imagen + Veo dispatcher with silent Veo->Imagen fallback."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

from google import genai

from . import config
from .director import Scene

logger = logging.getLogger(__name__)


@dataclass
class VisualResult:
    path: Path
    kind: str  # "image" or "video"
    fallback_used: bool = False


def generate_visual(
    scene: Scene,
    output_dir: Path,
    content_type: str | None = None,
    image_generator: str = "imagen",
) -> VisualResult:
    """Dispatch to Imagen / Veo / FLUX. Veo and FLUX failures fall back to Imagen.

    Args:
        scene: the Director's plan for this scene.
        output_dir: where to write the asset.
        content_type: forwarded so Imagen picks the right quality enhancer.
        image_generator: 'imagen' (default, reliable Google stack) or 'flux'
            (open-source via MCP/HuggingFace). FLUX failures (queue timeout,
            tool error, HF auth) silently fall back to Imagen 4.
    """
    if scene.generator == "veo":
        try:
            return _generate_with_veo(scene, output_dir)
        except Exception as e:
            logger.warning(
                "Veo failed for scene %d (%s). Falling back to Imagen.",
                scene.scene_index, e,
            )
            result = _generate_with_imagen(scene, output_dir, content_type)
            return VisualResult(path=result.path, kind=result.kind, fallback_used=True)

    # Image scene — pick from three image generators, all with Imagen 4 fallback.
    if image_generator == "flux":
        try:
            return _generate_with_flux(scene, output_dir)
        except Exception as e:
            logger.warning(
                "FLUX failed for scene %d (%s). Falling back to Imagen 4.",
                scene.scene_index, e,
            )
            result = _generate_with_imagen(scene, output_dir, content_type)
            return VisualResult(path=result.path, kind=result.kind, fallback_used=True)

    if image_generator == "nano_banana":
        try:
            return _generate_with_nano_banana(scene, output_dir, content_type)
        except Exception as e:
            logger.warning(
                "Nano Banana failed for scene %d (%s). Falling back to Imagen 4.",
                scene.scene_index, e,
            )
            result = _generate_with_imagen(scene, output_dir, content_type)
            return VisualResult(path=result.path, kind=result.kind, fallback_used=True)

    return _generate_with_imagen(scene, output_dir, content_type)


def _generate_with_nano_banana(scene: Scene, output_dir: Path, content_type: str | None = None) -> VisualResult:
    """Generate via the Nano Banana model chain (Gemini multimodal image) on Vertex AI.

    Walks NANO_BANANA_MODEL_CHAIN in order: 3.1 Flash Image (preview) ->
    3 Pro Image -> 2.5 Flash Image. Most projects fall through to 2.5 since
    3.x is preview-gated. Caller raises (handled by dispatcher fallback to
    Imagen 4) only if EVERY model in the chain fails.
    """
    client = genai.Client(
        vertexai=True,
        project=config.GOOGLE_CLOUD_PROJECT,
        location=config.GOOGLE_CLOUD_LOCATION,
    )

    enhancer = config.QUALITY_ENHANCERS.get(content_type or "general", config.QUALITY_ENHANCERS["general"])
    enhanced_prompt = f"{scene.visual_prompt}. {enhancer}"

    last_error: Exception | None = None
    for model_id in config.NANO_BANANA_MODEL_CHAIN:
        try:
            logger.info("Nano Banana: trying %s for scene %d", model_id, scene.scene_index)
            response = client.models.generate_content(
                model=model_id,
                contents=enhanced_prompt,
                config={"response_modalities": ["IMAGE"]},
            )
            img_bytes = _extract_inline_image(response)
            if not img_bytes:
                last_error = RuntimeError(f"{model_id} returned no image part.")
                continue
            if model_id != config.NANO_BANANA_MODEL_CHAIN[0]:
                logger.warning(
                    "Nano Banana fell back to %s (chain head was %s).",
                    model_id, config.NANO_BANANA_MODEL_CHAIN[0],
                )
            out_path = output_dir / f"scene_{scene.scene_index}.png"
            out_path.write_bytes(img_bytes)
            return VisualResult(path=out_path, kind="image")
        except Exception as e:
            logger.info("Nano Banana model %s unavailable (%s). Trying next.", model_id, type(e).__name__)
            last_error = e
            continue

    raise RuntimeError(
        f"All Nano Banana models in the chain failed. Last error: "
        f"{type(last_error).__name__}: {last_error}"
    ) from last_error


def _extract_inline_image(response) -> bytes | None:
    """Pull image bytes out of a multimodal Gemini response.

    Handles both raw-bytes and base64-string forms of inline_data across
    google-genai SDK minor versions.
    """
    if not getattr(response, "candidates", None):
        return None
    parts = response.candidates[0].content.parts or []
    for part in parts:
        inline = getattr(part, "inline_data", None)
        if inline and getattr(inline, "data", None):
            data = inline.data
            if isinstance(data, str):
                import base64
                return base64.b64decode(data)
            return data
    return None


def _generate_with_flux(scene: Scene, output_dir: Path) -> VisualResult:
    """Call FLUX via the MCP client. Used when the UI explicitly picks FLUX.

    The MCP client launches the configured mcp-hfspace subprocess, connects,
    calls the FLUX schnell tool, decodes the base64 PNG, saves it. 10-min cap.
    """
    from . import mcp_image_client

    out_path = output_dir / f"scene_{scene.scene_index}.png"
    mcp_image_client.generate_image_via_mcp(
        prompt=scene.visual_prompt,
        output_path=out_path,
        aspect_ratio=config.ASPECT_RATIO,
        force=True,  # UI explicitly requested FLUX; bypass the env default
    )
    return VisualResult(path=out_path, kind="image")


def _generate_with_imagen(scene: Scene, output_dir: Path, content_type: str | None = None) -> VisualResult:
    client = genai.Client(
        vertexai=True,
        project=config.GOOGLE_CLOUD_PROJECT,
        location=config.GOOGLE_CLOUD_LOCATION,
    )
    # Append a content-type-aware quality enhancer to the Director's visual prompt.
    # Keeps the schema-level prompt clean; the enhancer is a presentation-layer concern.
    enhancer = config.QUALITY_ENHANCERS.get(content_type or "general", config.QUALITY_ENHANCERS["general"])
    enhanced_prompt = f"{scene.visual_prompt}. {enhancer}"

    result = client.models.generate_images(
        model=config.IMAGEN_MODEL,
        prompt=enhanced_prompt,
        config={
            "number_of_images": 1,
            "aspect_ratio": config.ASPECT_RATIO,
            "person_generation": "allow_adult",
        },
    )
    if not result.generated_images:
        raise RuntimeError(f"Imagen returned no images for scene {scene.scene_index}.")

    out_path = output_dir / f"scene_{scene.scene_index}.png"
    result.generated_images[0].image.save(str(out_path))
    return VisualResult(path=out_path, kind="image")


def _generate_with_veo(scene: Scene, output_dir: Path) -> VisualResult:
    client = genai.Client(
        vertexai=True,
        project=config.GOOGLE_CLOUD_PROJECT,
        location=config.GOOGLE_CLOUD_LOCATION,
    )
    operation = client.models.generate_videos(
        model=config.VEO_MODEL,
        prompt=scene.visual_prompt,
        config={
            "aspect_ratio": config.ASPECT_RATIO,
            "duration_seconds": int(max(5, min(8, round(scene.duration_seconds)))),
            "number_of_videos": 1,
            "person_generation": "allow_adult",
            # Veo 3 generates ambient audio (footsteps, wind, environmental
            # sound) natively. The assembler mixes this under the TTS narration
            # at reduced volume so the result feels grounded, not silent.
            "generate_audio": True,
        },
    )

    deadline = time.time() + 300  # 5-minute poll budget
    while not operation.done:
        if time.time() > deadline:
            raise TimeoutError(f"Veo timed out for scene {scene.scene_index}.")
        time.sleep(10)
        operation = client.operations.get(operation)

    if not operation.response or not operation.response.generated_videos:
        raise RuntimeError(f"Veo returned no videos for scene {scene.scene_index}.")

    out_path = output_dir / f"scene_{scene.scene_index}.mp4"
    operation.response.generated_videos[0].video.save(str(out_path))
    return VisualResult(path=out_path, kind="video")
