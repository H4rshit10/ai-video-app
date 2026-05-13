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


def generate_visual(scene: Scene, output_dir: Path) -> VisualResult:
    """Dispatch to Imagen or Veo. Veo failures fall back to Imagen automatically."""
    if scene.generator == "veo":
        try:
            return _generate_with_veo(scene, output_dir)
        except Exception as e:
            logger.warning(
                "Veo failed for scene %d (%s). Falling back to Imagen.",
                scene.scene_index, e,
            )
            result = _generate_with_imagen(scene, output_dir)
            return VisualResult(path=result.path, kind=result.kind, fallback_used=True)
    return _generate_with_imagen(scene, output_dir)


def _generate_with_imagen(scene: Scene, output_dir: Path) -> VisualResult:
    client = genai.Client(
        vertexai=True,
        project=config.GOOGLE_CLOUD_PROJECT,
        location=config.GOOGLE_CLOUD_LOCATION,
    )
    result = client.models.generate_images(
        model=config.IMAGEN_MODEL,
        prompt=scene.visual_prompt,
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
