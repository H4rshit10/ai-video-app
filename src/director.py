"""Gemini-powered scene planner with structured output."""
from __future__ import annotations

from typing import Literal

from google import genai
from pydantic import BaseModel, Field

from . import config

SYSTEM_INSTRUCTIONS = """You are an expert educational video director.

You receive a TOPIC and an AUDIENCE. Plan a SHORT (15-30 second) explainer video.

Rules:
- 3 to 5 scenes total
- Each scene runs 4-7 seconds
- Pacing: hook -> build -> payoff (+ outro for stories)
- Vocabulary appropriate for the audience
- Every scene must include:
  * visual_prompt: vivid, self-contained, usable as-is by an image generator.
    Include style, subject, composition, lighting, mood.
    Do NOT include camera-jargon like "f/1.4" or shot numbers.
  * audio_dialogue: spoken narration. 1-2 short sentences. Natural English, no markdown,
    no asterisks, no parentheticals. Write it like spoken speech.
  * on_screen_text: a very short caption (1-5 words) OR null
  * duration_seconds: realistic to read the dialogue at a relaxed pace
- Identify content_type as one of: "math", "place", "story", "general"
- Choose the generator per scene:
  * math   -> "imagen" for every scene (precision matters; generative video morphs labels)
  * place  -> "imagen" for every scene (documentary photo style)
  * story  -> "veo" for narrative/action scenes, "imagen" for title and outro cards
  * general -> "imagen"

Visual style guidance by content_type:
- math: clean flat illustration, bold colors, large readable labels, minimal background, optional friendly mascot
- place: photorealistic documentary photography, warm color grade, golden hour, wide composition
- story: storybook illustration, watercolor or flat illustration, consistent character design, expressive, kid-friendly
- general: clean professional illustration

The image generator literally reads visual_prompt. Make every word count.
"""


class Scene(BaseModel):
    scene_index: int = Field(ge=0)
    visual_prompt: str = Field(min_length=20, max_length=600)
    generator: Literal["imagen", "veo"]
    audio_dialogue: str = Field(min_length=10, max_length=500)
    on_screen_text: str | None = Field(default=None, max_length=60)
    duration_seconds: float = Field(ge=3.0, le=10.0)


class VideoPlan(BaseModel):
    title: str = Field(max_length=80)
    content_type: Literal["math", "place", "story", "general"]
    scenes: list[Scene] = Field(min_length=3, max_length=6)


def plan_video(topic: str, audience: str, content_type_hint: str | None = None) -> VideoPlan:
    """One Gemini call -> a fully-formed multi-scene plan."""
    client = genai.Client(api_key=config.GEMINI_API_KEY)

    user_prompt = f"TOPIC: {topic}\nAUDIENCE: {audience}"
    if content_type_hint and content_type_hint != "auto":
        user_prompt += f"\nCONTENT_TYPE_HINT: {content_type_hint}"

    response = client.models.generate_content(
        model=config.DIRECTOR_MODEL,
        contents=user_prompt,
        config={
            "response_mime_type": "application/json",
            "response_schema": VideoPlan,
            "system_instruction": SYSTEM_INSTRUCTIONS,
            "temperature": 0.7,
        },
    )

    return VideoPlan.model_validate_json(response.text)
