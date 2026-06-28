"""Single source of truth for env vars, model IDs, and constants."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _required(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required env var: {name}. See .env.example.")
    return value


# --- Auth ---
GEMINI_API_KEY: str = _required("GEMINI_API_KEY")
GOOGLE_CLOUD_PROJECT: str = _required("GOOGLE_CLOUD_PROJECT")
GOOGLE_CLOUD_LOCATION: str = os.getenv("GOOGLE_CLOUD_LOCATION") or "us-central1"

# --- Optional Postgres backend (V2 — run history + pgvector RAG) ---
# When disabled (default), the pipeline runs entirely on local filesystem.
ENABLE_POSTGRES: bool = (os.getenv("ENABLE_POSTGRES") or "false").lower() == "true"
DATABASE_URL: str = os.getenv("DATABASE_URL") or ""
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL") or "text-embedding-005"
EMBEDDING_DIM: int = 768

# --- Optional MCP image generation client (V2 — call an external Imagen tool) ---
# The pipeline keeps using its native Imagen 4 SDK call by default. When this
# flag is on, src/mcp_image_client.py becomes an alternative path that talks to
# any external MCP server exposing a generate_image-like tool.
#
# Default points at the bundled sample server (mcp_servers/imagen_server.py)
# which exposes Imagen 4 Ultra — a higher-quality model than the default
# imagen-4.0-generate-001 used by visuals.py. Swap MCP_IMAGE_SERVER_COMMAND /
# MCP_IMAGE_SERVER_ARGS to point at any other MCP image server (community
# servers, your own builds, third-party hosted servers).
ENABLE_MCP_IMAGE: bool = (os.getenv("ENABLE_MCP_IMAGE") or "false").lower() == "true"
MCP_IMAGE_SERVER_COMMAND: str = os.getenv("MCP_IMAGE_SERVER_COMMAND") or "python"
MCP_IMAGE_SERVER_ARGS: list[str] = (
    os.getenv("MCP_IMAGE_SERVER_ARGS") or "-m mcp_servers.imagen_server"
).split()
MCP_IMAGE_TOOL_NAME: str = os.getenv("MCP_IMAGE_TOOL_NAME") or "generate_image"

# --- Models (verified working 2026-05-13) ---
DIRECTOR_MODEL: str = os.getenv("GEMINI_MODEL") or "gemini-2.5-flash"
IMAGEN_MODEL: str = os.getenv("IMAGEN_MODEL") or "imagen-4.0-generate-001"
# Nano Banana — Gemini's multimodal native image generation models. Routed via
# Vertex AI for consistent auth with Imagen 4. We try the chain in order:
# Gemini 3.1 Flash Image (preview) -> 3 Pro Image -> 2.5 Flash Image.
# Most projects fall through to 2.5-flash-image (Nano Banana 1) since 3.x is
# preview-gated; the chain auto-upgrades when the project gets allowlisted.
NANO_BANANA_MODEL_CHAIN: list[str] = [
    os.getenv("NANO_BANANA_MODEL") or "gemini-3.1-flash-image",
    "gemini-3-pro-image",
    "gemini-2.5-flash-image",
]
VEO_MODEL: str = os.getenv("VEO_MODEL") or "veo-3.0-generate-001"
TTS_VOICE_DEFAULT: str = os.getenv("TTS_VOICE") or "en-US-Neural2-C"
TTS_LANGUAGE_DEFAULT: str = "en-US"

# --- Video output params ---
VIDEO_WIDTH: int = 1920
VIDEO_HEIGHT: int = 1080
VIDEO_FPS: int = 24
ASPECT_RATIO: str = "16:9"

# --- TTS pacing ---
TTS_SPEAKING_RATE_KIDS: float = 0.92
TTS_SPEAKING_RATE_ADULT: float = 1.0

# --- Captions ---
CAPTION_FONT_PATH: str = os.getenv("CAPTION_FONT") or "C:/Windows/Fonts/arialbd.ttf"

# --- Output paths ---
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
OUTPUT_DIR: Path = PROJECT_ROOT / "outputs"
CACHE_DIR: Path = OUTPUT_DIR / "_cache"

# --- Cost estimates (USD, ballpark May 2026) ---
COST_PER_IMAGEN_USD: float = 0.04
COST_PER_TTS_CHAR_USD: float = 16e-6
COST_PER_VEO_SECOND_USD: float = 0.50
COST_PER_GEMINI_INPUT_TOKEN_USD: float = 3.0e-7
COST_PER_GEMINI_OUTPUT_TOKEN_USD: float = 2.5e-6

# --- Image quality enhancers (appended to every Imagen prompt by content type) ---
# These extend the Director's six-element template with quality-focused keywords
# that Imagen interprets reliably. Keep them short — over-prompting degrades output.
QUALITY_ENHANCERS: dict[str, str] = {
    "math": (
        "bold flat illustration, clean vector lines, friendly mascot design, "
        "vibrant primary colors, premium animation quality, soft drop shadows, "
        "professional design"
    ),
    "place": (
        "8K documentary photography, golden-hour or blue-hour lighting, "
        "cinematic shallow depth of field, sharp foreground focus, "
        "National Geographic quality, premium colour grading, warm earthy tones, "
        "subtle film grain"
    ),
    "story": (
        "Pixar-grade illustration, expressive character design, "
        "rich watercolor texture, masterpiece storybook art, "
        "painterly detail, soft natural lighting, vibrant saturated palette"
    ),
    "general": (
        "professional illustration, polished design, vibrant detail, "
        "premium quality, clean composition"
    ),
}

# Portrait-specific enhancer — preserved verbatim for any future scene where the
# Director marks the subject as a real person whose identity must be preserved.
# NOT used as a default; only when scene content explicitly calls for it.
PORTRAIT_ENHANCER: str = (
    "Enhance the portrait while strictly preserving the subject's identity with "
    "accurate facial geometry. Do not change expression or face shape. Only allow "
    "subtle feature cleanup without altering who they are. "
    "Shot on Sony A1, 85mm f1.4 lens at f1.6, ISO 100, 1/200 shutter, cinematic "
    "shallow depth of field, perfect facial focus, editorial-neutral color profile. "
    "Lighting: soft directional, warm highlights, cool shadows, deeper contrast, "
    "expanded dynamic range, micro-contrast boost, smooth gradations, zero harsh "
    "shadows. Neutral premium colour tone, cinematic contrast curve, natural "
    "saturation, real skin texture, subtle film grain. 4K resolution, 10-bit "
    "colour, cinematic editorial style, premium clarity. "
    "Negative: no background change, no overly dramatic lighting, no face morphing, "
    "no fake glow, no flat lighting, no over-smooth skin. Keep aspect ratio and "
    "frame identical to the reference."
)
