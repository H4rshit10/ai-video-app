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

# --- Models (verified working 2026-05-13) ---
DIRECTOR_MODEL: str = os.getenv("GEMINI_MODEL") or "gemini-2.5-flash"
IMAGEN_MODEL: str = os.getenv("IMAGEN_MODEL") or "imagen-4.0-generate-001"
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
