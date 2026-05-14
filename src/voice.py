"""Google Cloud Text-to-Speech with Chirp 3 HD primary and Neural2 silent fallback.

Voices are selected per content type (math -> Aoede warm; place -> Kore narrator;
story -> Leda expressive). On any Chirp 3 HD failure (voice unavailable, region
limits), the synthesiser transparently falls back to Neural2-C with SSML pacing.
"""
from __future__ import annotations

import logging
from pathlib import Path

from google.cloud import texttospeech

from . import config
from .director import Scene

logger = logging.getLogger(__name__)


# Content-type -> Chirp 3 HD voice mapping (en-US).
# Override anywhere by passing voice_name explicitly to synthesize_scene_audio().
CONTENT_TYPE_VOICE_MAP: dict[str, str] = {
    "math": "en-US-Chirp3-HD-Aoede",
    "place": "en-US-Chirp3-HD-Kore",
    "story": "en-US-Chirp3-HD-Leda",
    "general": "en-US-Chirp3-HD-Aoede",
}

FALLBACK_VOICE = "en-US-Neural2-C"


def voice_for_content_type(content_type: str) -> str:
    """Return the recommended Chirp 3 HD voice for a content type."""
    return CONTENT_TYPE_VOICE_MAP.get(content_type, CONTENT_TYPE_VOICE_MAP["general"])


def _build_ssml(dialogue: str, speaking_rate: float) -> str:
    """Wrap dialogue in SSML for Neural2 prosody control."""
    safe = dialogue.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    # Insert a small break between sentences for natural pacing.
    safe = safe.replace(". ", '. <break time="180ms"/> ')
    safe = safe.replace("? ", '? <break time="220ms"/> ')
    safe = safe.replace("! ", '! <break time="220ms"/> ')
    return f'<speak><prosody rate="{speaking_rate}">{safe}</prosody></speak>'


def synthesize_scene_audio(
    scene: Scene,
    output_dir: Path,
    voice_name: str = config.TTS_VOICE_DEFAULT,
    language_code: str = config.TTS_LANGUAGE_DEFAULT,
    speaking_rate: float = 1.0,
) -> tuple[Path, float, str]:
    """Synthesize one scene's dialogue. Returns (mp3_path, duration_seconds, voice_used).

    Tries the requested voice first (typically Chirp 3 HD). On any error, silently
    falls back to Neural2-C with SSML prosody control. The returned voice_used
    reflects what actually produced the audio.
    """
    client = texttospeech.TextToSpeechClient()
    out_path = output_dir / f"scene_{scene.scene_index}.mp3"
    is_chirp3 = "Chirp3-HD" in voice_name

    try:
        audio_bytes = _synthesize_one(
            client=client,
            text=scene.audio_dialogue,
            voice_name=voice_name,
            language_code=language_code,
            speaking_rate=speaking_rate,
            use_ssml=not is_chirp3,
        )
        out_path.write_bytes(audio_bytes)
        return out_path, _mp3_duration(out_path), voice_name
    except Exception as primary_error:
        if voice_name == FALLBACK_VOICE:
            raise
        logger.warning(
            "TTS voice %s failed (%s). Falling back to %s.",
            voice_name, type(primary_error).__name__, FALLBACK_VOICE,
        )
        audio_bytes = _synthesize_one(
            client=client,
            text=scene.audio_dialogue,
            voice_name=FALLBACK_VOICE,
            language_code="en-US",
            speaking_rate=speaking_rate,
            use_ssml=True,
        )
        out_path.write_bytes(audio_bytes)
        return out_path, _mp3_duration(out_path), FALLBACK_VOICE


def _synthesize_one(
    client: texttospeech.TextToSpeechClient,
    text: str,
    voice_name: str,
    language_code: str,
    speaking_rate: float,
    use_ssml: bool,
) -> bytes:
    """Single synth call. Chirp 3 HD uses plain text (limited SSML support); others use SSML."""
    if use_ssml:
        synth_input = texttospeech.SynthesisInput(ssml=_build_ssml(text, speaking_rate))
    else:
        synth_input = texttospeech.SynthesisInput(text=text)

    response = client.synthesize_speech(
        input=synth_input,
        voice=texttospeech.VoiceSelectionParams(
            language_code=language_code, name=voice_name,
        ),
        audio_config=texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3,
            speaking_rate=speaking_rate if not use_ssml else 1.0,
            sample_rate_hertz=24000,
        ),
    )
    return response.audio_content


def _mp3_duration(path: Path) -> float:
    from moviepy import AudioFileClip
    clip = AudioFileClip(str(path))
    try:
        return float(clip.duration)
    finally:
        clip.close()
