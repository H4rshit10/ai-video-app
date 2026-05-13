"""Google Cloud Text-to-Speech with SSML, one MP3 per scene."""
from __future__ import annotations

from pathlib import Path

from google.cloud import texttospeech

from . import config
from .director import Scene


def _build_ssml(dialogue: str, speaking_rate: float) -> str:
    safe = dialogue.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return f'<speak><prosody rate="{speaking_rate}">{safe}</prosody></speak>'


def synthesize_scene_audio(
    scene: Scene,
    output_dir: Path,
    voice_name: str = config.TTS_VOICE_DEFAULT,
    language_code: str = config.TTS_LANGUAGE_DEFAULT,
    speaking_rate: float = 1.0,
) -> tuple[Path, float]:
    """Synthesize one scene's dialogue. Returns (mp3_path, duration_seconds)."""
    client = texttospeech.TextToSpeechClient()
    ssml = _build_ssml(scene.audio_dialogue, speaking_rate)

    response = client.synthesize_speech(
        input=texttospeech.SynthesisInput(ssml=ssml),
        voice=texttospeech.VoiceSelectionParams(
            language_code=language_code, name=voice_name
        ),
        audio_config=texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3,
            speaking_rate=1.0,
            sample_rate_hertz=24000,
        ),
    )

    out_path = output_dir / f"scene_{scene.scene_index}.mp3"
    out_path.write_bytes(response.audio_content)

    return out_path, _mp3_duration(out_path)


def _mp3_duration(path: Path) -> float:
    from moviepy import AudioFileClip
    clip = AudioFileClip(str(path))
    try:
        return float(clip.duration)
    finally:
        clip.close()
