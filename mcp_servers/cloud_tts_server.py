"""Cloud TTS MCP server — exposes Google Cloud Text-to-Speech as MCP tools.

Single-purpose server: wraps Chirp 3 HD (with Neural2 silent fallback) behind
a standard MCP interface. Same architectural pattern as imagen_server.py —
different Google service.

Three tools exposed:
  - synthesize_speech(text, voice_name, language_code, speaking_rate)
        Main tool. Returns base64-encoded MP3 audio with metadata. Tries the
        requested voice first; on any error falls back silently to
        en-US-Neural2-C with SSML prosody.
  - list_voices(language_code)
        Discovery — returns every Cloud TTS voice for a given language.
  - voice_for_content_type(content_type)
        Recommends a Chirp 3 HD voice by content type
        (math -> Aoede, place -> Kore, story -> Leda, general -> Aoede).

Run standalone:
    python -m mcp_servers.cloud_tts_server

Wire into Claude Desktop (claude_desktop_config.json):
    {
      "mcpServers": {
        "cloud-tts": {
          "command": "python",
          "args": ["-m", "mcp_servers.cloud_tts_server"],
          "env": { "GOOGLE_CLOUD_PROJECT": "<your-project-id>" }
        }
      }
    }

After restart, Claude can say things like "read this paragraph in a warm
female voice" and pick the synthesize_speech tool automatically.
"""
from __future__ import annotations

import logging
import os
import sys

# CRITICAL for MCP servers running over stdio: stdout is reserved for JSON-RPC
# messages only. Any library that writes to stdout (gRPC's logs, Google client
# library warnings, deprecation notices) will corrupt the protocol and hang
# the client mid-call. Configure everything to stderr BEFORE importing the SDK.
logging.basicConfig(level=logging.WARNING, stream=sys.stderr, force=True)
for noisy in ("google", "grpc", "urllib3", "absl"):
    logging.getLogger(noisy).setLevel(logging.ERROR)
os.environ.setdefault("GRPC_VERBOSITY", "ERROR")
os.environ.setdefault("GLOG_minloglevel", "2")

try:
    from mcp.server.fastmcp import FastMCP
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "The 'mcp' package is not installed. Run:\n"
        "  .venv\\Scripts\\pip install mcp\n"
        "before launching this server."
    ) from e

from google.cloud import texttospeech
from google.cloud import texttospeech_v1

logger = logging.getLogger(__name__)


# --- 1. Server initialization -------------------------------------------------
server = FastMCP("cloud-tts-server")


# Content-type -> Chirp 3 HD voice mapping. Mirrors src/voice.py so the same
# curation logic is available to any MCP client (Claude Desktop, our own
# Python client, IDE plugins, etc.).
CONTENT_TYPE_VOICES: dict[str, str] = {
    "math": "en-US-Chirp3-HD-Aoede",
    "place": "en-US-Chirp3-HD-Kore",
    "story": "en-US-Chirp3-HD-Leda",
    "general": "en-US-Chirp3-HD-Aoede",
}
FALLBACK_VOICE = "en-US-Neural2-C"


# --- 4. External API connection (lazy) ---------------------------------------
def _tts_client() -> texttospeech.TextToSpeechClient:
    return texttospeech.TextToSpeechClient()


def _build_ssml(text: str, speaking_rate: float) -> str:
    """SSML wrapper used for Neural2 (Chirp 3 HD uses plain text)."""
    safe = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    safe = safe.replace(". ", '. <break time="180ms"/> ')
    safe = safe.replace("? ", '? <break time="220ms"/> ')
    safe = safe.replace("! ", '! <break time="220ms"/> ')
    return f'<speak><prosody rate="{speaking_rate}">{safe}</prosody></speak>'


def _synthesize_one(text: str, voice_name: str, language_code: str, speaking_rate: float) -> bytes:
    """Sync helper (still used for non-MCP direct calls + standalone tests)."""
    client = _tts_client()
    is_chirp = "Chirp3-HD" in voice_name
    if is_chirp:
        synth_input = texttospeech.SynthesisInput(text=text)
        rate = speaking_rate
    else:
        synth_input = texttospeech.SynthesisInput(ssml=_build_ssml(text, speaking_rate))
        rate = 1.0  # SSML already controls rate

    response = client.synthesize_speech(
        input=synth_input,
        voice=texttospeech.VoiceSelectionParams(
            language_code=language_code, name=voice_name,
        ),
        audio_config=texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3,
            speaking_rate=rate,
            sample_rate_hertz=24000,
        ),
    )
    return response.audio_content


async def _synthesize_one_async(text: str, voice_name: str, language_code: str, speaking_rate: float) -> bytes:
    """Async helper used by the MCP tool handler.

    Uses google-cloud-texttospeech's native async client (built on gRPC asyncio)
    rather than wrapping the sync client in a thread. This avoids the
    sync-gRPC-thread / asyncio.to_thread deadlock that hangs the MCP stdio
    transport.
    """
    client = texttospeech_v1.TextToSpeechAsyncClient()
    is_chirp = "Chirp3-HD" in voice_name
    if is_chirp:
        synth_input = texttospeech_v1.SynthesisInput(text=text)
        rate = speaking_rate
    else:
        synth_input = texttospeech_v1.SynthesisInput(ssml=_build_ssml(text, speaking_rate))
        rate = 1.0

    response = await client.synthesize_speech(
        request={
            "input": synth_input,
            "voice": texttospeech_v1.VoiceSelectionParams(
                language_code=language_code, name=voice_name,
            ),
            "audio_config": texttospeech_v1.AudioConfig(
                audio_encoding=texttospeech_v1.AudioEncoding.MP3,
                speaking_rate=rate,
                sample_rate_hertz=24000,
            ),
        }
    )
    return response.audio_content


# --- 2 + 3. Tool definitions + handlers --------------------------------------

import tempfile
import time as _time


@server.tool()
async def synthesize_speech(
    text: str,
    voice_name: str = "en-US-Chirp3-HD-Aoede",
    language_code: str = "en-US",
    speaking_rate: float = 1.0,
    output_path: str | None = None,
) -> dict:
    """Generate spoken audio from text via Google Cloud Text-to-Speech.

    Saves the resulting MP3 to disk and returns the path plus metadata.
    The on-disk pattern avoids stuffing large base64 payloads through the
    MCP stdio JSON-RPC frame — standard practice for any tool that returns
    binary media larger than a few KB.

    Args:
        text: words to read.
        voice_name: Cloud TTS voice ID. Default: en-US-Chirp3-HD-Aoede.
            Use list_voices to browse, voice_for_content_type for a recommend.
        language_code: BCP-47 language code matching the voice.
        speaking_rate: 0.25 to 4.0 (1.0 natural; 0.92 slower for young learners).
        output_path: optional explicit path to write the MP3. If None, a
            timestamped path under the system temp dir is generated.

    Returns:
        {
            "path": "<absolute path to the MP3>",
            "voice": "<actual voice used>",
            "language": "<language code>",
            "bytes": <int — file size on disk>,
            "mime_type": "audio/mp3",
        }

    Resilience: if the requested voice fails, falls back silently to
    en-US-Neural2-C with SSML prosody. `voice` reflects what actually ran.
    """
    # Offload the blocking gRPC call to a worker thread so FastMCP's asyncio
    # event loop can keep servicing stdio reads/writes. Without this, the
    # gRPC client's background thread and the MCP event loop can deadlock.
    # Use the native async gRPC client. asyncio.to_thread + sync gRPC
    # deadlocks the FastMCP event loop on Windows + Python 3.14.
    try:
        audio_bytes = await _synthesize_one_async(text, voice_name, language_code, speaking_rate)
        used_voice = voice_name
    except Exception as e:
        if voice_name == FALLBACK_VOICE:
            raise
        logger.warning(
            "Voice %s failed (%s) — falling back to %s",
            voice_name, type(e).__name__, FALLBACK_VOICE,
        )
        audio_bytes = await _synthesize_one_async(text, FALLBACK_VOICE, "en-US", speaking_rate)
        used_voice = FALLBACK_VOICE

    if output_path:
        out = output_path
    else:
        ts = int(_time.time() * 1000)
        out = os.path.join(tempfile.gettempdir(), f"cloud_tts_{ts}.mp3")

    with open(out, "wb") as f:
        f.write(audio_bytes)

    return {
        "path": os.path.abspath(out),
        "voice": used_voice,
        "language": language_code,
        "bytes": len(audio_bytes),
        "mime_type": "audio/mp3",
    }


@server.tool()
def list_voices(language_code: str = "en-US") -> list[dict]:
    """List every Cloud TTS voice available for a given language."""
    client = _tts_client()
    response = client.list_voices(language_code=language_code)
    return [
        {
            "name": v.name,
            "language_codes": list(v.language_codes),
            "gender": texttospeech.SsmlVoiceGender(v.ssml_gender).name,
            "natural_sample_rate_hertz": v.natural_sample_rate_hertz,
        }
        for v in response.voices
    ]


@server.tool()
def voice_for_content_type(content_type: str) -> str:
    """Recommend a Chirp 3 HD voice based on educational content type.

    math -> Aoede (warm)
    place -> Kore (narrator authority)
    story -> Leda (expressive)
    general / unknown -> Aoede
    """
    return CONTENT_TYPE_VOICES.get(content_type, CONTENT_TYPE_VOICES["general"])


if __name__ == "__main__":
    # Speak MCP over stdin/stdout. Any client (Claude Desktop, our Python
    # client, IDE plugin) can launch this and talk JSON-RPC.
    server.run(transport="stdio")
