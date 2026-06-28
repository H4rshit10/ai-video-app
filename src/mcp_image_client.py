"""MCP client wrapper — call any external image-generation MCP server.

The native Imagen 4 path in visuals.py uses google-genai directly. This module
is the OPTIONAL alternative path: it speaks Model Context Protocol over stdio
to any server that exposes a generate_image-like tool. The bundled sample
server is mcp_servers/imagen_server.py (Imagen 4 Ultra), but the same client
works with any other MCP image server — community-built, third-party, or your
own — by setting these env vars:

    ENABLE_MCP_IMAGE=true
    MCP_IMAGE_SERVER_COMMAND=<binary or interpreter to launch>
    MCP_IMAGE_SERVER_ARGS=<space-separated args>
    MCP_IMAGE_TOOL_NAME=generate_image    # or whatever tool the server exposes

This is the same wire protocol Claude Desktop uses to talk to its MCP servers.
"""
from __future__ import annotations

import asyncio
import base64
import logging
import os
from pathlib import Path

from . import config

logger = logging.getLogger(__name__)

# 10-minute generation cap. FLUX on HF Spaces free tier can queue for several
# minutes; this gives it a real chance without hanging the pipeline forever.
DEFAULT_TIMEOUT_SECONDS = 600


def generate_image_via_mcp(
    prompt: str,
    output_path: Path,
    aspect_ratio: str = "16:9",
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    force: bool = False,
) -> Path:
    """Sync wrapper — launch the configured MCP server, call its image tool, save the PNG.

    Args:
        prompt: text prompt.
        output_path: where to save the PNG.
        aspect_ratio: '16:9' / '1:1' / '9:16' etc. The arg builder translates
            this into the right per-tool params (e.g. width/height for FLUX).
        timeout_seconds: hard cap on the whole call (handshake + generation).
        force: bypass the ENABLE_MCP_IMAGE flag — used when the UI explicitly
            asks for FLUX even though the env default is off.

    Returns:
        Path to the saved image file.
    """
    if not config.ENABLE_MCP_IMAGE and not force:
        raise RuntimeError("MCP image client is disabled (ENABLE_MCP_IMAGE=false).")
    return asyncio.run(
        asyncio.wait_for(
            _call_image_tool_async(prompt, output_path, aspect_ratio),
            timeout=timeout_seconds,
        )
    )


def _build_tool_arguments(tool_name: str, prompt: str, aspect_ratio: str) -> dict:
    """Translate (prompt, aspect_ratio) into the kwargs each MCP tool expects.

    FLUX schnell on mcp-hfspace takes prompt + width/height/steps/seed.
    Imagen-style and most others take prompt + aspect_ratio.
    Extend this map as new tools are wired.
    """
    lower = tool_name.lower()
    if "flux" in lower or "schnell" in lower:
        sizes = {"16:9": (1280, 720), "9:16": (720, 1280), "1:1": (1024, 1024), "4:3": (1152, 864)}
        w, h = sizes.get(aspect_ratio, (1280, 720))
        return {
            "prompt": prompt,
            "width": w,
            "height": h,
            "num_inference_steps": 4,
            "randomize_seed": True,
            "seed": 0,
        }
    # Default shape — Imagen-flavoured servers (including our bundled sample)
    return {"prompt": prompt, "aspect_ratio": aspect_ratio}


async def _call_image_tool_async(prompt: str, output_path: Path, aspect_ratio: str) -> Path:
    # Deferred import — keeps the rest of the app usable without `mcp` installed.
    try:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
    except ImportError as e:
        raise RuntimeError(
            "The 'mcp' package is required. Install with:  pip install mcp"
        ) from e

    # Pass the parent's env explicitly so HF_TOKEN (and any GCP creds) reach
    # the subprocess. Some MCP server runtimes don't honour env=None reliably.
    params = StdioServerParameters(
        command=config.MCP_IMAGE_SERVER_COMMAND,
        args=config.MCP_IMAGE_SERVER_ARGS,
        env={**os.environ},
    )

    logger.info(
        "Launching MCP image server: %s %s",
        config.MCP_IMAGE_SERVER_COMMAND, " ".join(config.MCP_IMAGE_SERVER_ARGS),
    )

    async with stdio_client(params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            # Handshake — server advertises its tool list here.
            await session.initialize()

            # (Optional sanity check — confirm the configured tool exists)
            tools_response = await session.list_tools()
            tool_names = [t.name for t in tools_response.tools]
            if config.MCP_IMAGE_TOOL_NAME not in tool_names:
                raise RuntimeError(
                    f"MCP server does not expose tool '{config.MCP_IMAGE_TOOL_NAME}'. "
                    f"Available: {tool_names}"
                )

            # Build args matching the tool's expected schema, then invoke.
            tool_args = _build_tool_arguments(config.MCP_IMAGE_TOOL_NAME, prompt, aspect_ratio)
            result = await session.call_tool(
                config.MCP_IMAGE_TOOL_NAME,
                arguments=tool_args,
            )

            # MCP returns content blocks. Find the image (or base64 payload).
            img_bytes = _extract_image_bytes(result)
            output_path.write_bytes(img_bytes)
            logger.info("MCP image saved to %s (%d bytes)", output_path, len(img_bytes))
            return output_path


def _extract_image_bytes(result) -> bytes:
    """Pull image bytes out of an MCP tool-call result.

    Tries the common shapes: a structured `image` content block, or a JSON
    payload with {mime_type, data} fields. Falls back to raising if neither.
    """
    for block in getattr(result, "content", []):
        # Some servers return an `image` content block with raw base64 data.
        if getattr(block, "type", None) == "image":
            return base64.b64decode(block.data)
        # Some return a `text` block carrying JSON {mime_type, data}.
        if getattr(block, "type", None) == "text":
            try:
                import json
                payload = json.loads(block.text)
                if "data" in payload:
                    return base64.b64decode(payload["data"])
            except (json.JSONDecodeError, KeyError):
                continue
    raise RuntimeError(
        "MCP server response did not contain a recognisable image payload. "
        "Check the server's return shape."
    )
