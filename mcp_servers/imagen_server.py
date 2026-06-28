"""Sample Imagen MCP server — exposes Imagen 4 Ultra as a tool over MCP stdio.

Run standalone:
    python -m mcp_servers.imagen_server

The pipeline's MCP image client (src/mcp_image_client.py) launches this server
as a subprocess and talks to it over stdio. The same client can be pointed at
any other MCP server exposing a generate_image-like tool by swapping the
MCP_IMAGE_SERVER_COMMAND / MCP_IMAGE_SERVER_ARGS env vars.

Pattern reference:
    - Server init   : FastMCP("imagen-ultra-server")
    - Tool definition + handler : @server.tool decorator on generate_image()
    - External API connection   : genai.Client(vertexai=True).models.generate_images

The model exposed here is imagen-4.0-ultra-generate-001 — Google's highest-
fidelity Imagen variant, deliberately different from the imagen-4.0-generate-001
the main app uses by default. This lets you A/B the two image qualities and
also demonstrates the "swap in any image generator over MCP" extensibility.
"""
from __future__ import annotations

import base64
import os
from pathlib import Path

# MCP SDK — lazy import via direct module load so the static checker doesn't
# complain when the optional dependency isn't installed yet.
try:
    from mcp.server.fastmcp import FastMCP
except ImportError as e:
    raise SystemExit(
        "The 'mcp' package is not installed. Run:\n"
        "  .venv\\Scripts\\pip install mcp\n"
        "before launching this server."
    ) from e

from google import genai


# --- 1. Server initialization -------------------------------------------------
server = FastMCP("imagen-ultra-server")


# --- 4. External API connection (lazy — built on first call) ------------------
def _imagen_client() -> genai.Client:
    project = os.environ.get("GOOGLE_CLOUD_PROJECT")
    location = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
    if not project:
        raise RuntimeError(
            "GOOGLE_CLOUD_PROJECT env var is required. Set it before running this server."
        )
    return genai.Client(vertexai=True, project=project, location=location)


# --- 2 + 3. Tool definition + handler ----------------------------------------
@server.tool()
def generate_image(prompt: str, aspect_ratio: str = "16:9") -> dict:
    """Generate a single high-quality image with Imagen 4 Ultra.

    Args:
        prompt: Vivid description of the image to generate.
        aspect_ratio: One of '1:1', '16:9', '9:16', '4:3', '3:4'.

    Returns:
        {
            "mime_type": "image/png",
            "data": "<base64-encoded PNG>",
            "model": "imagen-4.0-ultra-generate-001"
        }
    """
    model_id = os.environ.get("MCP_SERVER_IMAGEN_MODEL", "imagen-4.0-ultra-generate-001")
    client = _imagen_client()
    result = client.models.generate_images(
        model=model_id,
        prompt=prompt,
        config={
            "number_of_images": 1,
            "aspect_ratio": aspect_ratio,
            "person_generation": "allow_adult",
        },
    )
    if not result.generated_images:
        raise RuntimeError("Imagen returned no images.")
    img_bytes = result.generated_images[0].image.image_bytes
    return {
        "mime_type": "image/png",
        "data": base64.b64encode(img_bytes).decode("ascii"),
        "model": model_id,
    }


@server.tool()
def list_supported_aspect_ratios() -> list[str]:
    """Return the aspect ratios this server accepts."""
    return ["1:1", "16:9", "9:16", "4:3", "3:4"]


if __name__ == "__main__":
    # Speak MCP over stdin/stdout. Claude Desktop, IDE plugins, or any MCP
    # client (including src/mcp_image_client.py) can launch this and talk JSON-RPC.
    server.run(transport="stdio")
