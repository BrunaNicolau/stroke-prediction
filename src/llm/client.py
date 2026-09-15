"""
src/llm/client.py

Thin wrapper around the Gemini API (google-genai) used by the rest of
src/llm to turn prompts into natural language text.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from google import genai

DEFAULT_MODEL = "gemini-3.5-flash"


def _load_environment() -> None:
    """
    Load environment variables from the project's .env file.

    Works both locally and in Google Colab/Google Drive.
    """
    project_root = Path(__file__).resolve().parents[2]
    env_file = project_root / ".env"

    load_dotenv(env_file)


_load_environment()


def configure_client() -> genai.Client:
    """
    Build a Gemini API client from the GOOGLE_API_KEY environment variable.

    Raises RuntimeError if GOOGLE_API_KEY is not set.
    """
    api_key = os.environ.get("GOOGLE_API_KEY")

    if not api_key:
        project_root = Path(__file__).resolve().parents[2]
        env_file = project_root / ".env"

        raise RuntimeError(
            "GOOGLE_API_KEY not set.\n"
            f"Expected .env file at: {env_file}\n"
            "Add GOOGLE_API_KEY to the .env file."
        )

    return genai.Client(api_key=api_key)


def generate(
    prompt: str,
    client: genai.Client | None = None,
    model: str | None = None,
    **generation_config,
) -> str:
    """
    Generate text from a prompt using the Gemini API.

    Args:
        prompt: The prompt to send to Gemini.
        client: A configured genai.Client. If None, one is built via
                configure_client().
        model: Model name to use. Defaults to the GEMINI_MODEL environment
               variable, falling back to DEFAULT_MODEL.
        **generation_config: Forwarded as generation config
                             (e.g. temperature=0.2, max_output_tokens=512).

    Returns:
        The generated text.
    """
    if client is None:
        client = configure_client()

    if model is None:
        model = os.environ.get("GEMINI_MODEL", DEFAULT_MODEL)

    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=generation_config or None,
    )

    return response.text