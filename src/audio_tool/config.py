"""Configuration and token handling."""

import os
from pathlib import Path


def get_huggingface_token() -> str | None:
    """
    Get HuggingFace token from environment or config file.

    Checks in order:
    1. HUGGINGFACE_TOKEN environment variable
    2. HF_TOKEN environment variable
    3. ~/.huggingface/token file
    """
    # Check environment variables
    token = os.environ.get("HUGGINGFACE_TOKEN")
    if token:
        return token

    token = os.environ.get("HF_TOKEN")
    if token:
        return token

    # Check ~/.huggingface/token file
    token_file = Path.home() / ".huggingface" / "token"
    if token_file.exists():
        token = token_file.read_text().strip()
        if token:
            return token

    return None
