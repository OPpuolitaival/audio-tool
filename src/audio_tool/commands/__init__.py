"""CLI commands for audio-tool."""

from .audio2json import audio2json
from .normalize import normalize

__all__ = ["audio2json", "normalize"]
