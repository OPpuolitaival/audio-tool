"""CLI commands for audio-tool."""

from .analyze import analyze
from .audio2json import audio2json
from .normalize import normalize
from .trim import trim

__all__ = ["analyze", "audio2json", "normalize", "trim"]
