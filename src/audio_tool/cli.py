"""Main CLI entry point for audio-tool."""

import click

from audio_tool import __version__
from audio_tool.commands import audio2json


@click.group()
@click.version_option(version=__version__, prog_name="audio-tool")
def main():
    """Audio processing toolkit.

    A collection of audio processing tools including transcription
    with speaker diarization.
    """
    pass


# Register commands
main.add_command(audio2json)


if __name__ == "__main__":
    main()
