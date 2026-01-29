"""audio2json command - transcribe audio to JSON with speaker diarization."""

import json
import logging
import sys
from pathlib import Path

import click

from audio_tool.config import get_huggingface_token
from audio_tool.transcriber import DiarizationModel, Transcriber, WhisperBackend


# Build choices from enums
WHISPER_CHOICES = [b.value for b in WhisperBackend]
DIARIZATION_CHOICES = [d.value for d in DiarizationModel]


@click.command()
@click.argument("audio_file", type=click.Path(exists=True))
@click.option(
    "--language", "-l",
    default="fi",
    help="Language code (default: fi)"
)
@click.option(
    "--output", "-o",
    type=click.Path(),
    help="Output JSON file (default: prints to stdout)"
)
@click.option(
    "--huggingface-token",
    envvar="HUGGINGFACE_TOKEN",
    help="HuggingFace API token (or set HUGGINGFACE_TOKEN/HF_TOKEN env var)"
)
@click.option(
    "--whisper-backend", "-w",
    type=click.Choice(WHISPER_CHOICES),
    default=WhisperBackend.WHISPER_TIMESTAMPED.value,
    help="Whisper backend to use"
)
@click.option(
    "--diarization-model", "-d",
    type=click.Choice(DIARIZATION_CHOICES),
    default=DiarizationModel.DIARIZATION_3_1.value,
    help="Pyannote diarization model to use"
)
@click.option(
    "--verbose", "-v",
    is_flag=True,
    help="Verbose output"
)
def audio2json(
    audio_file: str,
    language: str,
    output: str | None,
    huggingface_token: str | None,
    whisper_backend: str,
    diarization_model: str,
    verbose: bool,
):
    """
    Transcribe audio file to JSON with speaker diarization.

    Outputs JSON with metadata, segments (speech/non-speech), speaker embeddings,
    and statistics.

    \b
    Examples:
        # Basic usage
        audio-tool audio2json recording.mp3

        # With options
        audio-tool audio2json -l en -o output.json recording.wav

        # MLX backend (Mac)
        audio-tool audio2json -w mlx-turbo recording.mp3

        # Fastest combination on Mac
        audio-tool audio2json -w mlx-turbo -d pyannote/speaker-diarization-community-1 audio.mp3
    """
    # Get token from option, env, or config file
    token = huggingface_token or get_huggingface_token()
    if not token:
        raise click.UsageError(
            "HuggingFace token required. Provide via --huggingface-token, "
            "HUGGINGFACE_TOKEN/HF_TOKEN env var, or ~/.huggingface/token file"
        )

    # Setup logging
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=sys.stderr,
    )
    logger = logging.getLogger(__name__)

    audio_path = Path(audio_file)
    logger.info(f"Transcribing: {audio_path}")
    logger.info(f"Language: {language}")
    logger.info(f"Whisper backend: {whisper_backend}")
    logger.info(f"Diarization model: {diarization_model}")

    # Convert string choices to enums
    whisper_backend_enum = WhisperBackend(whisper_backend)
    diarization_model_enum = DiarizationModel(diarization_model)

    # Create transcriber and run
    transcriber = Transcriber(
        huggingface_token=token,
        language=language,
        whisper_backend=whisper_backend_enum,
        diarization_model=diarization_model_enum,
        logger=logger,
    )

    result = transcriber.transcribe(audio_path=audio_path, language=language)

    # Output results
    json_output = json.dumps(result, ensure_ascii=False, indent=2)

    if output:
        output_path = Path(output)
        output_path.write_text(json_output, encoding="utf-8")
        logger.info(f"Results saved to: {output_path}")
    else:
        print(json_output)

    # Print summary to stderr
    stats = result.get("statistics", {})
    meta = result.get("metadata", {})
    click.echo("\nSummary:", err=True)
    click.echo(f"  Whisper backend: {meta.get('whisper_backend', 'unknown')}", err=True)
    click.echo(f"  Diarization model: {meta.get('diarization_model', 'unknown')}", err=True)
    click.echo(f"  Speakers: {stats.get('total_speakers', 0)}", err=True)
    click.echo(f"  Speech segments: {stats.get('total_speech_segments', 0)}", err=True)
    click.echo(f"  Speech duration: {stats.get('total_speech_duration', 0):.1f}s", err=True)
