"""audio2json command - transcribe audio to JSON with speaker diarization."""

import json
import logging
import sys
from pathlib import Path

import click

from audio_tool.config import get_huggingface_token
from audio_tool.transcriber import (
    DIARIZATION_MODEL,
    MLX_AVAILABLE,
    MLX_MODEL_ALIASES,
    WHISPER_MODEL_MAP,
    Transcriber,
    WhisperBackend,
    resolve_mlx_model_path,
)

WHISPER_BACKEND_CHOICES = ["auto"] + [b.value for b in WhisperBackend]


@click.command()
@click.argument("audio_file", type=click.Path(exists=True))
@click.option(
    "--language",
    "-l",
    default="fi",
    help="Language code (default: fi). Pass an empty string to auto-detect.",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    help="Output JSON file (default: prints to stdout)",
)
@click.option(
    "--huggingface-token",
    envvar="HUGGINGFACE_TOKEN",
    help="HuggingFace API token (or set HUGGINGFACE_TOKEN/HF_TOKEN env var)",
)
@click.option(
    "--whisper-backend",
    "-b",
    type=click.Choice(WHISPER_BACKEND_CHOICES),
    default="auto",
    help="Whisper backend (default: auto = MLX large-v3 on Apple Silicon, whisper-timestamped elsewhere)",
)
@click.option(
    "--whisper-model",
    "-w",
    default="large-v3",
    help=(
        "MLX Whisper model. Aliases: "
        + "|".join(MLX_MODEL_ALIASES)
        + '. Or a full HuggingFace repo path (containing "/"). Default: large-v3.'
    ),
)
@click.option(
    "--skip-speaker-embedding",
    "-s",
    is_flag=True,
    help="Skip per-speaker embedding extraction (faster; diarization still runs)",
)
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
def audio2json(
    audio_file: str,
    language: str,
    output: str | None,
    huggingface_token: str | None,
    whisper_backend: str,
    whisper_model: str,
    skip_speaker_embedding: bool,
    verbose: bool,
):
    """
    Transcribe audio file to JSON with speaker diarization.

    Pipeline: pyannote diarization, Whisper per speech segment, recovery of
    speech diarization missed (Silero VAD over the gaps), decoder-loop
    collapsing, PANNs classification of non-speech and speaker embeddings.

    \b
    Examples:
        audio-tool audio2json recording.mp3
        audio-tool audio2json -l en -o output.json recording.wav
        audio-tool audio2json -w turbo recording.mp3   # faster MLX model (Mac)
    """
    token = huggingface_token or get_huggingface_token()
    if not token:
        raise click.UsageError(
            "HuggingFace token required. Provide via --huggingface-token, "
            "HUGGINGFACE_TOKEN/HF_TOKEN env var, or ~/.huggingface/token file"
        )

    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=sys.stderr,
    )
    logger = logging.getLogger(__name__)

    backend = None if whisper_backend == "auto" else WhisperBackend(whisper_backend)

    mlx_model_path = None
    if MLX_AVAILABLE and backend != WhisperBackend.WHISPER_TIMESTAMPED:
        try:
            mlx_model_path = resolve_mlx_model_path(whisper_model)
        except ValueError as e:
            raise click.UsageError(str(e))
        logger.info(
            f"MLX Whisper model: {mlx_model_path or WHISPER_MODEL_MAP[WhisperBackend.MLX_LARGE_V3]}"
        )

    audio_path = Path(audio_file)
    logger.info(f"Transcribing: {audio_path}")
    logger.info(f"Language: {language or 'auto-detect'}")

    transcriber = Transcriber(
        huggingface_token=token,
        language=language or None,
        whisper_backend=backend,
        logger=logger,
        mlx_model_path_override=mlx_model_path,
    )

    result = transcriber.transcribe(
        audio_path=audio_path,
        language=language or None,
        skip_speaker_embedding=skip_speaker_embedding,
    )

    json_output = json.dumps(result, ensure_ascii=False, indent=2)

    if output:
        output_path = Path(output)
        output_path.write_text(json_output, encoding="utf-8")
        logger.info(f"Results saved to: {output_path}")
    else:
        print(json_output)

    stats = result.get("statistics", {})
    meta = result.get("metadata", {})
    click.echo("\nSummary:", err=True)
    click.echo(f"  Whisper backend: {meta.get('whisper_backend', 'unknown')}", err=True)
    click.echo(f"  Diarization model: {DIARIZATION_MODEL}", err=True)
    click.echo(f"  Speakers: {stats.get('total_speakers', 0)}", err=True)
    click.echo(f"  Speech segments: {stats.get('total_speech_segments', 0)}", err=True)
    click.echo(
        f"  Speech duration: {stats.get('total_speech_duration', 0):.1f}s", err=True
    )
    click.echo(
        f"  Recovered from diarization gaps: {stats.get('recovered_speech_segments', 0)} segments "
        f"({stats.get('recovered_speech_duration', 0):.1f}s)",
        err=True,
    )
    stage_times = meta.get("stage_times_sec", {})
    if stage_times:
        click.echo("  Stage times (s):", err=True)
        for name, t_val in stage_times.items():
            click.echo(f"    {name:<20} {t_val:7.2f}", err=True)
    click.echo(
        f"  Total pipeline:      {meta.get('pipeline_time_sec', 0):7.2f}s", err=True
    )
