"""normalize command - normalize audio loudness using FFmpeg filters."""

import sys

import click

from audio_tool.normalization import (
    DynaudnormParams,
    LoudnormParams,
    SpeechnormParams,
    normalize_dynaudnorm,
    normalize_speechnorm,
)


@click.command()
@click.argument("audio_file", type=click.Path(exists=True))
@click.argument("output_file", type=click.Path())
@click.option(
    "--method", "-m",
    type=click.Choice(["speechnorm", "dynaudnorm"]),
    default="speechnorm",
    help="Normalization method (default: speechnorm)"
)
@click.option(
    "--lufs", "-I",
    type=float,
    default=-18.0,
    help="Target integrated loudness in LUFS (default: -18)"
)
@click.option(
    "--true-peak", "-TP",
    type=float,
    default=-1.0,
    help="Maximum true peak in dB (default: -1)"
)
@click.option(
    "--lra", "-LRA",
    type=float,
    default=4.0,
    help="Loudness range target (default: 4)"
)
@click.option(
    "--expansion", "-e",
    type=float,
    default=25.0,
    help="Speechnorm expansion factor (default: 25)"
)
@click.option(
    "--recovery", "-r",
    type=float,
    default=0.0001,
    help="Speechnorm recovery time (default: 0.0001)"
)
@click.option(
    "--frame-length", "-f",
    type=int,
    default=150,
    help="Dynaudnorm frame length in ms (default: 150)"
)
@click.option(
    "--gauss-size", "-g",
    type=int,
    default=15,
    help="Dynaudnorm gaussian window size, must be odd (default: 15)"
)
@click.option(
    "--max-gain",
    type=float,
    default=20.0,
    help="Dynaudnorm maximum gain in dB (default: 20)"
)
@click.option(
    "--compress", "-s",
    type=float,
    default=3.0,
    help="Dynaudnorm compression factor (default: 3)"
)
@click.option(
    "--codec",
    default=None,
    help="Audio codec (default: auto-detect from input)"
)
@click.option(
    "--bitrate",
    default=None,
    help="Audio bitrate for lossy codecs (default: auto-detect)"
)
def normalize(
    audio_file: str,
    output_file: str,
    method: str,
    lufs: float,
    true_peak: float,
    lra: float,
    expansion: float,
    recovery: float,
    frame_length: int,
    gauss_size: int,
    max_gain: float,
    compress: float,
    codec: str | None,
    bitrate: str | None,
):
    """
    Normalize audio loudness using two-stage processing.

    Uses FFmpeg filters to apply dynamic range compression followed by
    loudness normalization. Requires FFmpeg to be installed.

    \b
    Methods:
      speechnorm  Best for speech-heavy content (podcasts, radio)
      dynaudnorm  Best for mixed content (speech + music)

    \b
    Examples:
        # Basic usage (speechnorm method)
        audio-tool normalize input.mp3 output.mp3

        # Dynamic normalization for mixed content
        audio-tool normalize -m dynaudnorm input.mp3 output.mp3

        # Custom loudness target
        audio-tool normalize --lufs -16 input.wav output.wav

        # Gentler speech normalization
        audio-tool normalize -e 20 input.mp3 output.mp3
    """
    loudnorm_params = LoudnormParams(
        integrated_lufs=lufs,
        true_peak_db=true_peak,
        lra=lra,
    )

    try:
        if method == "speechnorm":
            speechnorm_params = SpeechnormParams(
                expansion=expansion,
                recovery=recovery,
            )
            click.echo(f"Normalizing with speechnorm + loudnorm...", err=True)
            click.echo(f"  Filter: {speechnorm_params.to_filter()},{loudnorm_params.to_filter()}", err=True)
            normalize_speechnorm(
                audio_file,
                output_file,
                speechnorm_params=speechnorm_params,
                loudnorm_params=loudnorm_params,
                audio_codec=codec,
                audio_bitrate=bitrate,
            )
        else:
            dynaudnorm_params = DynaudnormParams(
                frame_length=frame_length,
                gauss_size=gauss_size,
                max_gain=max_gain,
                compress=compress,
            )
            click.echo(f"Normalizing with dynaudnorm + loudnorm...", err=True)
            click.echo(f"  Filter: {dynaudnorm_params.to_filter()},{loudnorm_params.to_filter()}", err=True)
            normalize_dynaudnorm(
                audio_file,
                output_file,
                dynaudnorm_params=dynaudnorm_params,
                loudnorm_params=loudnorm_params,
                audio_codec=codec,
                audio_bitrate=bitrate,
            )

        click.echo(f"Done! Output saved to: {output_file}", err=True)

    except FileNotFoundError as e:
        raise click.ClickException(str(e))
    except ValueError as e:
        raise click.ClickException(str(e))
    except Exception as e:
        raise click.ClickException(f"Normalization failed: {e}")
