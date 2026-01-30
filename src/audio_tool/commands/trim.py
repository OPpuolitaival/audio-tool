"""trim command - trim silence or segments from audio files."""

from pathlib import Path

import click

from audio_tool.trimmer import detect_trim_points, trim_audio


@click.command()
@click.argument("audio_file", type=click.Path(exists=True))
@click.argument("output_file", type=click.Path(), required=False)
@click.option(
    "--start", "-s",
    type=float,
    help="Manual start time in seconds (default: auto-detect)"
)
@click.option(
    "--end", "-e",
    type=float,
    help="Manual end time in seconds (default: auto-detect)"
)
@click.option(
    "--silence-threshold",
    type=float,
    default=-40.0,
    help="Silence threshold in dB (default: -40)"
)
@click.option(
    "--keep-start",
    type=float,
    default=0.5,
    help="Silence to keep at start in seconds (default: 0.5)"
)
@click.option(
    "--keep-end",
    type=float,
    default=2.0,
    help="Silence to keep at end in seconds (default: 2.0)"
)
@click.option(
    "--analyze-only", "-a",
    is_flag=True,
    help="Only analyze and show recommended trim points"
)
@click.option(
    "--verbose", "-v",
    is_flag=True,
    help="Verbose output"
)
def trim(
    audio_file: str,
    output_file: str | None,
    start: float | None,
    end: float | None,
    silence_threshold: float,
    keep_start: float,
    keep_end: float,
    analyze_only: bool,
    verbose: bool,
):
    """
    Trim silence from audio files or apply manual trimming.

    Auto-detects silence at the start and end of audio files and trims
    while keeping a specified amount of silence. Requires FFmpeg.

    \b
    Examples:
        # Auto-detect and trim silence
        audio-tool trim input.mp3 output.mp3

        # Analyze only (show recommended trim points)
        audio-tool trim -a input.mp3

        # Manual trim from 5s to 120s
        audio-tool trim -s 5 -e 120 input.mp3 output.mp3

        # Custom silence threshold
        audio-tool trim --silence-threshold -50 input.mp3 output.mp3

        # Keep more silence at end (for fade-out)
        audio-tool trim --keep-end 3.0 input.mp3 output.mp3
    """
    import logging
    import sys

    # Setup logging
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(message)s",
        stream=sys.stderr,
    )

    try:
        if analyze_only:
            # Just analyze and show recommendations
            click.echo(f"Analyzing: {audio_file}", err=True)
            start_time, end_time = detect_trim_points(
                audio_file,
                silence_thresh_db=silence_threshold,
                wanted_silence_start=keep_start,
                wanted_silence_end=keep_end,
            )

            from audio_tool.analyzer import analyze_audio_quality
            result = analyze_audio_quality(audio_file, silence_thresh_db=silence_threshold)

            click.echo("")
            click.echo(f"File: {audio_file}")
            click.echo(f"Duration: {result.duration_sec:.2f} sec")
            click.echo("")
            click.echo("Detected silence:")
            click.echo(f"  Start: {result.start_silence_sec:.2f} sec")
            click.echo(f"  End: {result.end_silence_sec:.2f} sec")
            click.echo("")
            click.echo("Recommended trim points:")
            click.echo(f"  Start: {start_time:.2f} sec")
            click.echo(f"  End: {end_time:.2f} sec")
            click.echo(f"  New duration: {end_time - start_time:.2f} sec")
            click.echo(f"  Would remove: {result.duration_sec - (end_time - start_time):.2f} sec")

        else:
            if not output_file:
                raise click.UsageError("OUTPUT_FILE is required (or use --analyze-only)")

            click.echo(f"Trimming: {audio_file}", err=True)

            # Perform trimming
            output_result = trim_audio(
                input_path=audio_file,
                output_path=output_file,
                start_time=start,
                end_time=end,
                silence_thresh_db=silence_threshold,
                auto_detect=(start is None or end is None),
                wanted_silence_start=keep_start,
                wanted_silence_end=keep_end,
            )

            click.echo("", err=True)
            click.echo(f"Output saved to: {output_file}", err=True)
            click.echo(f"Duration: {output_result.duration_sec:.2f} sec", err=True)

    except FileNotFoundError as e:
        raise click.ClickException(str(e))
    except ValueError as e:
        raise click.ClickException(str(e))
    except Exception as e:
        raise click.ClickException(f"Trimming failed: {e}")
