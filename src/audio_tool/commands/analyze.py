"""analyze command - analyze audio quality metrics."""

from pathlib import Path

import click

from audio_tool.analyzer import (
    analyze_audio_quality,
    check_audio_quality,
    format_result_json,
    format_result_text,
)


@click.command()
@click.argument("audio_file", type=click.Path(exists=True))
@click.option(
    "--output", "-o", type=click.Path(), help="Output file path (default: stdout)"
)
@click.option(
    "--format",
    "-f",
    "output_format",
    type=click.Choice(["text", "txt", "json"]),
    default="text",
    help="Output format (default: text)",
)
@click.option(
    "--silence-threshold",
    type=float,
    default=-40.0,
    help="Silence threshold in dB (default: -40)",
)
@click.option(
    "--min-silence-length",
    type=float,
    default=0.1,
    help="Minimum silence length in seconds (default: 0.1)",
)
@click.option(
    "--include-waveform",
    is_flag=True,
    help="Include waveform timeline data (increases output size)",
)
@click.option(
    "--lufs-min",
    type=float,
    default=-18.5,
    help="Minimum acceptable LUFS (default: -18.5)",
)
@click.option(
    "--lufs-max",
    type=float,
    default=-17.5,
    help="Maximum acceptable LUFS (default: -17.5)",
)
def analyze(
    audio_file: str,
    output: str | None,
    output_format: str,
    silence_threshold: float,
    min_silence_length: float,
    include_waveform: bool,
    lufs_min: float,
    lufs_max: float,
):
    """
    Analyze audio file quality metrics.

    Analyzes silence, loudness (LUFS), clipping, and phase correlation
    in a single FFmpeg pass. Requires FFmpeg to be installed.

    \b
    Output formats:
      text/txt  Human-readable text report
      json      Machine-readable JSON with full metrics

    \b
    Examples:
        # Basic analysis to stdout
        audio-tool analyze recording.mp3

        # Save as JSON
        audio-tool analyze -f json -o report.json recording.mp3

        # Save as text file
        audio-tool analyze -f txt -o report.txt recording.mp3

        # Custom thresholds
        audio-tool analyze --lufs-min -20 --lufs-max -16 recording.mp3
    """
    # Normalize format
    if output_format == "txt":
        output_format = "text"

    click.echo(f"Analyzing: {audio_file}", err=True)

    try:
        # Run analysis
        result = analyze_audio_quality(
            audio_file,
            silence_thresh_db=silence_threshold,
            min_silence_len_sec=min_silence_length,
            include_waveform=include_waveform,
        )

        # Run quality checks
        checks = check_audio_quality(
            result,
            lufs_min=lufs_min,
            lufs_max=lufs_max,
        )

        # Format output
        if output_format == "json":
            formatted = format_result_json(result, checks)
        else:
            formatted = format_result_text(result, checks)

        # Write output
        if output:
            output_path = Path(output)
            output_path.write_text(formatted, encoding="utf-8")
            click.echo(f"Results saved to: {output_path}", err=True)
        else:
            print(formatted)

        # Summary to stderr
        click.echo("", err=True)
        if checks:
            errors = [c for c in checks if c.severity == "error"]
            warnings = [c for c in checks if c.severity == "warning"]
            if errors:
                click.echo(
                    f"Found {len(errors)} error(s), {len(warnings)} warning(s)",
                    err=True,
                )
            else:
                click.echo(f"Found {len(warnings)} warning(s)", err=True)
        else:
            click.echo("No issues detected", err=True)

    except FileNotFoundError as e:
        raise click.ClickException(str(e))
    except ValueError as e:
        raise click.ClickException(str(e))
    except Exception as e:
        raise click.ClickException(f"Analysis failed: {e}")
