# Audio-Tool Development Notes

## Project Overview
CLI tool for audio processing, starting with transcription and speaker diarization.

## Structure
```
audio-tool/
├── pyproject.toml              # uv package config
├── .python-version             # Python 3.13
├── src/audio_tool/
│   ├── __init__.py             # Version: 0.1.0
│   ├── cli.py                  # Click group entry point
│   ├── config.py               # HuggingFace token handling
│   ├── transcriber.py          # Transcription engine (Whisper + Pyannote)
│   ├── normalization.py        # Audio normalization (FFmpeg filters)
│   ├── analyzer.py             # Audio quality analysis (FFmpeg)
│   ├── trimmer.py              # Audio trimming (FFmpeg)
│   └── commands/
│       ├── __init__.py
│       ├── analyze.py          # analyze subcommand
│       ├── audio2json.py       # audio2json subcommand
│       ├── normalize.py        # normalize subcommand
│       └── trim.py             # trim subcommand
```

## Commands
- `audio-tool analyze <file>` - Analyze audio quality (silence, LUFS, clipping, phase)
- `audio-tool audio2json <file>` - Transcribe audio to JSON with speaker diarization
- `audio-tool normalize <input> <output>` - Normalize audio loudness (requires FFmpeg)
- `audio-tool trim <input> <output>` - Trim silence from audio files (requires FFmpeg)

## Key Dependencies
- **Whisper backends**: whisper-timestamped (CPU), mlx-whisper (Mac)
- **Diarization**: pyannote-audio
- **CLI**: click
- **speechbrain**: Custom fork from OPpuolitaival for pyannote-audio 4.x compatibility

## Development
```bash
# Install/sync dependencies
uv sync

# Run CLI
uv run audio-tool --help
uv run audio-tool audio2json --help

# Test transcription
uv run audio-tool audio2json -l en recording.mp3

# MLX backend (Mac, fastest)
uv run audio-tool audio2json -w mlx-turbo recording.mp3
```

## HuggingFace Token
Required for pyannote models. Sources checked in order:
1. `--huggingface-token` CLI option
2. `HUGGINGFACE_TOKEN` env var
3. `HF_TOKEN` env var
4. `~/.huggingface/token` file

## Adding New Commands
1. Create `src/audio_tool/commands/newcmd.py` with `@click.command()` function
2. Import and export in `src/audio_tool/commands/__init__.py`
3. Add `main.add_command(newcmd)` in `src/audio_tool/cli.py`
