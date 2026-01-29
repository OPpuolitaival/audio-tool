# audio-tool

Audio processing CLI with transcription and speaker diarization.

## Installation

Requires Python 3.13+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone <repo-url>
cd audio-tool
uv sync
```

## Setup

A HuggingFace token is required for pyannote speaker diarization models. Get one at https://huggingface.co/settings/tokens and accept the model licenses:
- https://huggingface.co/pyannote/speaker-diarization-3.1
- https://huggingface.co/pyannote/segmentation-3.0

Set the token via one of:
```bash
# Environment variable
export HUGGINGFACE_TOKEN=hf_xxx

# Or save to file
echo "hf_xxx" > ~/.huggingface/token
```

## Usage

### audio2json

Transcribe audio files to JSON with speaker diarization.

```bash
# Basic usage (Finnish, default)
uv run audio-tool audio2json recording.mp3

# English transcription
uv run audio-tool audio2json -l en recording.mp3

# Save to file
uv run audio-tool audio2json -o output.json recording.mp3

# Verbose output
uv run audio-tool audio2json -v recording.mp3
```

### Whisper Backends

```bash
# Default: whisper-timestamped (CPU, works everywhere)
uv run audio-tool audio2json recording.mp3

# MLX turbo (Mac only, fastest)
uv run audio-tool audio2json -w mlx-turbo recording.mp3

# MLX large-v3 (Mac only, highest quality)
uv run audio-tool audio2json -w mlx-large-v3 recording.mp3
```

For MLX backends on Mac, install mlx-whisper:
```bash
uv add mlx-whisper
```

### Diarization Models

```bash
# Default: pyannote 3.1 (faster loading)
uv run audio-tool audio2json recording.mp3

# Community model (faster inference)
uv run audio-tool audio2json -d pyannote/speaker-diarization-community-1 recording.mp3
```

### Fastest Setup (Mac)

```bash
uv run audio-tool audio2json -w mlx-turbo -d pyannote/speaker-diarization-community-1 recording.mp3
```

## Output Format

JSON output includes:
- `metadata` - Processing info (version, timestamps, models used)
- `segments` - Speech and non-speech segments with timestamps
- `speaker_embeddings` - Voice embeddings per speaker
- `statistics` - Summary (speaker count, speech duration)

Example segment:
```json
{
  "start": 0.5,
  "end": 3.2,
  "speaker": "SPEAKER_00",
  "type": "speech",
  "text": "Hello, welcome to the show.",
  "words": [
    {"word": "Hello,", "start": 0.5, "end": 0.8, "confidence": 0.95}
  ],
  "avg_confidence": 0.92
}
```

## Help

```bash
uv run audio-tool --help
uv run audio-tool audio2json --help
```
