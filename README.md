# audio-tool

Audio processing CLI with transcription, speaker diarization, and loudness normalization.

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

### analyze

Analyze audio quality metrics. Requires FFmpeg to be installed.

```bash
# Basic analysis (text output to stdout)
uv run audio-tool analyze recording.mp3

# JSON output
uv run audio-tool analyze -f json recording.mp3

# Save to file
uv run audio-tool analyze -f json -o report.json recording.mp3
uv run audio-tool analyze -f txt -o report.txt recording.mp3

# Custom LUFS thresholds
uv run audio-tool analyze --lufs-min -20 --lufs-max -16 recording.mp3
```

**Metrics analyzed:**
- Silence detection (start, end, middle gaps)
- Loudness (integrated LUFS, loudness range, true peak)
- Clipping detection (max volume, 0dB samples)
- Phase correlation (stereo files)

**Output formats:**
- `text` / `txt` - Human-readable report
- `json` - Full metrics for programmatic use

### normalize

Normalize audio loudness using FFmpeg filters. Requires FFmpeg to be installed.

```bash
# Basic usage (speechnorm method, best for speech)
uv run audio-tool normalize input.mp3 output.mp3

# Dynamic normalization (best for mixed content with music)
uv run audio-tool normalize -m dynaudnorm input.mp3 output.mp3

# Custom loudness target (-16 LUFS)
uv run audio-tool normalize --lufs -16 input.wav output.wav

# Gentler speech normalization
uv run audio-tool normalize -e 20 input.mp3 output.mp3
```

**Methods:**
- `speechnorm` - Best for speech-heavy content (podcasts, radio programs)
- `dynaudnorm` - Best for mixed content with speech and music

**Key options:**
- `--lufs` / `-I` - Target loudness in LUFS (default: -18)
- `--true-peak` / `-TP` - Maximum true peak in dB (default: -1)
- `--method` / `-m` - Normalization method

## Output Format (audio2json)

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
uv run audio-tool analyze --help
uv run audio-tool audio2json --help
uv run audio-tool normalize --help
```
