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

### Pipeline

The transcription pipeline:

1. pyannote speaker diarization (`speaker-diarization-3.1`) finds speech segments per speaker
2. Whisper transcribes each speech segment (MLX large-v3 on Apple Silicon, whisper-timestamped elsewhere)
3. Gap recovery: Silero VAD looks for speech in the stretches diarization did not call speech
   (speech over music, foreign-language speech) and Whisper transcribes it one 30 s window at a
   time, keeping only segments Whisper is confident about. Recovered segments carry `"recovered": true`
4. Decoder loops are collapsed (a phrase repeated 3+ times in a row is kept once); segments that
   still loop are dropped
5. PANNs classifies the non-speech gaps (music / silence / speech / unknown), and speaker
   embeddings are extracted

Measured against hand-corrected transcripts, gap recovery and loop collapsing brought WER
from 10.1 % to 7.9 %.

### Whisper backend and model

```bash
# Default: auto-selects MLX large-v3 on Apple Silicon, whisper-timestamped elsewhere
uv run audio-tool audio2json recording.mp3

# Force the CPU backend
uv run audio-tool audio2json -b whisper-timestamped recording.mp3

# Faster but less accurate MLX model (aliases: large-v3, turbo, medium, small, base, tiny,
# or a full HuggingFace repo path)
uv run audio-tool audio2json -w turbo recording.mp3

# Skip speaker embeddings (diarization still runs)
uv run audio-tool audio2json -s recording.mp3

# Auto-detect the language
uv run audio-tool audio2json -l "" recording.mp3
```

`mlx-whisper` is installed automatically on Apple Silicon Macs.

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

### trim

Trim silence from audio files. Requires FFmpeg to be installed.

```bash
# Auto-detect and trim silence
uv run audio-tool trim input.mp3 output.mp3

# Analyze only (show recommended trim points)
uv run audio-tool trim -a input.mp3

# Manual trim from 5s to 120s
uv run audio-tool trim -s 5 -e 120 input.mp3 output.mp3

# Custom silence threshold (-50 dB)
uv run audio-tool trim --silence-threshold -50 input.mp3 output.mp3

# Keep more silence at end (for fade-out)
uv run audio-tool trim --keep-end 3.0 input.mp3 output.mp3
```

**Features:**
- Auto-detects silence at start and end
- Keeps configurable amount of silence (default: 0.5s start, 2.0s end)
- Manual start/end time override
- Analyze-only mode to preview trim points

## Output Format (audio2json)

JSON output includes:
- `metadata` - Processing info (version, timestamps, models used)
- `segments` - Speech and non-speech segments with timestamps (non-speech segments carry a PANNs `classification`)
- `speaker_embeddings` - Voice embeddings per speaker
- `statistics` - Summary (speaker count, speech duration, recovered gap speech)

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
uv run audio-tool trim --help
```
