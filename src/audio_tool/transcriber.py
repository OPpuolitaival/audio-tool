"""
Transcription engine for audio-tool.

Uses Whisper for speech-to-text and Pyannote for speaker diarization.

Supported configurations:
- Mac (Apple Silicon): MLX Large-v3 + Diarization 3.1 (best quality, 3.2x realtime)
- Linux/CPU: Whisper-timestamped + Diarization 3.1 (7.13% WER, 1.8x realtime)
"""

# CRITICAL: Set environment variables BEFORE any imports
import os

os.environ.setdefault("TORCH_FORCE_WEIGHTS_ONLY_LOAD", "0")
# Suppress macOS MallocStackLogging noise from MLX/Metal/MPS libraries
os.environ.setdefault("MallocStackLogging", "0")

import gc
import logging
import time
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import librosa
import numpy as np
import torch
import whisper_timestamped
from faster_whisper.vad import VadOptions, get_speech_timestamps
from panns_inference import AudioTagging
from pyannote.audio import Pipeline as DiarizationPipeline
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding

from audio_tool.postprocess import (
    GAP_MIN_DURATION_SEC,
    GAP_VAD_MAX_CHUNK_SEC,
    REPEAT_DROP_COUNT,
    collapse_repeated_phrases,
    collapse_repeated_text,
    gaps_between,
    group_speech_pieces,
    join_words,
    max_consecutive_repeats,
    whisper_segment_is_reliable,
)

logger = logging.getLogger(__name__)


def load_audio_mono_sanitized(
    audio_path: Path,
    sample_rate: int,
    log: Optional[logging.Logger] = None,
) -> tuple[np.ndarray, int]:
    """Load audio as mono at ``sample_rate``, tolerating NaN/Inf samples.

    ``librosa.load(mono=True)`` runs ``to_mono`` -> ``util.valid_audio`` *inside*
    the load and raises ``ParameterError("Audio buffer is not finite
    everywhere")`` before returning, so sanitizing after the load never runs
    (some recorders emit invalid float samples — e.g. Eila_luotu.wav). We load
    raw (``mono=False, sr=None`` — no downmix, no resample, no validation),
    replace non-finite samples with silence, then downmix and resample ourselves.
    """
    log = log or logger
    audio, sr_native = librosa.load(str(audio_path), sr=None, mono=False)

    if not np.isfinite(audio).all():
        count = int((~np.isfinite(audio)).sum())
        log.warning(
            f"Audio contains {count} non-finite samples, replacing with silence"
        )
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)

    if audio.ndim > 1:
        audio = librosa.to_mono(audio)

    if sample_rate is not None and sr_native != sample_rate:
        audio = librosa.resample(audio, orig_sr=sr_native, target_sr=sample_rate)
        sr_native = sample_rate

    return audio, int(sr_native)


# Check if MLX is available (Mac only)
MLX_AVAILABLE = False
try:
    import mlx.core as mx
    import mlx_whisper

    MLX_AVAILABLE = True
except ImportError:
    mlx_whisper = None
    mx = None


# Monkey-patch torch.load to ALWAYS disable weights_only for pyannote models
_original_torch_load = torch.load


def _torch_load_without_weights_only(*args, **kwargs):
    """Wrapper for torch.load that forces weights_only=False for compatibility with pyannote models."""
    kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)


torch.load = _torch_load_without_weights_only


# AudioSet labels for PANNs classification
MUSIC_LABELS = {
    "Music",
    "Musical instrument",
    "Singing",
    "Song",
    "Guitar",
    "Piano",
    "Drum",
    "Bass",
    "Synthesizer",
    "Keyboard (musical)",
    "Orchestra",
    "Band",
    "Pop music",
    "Rock music",
    "Hip hop music",
    "Jazz",
    "Classical music",
    "Electronic music",
    "Techno",
    "House music",
    "Disco",
    "Funk",
    "Soul music",
    "Rhythm and blues",
    "Country music",
    "Folk music",
    "Reggae",
    "Ska",
    "Flamenco",
    "Blues",
    "Punk rock",
    "Heavy metal",
    "Grunge",
    "Progressive rock",
    "Psychedelic rock",
    "Ambient music",
    "Trance music",
    "Drum and bass",
    "Dubstep",
    "Electronica",
    "Dance music",
    "Soundtrack music",
    "Theme music",
    "Jingle (music)",
    "Background music",
    "Violin, fiddle",
    "Cello",
    "Harp",
    "Trumpet",
    "Trombone",
    "Saxophone",
    "Flute",
    "Clarinet",
    "Accordion",
    "Harmonica",
    "Banjo",
    "Mandolin",
    "Ukulele",
    "Organ",
    "Electric guitar",
    "Acoustic guitar",
    "Bass guitar",
    "Plucked string instrument",
    "Choir",
    "Chant",
    "Mantra",
}

SILENCE_LABELS = {"Silence", "White noise", "Pink noise"}

SPEECH_LABELS = {
    "Speech",
    "Conversation",
    "Narration, monologue",
    "Child speech, kid speaking",
}


def classify_panns_predictions(predictions: list[dict]) -> tuple[str, float]:
    """
    Map PANNs predictions to category (music/silence/speech/unknown).

    Returns:
        Tuple of (classification, confidence)
    """
    if not predictions:
        return "unknown", 0.0

    # Calculate scores for each category
    music_score = 0.0
    silence_score = 0.0
    speech_score = 0.0

    for pred in predictions[:10]:
        label = pred["label"]
        conf = pred["confidence"]

        if label in MUSIC_LABELS:
            music_score += conf
        if label in SILENCE_LABELS:
            silence_score += conf
        if label in SPEECH_LABELS:
            speech_score += conf

    top_conf = predictions[0]["confidence"]

    # Primary classification based on accumulated scores
    if music_score > 0.3:
        return "music", music_score
    elif silence_score > 0.3:
        return "silence", silence_score
    elif speech_score > 0.5:
        return "speech", speech_score
    elif top_conf < 0.1:
        return "silence", 1.0 - top_conf
    else:
        return "unknown", top_conf


class WhisperBackend(str, Enum):
    """Available Whisper backends."""

    WHISPER_TIMESTAMPED = "whisper-timestamped"  # CPU/Linux compatible
    MLX_LARGE_V3 = "mlx-large-v3"  # Mac MLX, best quality (6.30% WER)


# Diarization model - always use 3.1 (best accuracy)
DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"

# Model mappings
WHISPER_MODEL_MAP = {
    WhisperBackend.WHISPER_TIMESTAMPED: "large-v3",
    WhisperBackend.MLX_LARGE_V3: "mlx-community/whisper-large-v3-mlx",
}

# Convenience aliases for the ``--whisper-model`` CLI flag (MLX backend).
# Short alias → HuggingFace repo path understood by mlx_whisper.
MLX_MODEL_ALIASES = {
    "large-v3": "mlx-community/whisper-large-v3-mlx",
    "turbo": "mlx-community/whisper-large-v3-turbo",
    "medium": "mlx-community/whisper-medium-mlx",
    "small": "mlx-community/whisper-small-mlx",
    "base": "mlx-community/whisper-base-mlx",
    "tiny": "mlx-community/whisper-tiny-mlx",
}


def resolve_mlx_model_path(name_or_path: Optional[str]) -> Optional[str]:
    """Map a short alias (``'turbo'``, ``'large-v3'`` …) to a full MLX repo path.

    - ``None`` or empty string → ``None`` (caller should use the default).
    - A value containing ``'/'`` is assumed to already be an HF repo path and is returned unchanged.
    - An unknown short name raises ``ValueError``.
    """
    if not name_or_path:
        return None
    if "/" in name_or_path:
        return name_or_path
    try:
        return MLX_MODEL_ALIASES[name_or_path]
    except KeyError:
        raise ValueError(
            f"Unknown MLX Whisper model alias {name_or_path!r}. "
            f"Known aliases: {', '.join(MLX_MODEL_ALIASES)}. "
            'Or pass a full HuggingFace repo path containing "/".'
        )


def get_device():
    """Determine the best device for processing."""
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"


def check_repeated_words(text: str) -> int:
    """Largest back-to-back repeat count of any 1..8-word phrase."""
    if not text or not isinstance(text, str):
        return 0
    return max_consecutive_repeats(text.split())


class Transcriber:
    """
    Transcription engine using Whisper and Pyannote.

    Automatically selects the best backend:
    - Mac (Apple Silicon): MLX Large-v3 (6.30% WER, 3.2x realtime)
    - Linux/CPU: Whisper-timestamped (7.13% WER, 1.8x realtime)

    Always uses Pyannote speaker-diarization-3.1 for best accuracy.
    """

    VERSION = "2.3"

    def __init__(
        self,
        huggingface_token: str,
        model_name: str = "large-v3",
        language: Optional[str] = None,
        whisper_backend: WhisperBackend | None = None,
        logger: Optional[logging.Logger] = None,
        mlx_model_path_override: Optional[str] = None,
    ):
        self.model_name = model_name
        self.language = language
        self.sample_rate = 16000
        self.huggingface_token = huggingface_token
        self.logger = logger or logging.getLogger(__name__)
        self._mlx_model_path_override = mlx_model_path_override

        if not self.huggingface_token:
            raise ValueError("HUGGINGFACE_TOKEN is required for pyannote models")

        # Auto-select best backend if not specified
        if whisper_backend is None:
            if MLX_AVAILABLE:
                whisper_backend = WhisperBackend.MLX_LARGE_V3
                self.logger.info(
                    "Auto-selected MLX Large-v3 backend (best quality on Mac)"
                )
            else:
                whisper_backend = WhisperBackend.WHISPER_TIMESTAMPED
                self.logger.info(
                    "Auto-selected whisper-timestamped backend (Linux/CPU)"
                )

        self.whisper_backend = whisper_backend

        # Validate MLX backend availability
        if whisper_backend == WhisperBackend.MLX_LARGE_V3:
            if not MLX_AVAILABLE:
                raise ValueError(
                    f"MLX backend {whisper_backend.value} requested but mlx-whisper is not installed. "
                    "Install with: pip install mlx-whisper (Mac only)"
                )

        self.device = get_device()
        self.logger.info(f"Using device: {self.device}")
        self.logger.info(f"Whisper backend: {self.whisper_backend.value}")
        self.logger.info(f"Diarization model: {DIARIZATION_MODEL}")

        self.diarization_pipeline = None
        self.speaker_embedding_model = None
        self.whisper_model = None
        self._mlx_model_path = None  # For MLX model caching
        self.panns_model = None  # PANNs audio tagging model
        self._panns_labels = None

        self._load_models()

    def _load_models(self):
        """Load all required models."""
        self.logger.info(
            f"Loading models (whisper={self.whisper_backend.value}, diarization={DIARIZATION_MODEL})..."
        )

        # Load PyAnnote diarization (always use 3.1 for best accuracy)
        self.diarization_pipeline = DiarizationPipeline.from_pretrained(
            DIARIZATION_MODEL,
            token=self.huggingface_token,
        )

        if self.device and self.device != "cpu":
            self.diarization_pipeline.to(torch.device(self.device))
            self.logger.info(f"Moved diarization pipeline to {self.device}")

        # Load speaker embedding model
        self.speaker_embedding_model = PretrainedSpeakerEmbedding(
            "speechbrain/spkrec-ecapa-voxceleb",
            device=self.device
            if self.device != "mps"
            else "cpu",  # MPS not always supported
            token=self.huggingface_token,
        )
        self.logger.info("Loaded speaker embedding model")

        # Load Whisper based on backend
        if self.whisper_backend == WhisperBackend.WHISPER_TIMESTAMPED:
            # whisper-timestamped backend (CPU/Linux)
            self.whisper_model = whisper_timestamped.load_model(
                self.model_name, device="cpu"
            )
            self.logger.info(
                f"Loaded Whisper model (whisper-timestamped): {self.model_name}"
            )
        else:
            # MLX backend (Mac) - models are loaded on first use
            self._mlx_model_path = (
                self._mlx_model_path_override or WHISPER_MODEL_MAP[self.whisper_backend]
            )
            self.logger.info(f"Will use MLX Whisper model: {self._mlx_model_path}")
            # MLX keeps freed buffers in a cache that otherwise grows without bound (15 GB after six
            # files in one process, then swapping). Bound it here and empty it after every file
            # (see transcribe()).
            mx.set_cache_limit(2 << 30)

    def unload_models(self) -> None:
        """Drop every model and give the memory back.

        Useful when other large models need the memory, or when one process
        transcribes many files: memory creeps up across files (measured
        1.3 GB to 15 GB over six), so reloading every so often beats growing
        into swap.

        MLX Whisper's weights are the subtle one: they do not hang off this
        object at all but off ``mlx_whisper``'s own module-level holder, which
        never evicts, so they have to be cleared there.
        """
        self.diarization_pipeline = None
        self.speaker_embedding_model = None
        self.whisper_model = None
        self.panns_model = None
        self._panns_labels = None
        if MLX_AVAILABLE:
            try:
                from mlx_whisper.transcribe import ModelHolder

                ModelHolder.model = None
                ModelHolder.model_path = None
            except ImportError:
                pass
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()
        if MLX_AVAILABLE:
            mx.clear_cache()
        self.logger.info("Unloaded transcription models")

    def ensure_models(self) -> None:
        """Load the models again after ``unload_models``."""
        if self.diarization_pipeline is None:
            self._load_models()

    def _load_panns_model(self):
        """Lazy-load PANNs model for non-speech classification."""
        if self.panns_model is not None:
            return

        self.logger.info("Loading PANNs model for non-speech classification...")
        # PANNs may have issues with MPS, use CPU
        device = "cpu"
        self.panns_model = AudioTagging(checkpoint_path=None, device=device)

        # Get AudioSet labels
        try:
            from panns_inference import labels

            self._panns_labels = labels
        except (ImportError, AttributeError):
            self._panns_labels = ["Speech", "Music", "Singing", "Silence"]

        self.logger.info(f"PANNs model loaded on {device}")

    def _classify_with_panns(
        self, audio: np.ndarray, sr: int, top_k: int = 10
    ) -> list[dict]:
        """Classify audio segment using PANNs model."""
        self._load_panns_model()

        # Resample to 32kHz (PANNs requirement)
        if sr != 32000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=32000)

        # PANNs expects audio in shape (batch, samples)
        audio_input = audio[np.newaxis, :]
        clipwise_output, _ = self.panns_model.inference(audio_input)

        # Get top predictions
        probs = clipwise_output[0]
        top_indices = np.argsort(probs)[::-1][:top_k]

        predictions = []
        assert self._panns_labels is not None
        for idx in top_indices:
            if idx < len(self._panns_labels):
                predictions.append(
                    {
                        "label": self._panns_labels[idx],
                        "confidence": float(probs[idx]),
                    }
                )

        return predictions

    def _transcribe_with_whisper_timestamped(
        self, audio: np.ndarray, segment_start: float
    ) -> dict:
        """Transcribe using whisper-timestamped backend."""
        result = whisper_timestamped.transcribe(
            self.whisper_model,
            audio,
            language=self.language,
            compute_word_confidence=True,
            include_punctuation_in_confidence=True,
        )

        # Normalize output format
        text = result.get("text", "").strip()
        language = result.get("language", self.language)

        words = []
        if "segments" in result:
            for seg in result["segments"]:
                if "words" in seg:
                    for word_info in seg["words"]:
                        words.append(
                            {
                                "word": word_info["text"],
                                "start": word_info["start"] + segment_start,
                                "end": word_info["end"] + segment_start,
                                "confidence": word_info.get("confidence", 0.0),
                            }
                        )

        return {"text": text, "language": language, "words": words}

    def _transcribe_with_mlx(
        self, audio_input: "str | np.ndarray", segment_start: float
    ) -> dict:
        """Transcribe using MLX Whisper backend. Accepts a file path or a 16 kHz numpy array."""
        if not MLX_AVAILABLE:
            raise RuntimeError("MLX Whisper not available")

        result = mlx_whisper.transcribe(
            audio_input,
            path_or_hf_repo=self._mlx_model_path,
            language=self.language,
            word_timestamps=True,
        )

        # Normalize output format
        text = result.get("text", "").strip()
        language = result.get("language", self.language)

        words = []
        if "segments" in result:
            for seg in result["segments"]:
                if "words" in seg:
                    for word_info in seg["words"]:
                        # MLX uses 'word' or 'text' for the word text
                        word_text = word_info.get("word", word_info.get("text", ""))
                        # MLX uses 'probability' for confidence
                        confidence = word_info.get(
                            "probability", word_info.get("confidence", 0.0)
                        )
                        words.append(
                            {
                                "word": word_text,
                                "start": word_info["start"] + segment_start,
                                "end": word_info["end"] + segment_start,
                                "confidence": confidence,
                            }
                        )

        return {"text": text, "language": language, "words": words}

    def _whisper_raw(self, audio: np.ndarray) -> dict[str, Any]:
        """Whisper's own result dict: segments with confidence stats and word timings."""
        if self.whisper_backend == WhisperBackend.MLX_LARGE_V3:
            if not MLX_AVAILABLE:
                raise RuntimeError("MLX Whisper not available")
            return mlx_whisper.transcribe(
                audio,
                path_or_hf_repo=self._mlx_model_path,
                language=self.language,
                word_timestamps=True,
            )
        return whisper_timestamped.transcribe(
            self.whisper_model,
            audio,
            language=self.language,
            compute_word_confidence=True,
            include_punctuation_in_confidence=True,
        )

    @staticmethod
    def _neighbour_speaker(
        speech_segments: list[dict[str, Any]], gap_start: float, gap_end: float
    ) -> str:
        before = [s for s in speech_segments if s["end"] <= gap_start + 0.01]
        if before:
            return before[-1].get("speaker") or "SPEAKER_00"
        after = [s for s in speech_segments if s["start"] >= gap_end - 0.01]
        return (after[0].get("speaker") if after else None) or "SPEAKER_00"

    def recover_gap_speech(
        self, audio: np.ndarray, speech_segments: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Transcribe speech that diarization missed.

        pyannote labels speech over music, foreign-language speech and some read
        speech "non-speech", and those passages were simply absent from the
        transcript (29 % of the word errors measured on hand-corrected transcripts). Silero
        VAD finds the speech inside each gap, it is transcribed one Whisper window at
        a time, and only segments Whisper itself is confident about are kept.
        Recovered segments inherit the neighbouring speaker label.
        """
        sr = self.sample_rate
        speech = sorted(
            (s for s in speech_segments if s.get("type") == "speech"),
            key=lambda s: s["start"],
        )
        recovered: list[dict[str, Any]] = []
        vad_options = VadOptions()
        for gap_start, gap_end in gaps_between(
            speech, len(audio) / sr, GAP_MIN_DURATION_SEC
        ):
            gap_audio = audio[int(gap_start * sr) : int(gap_end * sr)]
            timestamps = get_speech_timestamps(gap_audio, vad_options, sampling_rate=sr)
            if not timestamps:
                continue
            speaker = self._neighbour_speaker(speech, gap_start, gap_end)
            for pieces in group_speech_pieces(timestamps, sr, GAP_VAD_MAX_CHUNK_SEC):
                chunk = np.concatenate(
                    [gap_audio[piece["start"] : piece["end"]] for piece in pieces]
                )
                if len(chunk) < sr * 0.5:
                    continue
                try:
                    result = self._whisper_raw(np.ascontiguousarray(chunk))
                except Exception as e:
                    self.logger.warning(
                        f"Gap transcription failed at {gap_start:.1f}s: {e}"
                    )
                    continue

                def to_original(t: float) -> float:
                    # chunk time -> file time; the chunk is the concatenation of the VAD pieces
                    acc = 0.0
                    for piece in pieces:
                        dur = (piece["end"] - piece["start"]) / sr
                        if t <= acc + dur:
                            return gap_start + piece["start"] / sr + (t - acc)
                        acc += dur
                    return gap_start + pieces[-1]["end"] / sr

                kept = [
                    seg
                    for seg in result.get("segments", [])
                    if whisper_segment_is_reliable(seg)
                ]
                if not kept:
                    continue
                words = []
                for seg in kept:
                    for w in seg.get("words", []):
                        words.append(
                            {
                                "word": w.get("word", w.get("text", "")),
                                "start": to_original(w["start"]),
                                "end": to_original(w["end"]),
                                "confidence": w.get(
                                    "probability", w.get("confidence", 0.0)
                                ),
                            }
                        )
                recovered.append(
                    {
                        "start": to_original(kept[0]["start"]),
                        "end": to_original(kept[-1]["end"]),
                        "speaker": speaker,
                        "type": "speech",
                        "confidence": 1.0,
                        "text": " ".join(seg["text"].strip() for seg in kept),
                        "language": result.get("language", self.language),
                        "words": words,
                        "avg_confidence": sum(w["confidence"] for w in words)
                        / len(words)
                        if words
                        else 0.0,
                        "recovered": True,
                    }
                )
        if recovered:
            self.logger.info(
                f"Recovered {len(recovered)} speech segments ({sum(s['end'] - s['start'] for s in recovered):.1f}s) from diarization gaps"
            )
        return recovered

    def collapse_loops(self, segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Keep one copy of a phrase Whisper repeated three or more times in a row (a decoder loop)."""
        for segment in segments:
            if segment.get("type") != "speech":
                continue
            words = segment.get("words") or []
            if words:
                kept = collapse_repeated_phrases(words, key=lambda w: w.get("word", ""))
                if len(kept) < len(words):
                    segment["words"] = kept
                    segment["text"] = join_words(kept)
                    segment["avg_confidence"] = (
                        sum(w["confidence"] for w in kept) / len(kept) if kept else 0.0
                    )
            else:
                segment["text"] = collapse_repeated_text(segment.get("text", ""))
        return segments

    def detect_speech_segments(
        self,
        audio_path: Path,
        min_segment_duration: float = 2.0,
        merge_gap_threshold: float = 4.0,
        max_merge_duration: float = 300,
        audio: Optional[np.ndarray] = None,
    ) -> list[dict[str, Any]]:
        """Detect speech segments using PyAnnote diarization."""
        self.logger.info("Detecting speech segments...")
        start_time = time.time()

        if audio is None:
            audio, sr = load_audio_mono_sanitized(
                audio_path, self.sample_rate, self.logger
            )
        duration = len(audio) / self.sample_rate
        self.logger.info(f"Audio duration: {duration:.2f} seconds")

        # Configure chunked processing for long files
        if duration > 60 * 15:
            chunk_size = 60 * 10
            overlap = 5
        else:
            chunk_size = int(duration) + 1
            overlap = 0

        chunk_segments = []
        for chunk_start in range(0, int(duration), max(1, chunk_size - overlap)):
            chunk_end = min(chunk_start + chunk_size, duration)
            self.logger.info(f"Processing chunk {chunk_start}s - {chunk_end}s")

            chunk_start_sample = int(chunk_start * self.sample_rate)
            chunk_end_sample = int(chunk_end * self.sample_rate)
            chunk_audio = audio[chunk_start_sample:chunk_end_sample]

            # Feed pyannote the waveform in-memory (shape: channels, samples) to skip disk I/O
            waveform = (
                torch.from_numpy(np.ascontiguousarray(chunk_audio)).unsqueeze(0).float()
            )
            diar_input = {"waveform": waveform, "sample_rate": self.sample_rate}

            assert self.diarization_pipeline is not None
            chunk_diarization = self.diarization_pipeline(diar_input)

            # Handle both pyannote-audio 3.x and 4.x APIs
            if hasattr(chunk_diarization, "speaker_diarization"):
                # pyannote-audio 4.x returns DiarizeOutput with speaker_diarization attribute
                diarization_iter = chunk_diarization.speaker_diarization
            elif hasattr(chunk_diarization, "itertracks"):
                # pyannote-audio 3.x returns Annotation with itertracks method
                diarization_iter = (
                    (turn, speaker)
                    for turn, _, speaker in chunk_diarization.itertracks(
                        yield_label=True
                    )
                )
            else:
                self.logger.warning(
                    f"Unknown diarization output type: {type(chunk_diarization)}"
                )
                diarization_iter = []

            for turn, speaker in diarization_iter:
                if chunk_start > 0 and turn.start < overlap and overlap > 0:
                    continue

                adjusted_start = chunk_start + turn.start
                adjusted_end = chunk_start + turn.end

                chunk_segments.append(
                    {
                        "start": adjusted_start,
                        "end": adjusted_end,
                        "speaker": speaker,
                        "type": "speech",
                        "confidence": 1.0,
                    }
                )

        # Sort and merge segments
        chunk_segments.sort(key=lambda s: s["start"])

        if chunk_segments:
            merged_segments = [chunk_segments[0]]
            for segment in chunk_segments[1:]:
                prev = merged_segments[-1]
                gap = segment["start"] - prev["end"]
                merged_duration = segment["end"] - prev["start"]

                should_merge = (
                    segment["speaker"] == prev["speaker"]
                    and merged_duration <= max_merge_duration
                    and (
                        gap < merge_gap_threshold
                        or (merged_duration < min_segment_duration and gap < 2.0)
                    )
                )

                if should_merge:
                    prev["end"] = segment["end"]
                else:
                    merged_segments.append(segment)

            segments = [
                s
                for s in merged_segments
                if (s["end"] - s["start"]) >= min_segment_duration
            ]
        else:
            segments = []

        self.logger.info(
            f"Detected {len(segments)} speech segments in {time.time() - start_time:.2f}s"
        )
        return segments

    def transcribe_segments(
        self,
        audio_path: Path,
        segments: list[dict[str, Any]],
        audio: Optional[np.ndarray] = None,
    ) -> list[dict[str, Any]]:
        """Transcribe speech segments with word-level timestamps."""
        self.logger.info("Transcribing segments...")

        if audio is None:
            audio, sr = load_audio_mono_sanitized(
                audio_path, self.sample_rate, self.logger
            )

        # MLX backend requires file paths, whisper-timestamped uses audio arrays
        use_mlx = self.whisper_backend == WhisperBackend.MLX_LARGE_V3

        for i, segment in enumerate(segments):
            if segment["type"] != "speech":
                continue

            self.logger.info(f"Transcribing segment {i + 1}/{len(segments)}...")
            seg_start_time = time.time()

            start_sample = int(segment["start"] * self.sample_rate)
            end_sample = int(segment["end"] * self.sample_rate)
            segment_audio = audio[start_sample:end_sample]

            try:
                if use_mlx:
                    # MLX accepts numpy arrays directly — skip temp-file I/O
                    result = self._transcribe_with_mlx(
                        np.ascontiguousarray(segment_audio), segment["start"]
                    )
                else:
                    result = self._transcribe_with_whisper_timestamped(
                        segment_audio, segment["start"]
                    )

                segment["text"] = result["text"]
                segment["language"] = result["language"]
                segment["words"] = result["words"]
                segment["avg_confidence"] = (
                    sum(w["confidence"] for w in result["words"]) / len(result["words"])
                    if result["words"]
                    else 0.0
                )

                seg_duration = round(time.time() - seg_start_time, 1)
                self.logger.info(
                    f"  {seg_duration}s, words:{len(result['words'])}, conf:{segment['avg_confidence']:.2f} - {segment['text'][:50]}..."
                )

            except Exception as e:
                self.logger.warning(f"Transcription failed for segment {i}: {e}")
                segment["text"] = ""
                segment["words"] = []
                segment["avg_confidence"] = 0.0

        return segments

    def filter_bad_segments(
        self, segments: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Remove segments that still loop after collapse_loops (nothing but repetition)."""
        return [
            s
            for s in segments
            if check_repeated_words(s.get("text", "")) < REPEAT_DROP_COUNT
        ]

    def analyse_non_speech(
        self,
        audio_path: Path,
        transcribed_segments: list[dict[str, Any]],
        audio: Optional[np.ndarray] = None,
    ) -> list[dict[str, Any]]:
        """Analyze gaps between speech segments using PANNs classifier."""
        self.logger.info("Analyzing non-speech segments with PANNs...")

        sr = self.sample_rate
        if audio is None:
            audio, sr = load_audio_mono_sanitized(
                audio_path, self.sample_rate, self.logger
            )
        total_duration = len(audio) / self.sample_rate

        speech_segments = sorted(
            [s for s in transcribed_segments if s.get("type") == "speech"],
            key=lambda x: x["start"],
        )

        non_speech_segments = []
        min_non_speech_duration = 2.0
        current_time = 0.0

        for speech_segment in speech_segments:
            gap_start = current_time
            gap_end = speech_segment["start"]
            gap_duration = gap_end - gap_start

            if gap_duration >= min_non_speech_duration:
                non_speech_segments.append(
                    {
                        "start": gap_start,
                        "end": gap_end,
                        "type": "non-speech",
                        "duration": gap_duration,
                    }
                )

            current_time = speech_segment["end"]

        # Check final gap
        if current_time < total_duration:
            gap_duration = total_duration - current_time
            if gap_duration >= min_non_speech_duration:
                non_speech_segments.append(
                    {
                        "start": current_time,
                        "end": total_duration,
                        "type": "non-speech",
                        "duration": gap_duration,
                    }
                )

        self.logger.info(f"Found {len(non_speech_segments)} non-speech segments")

        # Classify each non-speech segment with PANNs
        for i, segment in enumerate(non_speech_segments):
            start_sample = int(segment["start"] * self.sample_rate)
            end_sample = int(segment["end"] * self.sample_rate)
            segment_audio = audio[start_sample:end_sample]

            # Get PANNs predictions
            predictions = self._classify_with_panns(segment_audio, sr, top_k=10)

            # Map to category
            classification, confidence = classify_panns_predictions(predictions)

            segment["classification"] = classification
            segment["confidence"] = confidence
            segment["top_predictions"] = predictions[:3]  # Top 3 only

            if classification == "music":
                self.logger.info(
                    f"Music detected: {segment['start']:.1f}s - {segment['end']:.1f}s "
                    f"({predictions[0]['label']}: {predictions[0]['confidence']:.2f})"
                )

        # Combine all segments
        all_segments = transcribed_segments + non_speech_segments
        all_segments.sort(key=lambda x: x["start"])

        return all_segments

    def extract_speaker_embeddings(
        self,
        audio_path: Path,
        segments: list[dict[str, Any]],
        audio: Optional[np.ndarray] = None,
    ) -> dict[str, list[float]]:
        """Extract speaker embeddings."""
        assert self.speaker_embedding_model is not None
        self.logger.info("Extracting speaker embeddings...")
        start_time = time.time()

        if audio is None:
            audio, sr = load_audio_mono_sanitized(
                audio_path, self.sample_rate, self.logger
            )
        speaker_embeddings = {}

        for segment in segments:
            if segment["type"] == "speech" and "speaker" in segment:
                speaker_id = segment["speaker"]

                if speaker_id not in speaker_embeddings:
                    start_sample = int(segment["start"] * self.sample_rate)
                    end_sample = int(segment["end"] * self.sample_rate)
                    segment_audio = audio[start_sample:end_sample]

                    if len(segment_audio) > self.sample_rate * 0.5:
                        audio_tensor = (
                            torch.tensor(segment_audio).unsqueeze(0).unsqueeze(0)
                        )
                        embedding = self.speaker_embedding_model(audio_tensor)
                        if isinstance(embedding, torch.Tensor):
                            embedding = embedding.cpu().numpy()
                        speaker_embeddings[speaker_id] = embedding.tolist()

        self.logger.info(
            f"Extracted embeddings for {len(speaker_embeddings)} speakers in {time.time() - start_time:.2f}s"
        )
        return speaker_embeddings

    def transcribe(
        self,
        audio_path: Path,
        language: Optional[str] = None,
        skip_speaker_embedding: bool = False,
        mlx_model_path: Optional[str] = None,
    ) -> dict[str, Any]:
        """Complete audio processing pipeline.

        If ``language`` is ``None`` (or an empty string), Whisper auto-detects
        the spoken language.

        If ``skip_speaker_embedding`` is True, the ECAPA-VoxCeleb speaker
        embedding step is skipped (``speaker_embeddings`` will be empty).
        Diarization itself still runs so segments keep their speaker labels.

        If ``mlx_model_path`` is given (MLX backend only), it overrides the
        Whisper model for this call — e.g. a cheaper model
        (``mlx-community/whisper-large-v3-turbo``) per file without
        reloading diarization/embedding models. Ignored on other backends.
        """
        # Reloads the models if unload_models() was called.
        self.ensure_models()
        if language:
            self.language = language
        else:
            # Reset to auto-detect even when a previous call left
            # ``self.language`` set: a reused Transcriber would otherwise
            # force the previous file's language on this one.
            self.language = None

        # Per-call MLX model override: swap self._mlx_model_path for the call,
        # restore it in the finally block. Safe because MLX caches models per
        # path in a module-level ModelHolder, so switching paths is cheap.
        previous_mlx_path = self._mlx_model_path
        if mlx_model_path and self.whisper_backend == WhisperBackend.MLX_LARGE_V3:
            if mlx_model_path != self._mlx_model_path:
                self.logger.info(
                    f"Overriding MLX Whisper model for this file: {mlx_model_path}"
                )
            self._mlx_model_path = mlx_model_path
        elif mlx_model_path:
            self.logger.warning(
                f"Ignoring mlx_model_path={mlx_model_path!r}: backend is {self.whisper_backend.value}, not MLX."
            )

        self.logger.info(f"Processing: {audio_path}")
        self.logger.info("=" * 60)

        # Load audio ONCE and reuse across all pipeline steps. The helper loads
        # raw and replaces NaN/Inf with silence — librosa.load(mono=True) would
        # instead raise "Audio buffer is not finite everywhere" before returning.
        audio, sr = load_audio_mono_sanitized(audio_path, self.sample_rate, self.logger)

        duration = len(audio) / sr
        self.logger.info(f"Audio loaded: {duration:.1f}s, {len(audio)} samples")

        stage_times: dict[str, float] = {}
        pipeline_start = time.time()

        try:
            # 1. Detect speech segments (diarization)
            t = time.time()
            speech_segments = self.detect_speech_segments(audio_path, audio=audio)
            stage_times["diarization"] = time.time() - t

            # 2. Transcribe
            t = time.time()
            transcribed_segments = self.transcribe_segments(
                audio_path, speech_segments, audio=audio
            )
            stage_times["transcribe"] = time.time() - t

            # Guard against silent total failure: ``transcribe_segments`` catches
            # per-segment exceptions and stores an empty text. If Whisper rejected
            # every segment — most commonly because the language code was invalid
            # (e.g. ``'se'`` instead of ``'sv'``) — we previously returned a
            # "successful" transcript full of empty strings. Surface that here.
            speech_only = [s for s in transcribed_segments if s.get("type") == "speech"]
            if len(speech_only) >= 5 and all(
                not s.get("text", "").strip() for s in speech_only
            ):
                raise RuntimeError(
                    f"All {len(speech_only)} speech segments produced empty text. "
                    f"Whisper likely rejected every segment (language={self.language!r}). "
                    "Check the language code and the audio content."
                )

            # 3. Recover speech in the gaps diarization rejected
            t = time.time()
            recovered_segments = self.recover_gap_speech(audio, speech_segments)
            transcribed_segments = sorted(
                transcribed_segments + recovered_segments, key=lambda s: s["start"]
            )
            stage_times["gap_recovery"] = time.time() - t

            # 4. Collapse decoder loops, then drop what still loops
            t = time.time()
            transcribed_segments = self.collapse_loops(transcribed_segments)
            transcribed_segments = self.filter_bad_segments(transcribed_segments)
            stage_times["filter"] = time.time() - t

            # 5. Analyze non-speech (PANNs)
            t = time.time()
            all_segments = self.analyse_non_speech(
                audio_path, transcribed_segments, audio=audio
            )
            stage_times["non_speech"] = time.time() - t

            # 6. Extract speaker embeddings (optional)
            if skip_speaker_embedding:
                self.logger.info("Skipping speaker embedding extraction")
                speaker_embeddings: dict[str, list[float]] = {}
                stage_times["speaker_embedding"] = 0.0
            else:
                t = time.time()
                speaker_embeddings = self.extract_speaker_embeddings(
                    audio_path, speech_segments, audio=audio
                )
                stage_times["speaker_embedding"] = time.time() - t

            total_pipeline = time.time() - pipeline_start

            # 7. Compile results (use original sample rate for metadata)
            _, original_sr = librosa.load(str(audio_path), sr=None, duration=0.01)
            unique_speakers = {
                s.get("speaker")
                for s in all_segments
                if s.get("type") == "speech" and s.get("speaker")
            }
            results = {
                "metadata": {
                    "version": self.VERSION,
                    "input_file": str(audio_path),
                    "processing_timestamp": datetime.now().isoformat(),
                    "model_used": self.model_name,
                    "whisper_backend": self.whisper_backend.value,
                    "mlx_model_path": self._mlx_model_path,
                    "diarization_model": DIARIZATION_MODEL,
                    "duration_seconds": duration,
                    "sample_rate": original_sr,
                    "total_samples": int(duration * original_sr),
                    "language": self.language,
                    "skip_speaker_embedding": skip_speaker_embedding,
                    "stage_times_sec": {k: round(v, 2) for k, v in stage_times.items()},
                    "pipeline_time_sec": round(total_pipeline, 2),
                },
                "segments": all_segments,
                "speaker_embeddings": speaker_embeddings,
                "statistics": {
                    "total_speech_segments": len(
                        [s for s in all_segments if s["type"] == "speech"]
                    ),
                    "total_speakers": len(unique_speakers),
                    "total_speech_duration": sum(
                        s["end"] - s["start"]
                        for s in all_segments
                        if s["type"] == "speech"
                    ),
                    "recovered_speech_segments": len(recovered_segments),
                    "recovered_speech_duration": sum(
                        s["end"] - s["start"] for s in recovered_segments
                    ),
                },
            }

            self.logger.info("Processing complete!")
            self.logger.info(f"  Speakers: {results['statistics']['total_speakers']}")
            self.logger.info(
                f"  Speech duration: {results['statistics']['total_speech_duration']:.1f}s"
            )
            self.logger.info("  Stage times (s):")
            for name, t_val in stage_times.items():
                self.logger.info(f"    {name:<20} {t_val:7.2f}")
            self.logger.info(
                f"  Total pipeline:      {total_pipeline:7.2f}s ({total_pipeline / duration:.2f}x realtime)"
            )

            return results
        finally:
            # Restore the instance-level MLX model path if this call overrode it.
            self._mlx_model_path = previous_mlx_path
            # Explicitly free the large audio array and trigger cleanup
            del audio
            gc.collect()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            elif torch.cuda.is_available():
                torch.cuda.empty_cache()
            if MLX_AVAILABLE:
                mx.clear_cache()
