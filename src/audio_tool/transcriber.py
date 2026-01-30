"""
Transcription engine for audio-tool.

Uses Whisper for speech-to-text and Pyannote for speaker diarization.
Supports multiple backends for benchmarking and optimization.
"""

# CRITICAL: Set PyTorch environment variables BEFORE any imports
import os

os.environ.setdefault("TORCH_FORCE_WEIGHTS_ONLY_LOAD", "0")

import gc
import logging
import tempfile
import time
import uuid
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import librosa
import librosa.beat
import librosa.effects
import librosa.feature
import numpy as np
import requests
import soundfile as sf
import torch
import whisper_timestamped
from pyannote.audio import Pipeline as DiarizationPipeline
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
from scipy.signal import find_peaks


# Check if MLX is available (Mac only)
MLX_AVAILABLE = False
try:
    import mlx_whisper

    MLX_AVAILABLE = True
except ImportError:
    mlx_whisper = None


# Monkey-patch torch.load to ALWAYS disable weights_only for pyannote models
_original_torch_load = torch.load


def _torch_load_without_weights_only(*args, **kwargs):
    """Wrapper for torch.load that forces weights_only=False for compatibility with pyannote models."""
    kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)


torch.load = _torch_load_without_weights_only


class WhisperBackend(str, Enum):
    """Available Whisper backends."""

    WHISPER_TIMESTAMPED = "whisper-timestamped"  # CPU/Linux compatible, original
    MLX_LARGE_V3 = "mlx-large-v3"  # Mac MLX, full large-v3
    MLX_TURBO = "mlx-turbo"  # Mac MLX, faster turbo variant


class DiarizationModel(str, Enum):
    """Available diarization models."""

    DIARIZATION_3_1 = "pyannote/speaker-diarization-3.1"  # Original, faster load
    DIARIZATION_COMMUNITY = (
        "pyannote/speaker-diarization-community-1"  # Faster inference
    )


# Model mappings
WHISPER_MODEL_MAP = {
    WhisperBackend.WHISPER_TIMESTAMPED: "large-v3",
    WhisperBackend.MLX_LARGE_V3: "mlx-community/whisper-large-v3-mlx",
    WhisperBackend.MLX_TURBO: "mlx-community/whisper-large-v3-turbo",
}


def get_device():
    """Determine the best device for processing."""
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"


def check_repeated_words(text: str) -> int:
    """Check for consecutive repetitions of words or phrases."""
    if not text or not isinstance(text, str):
        return 0

    text = " ".join(text.lower().split())
    words = text.split()

    if not words:
        return 0

    max_count = 1

    # Check for single word repetitions
    i = 0
    while i < len(words):
        current_word = words[i]
        count = 1
        while i + count < len(words) and words[i + count] == current_word:
            count += 1
        max_count = max(max_count, count)
        i += count

    # Check for multi-word phrase repetitions
    for phrase_length in range(2, 4):
        if len(words) < phrase_length * 2:
            continue
        i = 0
        while i <= len(words) - phrase_length:
            phrase = tuple(words[i : i + phrase_length])
            phrase_count = 1
            next_pos = i + phrase_length
            while next_pos <= len(words) - phrase_length:
                next_phrase = tuple(words[next_pos : next_pos + phrase_length])
                if next_phrase == phrase:
                    phrase_count += 1
                    next_pos += phrase_length
                else:
                    break
            max_count = max(max_count, phrase_count)
            if phrase_count > 1:
                i = next_pos
            else:
                i += 1

    return max_count


def create_constellation_map(audio, sr):
    """Create constellation map of spectral peaks for fingerprinting."""
    stft = librosa.stft(audio, hop_length=512, n_fft=2048)
    magnitude = np.abs(stft)

    peaks = []
    for time_idx in range(magnitude.shape[1]):
        spectrum = magnitude[:, time_idx]
        peak_indices, _ = find_peaks(spectrum, height=np.max(spectrum) * 0.1)
        for peak_freq in peak_indices[:5]:
            peaks.append((time_idx, peak_freq, spectrum[peak_freq]))

    fingerprint_hashes = []
    for i, (t1, f1, _) in enumerate(peaks):
        for j in range(i + 1, min(i + 10, len(peaks))):
            t2, f2, _ = peaks[j]
            dt = t2 - t1
            df = f2 - f1
            hash_value = hash((df, dt, f1)) % (10**8)
            fingerprint_hashes.append(hash_value)

    return fingerprint_hashes[:100]


def extract_audio_fingerprint(audio_segment, sr):
    """Extract robust audio fingerprint."""
    if len(audio_segment.shape) > 1:
        audio_segment = np.mean(audio_segment, axis=1)

    chroma = librosa.feature.chroma_stft(
        y=audio_segment, sr=sr, hop_length=512, n_fft=2048
    )
    spectral_contrast = librosa.feature.spectral_contrast(y=audio_segment, sr=sr)
    mfccs = librosa.feature.mfcc(y=audio_segment, sr=sr, n_mfcc=20)
    fingerprint = create_constellation_map(audio_segment, sr)

    return {
        "fingerprint_hash": fingerprint,
        "chroma_mean": np.mean(chroma, axis=1).tolist(),
        "chroma_var": np.var(chroma, axis=1).tolist(),
        "spectral_contrast_mean": np.mean(spectral_contrast, axis=1).tolist(),
        "mfcc_mean": np.mean(mfccs, axis=1).tolist(),
        "tempo": float(librosa.beat.tempo(y=audio_segment, sr=sr)[0]),
    }


def extract_rhythm_features(audio, beats, sr):
    """Extract rhythm-specific features."""
    if len(beats) < 2:
        return np.zeros(5)
    beat_intervals = np.diff(beats) / sr
    return np.array(
        [
            np.mean(beat_intervals),
            np.var(beat_intervals),
            len(beats) / (len(audio) / sr),
            np.median(beat_intervals),
            np.std(beat_intervals),
        ]
    )


def extract_enhanced_music_embedding(audio_segment, sr):
    """Extract comprehensive music embedding for similarity comparison."""
    try:
        harmonic, percussive = librosa.effects.hpss(audio_segment)
    except Exception:
        harmonic = audio_segment
        percussive = audio_segment

    tempo, beats = librosa.beat.beat_track(y=audio_segment, sr=sr)
    rhythm_features = extract_rhythm_features(audio_segment, beats, sr)

    chroma = librosa.feature.chroma_stft(y=harmonic, sr=sr)
    tonnetz = librosa.feature.tonnetz(y=harmonic, sr=sr)
    mfccs = librosa.feature.mfcc(y=audio_segment, sr=sr, n_mfcc=20)
    spectral_centroids = librosa.feature.spectral_centroid(y=audio_segment, sr=sr)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_segment, sr=sr)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio_segment, sr=sr)
    zero_crossing_rate = librosa.feature.zero_crossing_rate(audio_segment)

    embedding = np.concatenate(
        [
            [tempo],
            rhythm_features,
            np.mean(chroma, axis=1),
            np.var(chroma, axis=1),
            np.mean(tonnetz, axis=1),
            np.mean(mfccs, axis=1),
            np.var(mfccs, axis=1),
            np.mean(spectral_centroids),
            np.var(spectral_centroids),
            np.mean(spectral_rolloff),
            np.mean(spectral_bandwidth),
            np.mean(zero_crossing_rate),
            [np.mean(harmonic**2), np.mean(percussive**2)],
        ]
    )

    return embedding


class Transcriber:
    """
    Transcription engine using Whisper and Pyannote.

    Supports multiple backends for benchmarking:
    - Whisper: whisper-timestamped (CPU), MLX large-v3, MLX turbo
    - Diarization: pyannote 3.1, pyannote community
    """

    VERSION = "2.1"

    def __init__(
        self,
        huggingface_token: str,
        model_name: str = "large-v3",
        language: str = "fi",
        whisper_backend: WhisperBackend = WhisperBackend.WHISPER_TIMESTAMPED,
        diarization_model: DiarizationModel = DiarizationModel.DIARIZATION_3_1,
        logger: Optional[logging.Logger] = None,
    ):
        self.model_name = model_name
        self.language = language
        self.sample_rate = 16000
        self.huggingface_token = huggingface_token
        self.whisper_backend = whisper_backend
        self.diarization_model = diarization_model
        self.logger = logger or logging.getLogger(__name__)

        if not self.huggingface_token:
            raise ValueError("HUGGINGFACE_TOKEN is required for pyannote models")

        # Validate MLX backend availability
        if whisper_backend in (WhisperBackend.MLX_LARGE_V3, WhisperBackend.MLX_TURBO):
            if not MLX_AVAILABLE:
                raise ValueError(
                    f"MLX backend {whisper_backend.value} requested but mlx-whisper is not installed. "
                    "Install with: pip install mlx-whisper (Mac only)"
                )

        self.device = get_device()
        self.logger.info(f"Using device: {self.device}")
        self.logger.info(f"Whisper backend: {self.whisper_backend.value}")
        self.logger.info(f"Diarization model: {self.diarization_model.value}")

        self.diarization_pipeline = None
        self.speaker_embedding_model = None
        self.whisper_model = None
        self._mlx_model_path = None  # For MLX model caching

        self._load_models()

    def _load_models(self):
        """Load all required models."""
        self.logger.info(
            f"Loading models (whisper={self.whisper_backend.value}, diarization={self.diarization_model.value})..."
        )

        # Load PyAnnote diarization
        self.diarization_pipeline = DiarizationPipeline.from_pretrained(
            self.diarization_model.value,
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
            # Original whisper-timestamped backend
            self.whisper_model = whisper_timestamped.load_model(
                self.model_name, device="cpu"
            )
            self.logger.info(
                f"Loaded Whisper model (whisper-timestamped): {self.model_name}"
            )
        else:
            # MLX backends - models are loaded on first use
            self._mlx_model_path = WHISPER_MODEL_MAP[self.whisper_backend]
            self.logger.info(f"Will use MLX Whisper model: {self._mlx_model_path}")

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

    def _transcribe_with_mlx(self, audio_path: str, segment_start: float) -> dict:
        """Transcribe using MLX Whisper backend."""
        if not MLX_AVAILABLE:
            raise RuntimeError("MLX Whisper not available")

        result = mlx_whisper.transcribe(
            audio_path,
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

    def download_audio(self, url: str) -> Path:
        """Download audio file from URL."""
        self.logger.info(f"Downloading: {url[:80]}...")

        # Create temp file with appropriate extension
        suffix = ".mp3"
        if ".wav" in url.lower():
            suffix = ".wav"
        elif ".m4a" in url.lower():
            suffix = ".m4a"
        elif ".flac" in url.lower():
            suffix = ".flac"

        temp_dir = Path(tempfile.gettempdir()) / "transcription_v2"
        temp_dir.mkdir(exist_ok=True)

        temp_path = temp_dir / f"audio_{uuid.uuid4().hex}{suffix}"

        try:
            response = requests.get(url, timeout=300, stream=True)
            response.raise_for_status()

            with open(temp_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
        except Exception:
            # Clean up partial download on failure
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except Exception:
                    pass
            raise

        self.logger.info(f"Downloaded to: {temp_path}")
        return temp_path

    def detect_speech_segments(
        self,
        audio_path: Path,
        min_segment_duration: float = 2.0,
        merge_gap_threshold: float = 4.0,
        max_merge_duration: float = 300,
    ) -> list[dict[str, Any]]:
        """Detect speech segments using PyAnnote diarization."""
        self.logger.info("Detecting speech segments...")
        start_time = time.time()

        audio, sr = librosa.load(str(audio_path), sr=self.sample_rate)
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

            # Use the same temp directory as download_audio so cleanup works correctly
            temp_dir = Path(tempfile.gettempdir()) / "transcription_v2"
            temp_dir.mkdir(exist_ok=True)
            temp_chunk_path = (
                temp_dir / f"transcribe_chunk_{chunk_start}_{chunk_end}.wav"
            )
            sf.write(str(temp_chunk_path), chunk_audio, self.sample_rate)

            try:
                chunk_diarization = self.diarization_pipeline(str(temp_chunk_path))

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
            finally:
                if temp_chunk_path.exists():
                    os.remove(temp_chunk_path)

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
        self, audio_path: Path, segments: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Transcribe speech segments with word-level timestamps."""
        self.logger.info("Transcribing segments...")

        audio, sr = librosa.load(str(audio_path), sr=self.sample_rate)

        # For MLX, we need to write temp files since it takes file paths
        use_mlx = self.whisper_backend in (
            WhisperBackend.MLX_LARGE_V3,
            WhisperBackend.MLX_TURBO,
        )

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
                    # MLX needs a file path, write temp file
                    temp_dir = Path(tempfile.gettempdir()) / "transcription_v2"
                    temp_dir.mkdir(exist_ok=True)
                    temp_segment_path = temp_dir / f"segment_{uuid.uuid4().hex}.wav"
                    sf.write(str(temp_segment_path), segment_audio, self.sample_rate)

                    try:
                        result = self._transcribe_with_mlx(
                            str(temp_segment_path), segment["start"]
                        )
                    finally:
                        if temp_segment_path.exists():
                            temp_segment_path.unlink()
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
                    f"  {seg_duration}s, words:{len(result['words'])}, "
                    f"conf:{segment['avg_confidence']:.2f} - {segment['text'][:50]}..."
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
        """Remove segments with too many repeated words."""
        return [s for s in segments if check_repeated_words(s.get("text", "")) < 4]

    def enhanced_audio_classification(self, segment_audio, sample_rate):
        """Classify audio segment as speech, music, or silence."""
        rms_energy = np.sqrt(np.mean(segment_audio**2))
        zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(segment_audio))
        spectral_centroid = np.mean(
            librosa.feature.spectral_centroid(y=segment_audio, sr=sample_rate)
        )
        spectral_rolloff = np.mean(
            librosa.feature.spectral_rolloff(y=segment_audio, sr=sample_rate)
        )
        spectral_bandwidth = np.mean(
            librosa.feature.spectral_bandwidth(y=segment_audio, sr=sample_rate)
        )
        tempo, _ = librosa.beat.beat_track(y=segment_audio, sr=sample_rate)

        try:
            harmonic, percussive = librosa.effects.hpss(segment_audio)
            harmonic_ratio = np.sum(harmonic**2) / (
                np.sum(harmonic**2) + np.sum(percussive**2)
            )
        except Exception:
            harmonic_ratio = 0.5

        chroma = np.mean(librosa.feature.chroma_stft(y=segment_audio, sr=sample_rate))

        features = {
            "rms_energy": float(rms_energy),
            "zero_crossing_rate": float(zero_crossing_rate),
            "spectral_centroid": float(spectral_centroid),
            "spectral_rolloff": float(spectral_rolloff),
            "spectral_bandwidth": float(spectral_bandwidth),
            "tempo": float(tempo),
            "harmonic_ratio": float(harmonic_ratio),
            "chroma": float(chroma),
        }

        if rms_energy < 0.001:
            return "silence", features
        elif harmonic_ratio > 0.7 and tempo > 60:
            return "music", features
        elif zero_crossing_rate < 0.1 and spectral_centroid > 1000:
            return "speech", features
        else:
            return "unknown", features

    def analyse_non_speech(
        self, audio_path: Path, transcribed_segments: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Analyze gaps between speech segments."""
        self.logger.info("Analyzing non-speech segments...")

        audio, sr = librosa.load(str(audio_path), sr=self.sample_rate)
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

        # Analyze each non-speech segment
        for segment in non_speech_segments:
            start_sample = int(segment["start"] * self.sample_rate)
            end_sample = int(segment["end"] * self.sample_rate)
            segment_audio = audio[start_sample:end_sample]

            category, features = self.enhanced_audio_classification(segment_audio, sr)
            segment["classification"] = category
            segment["features"] = features

            if segment["classification"] == "music":
                try:
                    fingerprint = extract_audio_fingerprint(
                        segment_audio, self.sample_rate
                    )
                    enhanced_embedding = extract_enhanced_music_embedding(
                        segment_audio, self.sample_rate
                    )
                    segment["music_fingerprint"] = fingerprint
                    segment["music_embedding"] = enhanced_embedding.tolist()
                    segment["confidence"] = min(features.get("rms_energy", 0) * 10, 1.0)
                    self.logger.info(
                        f"Music detected: {segment['start']:.1f}s - {segment['end']:.1f}s"
                    )
                except Exception as e:
                    self.logger.error(f"Error extracting music features: {e}")
                    segment["classification"] = "unknown"
                    segment["confidence"] = 0.5

        # Combine all segments
        all_segments = transcribed_segments + non_speech_segments
        all_segments.sort(key=lambda x: x["start"])

        return all_segments

    def extract_speaker_embeddings(
        self, audio_path: Path, segments: list[dict[str, Any]]
    ) -> dict[str, list[float]]:
        """Extract speaker embeddings."""
        self.logger.info("Extracting speaker embeddings...")

        audio, sr = librosa.load(str(audio_path), sr=self.sample_rate)
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

        self.logger.info(f"Extracted embeddings for {len(speaker_embeddings)} speakers")
        return speaker_embeddings

    def transcribe(self, audio_path: Path, language: str = None) -> dict[str, Any]:
        """Complete audio processing pipeline."""
        if language:
            self.language = language

        self.logger.info(f"Processing: {audio_path}")
        self.logger.info("=" * 60)

        # 1. Detect speech segments
        speech_segments = self.detect_speech_segments(audio_path)

        # 2. Transcribe
        transcribed_segments = self.transcribe_segments(audio_path, speech_segments)

        # 3. Filter bad segments
        transcribed_segments = self.filter_bad_segments(transcribed_segments)

        # 4. Analyze non-speech
        all_segments = self.analyse_non_speech(audio_path, transcribed_segments)

        # 5. Extract speaker embeddings
        speaker_embeddings = self.extract_speaker_embeddings(
            audio_path, speech_segments
        )

        # 6. Get audio info
        audio, sr = librosa.load(str(audio_path), sr=None)
        duration = len(audio) / sr

        # 7. Compile results
        results = {
            "metadata": {
                "version": self.VERSION,
                "input_file": str(audio_path),
                "processing_timestamp": datetime.now().isoformat(),
                "model_used": self.model_name,
                "whisper_backend": self.whisper_backend.value,
                "diarization_model": self.diarization_model.value,
                "duration_seconds": duration,
                "sample_rate": sr,
                "total_samples": len(audio),
                "language": self.language,
            },
            "segments": all_segments,
            "speaker_embeddings": speaker_embeddings,
            "statistics": {
                "total_speech_segments": len(
                    [s for s in all_segments if s["type"] == "speech"]
                ),
                "total_speakers": len(speaker_embeddings),
                "total_speech_duration": sum(
                    s["end"] - s["start"] for s in all_segments if s["type"] == "speech"
                ),
            },
        }

        self.logger.info("Processing complete!")
        self.logger.info(f"  Speakers: {results['statistics']['total_speakers']}")
        self.logger.info(
            f"  Speech duration: {results['statistics']['total_speech_duration']:.1f}s"
        )

        # Cleanup
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()

        return results
