"""
Audio Normalization Library

Two-stage normalization strategies that apply dynamic range compression
before loudnorm to achieve consistent loudness levels.

Strategies:
1. speechnorm + loudnorm: Best for speech-heavy content (radio programs, podcasts)
2. dynaudnorm + loudnorm: Best for mixed content with music and speech

Both strategies reduce the loudness range (LRA) before applying loudnorm,
which allows loudnorm to reach the target LUFS without peaks limiting gain.
"""

import json
import os
import subprocess
from dataclasses import dataclass
from typing import Optional

__all__ = [
    'SpeechnormParams',
    'DynaudnormParams',
    'LoudnormParams',
    'normalize_speechnorm',
    'normalize_dynaudnorm',
    'detect_audio_codec',
    'detect_sample_rate',
]


@dataclass
class LoudnormParams:
    """Parameters for loudnorm filter (final stage)."""

    integrated_lufs: float = -18.0  # Target integrated loudness (I)
    true_peak_db: float = -1.0  # Maximum true peak (TP)
    lra: float = 4.0  # Loudness range target (LRA) - lower than default for compressed audio

    def to_filter(self) -> str:
        """Return ffmpeg filter string."""
        return f'loudnorm=I={self.integrated_lufs}:TP={self.true_peak_db}:LRA={self.lra}'


@dataclass
class SpeechnormParams:
    """
    Parameters for speechnorm filter (speech normalization).

    speechnorm is optimized for speech content. It analyzes the audio
    and adjusts gain to normalize speech levels dynamically.

    Attributes:
        expansion: Amount of expansion/compression (1-50, default 25)
                   Higher values = more aggressive normalization
        recovery: Recovery time coefficient (0.0001-1, default 0.0001)
                  Lower values = faster recovery from loud passages
        lim: Enable limiter (0 or 1, default 1)
             Prevents clipping after normalization
    """

    expansion: float = 25.0  # e parameter - expansion factor
    recovery: float = 0.0001  # r parameter - recovery time
    lim: int = 1  # l parameter - limiter enabled

    def to_filter(self) -> str:
        """Return ffmpeg filter string."""
        return f'speechnorm=e={self.expansion}:r={self.recovery}:l={self.lim}'


@dataclass
class DynaudnormParams:
    """
    Parameters for dynaudnorm filter (dynamic audio normalizer).

    dynaudnorm normalizes audio by analyzing short time windows and
    adjusting gain dynamically. Works well for mixed content.

    Attributes:
        frame_length: Frame length in ms for analysis (10-8000, default 150)
                      Shorter = more responsive, longer = smoother
        gauss_size: Gaussian filter window size (3-301 odd, default 15)
                    Higher = smoother gain changes
        peak: Target peak level (0.0-1.0, default 0.95)
        max_gain: Maximum gain factor in dB (1-100, default 20)
        target_rms: Target RMS level (0.0-1.0, default not set)
                    Uses 0 to disable RMS targeting
        compress: Compression factor (0.0-30.0, default 3)
                  Higher = more compression
    """

    frame_length: int = 150  # f parameter - frame length in ms
    gauss_size: int = 15  # g parameter - gaussian window size
    peak: float = 0.95  # p parameter - target peak
    max_gain: float = 20.0  # m parameter - maximum gain in dB
    compress: float = 3.0  # s parameter - compression factor

    def __post_init__(self):
        # Validate gauss_size is odd and in valid range (FFmpeg requirement)
        if self.gauss_size < 3 or self.gauss_size > 301:
            raise ValueError(f'gauss_size must be between 3 and 301, got {self.gauss_size}')
        if self.gauss_size % 2 == 0:
            raise ValueError(f'gauss_size must be an odd number, got {self.gauss_size}')

    def to_filter(self) -> str:
        """Return ffmpeg filter string."""
        return f'dynaudnorm=f={self.frame_length}:g={self.gauss_size}:p={self.peak}:m={self.max_gain}:s={self.compress}'


def detect_audio_codec(input_path: str) -> tuple[str, Optional[str]]:
    """
    Detect the audio codec and bitrate of an audio file using ffprobe.

    Args:
        input_path: Path to input audio file

    Returns:
        Tuple of (codec_name, bitrate_string)
        - codec_name: FFmpeg codec name (e.g., 'pcm_s16le', 'libmp3lame', 'aac')
        - bitrate_string: Bitrate for lossy codecs (e.g., '320k') or None for lossless

    Raises:
        FileNotFoundError: If input file doesn't exist
        subprocess.CalledProcessError: If ffprobe fails
        ValueError: If codec cannot be determined
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f'Input file not found: {input_path}')

    cmd = ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_streams', '-select_streams', 'a:0', input_path]

    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    data = json.loads(result.stdout)

    if not data.get('streams'):
        raise ValueError(f'No audio streams found in {input_path}')

    audio_stream = data['streams'][0]
    codec_name = audio_stream.get('codec_name')

    if not codec_name:
        raise ValueError(f'Could not determine codec for {input_path}')

    # Map codec names to FFmpeg encoder names
    codec_map = {
        'mp3': 'libmp3lame',
        'aac': 'aac',
        'vorbis': 'libvorbis',
        'opus': 'libopus',
        'flac': 'flac',
        'pcm_s16le': 'pcm_s16le',
        'pcm_s24le': 'pcm_s24le',
        'pcm_s32le': 'pcm_s32le',
        'pcm_f32le': 'pcm_f32le',
        'pcm_s16be': 'pcm_s16be',
        'pcm_s24be': 'pcm_s24be',
    }

    encoder_name = codec_map.get(codec_name, codec_name)

    # Determine if codec is lossy and needs bitrate
    lossy_codecs = {'libmp3lame', 'aac', 'libvorbis', 'libopus'}
    bitrate = None

    if encoder_name in lossy_codecs:
        bit_rate = audio_stream.get('bit_rate')
        if bit_rate:
            bitrate_k = int(bit_rate) // 1000
            bitrate = f'{bitrate_k}k'

    return encoder_name, bitrate


def detect_sample_rate(input_path: str) -> Optional[int]:
    """
    Detect the sample rate of an audio file using ffprobe.

    Args:
        input_path: Path to input audio file

    Returns:
        Sample rate in Hz (e.g., 44100, 48000) or None if detection fails

    Raises:
        FileNotFoundError: If input file doesn't exist
        subprocess.CalledProcessError: If ffprobe fails
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f'Input file not found: {input_path}')

    cmd = ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_streams', '-select_streams', 'a:0', input_path]

    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    data = json.loads(result.stdout)

    if data.get('streams'):
        sample_rate = data['streams'][0].get('sample_rate')
        if sample_rate:
            return int(sample_rate)
    return None


def _run_normalization(
    input_path: str,
    output_path: str,
    filter_chain: str,
    audio_codec: Optional[str] = None,
    audio_bitrate: Optional[str] = None,
) -> None:
    """
    Internal function to run FFmpeg normalization.

    Args:
        input_path: Path to input audio file
        output_path: Path to output audio file
        filter_chain: Complete FFmpeg audio filter chain
        audio_codec: Audio codec for output (default: auto-detect)
        audio_bitrate: Audio bitrate for lossy codecs (default: auto-detect)
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f'Input file not found: {input_path}')

    # Auto-detect codec if not specified
    if audio_codec is None or audio_bitrate is None:
        detected_codec, detected_bitrate = detect_audio_codec(input_path)
        if audio_codec is None:
            audio_codec = detected_codec
        if audio_bitrate is None:
            audio_bitrate = detected_bitrate

    # Detect original sample rate to preserve it
    # The loudnorm filter internally upsamples to 192kHz for accurate true peak detection,
    # which can cause ~4x file size increase if not resampled back to original rate
    sample_rate = detect_sample_rate(input_path)
    if sample_rate:
        filter_chain = f'{filter_chain},aresample={sample_rate}'

    # Create output directory if needed
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Build ffmpeg command
    cmd = ['ffmpeg', '-y', '-i', input_path, '-af', filter_chain, '-c:a', audio_codec]

    if audio_bitrate:
        cmd.extend(['-b:a', audio_bitrate])

    cmd.append(output_path)

    subprocess.run(cmd, capture_output=True, text=True, check=True)


def normalize_speechnorm(
    input_path: str,
    output_path: str,
    speechnorm_params: Optional[SpeechnormParams] = None,
    loudnorm_params: Optional[LoudnormParams] = None,
    audio_codec: Optional[str] = None,
    audio_bitrate: Optional[str] = None,
) -> None:
    """
    Normalize audio using speechnorm + loudnorm.

    Best for speech-heavy content like radio programs, podcasts, and interviews.
    Speechnorm dynamically adjusts gain to normalize speech levels, then
    loudnorm ensures consistent integrated loudness.

    Args:
        input_path: Path to input audio file
        output_path: Path to output audio file
        speechnorm_params: Speechnorm parameters (uses defaults if None)
        loudnorm_params: Loudnorm parameters (uses defaults if None)
        audio_codec: Audio codec for output (default: auto-detect from input)
        audio_bitrate: Audio bitrate for lossy codecs (default: auto-detect)

    Example:
        # Default parameters (aggressive speech normalization)
        normalize_speechnorm('input.wav', 'output.wav')

        # Custom parameters
        normalize_speechnorm(
            'input.wav',
            'output.wav',
            speechnorm_params=SpeechnormParams(expansion=20.0),
            loudnorm_params=LoudnormParams(integrated_lufs=-16.0),
        )
    """
    if speechnorm_params is None:
        speechnorm_params = SpeechnormParams()
    if loudnorm_params is None:
        loudnorm_params = LoudnormParams()

    filter_chain = f'{speechnorm_params.to_filter()},{loudnorm_params.to_filter()}'

    _run_normalization(input_path, output_path, filter_chain, audio_codec, audio_bitrate)


def normalize_dynaudnorm(
    input_path: str,
    output_path: str,
    dynaudnorm_params: Optional[DynaudnormParams] = None,
    loudnorm_params: Optional[LoudnormParams] = None,
    audio_codec: Optional[str] = None,
    audio_bitrate: Optional[str] = None,
) -> None:
    """
    Normalize audio using dynaudnorm + loudnorm.

    Best for mixed content with both speech and music. Dynaudnorm analyzes
    short time windows and adjusts gain dynamically, providing smooth
    loudness normalization before loudnorm finalizes the target level.

    Args:
        input_path: Path to input audio file
        output_path: Path to output audio file
        dynaudnorm_params: Dynaudnorm parameters (uses defaults if None)
        loudnorm_params: Loudnorm parameters (uses defaults if None)
        audio_codec: Audio codec for output (default: auto-detect from input)
        audio_bitrate: Audio bitrate for lossy codecs (default: auto-detect)

    Example:
        # Default parameters (aggressive dynamic normalization)
        normalize_dynaudnorm('input.wav', 'output.wav')

        # Custom parameters for gentler processing
        normalize_dynaudnorm(
            'input.wav',
            'output.wav',
            dynaudnorm_params=DynaudnormParams(frame_length=300, gauss_size=10),
            loudnorm_params=LoudnormParams(lra=5.0),
        )
    """
    if dynaudnorm_params is None:
        dynaudnorm_params = DynaudnormParams()
    if loudnorm_params is None:
        loudnorm_params = LoudnormParams()

    filter_chain = f'{dynaudnorm_params.to_filter()},{loudnorm_params.to_filter()}'

    _run_normalization(input_path, output_path, filter_chain, audio_codec, audio_bitrate)
