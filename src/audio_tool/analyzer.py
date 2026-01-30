"""
Audio Quality Analyzer Library

Combines silence, LUFS, and clipping analysis in a single FFmpeg pass for maximum efficiency.
"""

import json
import os
import re
import subprocess
from dataclasses import asdict, dataclass
from typing import Any, Optional

TARGET_LUFS = -18.0
LUFS_RANGE = 0.5
LUFS_MIN = TARGET_LUFS - LUFS_RANGE
LUFS_MAX = TARGET_LUFS + LUFS_RANGE

AUDIO_ANALYSE_VERSION = 3


@dataclass
class AudioQualityResult:
    """Complete audio quality analysis result."""

    file: str
    duration_sec: float

    # Silence metrics
    start_silence_sec: float
    end_silence_sec: float
    longest_middle_silence_sec: float
    silence_periods_count: int

    # LUFS metrics (from ebur128)
    integrated_lufs: Optional[float]
    loudness_range_lu: Optional[float]
    true_peak_db: Optional[float]

    # Volume/clipping metrics
    max_volume_db: float
    mean_volume_db: float
    histogram_0db: Optional[int]

    # Metadata (from ffprobe)
    metadata: dict[str, Any]

    # Phase correlation (stereo only)
    phase_correlation: Optional[float]

    # LUFS timeline data (optional, for visualization)
    lufs_timeline: Optional[list[dict[str, float]]] = None

    # Waveform timeline data (optional, for visualization)
    waveform_timeline: Optional[list[dict[str, float]]] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class QualityCheck:
    """Single quality check result with short and detailed descriptions."""

    check_type: str  # e.g., 'silence', 'lufs', 'clipping'
    severity: str  # 'warning', 'error', 'info'
    short_text: str  # Brief message for UI display
    detailed_text: str  # Longer explanation of the issue
    value: Optional[float] = None  # The actual measured value

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


def get_audio_metadata(audio_path: str) -> tuple[float, dict[str, Any]]:
    """
    Get audio file metadata using ffprobe.

    Returns:
        Tuple of (duration_sec, metadata_dict)
    """
    try:
        cmd = [
            "ffprobe",
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            audio_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)

        # Extract duration
        duration_sec = float(data.get("format", {}).get("duration", 0))

        # Build simplified metadata dict
        metadata = {
            "format": data.get("format", {}),
            "streams": data.get("streams", []),
        }

        # Extract useful audio stream info if available
        audio_stream = next(
            (s for s in data.get("streams", []) if s.get("codec_type") == "audio"), None
        )
        if audio_stream:
            metadata["audio"] = {
                "codec": audio_stream.get("codec_name"),
                "sample_rate": audio_stream.get("sample_rate"),
                "channels": audio_stream.get("channels"),
                "bit_rate": audio_stream.get("bit_rate"),
                "bits_per_sample": audio_stream.get("bits_per_sample"),
            }

        return duration_sec, metadata
    except (subprocess.CalledProcessError, json.JSONDecodeError, Exception) as e:
        import logging

        logger = logging.getLogger(__name__)
        logger.warning(f"ffprobe failed for {audio_path}: {e}")
        return 0.0, {}


def calculate_phase_correlation(
    audio_path: str, metadata: dict[str, Any]
) -> Optional[float]:
    """
    Calculate phase correlation for stereo audio files.

    Phase correlation measures the similarity between left and right channels:
    - +1.0: Channels are identical (mono compatibility)
    - 0.0: Channels are uncorrelated
    - -1.0: Channels are perfectly inverted (severe mono cancellation)
    """
    import numpy as np
    import soundfile as sf

    # Only analyze stereo files
    channels = metadata.get("audio", {}).get("channels")
    if channels != 2:
        return None

    try:
        # Read first 30 seconds (sufficient for phase analysis)
        with sf.SoundFile(audio_path) as f:
            sample_rate = f.samplerate
            frames_to_read = min(30 * sample_rate, len(f))
            audio_data = f.read(frames_to_read)

        left = audio_data[:, 0]
        right = audio_data[:, 1]

        # Calculate phase correlation using Pearson correlation
        correlation = np.corrcoef(left, right)[0, 1]

        return float(correlation)

    except Exception:
        return None


def calculate_waveform_timeline(
    audio_path: str, duration_sec: float, target_points_per_minute: int = 1000
) -> Optional[list[dict[str, float]]]:
    """
    Calculate waveform timeline data for visualization.

    Downsamples audio to ~1000 points/minute by finding min/max peaks.
    """
    import numpy as np
    import soundfile as sf

    try:
        if duration_sec <= 0:
            return None

        waveform_timeline = []

        with sf.SoundFile(audio_path) as f:
            sample_rate = f.samplerate
            total_frames = len(f)
            channels = f.channels

            target_points = int(duration_sec * target_points_per_minute / 60)
            samples_per_window = max(1, total_frames // target_points)

            current_frame = 0
            while current_frame < total_frames:
                frames_to_read = min(samples_per_window, total_frames - current_frame)
                window = f.read(frames_to_read, always_2d=True)

                if len(window) == 0:
                    break

                time = current_frame / sample_rate

                left_data = window[:, 0]
                data_point = {
                    "time": round(time, 3),
                    "left_min": round(float(np.min(left_data)), 4),
                    "left_max": round(float(np.max(left_data)), 4),
                }

                if channels >= 2:
                    right_data = window[:, 1]
                    data_point["right_min"] = round(float(np.min(right_data)), 4)
                    data_point["right_max"] = round(float(np.max(right_data)), 4)

                waveform_timeline.append(data_point)
                current_frame += frames_to_read

        return waveform_timeline

    except Exception:
        return None


def analyze_audio_quality(
    audio_path: str,
    silence_thresh_db: float = -40.0,
    min_silence_len_sec: float = 0.1,
    include_waveform: bool = False,
) -> AudioQualityResult:
    """
    Analyze audio file for all quality metrics in a single FFmpeg pass.

    Combines silence detection, LUFS measurement, clipping detection,
    and metadata extraction.

    Args:
        audio_path: Path to audio file
        silence_thresh_db: Silence threshold in dB (default: -40.0)
        min_silence_len_sec: Minimum silence length in seconds (default: 0.1)
        include_waveform: Include waveform timeline data (default: False)

    Returns:
        AudioQualityResult dataclass with all metrics
    """
    # Validate parameters
    if not isinstance(silence_thresh_db, (int, float)) or not isinstance(
        min_silence_len_sec, (int, float)
    ):
        raise ValueError("silence_thresh_db and min_silence_len_sec must be numeric")

    silence_thresh_db = max(-100.0, min(0.0, float(silence_thresh_db)))
    min_silence_len_sec = max(0.01, min(60.0, float(min_silence_len_sec)))

    # Prevent filename injection
    safe_path = audio_path
    if audio_path.startswith("-"):
        safe_path = "./" + audio_path

    try:
        # Get duration and metadata first
        duration_sec, metadata = get_audio_metadata(audio_path)

        audio_info = metadata.get("audio", {})
        if not audio_info or duration_sec <= 0:
            import logging

            logger = logging.getLogger(__name__)
            logger.warning(f"Invalid audio file or ffprobe failed for {audio_path}")
            return AudioQualityResult(
                file=os.path.basename(audio_path),
                duration_sec=duration_sec,
                start_silence_sec=0.0,
                end_silence_sec=0.0,
                longest_middle_silence_sec=0.0,
                silence_periods_count=0,
                integrated_lufs=None,
                loudness_range_lu=None,
                true_peak_db=None,
                max_volume_db=0.0,
                mean_volume_db=0.0,
                histogram_0db=None,
                metadata=metadata,
                phase_correlation=None,
            )

        # Calculate phase correlation for stereo files
        phase_correlation = calculate_phase_correlation(audio_path, metadata)

        # Calculate waveform timeline if requested
        waveform_timeline = None
        if include_waveform:
            waveform_timeline = calculate_waveform_timeline(audio_path, duration_sec)

        # Run FFmpeg with all filters chained
        cmd = [
            "ffmpeg",
            "-v",
            "verbose",
            "-i",
            safe_path,
            "-af",
            f"silencedetect=noise={silence_thresh_db}dB:d={min_silence_len_sec},volumedetect,ebur128=peak=true",
            "-f",
            "null",
            "-",
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise ValueError(
                f"FFmpeg analysis failed for {audio_path} (exit code {result.returncode})"
            )

        output = result.stderr

        # Parse silence periods
        silence_periods = []
        silence_start = None

        for line in output.split("\n"):
            if "silencedetect" in line:
                start_match = re.search(r"silence_start:\s*([\d.]+)", line)
                if start_match:
                    silence_start = float(start_match.group(1))

                end_match = re.search(r"silence_end:\s*([\d.]+)", line)
                if end_match and silence_start is not None:
                    silence_end = float(end_match.group(1))
                    silence_periods.append((silence_start, silence_end))
                    silence_start = None

        # Calculate silence metrics
        start_silence_sec = 0
        end_silence_sec = 0
        longest_middle_silence_sec = 0

        if silence_periods:
            if silence_periods[0][0] < 0.1:
                start_silence_sec = silence_periods[0][1] - silence_periods[0][0]

            if abs(silence_periods[-1][1] - duration_sec) < 0.1:
                end_silence_sec = silence_periods[-1][1] - silence_periods[-1][0]

            middle_silences = []
            for i, (start, end) in enumerate(silence_periods):
                if i == 0 and start < 0.1:
                    continue
                if i == len(silence_periods) - 1 and abs(end - duration_sec) < 0.1:
                    continue
                middle_silences.append(end - start)

            longest_middle_silence_sec = max(middle_silences) if middle_silences else 0

        # Parse volume metrics
        max_volume_db = None
        mean_volume_db = None
        histogram_0db = None

        for line in output.split("\n"):
            if "max_volume:" in line:
                match = re.search(r"max_volume:\s*([-\d.]+)\s*dB", line)
                if match:
                    max_volume_db = float(match.group(1))

            if "mean_volume:" in line:
                match = re.search(r"mean_volume:\s*([-\d.]+)\s*dB", line)
                if match:
                    mean_volume_db = float(match.group(1))

            if "histogram_0db:" in line:
                match = re.search(r"histogram_0db:\s*(\d+)", line)
                if match:
                    histogram_0db = int(match.group(1))

        # Parse EBUR128 metrics and timeline data
        integrated_lufs = None
        loudness_range_lu = None
        true_peak_db = None
        lufs_timeline = []

        timeline_pattern = re.compile(
            r"t:\s*([\d.]+)\s+.*?M:\s*(-?[\d.]+)\s+S:\s*(-?[\d.]+)\s+I:\s*(-?[\d.]+)"
        )

        for line in output.split("\n"):
            if "Parsed_ebur128" in line and "t:" in line:
                match = timeline_pattern.search(line)
                if match:
                    t = float(match.group(1))
                    m = float(match.group(2))
                    s = float(match.group(3))
                    i = float(match.group(4))

                    if m > -70.0:
                        lufs_timeline.append(
                            {
                                "time": round(t, 2),
                                "momentary": round(m, 2),
                                "shortterm": round(s, 2),
                                "integrated": round(i, 2),
                            }
                        )
                        integrated_lufs = i

        for line in output.split("\n"):
            if "I:" in line and "LUFS" in line:
                match = re.search(r"I:\s*([-\d.]+)\s*LUFS", line)
                if match:
                    integrated_lufs = float(match.group(1))

            if "LRA:" in line and "LU" in line:
                match = re.search(r"LRA:\s*([-\d.]+)\s*LU", line)
                if match:
                    loudness_range_lu = float(match.group(1))

            if "Peak:" in line and "dBFS" in line:
                match = re.search(r"Peak:\s*([-\d.]+)\s*dBFS", line)
                if match:
                    true_peak_db = float(match.group(1))

        if max_volume_db is None or mean_volume_db is None:
            raise ValueError(f"Could not parse volume metrics from {audio_path}")

        return AudioQualityResult(
            file=os.path.basename(audio_path),
            duration_sec=duration_sec,
            start_silence_sec=start_silence_sec,
            end_silence_sec=end_silence_sec,
            longest_middle_silence_sec=longest_middle_silence_sec,
            silence_periods_count=len(silence_periods),
            integrated_lufs=integrated_lufs,
            loudness_range_lu=loudness_range_lu,
            true_peak_db=true_peak_db,
            max_volume_db=max_volume_db,
            mean_volume_db=mean_volume_db,
            histogram_0db=histogram_0db,
            metadata=metadata,
            phase_correlation=phase_correlation,
            lufs_timeline=lufs_timeline if lufs_timeline else None,
            waveform_timeline=waveform_timeline,
        )

    except Exception as e:
        import logging

        logger = logging.getLogger(__name__)
        logger.error(f"Audio analysis failed for {audio_path}: {e}", exc_info=True)
        return AudioQualityResult(
            file=os.path.basename(audio_path),
            duration_sec=0.0,
            start_silence_sec=0.0,
            end_silence_sec=0.0,
            longest_middle_silence_sec=0.0,
            silence_periods_count=0,
            integrated_lufs=None,
            loudness_range_lu=None,
            true_peak_db=None,
            max_volume_db=0.0,
            mean_volume_db=0.0,
            histogram_0db=None,
            metadata={},
            phase_correlation=None,
        )


def check_audio_quality(
    result: AudioQualityResult,
    lufs_min: float = LUFS_MIN,
    lufs_max: float = LUFS_MAX,
    start_silence_warning: float = 3.0,
    end_silence_warning: float = 3.0,
    middle_silence_warning: float = 5.0,
    clipping_samples_warning: int = 0,
    max_volume_warning: float = -0.1,
    true_peak_warning: float = -1.0,
    phase_correlation_warning: float = -0.5,
    loudness_range_warning: float = 8.0,
) -> list[QualityCheck]:
    """
    Evaluate AudioQualityResult against quality standards and return list of issues.

    Returns:
        List of QualityCheck objects describing any detected issues.
        Empty list if no issues found.
    """
    checks = []

    # Silence checks
    if result.start_silence_sec > start_silence_warning:
        checks.append(
            QualityCheck(
                check_type="silence",
                severity="warning",
                short_text=f"Long start silence ({result.start_silence_sec:.2f}s)",
                detailed_text=(
                    f"File begins with {result.start_silence_sec:.2f} seconds of silence, "
                    f"exceeding the recommended {start_silence_warning:.1f}-second threshold."
                ),
                value=result.start_silence_sec,
            )
        )

    if result.end_silence_sec > end_silence_warning:
        checks.append(
            QualityCheck(
                check_type="silence",
                severity="warning",
                short_text=f"Long end silence ({result.end_silence_sec:.2f}s)",
                detailed_text=(
                    f"File ends with {result.end_silence_sec:.2f} seconds of silence, "
                    f"exceeding the recommended {end_silence_warning:.1f}-second threshold."
                ),
                value=result.end_silence_sec,
            )
        )

    if result.longest_middle_silence_sec > middle_silence_warning:
        checks.append(
            QualityCheck(
                check_type="silence",
                severity="warning",
                short_text=f"Long middle silence ({result.longest_middle_silence_sec:.2f}s)",
                detailed_text=(
                    f"A {result.longest_middle_silence_sec:.2f}-second silence gap was detected "
                    f"within the audio, exceeding the {middle_silence_warning:.1f}-second threshold."
                ),
                value=result.longest_middle_silence_sec,
            )
        )

    # LUFS checks
    if result.integrated_lufs is not None:
        if result.integrated_lufs < lufs_min:
            checks.append(
                QualityCheck(
                    check_type="lufs",
                    severity="warning",
                    short_text=f"Too quiet ({result.integrated_lufs:.2f} LUFS)",
                    detailed_text=(
                        f"Integrated loudness is {result.integrated_lufs:.2f} LUFS, below the "
                        f"recommended {lufs_min:.1f} LUFS minimum."
                    ),
                    value=result.integrated_lufs,
                )
            )
        elif result.integrated_lufs > lufs_max:
            checks.append(
                QualityCheck(
                    check_type="lufs",
                    severity="warning",
                    short_text=f"Too loud ({result.integrated_lufs:.2f} LUFS)",
                    detailed_text=(
                        f"Integrated loudness is {result.integrated_lufs:.2f} LUFS, exceeding the "
                        f"recommended {lufs_max:.1f} LUFS maximum."
                    ),
                    value=result.integrated_lufs,
                )
            )

    # Loudness Range check
    if (
        result.loudness_range_lu is not None
        and result.loudness_range_lu > loudness_range_warning
    ):
        checks.append(
            QualityCheck(
                check_type="loudness_range",
                severity="warning",
                short_text=f"High loudness range ({result.loudness_range_lu:.1f} LU)",
                detailed_text=(
                    f"Loudness range is {result.loudness_range_lu:.1f} LU, exceeding the recommended "
                    f"{loudness_range_warning:.1f} LU maximum. Consider applying compression."
                ),
                value=result.loudness_range_lu,
            )
        )

    # True Peak check
    if result.true_peak_db is not None and result.true_peak_db > true_peak_warning:
        checks.append(
            QualityCheck(
                check_type="peak",
                severity="warning",
                short_text=f"True peak risk ({result.true_peak_db:.2f} dBFS)",
                detailed_text=(
                    f"True peak level of {result.true_peak_db:.2f} dBFS exceeds the recommended "
                    f"{true_peak_warning:.1f} dBFS maximum."
                ),
                value=result.true_peak_db,
            )
        )

    # Clipping checks
    if result.max_volume_db >= max_volume_warning:
        checks.append(
            QualityCheck(
                check_type="clipping",
                severity="error",
                short_text=f"Clipping detected (max: {result.max_volume_db:.2f} dB)",
                detailed_text=(
                    f"Maximum volume level of {result.max_volume_db:.2f} dB indicates hard clipping."
                ),
                value=result.max_volume_db,
            )
        )

    if (
        result.histogram_0db is not None
        and result.histogram_0db > clipping_samples_warning
    ):
        checks.append(
            QualityCheck(
                check_type="clipping",
                severity="error",
                short_text=f"{result.histogram_0db} samples clipped at 0dB",
                detailed_text=(
                    f"FFmpeg detected {result.histogram_0db} audio samples at exactly 0 dB, "
                    "indicating hard clipping."
                ),
                value=float(result.histogram_0db),
            )
        )

    # Phase correlation check
    if (
        result.phase_correlation is not None
        and result.phase_correlation < phase_correlation_warning
    ):
        checks.append(
            QualityCheck(
                check_type="phase",
                severity="error",
                short_text=f"Phase inverted channels (correlation: {result.phase_correlation:.2f})",
                detailed_text=(
                    f"Stereo channels are phase-inverted (correlation: {result.phase_correlation:.2f}). "
                    "When summed to mono, channels will cancel out."
                ),
                value=result.phase_correlation,
            )
        )

    return checks


def format_result_text(result: AudioQualityResult, checks: list[QualityCheck]) -> str:
    """Format analysis result as human-readable text."""
    lines = []

    lines.append(f"File: {result.file}")
    lines.append(f"Duration: {result.duration_sec:.2f} sec")

    lines.append("")
    lines.append("Silence Analysis:")
    lines.append(f"  Start silence: {result.start_silence_sec:.2f} sec")
    lines.append(f"  End silence: {result.end_silence_sec:.2f} sec")
    lines.append(
        f"  Longest middle silence: {result.longest_middle_silence_sec:.2f} sec"
    )
    lines.append(f"  Total periods: {result.silence_periods_count}")

    lines.append("")
    lines.append("LUFS Analysis:")
    if result.integrated_lufs is not None:
        lines.append(f"  Integrated LUFS: {result.integrated_lufs:.2f} LUFS")
    if result.loudness_range_lu is not None:
        lines.append(f"  Loudness Range: {result.loudness_range_lu:.2f} LU")
    if result.true_peak_db is not None:
        lines.append(f"  True Peak: {result.true_peak_db:.2f} dBFS")

    lines.append("")
    lines.append("Volume Analysis:")
    lines.append(f"  Max Volume: {result.max_volume_db:.2f} dB")
    lines.append(f"  Mean Volume: {result.mean_volume_db:.2f} dB")
    if result.histogram_0db is not None and result.histogram_0db > 0:
        lines.append(f"  Clipped Samples: {result.histogram_0db}")

    if "audio" in result.metadata:
        lines.append("")
        lines.append("Metadata:")
        audio = result.metadata["audio"]
        lines.append(f"  Codec: {audio.get('codec', 'N/A')}")
        lines.append(f"  Sample Rate: {audio.get('sample_rate', 'N/A')} Hz")
        lines.append(f"  Channels: {audio.get('channels', 'N/A')}")
        if audio.get("bit_rate"):
            lines.append(f"  Bit Rate: {int(audio['bit_rate']) // 1000} kbps")

    if result.phase_correlation is not None:
        lines.append("")
        lines.append("Phase Correlation:")
        lines.append(f"  Correlation: {result.phase_correlation:.3f}")
        if result.phase_correlation >= 0.9:
            lines.append("  Status: Excellent mono compatibility")
        elif result.phase_correlation >= 0.5:
            lines.append("  Status: Good")
        elif result.phase_correlation >= 0:
            lines.append("  Status: Acceptable (uncorrelated)")
        elif result.phase_correlation >= -0.5:
            lines.append("  Status: Warning (partially inverted)")
        else:
            lines.append("  Status: CRITICAL - Inverted phase!")

    lines.append("")
    lines.append("Quality Checks:")
    if checks:
        for check in checks:
            icon = "[ERROR]" if check.severity == "error" else "[WARN]"
            lines.append(f"  {icon} [{check.check_type.upper()}] {check.short_text}")
    else:
        lines.append("  [OK] No issues detected")

    return "\n".join(lines)


def format_result_json(result: AudioQualityResult, checks: list[QualityCheck]) -> str:
    """Format analysis result as JSON."""
    output = {
        "result": result.to_dict(),
        "checks": [c.to_dict() for c in checks],
    }
    return json.dumps(output, ensure_ascii=False, indent=2)
