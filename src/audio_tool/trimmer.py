"""
Audio Trimming Library

Automatically trims silence from audio files or applies manual start/end time trimming.
Supports removing segments with crossfade for seamless editing.
"""

import logging
import os
import subprocess
from typing import Optional

from audio_tool.analyzer import AudioQualityResult, analyze_audio_quality, check_audio_quality

log = logging.getLogger(__name__)


def detect_trim_points(
    audio_path: str,
    silence_thresh_db: float = -40.0,
    wanted_silence_start: float = 0.5,
    wanted_silence_end: float = 2.0,
) -> tuple[float, float]:
    """
    Analyze audio file and detect recommended trim points based on silence.

    Args:
        audio_path: Path to audio file
        silence_thresh_db: Silence threshold in dB (default: -40.0)
        wanted_silence_start: Desired silence to keep at start in seconds (default: 0.5)
        wanted_silence_end: Desired silence to keep at end in seconds (default: 2.0)

    Returns:
        Tuple of (start_time, end_time) in seconds
    """
    log.debug(f'Analyzing audio file: {audio_path}')
    result = analyze_audio_quality(audio_path, silence_thresh_db=silence_thresh_db)

    log.debug('Detected silence:')
    log.debug(f'  Start silence: {result.start_silence_sec:.2f} sec')
    log.debug(f'  End silence: {result.end_silence_sec:.2f} sec')
    log.debug(f'  Duration: {result.duration_sec:.2f} sec')

    # Calculate trim points, keeping wanted silence
    if result.start_silence_sec > 0.1:
        start_time = max(0, result.start_silence_sec - wanted_silence_start)
    else:
        start_time = 0

    if result.end_silence_sec > 0.1:
        end_time = min(result.duration_sec, result.duration_sec - result.end_silence_sec + wanted_silence_end)
    else:
        end_time = result.duration_sec

    trimmed_duration = end_time - start_time

    log.debug('Recommended trim points:')
    log.debug(f'  Start: {start_time:.2f} sec (keeping {wanted_silence_start:.2f} sec silence)')
    log.debug(f'  End: {end_time:.2f} sec (keeping {wanted_silence_end:.2f} sec silence)')
    log.debug(f'  New duration: {trimmed_duration:.2f} sec')
    log.debug(f'  Removed: {result.duration_sec - trimmed_duration:.2f} sec')

    return start_time, end_time


def trim_audio(
    input_path: str,
    output_path: str,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    silence_thresh_db: float = -40.0,
    auto_detect: bool = True,
    wanted_silence_start: float = 0.5,
    wanted_silence_end: float = 2.0,
) -> AudioQualityResult:
    """
    Trim audio file by removing silence or using manual time points.

    Args:
        input_path: Path to input audio file
        output_path: Path to output audio file
        start_time: Manual start time in seconds (None for auto-detect)
        end_time: Manual end time in seconds (None for auto-detect)
        silence_thresh_db: Silence threshold for auto-detection (default: -40.0)
        auto_detect: Whether to auto-detect trim points from silence
        wanted_silence_start: Desired silence to keep at start in seconds (default: 0.5)
        wanted_silence_end: Desired silence to keep at end in seconds (default: 2.0)

    Returns:
        AudioQualityResult of the output file
    """
    if not os.path.exists(input_path):
        raise ValueError(f'Input file not found: {input_path}')

    # Determine trim points
    if auto_detect or (start_time is None and end_time is None):
        detected_start, detected_end = detect_trim_points(
            input_path, silence_thresh_db, wanted_silence_start, wanted_silence_end
        )
        start_time = start_time if start_time is not None else detected_start
        end_time = end_time if end_time is not None else detected_end

    if start_time is None or end_time is None:
        raise ValueError('Either enable auto_detect or provide both start_time and end_time')

    if start_time < 0 or end_time <= start_time:
        raise ValueError(f'Invalid trim times: start={start_time}, end={end_time}')

    duration = end_time - start_time

    log.debug('Trimming audio:')
    log.debug(f'  Input: {input_path}')
    log.debug(f'  Output: {output_path}')
    log.debug(f'  Start: {start_time:.2f} sec')
    log.debug(f'  Duration: {duration:.2f} sec')

    # Create output directory if needed
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Trim using FFmpeg - try copy first, then re-encode if needed
    cmd = [
        'ffmpeg',
        '-i',
        input_path,
        '-ss',
        str(start_time),
        '-t',
        str(duration),
        '-c:a',
        'copy',
        '-y',
        output_path,
    ]

    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError:
        log.debug('  Codec copy failed, re-encoding...')
        # Determine codec based on extension
        output_ext = os.path.splitext(output_path)[1].lower()
        if output_ext == '.mp3':
            cmd[7] = 'libmp3lame'
            cmd.extend(['-q:a', '0'])
        elif output_ext == '.wav':
            cmd[7] = 'pcm_s16le'
        elif output_ext == '.flac':
            cmd[7] = 'flac'
        else:
            cmd[7] = 'libmp3lame'
            cmd.extend(['-q:a', '0'])
        subprocess.run(cmd, capture_output=True, text=True, check=True)

    log.debug('  Trimming complete!')

    # Analyze output file
    output_result = analyze_audio_quality(output_path, silence_thresh_db=silence_thresh_db)

    log.debug('Output file analysis:')
    log.debug(f'  Duration: {output_result.duration_sec:.2f} sec')
    log.debug(f'  Start silence: {output_result.start_silence_sec:.2f} sec')
    log.debug(f'  End silence: {output_result.end_silence_sec:.2f} sec')

    return output_result


def remove_segments_with_crossfade(
    input_path: str,
    output_path: str,
    segments_to_remove: list[tuple[float, float]],
    crossfade_duration: float = 0.03,
) -> AudioQualityResult:
    """
    Remove segments from audio file and join remaining parts with crossfade.

    Useful for removing unwanted sections like coach instructions,
    word repetitions, coughs, background noise, etc.

    Args:
        input_path: Path to input audio file
        output_path: Path to output audio file
        segments_to_remove: List of (start_time, end_time) tuples in seconds to remove
        crossfade_duration: Duration of crossfade in seconds (default: 0.03 = 30ms)

    Returns:
        AudioQualityResult of the output file

    Example:
        # Remove two segments: 5.2-7.8s and 15.0-16.5s
        remove_segments_with_crossfade(
            'input.mp3',
            'output.mp3',
            [(5.2, 7.8), (15.0, 16.5)],
            crossfade_duration=0.03
        )
    """
    if not os.path.exists(input_path):
        raise ValueError(f'Input file not found: {input_path}')

    if not segments_to_remove:
        raise ValueError('No segments to remove provided')

    # Sort segments by start time
    segments_to_remove = sorted(segments_to_remove, key=lambda x: x[0])

    # Validate segments don't overlap
    for i in range(len(segments_to_remove) - 1):
        if segments_to_remove[i][1] > segments_to_remove[i + 1][0]:
            raise ValueError(f'Overlapping segments: {segments_to_remove[i]} and {segments_to_remove[i + 1]}')

    # Get input duration
    input_result = analyze_audio_quality(input_path)
    total_duration = input_result.duration_sec

    log.info(f'Removing {len(segments_to_remove)} segments from audio ({total_duration:.2f}s)')
    for i, (start, end) in enumerate(segments_to_remove):
        log.info(f'  Segment {i + 1}: {start:.3f}s - {end:.3f}s ({end - start:.3f}s)')

    # Calculate segments to KEEP (inverse of segments to remove)
    segments_to_keep = []
    current_pos = 0.0

    for remove_start, remove_end in segments_to_remove:
        if remove_start < 0 or remove_end > total_duration:
            raise ValueError(f'Segment {remove_start}-{remove_end} is outside audio duration {total_duration}')
        if remove_end <= remove_start:
            raise ValueError(f'Invalid segment: end ({remove_end}) must be greater than start ({remove_start})')

        if remove_start > current_pos:
            segments_to_keep.append((current_pos, remove_start))

        current_pos = remove_end

    if current_pos < total_duration:
        segments_to_keep.append((current_pos, total_duration))

    if not segments_to_keep:
        raise ValueError('No audio would remain after removing all segments')

    log.info(f'Keeping {len(segments_to_keep)} segments:')
    for i, (start, end) in enumerate(segments_to_keep):
        log.info(f'  Part {i + 1}: {start:.3f}s - {end:.3f}s ({end - start:.3f}s)')

    # Create output directory if needed
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # If only one segment remains, just trim it
    if len(segments_to_keep) == 1:
        start, end = segments_to_keep[0]
        return trim_audio(
            input_path=input_path,
            output_path=output_path,
            start_time=start,
            end_time=end,
            auto_detect=False,
        )

    # Build FFmpeg filter_complex for removing segments with crossfade
    filter_parts = []
    for i, (start, end) in enumerate(segments_to_keep):
        filter_parts.append(f'[0:a]atrim=start={start:.6f}:end={end:.6f},asetpts=PTS-STARTPTS[seg{i}]')

    # Chain acrossfade filters between segments
    current_stream = '[seg0]'
    for i in range(1, len(segments_to_keep)):
        next_stream = f'[seg{i}]'
        output_stream = f'[cf{i}]' if i < len(segments_to_keep) - 1 else '[aout]'
        filter_parts.append(
            f'{current_stream}{next_stream}acrossfade=d={crossfade_duration:.6f}:c1=tri:c2=tri{output_stream}'
        )
        current_stream = output_stream

    filter_complex = ';'.join(filter_parts)

    # Determine output codec based on file extension
    output_ext = os.path.splitext(output_path)[1].lower()
    if output_ext == '.mp3':
        codec_params = ['-c:a', 'libmp3lame', '-q:a', '0']
    elif output_ext == '.wav':
        codec_params = ['-c:a', 'pcm_s16le']
    elif output_ext == '.flac':
        codec_params = ['-c:a', 'flac']
    else:
        codec_params = ['-c:a', 'libmp3lame', '-q:a', '0']

    cmd = [
        'ffmpeg',
        '-i',
        input_path,
        '-filter_complex',
        filter_complex,
        '-map',
        '[aout]',
        *codec_params,
        '-y',
        output_path,
    ]

    log.info('Running FFmpeg with acrossfade filter...')
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        log.error(f'FFmpeg error: {result.stderr}')
        raise subprocess.CalledProcessError(result.returncode, cmd, result.stdout, result.stderr)

    log.info('Segment removal complete!')

    # Analyze output file
    output_result = analyze_audio_quality(output_path)

    log.info(f'Output duration: {output_result.duration_sec:.2f}s')
    total_removed = sum(end - start for start, end in segments_to_remove)
    log.info(f'Total removed: {total_removed:.2f}s')

    return output_result
