"""Pure post-processing helpers for the transcription pipeline.

Kept free of torch/mlx imports so they stay cheap to import and test.
Thresholds were fitted against hand-corrected transcripts.
"""

import re
from typing import Any

# A pyannote gap shorter than this is not worth a Whisper call.
GAP_MIN_DURATION_SEC = 1.0
# Speech found inside a gap is transcribed in chunks of at most one Whisper window:
# longer chunks brought Whisper's long-form failure modes (loops, hallucinated passages) back.
GAP_VAD_MAX_CHUNK_SEC = 30.0
# Whisper segments from gap audio are kept only when Whisper itself is confident.
GAP_NO_SPEECH_MAX = 0.6
GAP_AVG_LOGPROB_MIN = -0.6
# A 1..8-word phrase repeated this many times in a row is a decoder loop, not speech.
REPEAT_PHRASE_MAX_WORDS = 8
REPEAT_COLLAPSE_COUNT = 3
REPEAT_DROP_COUNT = 4

_NON_WORD = re.compile(r"[^\w]+", re.UNICODE)


def _key(word: str) -> str:
    return _NON_WORD.sub("", word.lower())


def gaps_between(
    speech_segments: list[dict[str, Any]], total_duration: float, min_duration: float
) -> list[tuple[float, float]]:
    """Time ranges not covered by any speech segment, at least ``min_duration`` long."""
    gaps: list[tuple[float, float]] = []
    current = 0.0
    for seg in sorted(speech_segments, key=lambda s: s["start"]):
        if seg["start"] - current >= min_duration:
            gaps.append((current, seg["start"]))
        current = max(current, seg["end"])
    if total_duration - current >= min_duration:
        gaps.append((current, total_duration))
    return gaps


def group_speech_pieces(
    pieces: list[dict[str, Any]], sample_rate: int, max_chunk_sec: float
) -> list[list[dict[str, Any]]]:
    """Group consecutive VAD pieces ({'start','end'} in samples) into chunks of at most ``max_chunk_sec``.

    A single piece longer than the limit is kept whole as its own chunk: cutting inside
    continuous speech hurt more than a long window did in measurements. (faster-whisper's
    ``collect_chunks`` drops the piece metadata in that case, which is why this exists.)
    """
    limit = max_chunk_sec * sample_rate
    groups: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_len = 0
    for piece in pieces:
        length = piece["end"] - piece["start"]
        if current and current_len + length > limit:
            groups.append(current)
            current, current_len = [], 0
        current.append(piece)
        current_len += length
    if current:
        groups.append(current)
    return groups


def max_consecutive_repeats(
    tokens: list[str], max_phrase_words: int = REPEAT_PHRASE_MAX_WORDS
) -> int:
    """Largest number of times any 1..max_phrase_words phrase occurs back to back."""
    keys = [_key(t) for t in tokens]
    best = 1
    for n in range(1, max_phrase_words + 1):
        i = 0
        while i + n <= len(keys):
            count = 1
            while keys[i : i + n] == keys[i + n * count : i + n * (count + 1)]:
                count += 1
            best = max(best, count)
            i += n * count if count > 1 else 1
    return best


def collapse_repeated_phrases(
    items: list[Any],
    key=lambda item: item,
    max_phrase_words: int = REPEAT_PHRASE_MAX_WORDS,
    min_count: int = REPEAT_COLLAPSE_COUNT,
) -> list[Any]:
    """Keep one occurrence of any phrase that repeats ``min_count`` or more times in a row.

    Works on any item list (word dicts or plain strings) through ``key``. Longest
    phrases first, so ``a b a b a b`` collapses to ``a b`` rather than to ``a b a b``.
    """
    items = list(items)
    keys = [_key(key(it)) for it in items]
    changed = True
    while changed:
        changed = False
        for n in range(max_phrase_words, 0, -1):
            i = 0
            while i + n * min_count <= len(keys):
                count = 1
                while keys[i : i + n] == keys[i + n * count : i + n * (count + 1)]:
                    count += 1
                if count >= min_count:
                    del items[i + n : i + n * count]
                    del keys[i + n : i + n * count]
                    changed = True
                i += 1
    return items


def collapse_repeated_text(text: str) -> str:
    return " ".join(collapse_repeated_phrases(text.split()))


def whisper_segment_is_reliable(segment: dict[str, Any]) -> bool:
    """Filter for Whisper output on audio that pyannote did not call speech."""
    no_speech = segment.get("no_speech_prob")
    if no_speech is not None and no_speech > GAP_NO_SPEECH_MAX:
        return False
    logprob = segment.get("avg_logprob")
    if logprob is not None and logprob < GAP_AVG_LOGPROB_MIN:
        return False
    text = (segment.get("text") or "").strip()
    if not text:
        return False
    return max_consecutive_repeats(text.split()) < REPEAT_DROP_COUNT


def join_words(words: list[dict[str, Any]]) -> str:
    """Segment text from its word list (Whisper words carry their own punctuation)."""
    return " ".join(w["word"].strip() for w in words if w.get("word", "").strip())
