from __future__ import annotations

import io
from typing import Iterable, List

import numpy as np
from pydub import AudioSegment
import soundfile as sf


DEFAULT_SAMPLE_RATE = 24000


def numpy_to_segment(audio: np.ndarray, sample_rate: int) -> AudioSegment:
    """Convierte un array numpy (float -1..1) en AudioSegment."""

    if audio.ndim > 1:
        audio = audio.squeeze()
    # Normalizar a int16
    audio16 = np.clip(audio, -1.0, 1.0)
    audio16 = (audio16 * 32767.0).astype(np.int16)
    buf = io.BytesIO()
    sf.write(buf, audio16, samplerate=sample_rate, format="WAV", subtype="PCM_16")
    buf.seek(0)
    return AudioSegment.from_file(buf, format="wav")


def concatenate_with_pause(segments: Iterable[AudioSegment], pause_ms: int) -> AudioSegment:
    pause = AudioSegment.silent(duration=pause_ms)
    combined = AudioSegment.silent(duration=0)
    for idx, seg in enumerate(segments):
        if idx > 0:
            combined += pause
        combined += seg
    return combined


def merge_numpy(chunks: List[np.ndarray]) -> np.ndarray:
    if not chunks:
        return np.array([], dtype=np.float32)
    return np.concatenate(chunks).astype(np.float32)
