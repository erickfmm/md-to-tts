from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
from pydub import AudioSegment

from .audio_utils import merge_numpy, numpy_to_segment

logger = logging.getLogger(__name__)


class TtsEngine:
    name: str

    def synthesize(self, text: str) -> AudioSegment:  # pragma: no cover - interface
        raise NotImplementedError


@dataclass
class MmsSpanishEngine(TtsEngine):
    device: str = "cpu"

    def __post_init__(self):
        from transformers import AutoTokenizer, VitsModel
        import torch

        self.tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-spa")
        self.model = VitsModel.from_pretrained("facebook/mms-tts-spa")
        self.model.to(self.device)
        self.name = "facebook/mms-tts-spa"

    def synthesize(self, text: str) -> AudioSegment:
        import torch

        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output = self.model(**inputs).waveform.squeeze(0).cpu().numpy()
        return numpy_to_segment(output, sample_rate=self.model.config.sampling_rate)


@dataclass
class KokoroEngine(TtsEngine):
    lang_code: str = "a"  # auto-detect / multilingual

    def __post_init__(self):
        from kokoro import KPipeline

        self.pipeline = KPipeline(lang_code=self.lang_code)
        self.name = "hexgrad/Kokoro-82M"

    def synthesize(self, text: str) -> AudioSegment:
        chunks = []
        for _, _, audio in self.pipeline(text, voice="af_heart"):
            chunks.append(audio)
        merged = merge_numpy([np.array(c, dtype=np.float32) for c in chunks])
        if merged.size == 0:
            raise RuntimeError("No se produjo audio con Kokoro")
        return numpy_to_segment(merged, sample_rate=24000)


@dataclass
class ChatterboxEngine(TtsEngine):
    device: Optional[str] = None

    def __post_init__(self):
        from chatterbox.tts import ChatterboxTTS

        self.model = ChatterboxTTS.from_pretrained(device=self.device or "cpu")
        self.name = "ResembleAI/chatterbox"

    def synthesize(self, text: str) -> AudioSegment:
        wav = self.model.generate(text)
        data = wav.cpu().numpy() if hasattr(wav, "cpu") else np.asarray(wav)
        return numpy_to_segment(data.squeeze(), sample_rate=self.model.sr)


@dataclass
class PlaceholderEngine(TtsEngine):
    reason: str
    model_id: str

    def synthesize(self, text: str) -> AudioSegment:
        raise NotImplementedError(
            f"El modelo {self.model_id} aún no está implementado en este script: {self.reason}"
        )


ENGINE_REGISTRY = {
    "mms": MmsSpanishEngine,
    "kokoro": KokoroEngine,
    "chatterbox": ChatterboxEngine,
    "vibevoice": lambda **_: PlaceholderEngine(
        model_id="microsoft/VibeVoice-Realtime-0.5B",
        reason="La integración requiere el stack oficial de streaming y no está empaquetada en pip aún.",
    ),
    "cosyvoice": lambda **_: PlaceholderEngine(
        model_id="FunAudioLLM/Fun-CosyVoice3-0.5B-2512",
        reason="La inferencia completa necesita el framework CosyVoice; se recomienda usar su servidor dedicado.",
    ),
}


def build_engine(name: str, **kwargs) -> TtsEngine:
    key = name.lower()
    if key not in ENGINE_REGISTRY:
        raise KeyError(f"Engine desconocido: {name}. Opciones: {list(ENGINE_REGISTRY)}")
    engine_cls = ENGINE_REGISTRY[key]
    return engine_cls(**kwargs)
