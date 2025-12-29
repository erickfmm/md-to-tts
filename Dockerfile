FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Dependencias del sistema (ffmpeg para pydub, espeak-ng para kokoro español)
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg espeak-ng && \
    rm -rf /var/lib/apt/lists/*

# Actualizar pip primero
RUN python3 -m pip install --upgrade pip

# Copiar solo archivos de dependencias y instalarlas (aprovecha cache de Docker)
COPY pyproject.toml /app/
RUN python3 -m pip install --no-cache-dir \
    "transformers>=4.46.0" \
    "accelerate>=0.30.0" \
    "huggingface_hub>=0.24.0" \
    "torch>=2.1.0" \
    "torchaudio>=2.1.0" \
    "pydub>=0.25.1" \
    "soundfile>=0.12.1" \
    "numpy>=1.25.0" \
    "tqdm>=4.66.1" \
    "kokoro>=0.9.2" \
    "chatterbox-tts>=0.1.6" 
# Ahora copiar el código fuente e instalar en modo editable
COPY README.md /app/
COPY src /app/src
RUN python3 -m pip install --no-cache-dir -e .

ENTRYPOINT ["md-tts"]
