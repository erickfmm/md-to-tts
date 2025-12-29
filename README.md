# Markdown a audio (español)

Convierte todos los archivos Markdown del directorio `modificacion1/` en audios WAV, dividiendo el texto por secciones (`#`) y párrafos. Cada párrafo se sintetiza con un modelo TTS y se unen los fragmentos con una pausa corta para obtener un solo audio por archivo.

## Modelos soportados

- `mms` → [facebook/mms-tts-spa](https://huggingface.co/facebook/mms-tts-spa) (recomendado, español nativo).
- `kokoro` → [hexgrad/Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M).
- `chatterbox` → [ResembleAI/chatterbox](https://huggingface.co/ResembleAI/chatterbox).
- `vibevoice` → [microsoft/VibeVoice-Realtime-0.5B](https://huggingface.co/microsoft/VibeVoice-Realtime-0.5B) *(placeholder: requiere su stack de streaming oficial).* 
- `cosyvoice` → [FunAudioLLM/Fun-CosyVoice3-0.5B-2512](https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512) *(placeholder: integra CosyVoice completo o servidor dedicado).* 

Para uso inmediato en español, elija `--engine mms`. `kokoro` y `chatterbox` funcionan, pero sus descargas son pesadas y pueden requerir GPU para buen rendimiento.

## Requisitos

- Python 3.10+
- FFmpeg instalado (necesario para `pydub`).
- Conexión a internet para descargar pesos de los modelos la primera vez.

## Instalación rápida

```bash
pip install --upgrade pip
pip install -e .
```

Si falta FFmpeg (Linux/Docker):

```bash
sudo apt-get update && sudo apt-get install -y ffmpeg
```

## Uso

```bash
# Convierte todos los .md de modificacion1/ usando MMS (español)
md-tts --input-dir modificacion1 --output-dir output_audio --engine mms --pause-ms 500

# Usar Kokoro (multilingüe), intentando auto-voz
md-tts --engine kokoro --input-dir modificacion1 --output-dir output_audio_kokoro

# Usar Chatterbox (multilingüe) en GPU
md-tts --engine chatterbox --device cuda
```

- Los WAV resultantes se guardan en `--output-dir` con el mismo nombre base del Markdown.
- `--pause-ms` controla el silencio entre párrafos (por defecto 500 ms).
- `--device` acepta `cpu` o `cuda` (si está disponible). Algunos modelos requieren GPU para velocidad razonable.

## Estructura

```
src/md_tts/
  parser.py       # Divide Markdown en secciones y párrafos
  tts.py          # Motores TTS
  audio_utils.py  # Conversión y unión de audio
  cli.py          # Punto de entrada de línea de comandos
project.toml      # Metadatos y dependencias
Dockerfile        # Imagen lista para ejecutar
```

## Notas y limitaciones

- VibeVoice y CosyVoice aparecen como placeholders: el CLI informa que la integración no está implementada aún y sugiere usar los runtimes oficiales.
- Los modelos son pesados; la primera ejecución descargará varios cientos de MB.
- El proyecto genera WAV (16-bit PCM). Ajusta a otro formato exportando con `pydub` si lo necesitas.
- El texto se sintetiza en el orden original: encabezado seguido de sus párrafos, separados por silencios.

## Docker

```bash
docker build -t md-tts .
docker run --rm -v $(pwd):/app md-tts md-tts --input-dir modificacion1 --output-dir output_audio --engine mms
```

## Licencias de modelos

Revisa las licencias de cada modelo antes de uso en producción:
- facebook/mms-tts-spa → CC-BY-NC 4.0 (no comercial).
- hexgrad/Kokoro-82M → Apache 2.0.
- ResembleAI/chatterbox → MIT con watermarking.
- microsoft/VibeVoice-Realtime-0.5B → MIT (uso responsable, streaming).
- FunAudioLLM/Fun-CosyVoice3-0.5B-2512 → Apache 2.0.
