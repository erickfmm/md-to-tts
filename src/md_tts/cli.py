from __future__ import annotations

import argparse
import logging
import tempfile
from pathlib import Path
from typing import List

from pydub import AudioSegment
from tqdm import tqdm

from . import parser as md_parser
from .audio_utils import concatenate_with_pause
from .tts import build_engine

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("md_tts")


def process_markdown_file(
    md_file: Path,
    output_dir: Path,
    engine_name: str,
    pause_ms: int,
    device: str | None,
    cosyvoice_model_dir: Path | None = None,
    cosyvoice_prompt_wav: Path | None = None,
    save_fragments: bool = False,
) -> Path:
    paragraphs = md_parser.split_markdown(md_file)
    logger.info("%s -> %d fragmentos", md_file.name, len(paragraphs))

    engine = build_engine(
        engine_name,
        device=device,
        cosyvoice_model_dir=cosyvoice_model_dir,
        cosyvoice_prompt_wav=cosyvoice_prompt_wav,
    )
    segments: List[AudioSegment] = []

    # Crear carpeta temporal para fragmentos si se solicita
    temp_dir = None
    if save_fragments:
        temp_dir = output_dir / "fragments" / md_file.stem
        temp_dir.mkdir(parents=True, exist_ok=True)

    for para in tqdm(paragraphs, desc=f"{md_file.name}", unit="p"):
        text = para.text.strip()
        if not text:
            continue
        try:
            seg = engine.synthesize(text)
            segments.append(seg)
            
            # Guardar fragmento individual si se solicita
            if temp_dir:
                fragment_name = f"{para.index:04d}_{para.section[:30] if para.section else 'intro'}.wav"
                fragment_path = temp_dir / fragment_name
                seg.export(fragment_path, format="wav")
        except Exception as exc:  # pragma: no cover - externo
            logger.error("Error sintetizando párrafo %s: %s", para.index, exc)
            continue

    if not segments:
        raise RuntimeError(f"No se generaron segmentos de audio para {md_file}")

    combined = concatenate_with_pause(segments, pause_ms=pause_ms)
    output_path = output_dir / f"{md_file.stem}.wav"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.export(output_path, format="wav")
    return output_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convierte Markdown a audio en español.")
    repo_root = Path(__file__).resolve().parent.parent.parent
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=repo_root / "modificacion1",
        help="Directorio con archivos .md",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "output_audio",
        help="Directorio donde guardar los .wav",
    )
    parser.add_argument(
        "--engine",
        choices=["mms", "kokoro", "chatterbox", "vibevoice", "cosyvoice"],
        default="mms",
        help="Motor TTS a usar",
    )
    parser.add_argument(
        "--pause-ms",
        type=int,
        default=500,
        help="Pausa (ms) entre párrafos",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="cpu o cuda (si disponible)",
    )
    parser.add_argument(
        "--cosyvoice-model-dir",
        type=Path,
        default=None,
        help="Ruta local del modelo CosyVoice (opcional; si no, descarga).",
    )
    parser.add_argument(
        "--cosyvoice-prompt-wav",
        type=Path,
        default=None,
        help="WAV de referencia de voz para CosyVoice (opcional).",
    )
    parser.add_argument(
        "--save-fragments",
        action="store_true",
        help="Guardar fragmentos individuales en output_dir/fragments/",
    )
    return parser


def main(argv: List[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)

    input_dir: Path = args.input_dir
    output_dir: Path = args.output_dir
    if not input_dir.exists():
        raise FileNotFoundError(f"No existe el directorio de entrada: {input_dir}")

    for md_file in md_parser.iter_markdown_files(input_dir):
        out_path = process_markdown_file(
            md_file=md_file,
            output_dir=output_dir,
            engine_name=args.engine,
            pause_ms=args.pause_ms,
            device=args.device,
            cosyvoice_model_dir=args.cosyvoice_model_dir,
            cosyvoice_prompt_wav=args.cosyvoice_prompt_wav,
            save_fragments=args.save_fragments,
        )
        logger.info("Audio generado: %s", out_path)
        if args.save_fragments:
            logger.info("Fragmentos guardados en: %s/fragments/%s", output_dir, md_file.stem)


if __name__ == "__main__":
    main()
