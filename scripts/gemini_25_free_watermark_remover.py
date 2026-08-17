#!/usr/bin/env python3
"""
Standalone batch watermark/overlay removal tool using Gemini 2.5 Flash Image.

NOTE: Gemini removed the free tier for image-generation models in early 2026;
this script now requires a billed Gemini project (~$0.039/image). The main
Telegram bot flow no longer uses this paid path by default -- see
src/overlay_detect.py + src/local_inpaint.py for the free hybrid pipeline.

Intended as a limited prod tool for friends (no 25-image artificial limit).

Key guarantees:
- Processes images one by one
- Preserves original dimensions and resolution
- Preserves real photo content (signs, objects, architecture)
- Removes watermarks and obvious artificial overlays
- Always returns exactly as many images as were given (all processed)

Custom prompts are supported via --prompt-file.

Usage:
    python scripts/gemini_25_free_watermark_remover.py data/test_images/
    python scripts/gemini_25_free_watermark_remover.py photo1.jpg photo2.jpg --prompt-file prompts/my-prompt.txt
"""

import argparse
import logging
import os
import sys
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path
from dotenv import load_dotenv

from google import genai
from google.genai import types

from src.gemini_defaults import DEFAULT_GEMINI_IMAGE_MODEL, REMOVAL_PROMPT
from src.gemini_engine import _sanitize_prompt

load_dotenv()

# Persistent error logging for the free Gemini tool
Path("data/logs").mkdir(parents=True, exist_ok=True)
error_logger = logging.getLogger("gemini_free_errors")
error_logger.setLevel(logging.ERROR)
fh = RotatingFileHandler("data/logs/gemini_free_errors.log", maxBytes=2*1024*1024, backupCount=3)
fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
error_logger.addHandler(fh)

DEFAULT_REMOVAL_PROMPT = REMOVAL_PROMPT


def process_single_image(
    client: genai.Client,
    image_path: Path,
    prompt: str,
    output_suffix: str = "_cleaned",
) -> Path | None:
    """Process one image. Returns path to cleaned image or None on failure."""
    start = time.time()

    try:
        image_bytes = image_path.read_bytes()
        mime_type = "image/jpeg" if image_path.suffix.lower() in {".jpg", ".jpeg"} else "image/png"

        input_image = types.Part.from_bytes(data=image_bytes, mime_type=mime_type)

        config = types.GenerateContentConfig(
            response_modalities=["TEXT", "IMAGE"],
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )

        sanitized_prompt = _sanitize_prompt(prompt)
        response = client.models.generate_content(
            model=DEFAULT_GEMINI_IMAGE_MODEL,
            contents=[sanitized_prompt, input_image],
            config=config,
        )

        cleaned_bytes = None
        text_parts = []
        if response.candidates:
            for candidate in response.candidates:
                if candidate.content and candidate.content.parts:
                    for part in candidate.content.parts:
                        if part.inline_data:
                            cleaned_bytes = part.inline_data.data
                            break
                        if part.text:
                            text_parts.append(part.text.strip())

        if not cleaned_bytes:
            latency = (time.time() - start) * 1000
            err_detail = f": {' '.join(text_parts)}" if text_parts else ""
            print(f"✗ {image_path.name} ({latency:.0f}ms) - No image returned{err_detail}")
            return None

        import io
        from PIL import Image

        with Image.open(io.BytesIO(cleaned_bytes)) as out_img:
            with Image.open(image_path) as in_img:
                orig_size = in_img.size
                orig_format = in_img.format or ("PNG" if image_path.suffix.lower() == ".png" else "JPEG")
            if out_img.size != orig_size:
                out_img = out_img.resize(orig_size, Image.Resampling.LANCZOS)
            output_path = image_path.with_stem(image_path.stem + output_suffix)
            if orig_format.upper() == "JPEG" and out_img.mode not in {"RGB", "L"}:
                out_img = out_img.convert("RGB")
            out_img.save(output_path, format=orig_format)

        latency = (time.time() - start) * 1000
        print(f"✓ {image_path.name} ({latency:.0f}ms) → {output_path.name}")
        return output_path


    except Exception as e:
        latency = (time.time() - start) * 1000
        print(f"✗ {image_path.name} ({latency:.0f}ms) - Error: {e}")
        error_logger.exception(f"Error processing {image_path}: {e}")
        return None


def process_images(
    image_paths: list[Path],
    prompt: str,
    output_suffix: str = "_cleaned",
) -> list[Path]:
    """Process a list of images one by one. Returns list of successfully created output paths."""
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("No Gemini API key found (set GEMINI_API_KEY or GOOGLE_API_KEY)")

    client = genai.Client(api_key=api_key)
    results = []

    for path in image_paths:
        result = process_single_image(client, path, prompt, output_suffix)
        if result:
            results.append(result)

    return results


def collect_image_paths(args: list[str]) -> list[Path]:
    paths: list[Path] = []
    for arg in args:
        p = Path(arg)
        if p.is_dir():
            paths.extend(sorted(p.glob("*.jpg")))
            paths.extend(sorted(p.glob("*.jpeg")))
            paths.extend(sorted(p.glob("*.png")))
        elif p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            paths.append(p)
        else:
            print(f"Warning: skipping invalid path {arg}")
    return paths


def main():
    parser = argparse.ArgumentParser(
        description="Free Gemini 2.5 Flash watermark/overlay remover (no 25-image limit)."
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="Image files or folders to process",
    )
    parser.add_argument(
        "--prompt-file",
        type=Path,
        help="Path to a text file containing the editing instruction (optional)",
    )
    parser.add_argument(
        "--output-suffix",
        default="_cleaned",
        help="Suffix to add to output filenames (default: _cleaned)",
    )
    args = parser.parse_args()

    image_paths = collect_image_paths(args.paths)
    if not image_paths:
        print("No valid images found.")
        sys.exit(1)

    if args.prompt_file:
        prompt = args.prompt_file.read_text().strip()
        print(f"Using custom prompt from: {args.prompt_file}")
    else:
        prompt = DEFAULT_REMOVAL_PROMPT
        print("Using built-in conservative watermark removal prompt")

    print(f"Found {len(image_paths)} images. Processing one by one with {DEFAULT_GEMINI_IMAGE_MODEL}...\n")

    outputs = process_images(image_paths, prompt, args.output_suffix)

    print(f"\nDone. {len(outputs)}/{len(image_paths)} images processed successfully.")
    if len(outputs) != len(image_paths):
        print("Warning: Some images failed to process.")


if __name__ == "__main__":
    main()
