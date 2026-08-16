#!/usr/bin/env python3
"""Run a small Gemini upscale-and-overlay-removal experiment on local images."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image

from src.gemini_defaults import DEFAULT_GEMINI_IMAGE_MODEL

PROMPT = """Edit this real-estate photo in one pass.

1. Remove artificial watermarks, brokerage marks, labels, text overlays, and post-added graphics only. Do not remove real architectural text, building signage, unit numbers, entrance signs, or physical scene details.
2. Make the photo look naturally higher-resolution and cleaner, as if it had been captured at a larger size with a better camera. Reduce compression artifacts, blur, ringing, jagged edges, and upscaling artifacts.
3. Preserve the exact scene, perspective, composition, lighting, colors, and all real-world content.
4. Preserve the original aspect ratio. You may return a larger image if it improves quality, but do not crop, stretch, pad, rotate, or change the framing.
5. Return only the edited image."""


def _mime_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix == ".webp":
        return "image/webp"
    return "image/png"


def _extension(mime_type: str | None) -> str:
    if mime_type == "image/jpeg":
        return ".jpg"
    if mime_type == "image/webp":
        return ".webp"
    return ".png"


def run(paths: list[Path], output_dir: Path, model: str) -> list[dict]:
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("No GEMINI_API_KEY or GOOGLE_API_KEY set")

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "prompt.txt").write_text(PROMPT, encoding="utf-8")
    client = genai.Client(api_key=api_key)
    manifest = []

    for path in paths:
        with Image.open(path) as image:
            input_size = image.size
            input_format = image.format

        part = types.Part.from_bytes(data=path.read_bytes(), mime_type=_mime_type(path))
        config = types.GenerateContentConfig(
            response_modalities=["TEXT", "IMAGE"],
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )
        response = client.models.generate_content(
            model=model,
            contents=[PROMPT, part],
            config=config,
        )

        output_path = None
        for candidate in response.candidates or []:
            if not candidate.content:
                continue
            for response_part in candidate.content.parts or []:
                inline_data = response_part.inline_data
                if not inline_data:
                    continue
                output_path = output_dir / f"{path.stem}__gemini-natural-upscale{_extension(inline_data.mime_type)}"
                output_path.write_bytes(inline_data.data)
                break
            if output_path:
                break

        if not output_path:
            raise RuntimeError(f"Gemini returned no image for {path}")

        with Image.open(output_path) as image:
            output_size = image.size
            output_format = image.format

        entry = {
            "input": str(path),
            "input_size": input_size,
            "input_format": input_format,
            "output": str(output_path),
            "output_size": output_size,
            "output_format": output_format,
            "model": model,
        }
        manifest.append(entry)
        print(f"{path.name} {input_size} -> {output_path} {output_size} {output_format}")

    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_GEMINI_IMAGE_MODEL)
    args = parser.parse_args()
    run(args.paths, args.output_dir, args.model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
