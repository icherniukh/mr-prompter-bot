#!/usr/bin/env python3
"""Offline-first harness for visual review of image upscaling strategies.

Default runs are local-only. Gemini repair/finish strategies are included in
the manifest as skipped unless --enable-gemini is passed and an API key exists.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from src.gemini_defaults import DEFAULT_GEMINI_IMAGE_MODEL
except Exception:  # pragma: no cover - keeps the script usable as a loose file
    DEFAULT_GEMINI_IMAGE_MODEL = "gemini-2.5-flash-image"


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
DEFAULT_FACTORS = [2.0, 3.0, 4.0]
DEFAULT_RESAMPLERS = ["lanczos", "bicubic"]
DEFAULT_PARTIAL_FACTORS = [1.5, 2.0]
DEFAULT_OUTPUT_FORMAT = "jpeg"
VALID_OUTPUT_FORMATS = {"jpeg", "png", "webp"}
GEMINI_REPAIR_PROMPT = (
    "Improve this already-upscaled image by reducing ringing, aliasing, blur, "
    "stair-step edges, block artifacts, and other upscaling-related degradation. "
    "Preserve the exact scene, content, layout, identity, and intent. Return only "
    "the improved image."
)
GEMINI_FINISH_PROMPT = (
    "Upscale and finish this partially-upscaled image to the target scale factor "
    "{target_factor:g}x relative to the original source. Reduce blur, aliasing, "
    "ringing, and block artifacts while preserving the exact scene, content, "
    "layout, identity, and intent. Return only the finished image."
)


@dataclass
class OutputEntry:
    source: str
    strategy: str
    target_factor: float
    local_factor: float
    resampler: str
    output_path: str | None
    status: str
    elapsed_ms: int
    prompt: str | None = None
    model: str | None = None
    error: str | None = None


def safe_stem(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-")


def format_factor(value: float) -> str:
    return f"{value:g}x"


def safe_factor(value: float) -> str:
    return format_factor(value).replace(".", "p")


def parse_float_list(value: str, *, label: str) -> list[float]:
    parsed: list[float] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            number = float(item)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"{label} must contain numbers: {value}") from exc
        if number <= 1:
            raise argparse.ArgumentTypeError(f"{label} values must be greater than 1: {value}")
        parsed.append(number)
    if not parsed:
        raise argparse.ArgumentTypeError(f"{label} cannot be empty")
    return parsed


def parse_factors(value: str) -> list[float]:
    factors = parse_float_list(value, label="--factors")
    if len(factors) not in {2, 3}:
        raise argparse.ArgumentTypeError("--factors must include 2 or 3 comma-separated K values")
    return factors


def parse_resamplers(value: str) -> list[str]:
    resamplers = [item.strip().lower() for item in value.split(",") if item.strip()]
    if not resamplers:
        raise argparse.ArgumentTypeError("--resamplers cannot be empty")
    valid = {"nearest", "box", "bilinear", "hamming", "bicubic", "lanczos"}
    invalid = sorted(set(resamplers) - valid)
    if invalid:
        raise argparse.ArgumentTypeError(f"unknown resampler(s): {', '.join(invalid)}")
    return resamplers


def collect_image_paths(paths: list[str]) -> list[Path]:
    images: list[Path] = []
    for raw_path in paths:
        path = Path(raw_path)
        if path.is_dir():
            for child in sorted(path.iterdir()):
                if child.is_file() and child.suffix.lower() in IMAGE_EXTENSIONS:
                    images.append(child)
        elif path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            images.append(path)
        else:
            print(f"Warning: skipping invalid image path {raw_path}", file=sys.stderr)
    return images


def load_pillow_image_module() -> Any:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(
            "Pillow is required for local upscaling. Install it with: pip install pillow"
        ) from exc
    return Image


def pillow_resample_filter(image_module: Any, resampler: str) -> int:
    resampling = getattr(image_module, "Resampling", image_module)
    mapping = {
        "nearest": resampling.NEAREST,
        "box": resampling.BOX,
        "bilinear": resampling.BILINEAR,
        "hamming": resampling.HAMMING,
        "bicubic": resampling.BICUBIC,
        "lanczos": resampling.LANCZOS,
    }
    return mapping[resampler]


def upscale_local_image(
    input_path: Path,
    output_path: Path,
    factor: float,
    resampler: str,
    output_format: str = "source",
    *,
    image_module: Any | None = None,
) -> None:
    image_module = image_module or load_pillow_image_module()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with image_module.open(input_path) as image:
        width, height = image.size
        target_size = (max(1, round(width * factor)), max(1, round(height * factor)))
        resized = image.resize(target_size, pillow_resample_filter(image_module, resampler))
        image_format = pillow_format_for_output(output_format)
        if image_format == "JPEG" and resized.mode not in {"RGB", "L"}:
            resized = resized.convert("RGB")
        save_kwargs = {"quality": 95} if image_format in {"JPEG", "WEBP"} else {}
        resized.save(output_path, format=image_format, **save_kwargs)


def mime_type_for(path: Path) -> str:
    if path.suffix.lower() in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if path.suffix.lower() == ".webp":
        return "image/webp"
    return "image/png"


def extension_for_mime_type(mime_type: str | None) -> str:
    if mime_type == "image/jpeg":
        return ".jpg"
    if mime_type == "image/webp":
        return ".webp"
    return ".png"


def extension_for_output_format(output_format: str) -> str:
    if output_format == "jpeg":
        return ".jpg"
    return f".{output_format}"


def pillow_format_for_output(output_format: str) -> str:
    if output_format == "jpeg":
        return "JPEG"
    return output_format.upper()


def gemini_edit_image(
    input_path: Path,
    output_stem: Path,
    prompt: str,
    *,
    api_key: str,
    model: str,
) -> Path:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    image_part = types.Part.from_bytes(data=input_path.read_bytes(), mime_type=mime_type_for(input_path))
    config = types.GenerateContentConfig(
        response_modalities=["TEXT", "IMAGE"],
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )
    response = client.models.generate_content(
        model=model,
        contents=[prompt, image_part],
        config=config,
    )

    if response.candidates:
        for candidate in response.candidates:
            if candidate.content and candidate.content.parts:
                for part in candidate.content.parts:
                    inline_data = part.inline_data
                    if inline_data:
                        output_path = output_stem.with_suffix(extension_for_mime_type(inline_data.mime_type))
                        output_path.write_bytes(inline_data.data)
                        return output_path

    raise RuntimeError("Gemini returned no image")


def local_output_path(
    output_dir: Path,
    source: Path,
    strategy: str,
    factor: float,
    resampler: str,
    output_format: str,
) -> Path:
    name = (
        f"{safe_stem(source.stem)}__{strategy}__{resampler}"
        f"__{safe_factor(factor)}{extension_for_output_format(output_format)}"
    )
    return output_dir / "outputs" / name


def gemini_output_stem(
    output_dir: Path,
    source: Path,
    strategy: str,
    target_factor: float,
    local_factor: float,
    resampler: str,
) -> Path:
    name = (
        f"{safe_stem(source.stem)}__{strategy}__{resampler}"
        f"__local-{safe_factor(local_factor)}__target-{safe_factor(target_factor)}"
    )
    return output_dir / "outputs" / name


def record_entry(entries: list[OutputEntry], entry: OutputEntry) -> None:
    entries.append(entry)
    marker = "ok" if entry.status == "ok" else entry.status
    target = entry.output_path or entry.error or "not run"
    print(f"{marker}: {Path(entry.source).name} {entry.strategy} {format_factor(entry.target_factor)} {entry.resampler} -> {target}")


def run_evaluation(
    image_paths: list[Path],
    output_dir: Path,
    *,
    factors: list[float],
    resamplers: list[str],
    partial_factors: list[float],
    output_format: str,
    enable_gemini: bool,
    api_key: str | None,
    model: str,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "outputs").mkdir(parents=True, exist_ok=True)
    entries: list[OutputEntry] = []

    for source in image_paths:
        for target_factor in factors:
            for resampler in resamplers:
                start = time.time()
                manual_path = local_output_path(
                    output_dir, source, "manual", target_factor, resampler, output_format
                )
                try:
                    upscale_local_image(source, manual_path, target_factor, resampler, output_format)
                    record_entry(
                        entries,
                        OutputEntry(
                            source=str(source),
                            strategy="manual",
                            target_factor=target_factor,
                            local_factor=target_factor,
                            resampler=resampler,
                            output_path=str(manual_path),
                            status="ok",
                            elapsed_ms=round((time.time() - start) * 1000),
                        ),
                    )
                except Exception as exc:
                    record_entry(
                        entries,
                        OutputEntry(
                            source=str(source),
                            strategy="manual",
                            target_factor=target_factor,
                            local_factor=target_factor,
                            resampler=resampler,
                            output_path=None,
                            status="error",
                            elapsed_ms=round((time.time() - start) * 1000),
                            error=str(exc),
                        ),
                    )
                    continue

                repair_prompt = GEMINI_REPAIR_PROMPT
                repair_stem = gemini_output_stem(
                    output_dir, source, "manual-gemini-repair", target_factor, target_factor, resampler
                )
                run_gemini_strategy(
                    entries,
                    input_path=manual_path,
                    output_stem=repair_stem,
                    source=source,
                    strategy="manual-gemini-repair",
                    target_factor=target_factor,
                    local_factor=target_factor,
                    resampler=resampler,
                    prompt=repair_prompt,
                    enable_gemini=enable_gemini,
                    api_key=api_key,
                    model=model,
                )

                for partial_factor in partial_factors:
                    if partial_factor >= target_factor:
                        continue
                    start = time.time()
                    partial_path = local_output_path(
                        output_dir,
                        source,
                        f"partial-{format_factor(partial_factor)}",
                        partial_factor,
                        resampler,
                        output_format,
                    )
                    try:
                        upscale_local_image(source, partial_path, partial_factor, resampler, output_format)
                    except Exception as exc:
                        record_entry(
                            entries,
                            OutputEntry(
                                source=str(source),
                                strategy="partial-local",
                                target_factor=target_factor,
                                local_factor=partial_factor,
                                resampler=resampler,
                                output_path=None,
                                status="error",
                                elapsed_ms=round((time.time() - start) * 1000),
                                error=str(exc),
                            ),
                        )
                        continue

                    finish_prompt = GEMINI_FINISH_PROMPT.format(target_factor=target_factor)
                    finish_stem = gemini_output_stem(
                        output_dir,
                        source,
                        "partial-gemini-finish",
                        target_factor,
                        partial_factor,
                        resampler,
                    )
                    run_gemini_strategy(
                        entries,
                        input_path=partial_path,
                        output_stem=finish_stem,
                        source=source,
                        strategy="partial-gemini-finish",
                        target_factor=target_factor,
                        local_factor=partial_factor,
                        resampler=resampler,
                        prompt=finish_prompt,
                        enable_gemini=enable_gemini,
                        api_key=api_key,
                        model=model,
                    )

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "factors": factors,
            "resamplers": resamplers,
            "partial_factors": partial_factors,
            "output_format": output_format,
            "enable_gemini": enable_gemini,
            "gemini_model": model,
        },
        "inputs": [str(path) for path in image_paths],
        "entries": [asdict(entry) for entry in entries],
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\nManifest: {manifest_path}")
    return manifest


def run_gemini_strategy(
    entries: list[OutputEntry],
    *,
    input_path: Path,
    output_stem: Path,
    source: Path,
    strategy: str,
    target_factor: float,
    local_factor: float,
    resampler: str,
    prompt: str,
    enable_gemini: bool,
    api_key: str | None,
    model: str,
) -> None:
    if not enable_gemini:
        record_entry(
            entries,
            OutputEntry(
                source=str(source),
                strategy=strategy,
                target_factor=target_factor,
                local_factor=local_factor,
                resampler=resampler,
                output_path=None,
                status="skipped",
                elapsed_ms=0,
                prompt=prompt,
                model=model,
                error="Gemini disabled; pass --enable-gemini to run this strategy",
            ),
        )
        return
    if not api_key:
        record_entry(
            entries,
            OutputEntry(
                source=str(source),
                strategy=strategy,
                target_factor=target_factor,
                local_factor=local_factor,
                resampler=resampler,
                output_path=None,
                status="skipped",
                elapsed_ms=0,
                prompt=prompt,
                model=model,
                error="No GEMINI_API_KEY or GOOGLE_API_KEY set",
            ),
        )
        return

    start = time.time()
    try:
        output_path = gemini_edit_image(input_path, output_stem, prompt, api_key=api_key, model=model)
        record_entry(
            entries,
            OutputEntry(
                source=str(source),
                strategy=strategy,
                target_factor=target_factor,
                local_factor=local_factor,
                resampler=resampler,
                output_path=str(output_path),
                status="ok",
                elapsed_ms=round((time.time() - start) * 1000),
                prompt=prompt,
                model=model,
            ),
        )
    except Exception as exc:
        record_entry(
            entries,
            OutputEntry(
                source=str(source),
                strategy=strategy,
                target_factor=target_factor,
                local_factor=local_factor,
                resampler=resampler,
                output_path=None,
                status="error",
                elapsed_ms=round((time.time() - start) * 1000),
                prompt=prompt,
                model=model,
                error=str(exc),
            ),
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare local and opt-in Gemini upscaling strategies on image files."
    )
    parser.add_argument("paths", nargs="+", help="Image files or directories to evaluate")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/upscaling-evals") / datetime.now().strftime("%Y%m%d-%H%M%S"),
        help="Folder for generated images and manifest.json",
    )
    parser.add_argument(
        "--factors",
        type=parse_factors,
        default=DEFAULT_FACTORS,
        help="Two or three target K values, comma-separated (default: 2,3,4)",
    )
    parser.add_argument(
        "--partial-factors",
        type=lambda value: parse_float_list(value, label="--partial-factors"),
        default=DEFAULT_PARTIAL_FACTORS,
        help="Local N values for partial local upscale before Gemini finish (default: 1.5,2)",
    )
    parser.add_argument(
        "--resamplers",
        type=parse_resamplers,
        default=DEFAULT_RESAMPLERS,
        help="Comma-separated Pillow resampling methods (default: lanczos,bicubic)",
    )
    parser.add_argument(
        "--output-format",
        choices=sorted(VALID_OUTPUT_FORMATS),
        default=DEFAULT_OUTPUT_FORMAT,
        help=f"Output image format for local upscales (default: {DEFAULT_OUTPUT_FORMAT})",
    )
    parser.add_argument(
        "--enable-gemini",
        action="store_true",
        help="Opt in to real Gemini calls for repair/finish strategies",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_GEMINI_IMAGE_MODEL,
        help=f"Gemini model for opt-in calls (default: {DEFAULT_GEMINI_IMAGE_MODEL})",
    )
    return parser


def main() -> int:
    load_dotenv()
    parser = build_parser()
    args = parser.parse_args()

    image_paths = collect_image_paths(args.paths)
    if not image_paths:
        print("No valid input images found.", file=sys.stderr)
        return 1

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    manifest = run_evaluation(
        image_paths,
        args.output_dir,
        factors=args.factors,
        resamplers=args.resamplers,
        partial_factors=args.partial_factors,
        output_format=args.output_format,
        enable_gemini=args.enable_gemini,
        api_key=api_key,
        model=args.model,
    )
    failures = [entry for entry in manifest["entries"] if entry["status"] == "error"]
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
