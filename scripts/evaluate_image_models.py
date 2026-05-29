#!/usr/bin/env python3
"""
Evaluate OpenRouter image-output models on real watermark/overlay removal.

Usage examples:
    # Paid / cheap models (OpenRouter)
    python scripts/evaluate_image_models.py data/test_images/
    python scripts/evaluate_image_models.py data/test_images/ --cheap --image-config '{"strength": 0.25}'

    # Free tier - Gemini 3.1 Flash Image (Nano Banana 2)
    export GOOGLE_API_KEY=your_key_here
    python scripts/evaluate_image_models.py data/test_images/ --provider gemini

    # Compare several free Gemini models for quality + speed
    python scripts/evaluate_image_models.py data/test_images/ --provider gemini \
        --gemini-model "gemini-3.1-flash-image-preview,gemini-2.5-flash-image" \
        --instruction-file prompts/conservative-watermark-removal.txt

The script:
- Supports multiple providers via --provider:
    - openrouter (default, paid)
    - gemini (Google Gemini free tier — best current free option for volume)
- Uses the project's src.engine.remove_overlays for OpenRouter
- For Gemini: direct call to Gemini image editing API (images only)
- Saves successful cleaned images + cost logs to data/test_results/
- Is safe: never sends multiple images in one request
"""

import argparse
import asyncio
import csv
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional

# Make sure we can import the project
sys.path.insert(0, str(Path(__file__).parent.parent))

# The project config requires these at import time. Provide dummies for the
# evaluation script (we only care about the model list + engine).
os.environ.setdefault("TELEGRAM_BOT_TOKEN", "test:token")
os.environ.setdefault("ENCRYPTION_KEY", "test-encryption-key-for-script-only")

from dotenv import load_dotenv
import httpx
import requests

from src.config import MODEL_SHORTLIST, CHEAP_MODELS
from src.engine import ProcessingError, remove_overlays

load_dotenv()

RESULTS_DIR = Path("data/test_results")


# =============================================================================
# FREE TIER: Google Gemini (Nano Banana 2 / Gemini 3.1 Flash Image)
# =============================================================================
#
# Model: gemini-3.1-flash-image-preview
#
# This is currently one of the strongest free-tier models for prompt-driven
# image editing / watermark removal (as of May 2026).
#
# Free tier rate limits (approximate, dynamic - always verify in AI Studio):
#   - ~10 RPM (requests per minute)
#   - ~500 RPD (requests per day) for image generation/editing on Flash image models
#   - Resets at midnight Pacific Time
#
# Check your exact current limits here:
#   https://aistudio.google.com/rate-limit
#
# How to get a key:
#   https://aistudio.google.com/app/apikey  (free, no credit card required for base tier)
#
# Recommended prompt style for watermark removal:
#   "Remove only the artificial watermark/logo in the bottom right.
#    Seamlessly reconstruct the background matching lighting, texture, and perspective.
#    Do not change any real scene text, signs, or architectural details.
#    Preserve exact original image dimensions and composition."
#
# To compare several free Gemini models (recommended):
#   --gemini-model "gemini-3.1-flash-image-preview,gemini-2.5-flash-image"
#
# Limitations on free tier:
#   - Daily quota is limited (hundreds of images/day at best for Flash Image models)
#   - Occasional throttling during peak times
#   - For higher volume, consider paid tiers or self-hosted alternatives (ComfyUI + FLUX)
# =============================================================================

def sanitize_model_name(model: str) -> str:
    return model.replace("/", "_").replace(".", "-")


def call_gemini_for_editing(
    api_key: str,
    image_bytes: bytes,
    mime_type: str,
    prompt: str,
    model: str = "gemini-3.1-flash-image-preview",
) -> bytes:
    """
    Call Google Gemini free tier (Nano Banana 2) for image editing / watermark removal
    using the official google-genai SDK (the proper way as per 2026 docs).

    Model: gemini-3.1-flash-image-preview (also known as Gemini 3.1 Flash Image / Nano Banana 2)

    This is currently one of the best free-tier models for prompt-based image editing.
    It supports natural language instructions like "remove the watermark/logo in the bottom right
    and seamlessly reconstruct the background matching lighting and texture".

    Free tier rate limits (as of May 2026, dynamic):
    - Roughly 10 RPM (requests per minute)
    - Roughly 500 RPD (requests per day) for image generation/editing on Flash image models
    - Limits reset at midnight Pacific Time
    - Check exact current quotas in Google AI Studio: https://aistudio.google.com/rate-limit

    Note: Limits have been reduced over time. For higher volume, consider paid tiers or self-hosted alternatives.
    """
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)

    # Load the input image as a Part
    input_image = types.Part.from_bytes(data=image_bytes, mime_type=mime_type)

    config = types.GenerateContentConfig(
        response_modalities=["TEXT", "IMAGE"],   # Include TEXT to avoid issues on many models
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),  # Critical fix for MALFORMED_FUNCTION_CALL
    )

    response = client.models.generate_content(
        model=model,
        contents=[prompt, input_image],
        config=config,
    )

    # Extract the generated/edited image - try multiple access patterns
    # because Gemini responses can vary in structure
    try:
        # Preferred path
        if response.parts:
            for part in response.parts:
                if part.inline_data is not None:
                    return part.inline_data.data

        # Alternative path via candidates (more reliable on some versions)
        if response.candidates:
            for candidate in response.candidates:
                if candidate.content and candidate.content.parts:
                    for part in candidate.content.parts:
                        if part.inline_data is not None:
                            return part.inline_data.data
    except Exception as e:
        print(f"[Gemini Debug] Error while extracting image: {e}")

    # If we reach here, no image was returned - dump useful info for debugging
    debug_info = {
        "finish_reason": getattr(response, 'finish_reason', None),
        "candidates": [],
    }

    try:
        if response.candidates:
            for c in response.candidates:
                debug_info["candidates"].append({
                    "finish_reason": getattr(c, 'finish_reason', None),
                    "finish_message": getattr(c, 'finish_message', None),
                    "content_parts": str(c.content.parts) if c.content and c.content.parts else None,
                })
    except Exception:
        pass

    raise RuntimeError(f"No image returned by Gemini.\nDebug info: {debug_info}\nFull response: {response}")


async def evaluate_one(
    image_path: Path,
    model: str,
    api_key: str,
    semaphore: asyncio.Semaphore,
    image_config: dict[str, Any] | None = None,
    instruction: str | None = None,
    output_modalities: list[str] | None = None,
    provider: str = "openrouter",
    gemini_model: str = "gemini-3.1-flash-image-preview",
) -> dict:
    """Try to clean one image with one model. Returns a result dict."""
    async with semaphore:
        result = {
            "image": image_path.name,
            "model": model,
            "success": False,
            "error": None,
            "output_path": None,
            "http_status": None,
        }

        try:
            image_bytes = image_path.read_bytes()
            mime = "image/jpeg" if image_path.suffix.lower() in {".jpg", ".jpeg"} else "image/png"

            start_time = time.time()

            if provider == "gemini":
                # Free Gemini path - Nano Banana 2 (Gemini 3.1 Flash Image Preview)
                effective_prompt = instruction or "Remove only the artificial watermark/logo. Preserve all real scene content and exact original dimensions."
                cleaned = call_gemini_for_editing(
                    api_key,
                    image_bytes,
                    mime,
                    effective_prompt,
                    model=gemini_model,
                )
                metadata = {"provider": "gemini", "model": gemini_model, "cost": 0.0}
            else:
                # Default OpenRouter path
                cleaned, metadata = await remove_overlays(
                    api_key,
                    image_bytes,
                    model,
                    mime,
                    image_config=image_config,
                    instruction=instruction,
                    output_modalities=output_modalities,
                )

            latency_ms = (time.time() - start_time) * 1000
            metadata["latency_ms"] = round(latency_ms, 1)

            # Success!
            out_dir = RESULTS_DIR / sanitize_model_name(model)
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{image_path.stem}_cleaned{image_path.suffix}"
            out_path.write_bytes(cleaned)

            result["success"] = True
            result["output_path"] = str(out_path)
            result["cost"] = metadata.get("cost")
            result["generation_id"] = metadata.get("generation_id")
            result["latency_ms"] = metadata.get("latency_ms")
            result["metadata"] = metadata

            cost_str = f" ${metadata.get('cost'):.4f}" if metadata.get("cost") is not None else ""
            latency_str = f" ({metadata.get('latency_ms')}ms)" if metadata.get("latency_ms") else ""
            print(f"  ✓ {model} on {image_path.name}{cost_str}{latency_str} → {out_path}")

        except ProcessingError as e:
            result["error"] = str(e)
            print(f"  ✗ {model} on {image_path.name}: {e}")
        except httpx.HTTPStatusError as e:
            result["http_status"] = e.response.status_code
            result["error"] = f"HTTP {e.response.status_code}: {e.response.text[:200]}"
            print(f"  ✗ {model} on {image_path.name}: HTTP {e.response.status_code}")
        except Exception as e:
            result["error"] = f"{type(e).__name__}: {e}"
            print(f"  ✗ {model} on {image_path.name}: {e}")

        return result


async def main():
    parser = argparse.ArgumentParser(description="Evaluate image models for watermark removal")
    parser.add_argument("images", nargs="+", help="Image file(s) or directory containing images")
    parser.add_argument(
        "--models",
        help="Comma-separated list of models to test (default: current MODEL_SHORTLIST)",
    )
    parser.add_argument(
        "--cheap",
        action="store_true",
        help="Use only the recommended cheap models (flux.2-klein-4b, riverflow-v2-fast, recraft-v4.1-utility)",
    )
    parser.add_argument(
        "--provider",
        default="openrouter",
        choices=["openrouter", "gemini"],
        help="Provider backend. 'gemini' = Google Gemini free tier using Nano Banana 2 (gemini-3.1-flash-image-preview).",
    )
    parser.add_argument(
        "--gemini-model",
        default="gemini-3.1-flash-image-preview",
        help="Comma-separated list of Gemini models to test on free tier. "
             "Examples: 'gemini-3.1-flash-image-preview,gemini-2.5-flash-image'. "
             "Default focuses on Nano Banana 2 (Gemini 3.1 Flash Image).",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=2,
        help="Max concurrent requests (be gentle with image models)",
    )
    parser.add_argument(
        "--image-config",
        dest="image_config_json",
        help="JSON string or path to JSON file with image_config (e.g. '{\"aspect_ratio\": \"4:3\"}'). "
             "Passed to OpenRouter for dimension control, strength, etc.",
    )
    parser.add_argument(
        "--instruction-file",
        help="Path to a text file containing a custom removal instruction (for prompt A/B testing).",
    )
    parser.add_argument(
        "--output-modalities",
        default="image",
        help="Comma-separated output modalities, e.g. 'image' or 'image,text'. Default: 'image' (pure image output).",
    )
    args = parser.parse_args()

    # Resolve images
    image_paths: List[Path] = []
    for item in args.images:
        p = Path(item)
        if p.is_dir():
            image_paths.extend(sorted(p.glob("*.jpg")))
            image_paths.extend(sorted(p.glob("*.jpeg")))
            image_paths.extend(sorted(p.glob("*.png")))
        elif p.is_file():
            image_paths.append(p)
        else:
            print(f"Warning: {item} not found, skipping")

    if not image_paths:
        print("No images found.")
        sys.exit(1)

    # Resolve models
    if args.provider == "gemini":
        gemini_models = [m.strip() for m in args.gemini_model.split(",") if m.strip()]
        models = gemini_models
        print(f"Using Gemini free tier models: {models}")
    elif args.cheap:
        models = CHEAP_MODELS
        print("Using --cheap mode: only recommended low-cost models")
    elif args.models:
        models = [m.strip() for m in args.models.split(",") if m.strip()]
    else:
        models = MODEL_SHORTLIST

    # Load optional image_config
    image_config: dict[str, Any] | None = None
    if args.image_config_json:
        cfg_path = Path(args.image_config_json)
        if cfg_path.is_file():
            image_config = json.loads(cfg_path.read_text())
        else:
            image_config = json.loads(args.image_config_json)
        print(f"Using image_config: {image_config}")

    # Load optional custom instruction
    custom_instruction: str | None = None
    if args.instruction_file:
        custom_instruction = Path(args.instruction_file).read_text().strip()
        print(f"Using custom instruction from: {args.instruction_file}\n")

    # Parse output modalities
    output_modalities = [m.strip() for m in args.output_modalities.split(",") if m.strip()]
    print(f"Using output_modalities: {output_modalities}\n")

    provider = args.provider
    print(f"Using provider: {provider}\n")

    print(f"Testing {len(models)} models on {len(image_paths)} images:")
    for m in models:
        print(f"  - {m}")
    print()

    # Get API key depending on provider
    if provider == "gemini":
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            print("ERROR: No Gemini API key found.")
            print("Set GOOGLE_API_KEY or GEMINI_API_KEY in environment.")
            sys.exit(1)
        print(f"Using Gemini free tier (key: {api_key[:8]}...)\n")
    else:
        api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("HOST_OPENROUTER_KEY")
        if not api_key:
            print("ERROR: No OpenRouter key found.")
            print("Set OPENROUTER_API_KEY or HOST_OPENROUTER_KEY in environment or .env")
            sys.exit(1)
        print(f"Using key: {api_key[:12]}... (length {len(api_key)})\n")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    semaphore = asyncio.Semaphore(args.concurrency)

    tasks = []
    for img in image_paths:
        for model in models:
            tasks.append(
                evaluate_one(
                    img,
                    model,
                    api_key,
                    semaphore,
                    image_config=image_config,
                    instruction=custom_instruction,
                    output_modalities=output_modalities,
                    provider=provider,
                    gemini_model=model if provider == "gemini" else args.gemini_model,
                )
            )

    results = await asyncio.gather(*tasks)

    # Summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    for img in image_paths:
        print(f"\n{img.name}:")
        for r in results:
            if r["image"] == img.name:
                status = "SUCCESS" if r["success"] else f"FAILED ({r['error'] or r.get('http_status')})"
                print(f"  {r['model']}: {status}")
                if r["output_path"]:
                    print(f"      → {r['output_path']}")

    # Overall success counts per model
    print("\n" + "-" * 70)
    print("Success rate per model:")
    for model in models:
        successes = sum(1 for r in results if r["model"] == model and r["success"])
        total = sum(1 for r in results if r["model"] == model)
        print(f"  {model}: {successes}/{total}")

    # === Cost logging ===
    print("\n" + "=" * 70)
    print("COST + LATENCY SUMMARY (per request)")
    print("=" * 70)

    total_cost = 0.0
    cost_rows = []

    for r in results:
        cost = r.get("cost")
        if cost is not None:
            total_cost += cost
        cost_rows.append({
            "image": r["image"],
            "model": r["model"],
            "success": r["success"],
            "cost_usd": cost,
            "latency_ms": r.get("latency_ms"),
            "generation_id": r.get("generation_id"),
            "prompt_file": args.instruction_file or "default",
            "image_config": json.dumps(image_config) if image_config else "{}",
        })

    # Print per-request costs + latency
    for row in cost_rows:
        cost_str = f"${row['cost_usd']:.4f}" if row['cost_usd'] is not None else "N/A"
        latency_str = f"{row['latency_ms']}ms" if row.get('latency_ms') else "N/A"
        status = "OK" if row["success"] else "FAIL"
        print(f"  {row['model']:45} | {row['image']:35} | {cost_str:>10} | {latency_str:>8} | {status}")

    print(f"\nTotal cost for this run: ${total_cost:.4f}")

    # Average latency per model (very useful for comparing free tier models)
    print("\nAverage latency per model:")
    for m in set(r["model"] for r in results if r.get("latency_ms")):
        latencies = [r["latency_ms"] for r in results if r["model"] == m and r.get("latency_ms")]
        if latencies:
            avg = sum(latencies) / len(latencies)
            print(f"  {m:45} : {avg:.1f} ms (n={len(latencies)})")

    # Save detailed cost log to CSV
    costs_dir = RESULTS_DIR / "costs"
    costs_dir.mkdir(parents=True, exist_ok=True)

    # Create a nice filename based on run parameters
    run_tag = "cheap" if args.cheap else "custom"
    if args.instruction_file:
        run_tag += f"-{Path(args.instruction_file).stem}"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = costs_dir / f"costs_{run_tag}_{timestamp}.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "image", "model", "success", "cost_usd", "latency_ms", "generation_id", "prompt_file", "image_config"
        ])
        writer.writeheader()
        writer.writerows(cost_rows)

    print(f"\nDetailed cost log saved to: {csv_path}")

    print("\nCleaned images (when successful) are in data/test_results/<model>/")


if __name__ == "__main__":
    asyncio.run(main())
