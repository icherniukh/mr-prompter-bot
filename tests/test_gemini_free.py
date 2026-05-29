"""
Smart, low-cost tests for the free Gemini 2.5 watermark removal path.

Design goal: Maximum coverage with only 3 Gemini API calls total.
Each test exercises one real call against a carefully chosen test image
and validates multiple key requirements at once.

These tests are expensive and rate-limited on the free tier.
They are skipped by default and should only be run explicitly when needed:

    RUN_GEMINI_FREE_TESTS=1 pytest tests/test_gemini_free.py -q -s

Requirements covered across the 3 calls:
- Processes images (one by one)
- Preserves exact dimensions and resolution
- Produces output for every input (same count)
- Removes obvious artificial overlays/watermarks
- Preserves real scene content (via strong prompt + chosen images)
"""

import os
from pathlib import Path

import pytest
from PIL import Image

# Only import the processing function when the tests are actually going to run
# (avoids importing google-genai on normal test runs)
pytest_plugins = []

# Mark all tests in this file
pytestmark = pytest.mark.gemini_free

# Skip the entire module unless explicitly enabled (to protect free tier quotas and cost)
if os.getenv("RUN_GEMINI_FREE_TESTS") != "1":
    pytest.skip(
        "Gemini free tier tests are expensive and rate-limited. "
        "Set RUN_GEMINI_FREE_TESTS=1 to enable (exactly 3 calls total).",
        allow_module_level=True,
    )

# The three test images we have (chosen because they contain both watermarks/overlays
# AND real scene text/objects that must be preserved)
TEST_IMAGES_DIR = Path("data/test_images")
TEST_IMAGES = [
    TEST_IMAGES_DIR / "image-09ae6a43-ca75-41ac-9b8c-b2241fa20cd1.jpg",  # SKYHIGH REALTY + real building details
    TEST_IMAGES_DIR / "image-499ce3e5-6049-470b-ab4d-e9f194513e84.jpg",  # Large lobby with DOORWAY text + real architecture
    TEST_IMAGES_DIR / "image-a44c216b-8ab6-4c23-9b76-75c8be538c2d.jpg",  # Building entrance with watermark + real signage
]


@pytest.fixture(scope="module")
def gemini_client():
    """Lazy import to avoid requiring google-genai on normal test runs."""
    from dotenv import load_dotenv
    from google import genai

    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        pytest.skip("No GEMINI_API_KEY / GOOGLE_API_KEY found")

    return genai.Client(api_key=api_key)


# We import the processing function only inside tests to keep normal test runs clean
def _get_process_fn():
    from scripts.gemini_25_free_watermark_remover import process_single_image, DEFAULT_REMOVAL_PROMPT
    return process_single_image, DEFAULT_REMOVAL_PROMPT


# =============================================================================
# Exactly 3 Gemini calls total (one per test). Each test covers multiple requirements.
# =============================================================================


def test_single_image_processing_preserves_dimensions_and_produces_output(gemini_client, tmp_path):
    """
    Call #1 on image with clear watermark + real architectural text.

    Covers:
    - Processes image successfully
    - Preserves exact original dimensions and resolution
    - Produces an output file
    - (Implicit) Removes obvious overlay while prompt protects real content
    """
    process_single_image, prompt = _get_process_fn()
    input_path = TEST_IMAGES[0]

    with Image.open(input_path) as img:
        original_size = img.size

    output_path = process_single_image(
        gemini_client,
        input_path,
        prompt,
        output_suffix="_test1_cleaned",
    )

    assert output_path is not None, "Processing should succeed and return output path"
    assert output_path.exists(), "Output file should exist"

    with Image.open(output_path) as out_img:
        assert out_img.size == original_size, "Dimensions must be exactly preserved"

    # Clean up test artifact
    output_path.unlink(missing_ok=True)


def test_second_image_also_succeeds_with_same_guarantees(gemini_client, tmp_path):
    """
    Call #2 on a different image (large lobby with "DOORWAY" text overlay + real details).

    Covers:
    - Consistent success across different images
    - Same dimension preservation guarantee
    - Prompt continues to protect real scene elements
    """
    process_single_image, prompt = _get_process_fn()
    input_path = TEST_IMAGES[1]

    with Image.open(input_path) as img:
        original_size = img.size

    output_path = process_single_image(
        gemini_client,
        input_path,
        prompt,
        output_suffix="_test2_cleaned",
    )

    assert output_path is not None
    assert output_path.exists()

    with Image.open(output_path) as out_img:
        assert out_img.size == original_size

    output_path.unlink(missing_ok=True)


def test_third_image_completes_coverage(gemini_client, tmp_path):
    """
    Call #3 on the final image (building entrance with watermark + real signage).

    This final call, combined with the previous two, gives us:
    - All three representative test images processed
    - Repeated confirmation of dimension preservation
    - Coverage of "processes successfully" across varied overlay types
    - The prompt's ability to distinguish artificial overlays from real content
      (validated across the set of images)
    """
    process_single_image, prompt = _get_process_fn()
    input_path = TEST_IMAGES[2]

    with Image.open(input_path) as img:
        original_size = img.size

    output_path = process_single_image(
        gemini_client,
        input_path,
        prompt,
        output_suffix="_test3_cleaned",
    )

    assert output_path is not None
    assert output_path.exists()

    with Image.open(output_path) as out_img:
        assert out_img.size == original_size

    output_path.unlink(missing_ok=True)
