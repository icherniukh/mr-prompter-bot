DEFAULT_GEMINI_IMAGE_MODEL = "gemini-2.5-flash-image"

REMOVAL_PROMPT = (
    "Clean this image by removing all superimposed graphics, digital overlay elements, "
    "corner stamps, labels, and artificial text banners that were added on top of the original photo.\n\n"
    "Reconstruct the background seamlessly behind removed elements using surrounding textures, "
    "lighting, shadows, colors, perspective, and material properties so the result looks completely natural.\n\n"
    "Keep all real-world architectural signs, entrance text, building names, address numbers, "
    "and physical scene details completely intact.\n\n"
    "CRITICAL: Preserve the exact original image dimensions, aspect ratio, resolution, "
    "composition, framing, and pixel fidelity. Do not crop, pad, resize, rotate, recolor, "
    "stylize, or change anything about the overall image structure."
)
