"""Shared, bounded image decoding for inspection inputs."""

from io import BytesIO
from os import PathLike
from pathlib import Path
from typing import Union

from PIL import Image, ImageOps, UnidentifiedImageError

MAX_IMAGE_BYTES = 20 * 1024 * 1024
MAX_IMAGE_PIXELS = 40_000_000
SUPPORTED_IMAGE_FORMATS = {"JPEG", "PNG", "WEBP"}


def _open_image(source):
    try:
        return Image.open(source)
    except ModuleNotFoundError as exc:
        if exc.name != "pi_heif":
            raise
        raise UnidentifiedImageError("Image data is corrupt or in an unsupported format.") from exc


def load_rgb_image(image_input: Union[str, PathLike, bytes, Image.Image]) -> Image.Image:
    """Decode a supported image and return an EXIF-oriented RGB copy."""
    if isinstance(image_input, Image.Image):
        image = image_input.copy()
        image_format = image.format
    elif isinstance(image_input, bytes):
        if not image_input:
            raise ValueError("Image data is empty.")
        if len(image_input) > MAX_IMAGE_BYTES:
            raise ValueError(f"Image exceeds the {MAX_IMAGE_BYTES // (1024 * 1024)} MiB limit.")
        image = _open_image(BytesIO(image_input))
        image_format = image.format
    elif isinstance(image_input, (str, PathLike)):
        path = Path(image_input)
        if not path.is_file():
            raise ValueError("Image file does not exist.")
        if path.stat().st_size == 0:
            raise ValueError("Image file is empty.")
        if path.stat().st_size > MAX_IMAGE_BYTES:
            raise ValueError(f"Image exceeds the {MAX_IMAGE_BYTES // (1024 * 1024)} MiB limit.")
        image = _open_image(path)
        image_format = image.format
    else:
        raise ValueError("image_input must be a path, bytes, or PIL Image.")

    if image_format is not None and image_format.upper() not in SUPPORTED_IMAGE_FORMATS:
        image.close()
        raise ValueError("Unsupported image format. Use JPEG, PNG, or WebP.")

    try:
        width, height = image.size
        if width <= 0 or height <= 0:
            raise ValueError("Image dimensions must be positive.")
        if width * height > MAX_IMAGE_PIXELS:
            raise ValueError("Image dimensions exceed the 40-megapixel processing limit.")
        image.load()
        oriented = ImageOps.exif_transpose(image)
        return oriented.convert("RGB")
    finally:
        image.close()
