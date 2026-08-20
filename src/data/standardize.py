"""Image standardization: the bias-prevention step applied identically to
real and AI-generated images. Without this, a model can learn trivial shortcuts
like "PNG = AI" or "this resolution = real" instead of actual generation
artifacts. This dataset's raw images concretely have that problem: AI
images are 128x128 PNG, real images are variable-size JPEG.

Steps, applied the same way regardless of source:
1. Resize to a fixed target resolution
2. Convert to a fixed RGB color mode
3. Strip EXIF/metadata
4. Re-encode with the same compression settings (JPEG)
"""

from pathlib import Path
from PIL import Image


class CorruptImageError(Exception):
    """Raised when a source image can't be opened or standardized."""


def standardize_image(src_path: Path | str, dst_path: Path | str, image_size: int = 224, color_mode: str = "RGB", jpeg_quality: int = 90,) -> None:
    """Resize and re-encode an image using consistent settings.

    Args:
        src_path: Path to the source image.
        dst_path: Path where the standardized image will be saved.
        image_size: Target width and height in pixels.
        color_mode: Target image color mode.
        jpeg_quality: JPEG compression quality.

    Raises:
        CorruptImageError: If the image cannot be opened or standardized.
    """
    try:
        with Image.open(src_path) as img:
            img = img.convert(color_mode)
            img = img.resize((image_size, image_size), Image.BICUBIC) 
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            img.save(dst_path, format="JPEG", quality=jpeg_quality)
    except Exception as e:
        raise CorruptImageError(f"Failed to standardize image {src_path}: {e}") from e
