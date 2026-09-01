"""Frequency-domain feature extraction for RGB images."""

import numpy as np
from numpy.typing import NDArray
from PIL import Image


FloatArray = NDArray[np.floating]


def image_to_luminance(image: Image.Image | NDArray[np.number]) -> FloatArray:
    """Convert a PIL image or RGB array to a two-dimensional luminance array.

    The conversion uses Rec. 709/sRGB luminance coefficients. Grayscale arrays
    are accepted unchanged apart from conversion to ``float32``.

    Args:
        image: PIL image or NumPy array shaped ``(height, width)`` or
            ``(height, width, 3)``.

    Returns:
        Two-dimensional ``float32`` luminance values.

    Raises:
        ValueError: If the input is not a non-empty grayscale or RGB image.
    """
    if isinstance(image, Image.Image):
        array = np.asarray(image.convert("RGB"), dtype=np.float32)
    else:
        array = np.asarray(image, dtype=np.float32)

    if array.size == 0:
        raise ValueError("The image must not be empty.")

    if array.ndim == 2:
        return array
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(
            "Expected a grayscale array or an RGB array with shape (H, W, 3)."
        )

    coefficients = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    return np.sum(array * coefficients, axis=2, dtype=np.float32)


def compute_log_magnitude_spectrum(
    luminance: NDArray[np.number], *, apply_hann_window: bool = True
) -> FloatArray:
    """Compute a centered log-magnitude spectrum from a luminance image.

    The mean is removed so that overall brightness does not dominate the DC
    component. By default, a two-dimensional Hann window reduces artificial
    high-frequency energy caused by discontinuities at image boundaries.

    Args:
        luminance: Non-empty two-dimensional image array.
        apply_hann_window: Whether to apply a separable two-dimensional Hann
            window before the Fourier transform.

    Returns:
        Centered ``log(1 + magnitude)`` spectrum with the input spatial shape.

    Raises:
        ValueError: If the input is not a finite two-dimensional image.
    """
    values = np.asarray(luminance, dtype=np.float32)
    if values.ndim != 2 or values.size == 0:
        raise ValueError("Luminance input must be a non-empty 2D array.")
    if not np.isfinite(values).all():
        raise ValueError("Luminance input must contain only finite values.")

    centered = values - values.mean(dtype=np.float64)
    if apply_hann_window:
        height_window = np.hanning(values.shape[0])
        width_window = np.hanning(values.shape[1])
        centered = centered * np.outer(height_window, width_window)

    fourier = np.fft.fftshift(np.fft.fft2(centered))
    return np.log1p(np.abs(fourier)).astype(np.float32)


def compute_normalized_power_spectrum(
    luminance: NDArray[np.number], *, apply_hann_window: bool = True
) -> FloatArray:
    """Compute a centered power spectrum normalized to unit total energy.

    Unit-energy normalization makes spectra more comparable across images with
    different brightness and contrast before group averaging.

    Args:
        luminance: Non-empty two-dimensional image array.
        apply_hann_window: Whether to apply a separable two-dimensional Hann
            window before the Fourier transform.

    Returns:
        Centered ``float32`` power spectrum whose values sum to one.

    Raises:
        ValueError: If the input is invalid or has no frequency energy after
            mean removal.
    """
    values = np.asarray(luminance, dtype=np.float32)
    if values.ndim != 2 or values.size == 0:
        raise ValueError("Luminance input must be a non-empty 2D array.")
    if not np.isfinite(values).all():
        raise ValueError("Luminance input must contain only finite values.")

    centered = values - values.mean(dtype=np.float64)
    if apply_hann_window:
        height_window = np.hanning(values.shape[0])
        width_window = np.hanning(values.shape[1])
        centered = centered * np.outer(height_window, width_window)

    fourier = np.fft.fftshift(np.fft.fft2(centered))
    power = np.abs(fourier) ** 2
    total_energy = power.sum(dtype=np.float64)
    if not np.isfinite(total_energy) or total_energy <= 0:
        raise ValueError("The image has no finite frequency energy to normalize.")

    return (power / total_energy).astype(np.float32)


def compute_radial_energy_profile(
    normalized_power: NDArray[np.number], *, bins: int = 64
) -> FloatArray:
    """Summarize a centered power spectrum into concentric energy bands.

    Only the largest centered circle supported in every direction is used, so
    all radial bins have full angular coverage. Energy is summed within each
    annulus and renormalized to produce a profile that sums to one.

    Args:
        normalized_power: Centered, non-negative two-dimensional power array.
        bins: Number of equally spaced radial frequency bands.

    Returns:
        One-dimensional ``float32`` radial energy profile of length ``bins``.

    Raises:
        ValueError: If the spectrum or number of bins is invalid.
    """
    power = np.asarray(normalized_power, dtype=np.float64)
    if power.ndim != 2 or power.size == 0:
        raise ValueError("Power spectrum must be a non-empty 2D array.")
    if bins <= 0:
        raise ValueError("Number of radial bins must be positive.")
    if not np.isfinite(power).all() or np.any(power < 0):
        raise ValueError("Power spectrum must contain finite non-negative values.")

    height, width = power.shape
    y, x = np.indices(power.shape, dtype=np.float64)
    radius = np.sqrt((y - height // 2) ** 2 + (x - width // 2) ** 2)
    maximum_radius = min(height, width) / 2.0
    inside_circle = radius < maximum_radius
    radial_bins = np.floor(radius[inside_circle] / maximum_radius * bins).astype(int)

    profile = np.bincount(
        radial_bins,
        weights=power[inside_circle],
        minlength=bins,
    )[:bins]
    included_energy = profile.sum(dtype=np.float64)
    if not np.isfinite(included_energy) or included_energy <= 0:
        raise ValueError("Power spectrum has no finite energy inside the radial region.")

    return (profile / included_energy).astype(np.float32)


def compute_log_power_grid(
    normalized_power: NDArray[np.number], *, grid_size: int = 16
) -> FloatArray:
    """Compress a centered 2D power spectrum into a coarse log-power grid.

    Non-overlapping blocks are summed rather than averaged so that each grid
    cell represents its share of total spectral energy. Log scaling reduces
    dynamic range while retaining directional and periodic structure.

    Args:
        normalized_power: Centered, non-negative two-dimensional power array.
        grid_size: Number of output cells along each spatial dimension.

    Returns:
        Two-dimensional ``float32`` log10 power grid.

    Raises:
        ValueError: If the spectrum is invalid or cannot be evenly divided
            into the requested grid.
    """
    power = np.asarray(normalized_power, dtype=np.float64)
    if power.ndim != 2 or power.size == 0:
        raise ValueError("Power spectrum must be a non-empty 2D array.")
    if grid_size <= 0:
        raise ValueError("Grid size must be positive.")
    if not np.isfinite(power).all() or np.any(power < 0):
        raise ValueError("Power spectrum must contain finite non-negative values.")

    height, width = power.shape
    if height % grid_size != 0 or width % grid_size != 0:
        raise ValueError(
            "Power spectrum dimensions must be divisible by the grid size."
        )

    block_height = height // grid_size
    block_width = width // grid_size
    grid = power.reshape(
        grid_size, block_height, grid_size, block_width
    ).sum(axis=(1, 3))
    return np.log10(grid + 1e-12).astype(np.float32)
