"""
Image perturbation functions for Medical VQA experiment.
All functions take numpy array (H, W, 3) uint8 and return same format.
Applied before model's image processor.
"""

import numpy as np
import cv2


def apply_perturbation(image: np.ndarray, condition: str, **kwargs) -> np.ndarray:
    """Dispatch to the appropriate perturbation function."""
    if condition == "original":
        return image.copy()
    elif condition == "black":
        return apply_black(image)
    elif condition == "lpf":
        return apply_lpf(image, sigma=kwargs["sigma"])
    elif condition == "hpf":
        return apply_hpf(image, sigma=kwargs["sigma"])
    elif condition == "patch_shuffle":
        return apply_patch_shuffle(
            image,
            patch_size=kwargs.get("patch_size", 16),
            seed=kwargs.get("seed", 42),
        )
    else:
        raise ValueError(f"Unknown condition: {condition}")


def apply_black(image: np.ndarray) -> np.ndarray:
    """Generate a black image with the same resolution as the original."""
    return np.zeros_like(image)


def apply_lpf(image: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian Low-Pass Filter."""
    return cv2.GaussianBlur(image, (0, 0), sigma)


def apply_hpf(image: np.ndarray, sigma: float) -> np.ndarray:
    """High-Pass Filter = Original - LPF + 128."""
    lpf = cv2.GaussianBlur(image, (0, 0), sigma)
    hpf = image.astype(np.float32) - lpf.astype(np.float32) + 128.0
    return np.clip(hpf, 0, 255).astype(np.uint8)


def apply_patch_shuffle(
    image: np.ndarray, patch_size: int = 16, seed: int = 42
) -> np.ndarray:
    """
    Shuffle patch_size x patch_size patches randomly within the image.
    Edges that don't divide evenly are cropped (center crop) and zero-padded back.
    """
    rng = np.random.RandomState(seed)
    H, W, C = image.shape
    nH, nW = H // patch_size, W // patch_size

    # Center crop
    top = (H - nH * patch_size) // 2
    left = (W - nW * patch_size) // 2
    cropped = image[top : top + nH * patch_size, left : left + nW * patch_size]

    # Split into patches
    patches = cropped.reshape(nH, patch_size, nW, patch_size, C)
    patches = patches.transpose(0, 2, 1, 3, 4)  # (nH, nW, ps, ps, C)
    patches = patches.reshape(-1, patch_size, patch_size, C)

    # Shuffle
    indices = rng.permutation(len(patches))
    patches = patches[indices]

    # Reassemble
    patches = patches.reshape(nH, nW, patch_size, patch_size, C)
    patches = patches.transpose(0, 2, 1, 3, 4)
    result = patches.reshape(nH * patch_size, nW * patch_size, C)

    # Pad back to original size
    output = np.zeros_like(image)
    output[top : top + nH * patch_size, left : left + nW * patch_size] = result
    return output
