"""
utils.py

Helper utilities for UrbanBrush: Neural Style Transfer for Cityscapes.

- All function names follow snake_case (ab_cd) as required.
- Detailed docstrings explain behavior and expected inputs/outputs.
- Utilities include: loading, preprocessing, batch loading, conversions,
  saving and a small visualization helper for notebooks.

NOTE:
- This file expects Pillow, numpy, tensorflow and matplotlib to be available.
- Keep indentation as 4 spaces. Save file encoding as UTF-8.
"""

import os
from typing import List, Tuple
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt


def load_and_process_image(image_path: str,
                           target_size: Tuple[int, int] = (512, 512),
                           normalize: bool = True) -> np.ndarray:
    """
    Load image from disk, convert to RGB, resize and optionally normalize.

    Parameters
    ----------
    image_path : str
        Path to the image file (jpg/png/etc).
    target_size : tuple (width, height)
        Resize target. Default (512, 512).
    normalize : bool
        If True, scale pixel values to [0, 1]. If False, keep [0,255] floats.

    Returns
    -------
    image_array : np.ndarray
        NumPy array of shape (H, W, 3), dtype float32.
        Pixel range is [0,1] if normalize else [0,255].
    """
    # Open image with Pillow and ensure RGB
    image = Image.open(image_path).convert("RGB")
    # Use LANCZOS resampling for high-quality resizing
    image = image.resize(target_size, Image.LANCZOS)
    image_array = np.asarray(image, dtype=np.float32)

    if normalize:
        image_array = image_array / 255.0

    return image_array


def image_to_tensor(image_array: np.ndarray) -> tf.Tensor:
    """
    Convert a NumPy image array (H, W, C) to a TensorFlow tensor (1, H, W, C).

    Parameters
    ----------
    image_array : np.ndarray
        Image array in float32, pixel range usually [0,1] or [0,255].

    Returns
    -------
    tensor : tf.Tensor
        TensorFlow float32 tensor with batch dim added.
    """
    tensor = tf.convert_to_tensor(image_array, dtype=tf.float32)
    tensor = tf.expand_dims(tensor, axis=0)  # batch dimension
    return tensor


def tensor_to_image(tensor: tf.Tensor) -> Image.Image:
    """
    Convert a TensorFlow tensor (1, H, W, C) or (H, W, C) to a PIL Image.

    This function clips values to [0,1] if floats and transforms to uint8.

    Parameters
    ----------
    tensor : tf.Tensor
        Tensor with shape (1,H,W,C) or (H,W,C).

    Returns
    -------
    image : PIL.Image.Image
        PIL Image in RGB mode.
    """
    # Remove batch dim if present
    if len(tensor.shape) == 4:
        tensor = tf.squeeze(tensor, axis=0)

    # Ensure the values are in [0,1]
    tensor = tf.clip_by_value(tensor, 0.0, 1.0)
    array = (tensor.numpy() * 255.0).astype(np.uint8)
    return Image.fromarray(array)


def save_image(image: Image.Image, save_path: str, optimize: bool = True) -> None:
    """
    Save a PIL Image to disk (creates directories if needed).

    Parameters
    ----------
    image : PIL.Image.Image
        Image to save.
    save_path : str
        Full file path where the image will be saved.
    optimize : bool
        If True, use Pillow's optimize flag for jpeg/png (may slow save).
    """
    folder = os.path.dirname(save_path)
    if folder:
        os.makedirs(folder, exist_ok=True)

    # Use Pillow save with optimization where available
    save_kwargs = {}
    if optimize:
        save_kwargs["optimize"] = True

    image.save(save_path, **save_kwargs)


def load_images_from_folder(folder_path: str,
                            target_size: Tuple[int, int] = (512, 512),
                            extensions: Tuple[str, ...] = ('.jpg', '.jpeg', '.png'),
                            normalize: bool = True) -> List[Tuple[str, np.ndarray]]:
    """
    Load all images in a folder with given extensions, preprocess and return list.

    Parameters
    ----------
    folder_path : str
        Directory path containing images.
    target_size : tuple
        Resize size in (width, height).
    extensions : tuple
        File extensions to consider.
    normalize : bool
        Whether to normalize images to [0,1].

    Returns
    -------
    images : list of (filename, image_array)
        Each item is a tuple (filename, numpy_image_array).
    """
    items = []
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    for fname in sorted(os.listdir(folder_path)):
        lower = fname.lower()
        if any(lower.endswith(ext) for ext in extensions):
            full_path = os.path.join(folder_path, fname)
            try:
                img = load_and_process_image(full_path, target_size=target_size, normalize=normalize)
                items.append((fname, img))
            except Exception as e:
                # Continue loading others but print a warning for the failed file
                print(f"[warning] failed to load {full_path}: {e}")

    return items


def visualize_image_grid(images: List[np.ndarray],
                         titles: List[str] = None,
                         cols: int = 4,
                         figsize: Tuple[int, int] = (12, 8)) -> None:
    """
    Visualize a list of images (numpy arrays) in a grid using matplotlib.

    Parameters
    ----------
    images : list of np.ndarray
        Each image as HxWx3 float32 array (values in [0,1] recommended).
    titles : list of str
        Optional titles for each grid cell.
    cols : int
        Number of columns in the grid.
    figsize : tuple
        Figure size in inches for matplotlib.
    """
    if len(images) == 0:
        print("[info] No images to display.")
        return

    n = len(images)
    cols = max(1, cols)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    # axes may be 2D or 1D depending on rows, cols
    axes = np.array(axes).reshape(-1)

    for i in range(rows * cols):
        ax = axes[i]
        ax.axis('off')  # hide axis ticks
        if i < n:
            img = images[i]
            # If tensor-like scaled [0,1], convert for plt.imshow
            if img.dtype != np.uint8 and img.max() <= 1.0:
                ax.imshow(img)
            else:
                ax.imshow(img.astype(np.uint8))
            if titles and i < len(titles):
                ax.set_title(titles[i], fontsize=9)
        else:
            ax.set_visible(False)

    plt.tight_layout()
    plt.show()
