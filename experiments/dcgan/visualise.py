"""
============================================================
BrailleVisionAI — DCGAN Experiment  |  Visualisation Utils
experiments/dcgan/visualise.py
============================================================

PURPOSE
-------
Helper functions for visualising training data samples, generated
images, and training loss curves.

HOW TO USE
----------
    from experiments.dcgan.dataset_loader import BrailleDatasetLoader
    from experiments.dcgan.visualise import display_image_grid, plot_losses

    loader  = BrailleDatasetLoader('/path/to/Braille Dataset')
    x_train = loader.get_numpy_array()
    display_image_grid(x_train[:20], title='Real Braille Samples')
============================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional


def display_image_grid(
    images:  np.ndarray,
    title:   str = "Image Grid",
    cols:    int = 5,
    cmap:    str = "gray",
    vmin: Optional[float] = -1,
    vmax: Optional[float] =  1,
) -> None:
    """
    Display a grid of images using matplotlib.

    Args:
        images: Array of shape (N, H, W, 1) or (N, H, W, 3).
                Values should be in [-1, 1] (normalised) or [0, 255].
        title:  Figure super-title.
        cols:   Number of columns in the grid.
        cmap:   Colormap ('gray' for grayscale images).
        vmin:   Minimum pixel value for colormap scaling.
        vmax:   Maximum pixel value for colormap scaling.
    """
    n    = images.shape[0]
    rows = (n + cols - 1) // cols

    plt.figure(figsize=(cols * 2.5, rows * 2.5))
    plt.suptitle(title, fontsize=13, y=1.01)

    for i in range(n):
        plt.subplot(rows, cols, i + 1)
        if images.shape[-1] == 1:
            plt.imshow(images[i, :, :, 0], cmap=cmap, vmin=vmin, vmax=vmax)
        else:
            # RGB — rescale if normalised
            img = images[i]
            if vmin == -1:
                img = ((img + 1) * 127.5).astype(np.uint8)
            plt.imshow(img)
        plt.axis("off")

    plt.tight_layout()
    plt.show()


def plot_losses(
    gen_losses:  List[float],
    disc_losses: Optional[List[float]] = None,
) -> None:
    """
    Plot Generator (and optionally Discriminator) loss curves.

    Args:
        gen_losses:   List of per-step or per-epoch generator losses.
        disc_losses:  Optional list of per-step discriminator losses.
    """
    plt.figure(figsize=(10, 4))
    plt.title("DCGAN Training Loss")
    plt.plot(gen_losses, label="Generator loss", color="royalblue")
    if disc_losses is not None:
        plt.plot(disc_losses, label="Discriminator loss", color="tomato")
    plt.xlabel("Step / Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.show()
