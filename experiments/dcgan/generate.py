"""
============================================================
BrailleVisionAI — DCGAN Experiment  |  Image Generation
experiments/dcgan/generate.py
============================================================

PURPOSE
-------
Loads a trained Generator and produces synthetic Braille images.

WHAT THIS DOES
--------------
  1. Load the saved Generator weights from a .keras / .h5 file.
  2. Sample a batch of random noise vectors from N(0,1).
  3. Forward pass through the Generator → synthetic Braille images.
  4. Rescale from [-1, 1] → [0, 255] for display/saving.
  5. Optionally display a grid plot or save images to disk.

IMPORTANT: DENORMALISATION
--------------------------
  Training data was normalised: (pixel - 127.5) / 127.5 → [-1, 1]
  To recover uint8 images:
      pixel = (generated + 1) * 127.5   (→ [0, 255])

HOW TO USE
----------
  # Quick display (20 samples)
    from experiments.dcgan.generate import BrailleGenerator
    gen = BrailleGenerator('experiments/dcgan/checkpoints/best_generator.keras')
    gen.display_samples(n=20)

  # Save to disk
    gen.save_samples(n=50, output_dir='experiments/dcgan/generated/')

HACKATHON QUICK-START
---------------------
  If you do NOT have a trained model yet, use the provided helper
  `generate_random_grid()` which renders a labelled placeholder grid
  so you can demo the generation pipeline without GPU training.
============================================================
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import load_model


# ── Default config ────────────────────────────────────────────
LATENT_DIM     = 100   # must match the latent_dim used during training
IMG_SHAPE      = (28, 28, 1)


class BrailleGenerator:
    """
    Loads a trained DCGAN Generator and generates synthetic Braille images.

    Args:
        model_path (str): Path to the saved Keras model (.keras or .h5).
        latent_dim (int): Size of the noise vector (must match training; default 100).
    """

    def __init__(self, model_path: str, latent_dim: int = LATENT_DIM) -> None:
        self.latent_dim = latent_dim
        print(f"Loading generator from: {model_path}")
        self.generator = load_model(model_path, compile=False)
        print("Generator loaded successfully.")

    # ── Core generation ───────────────────────────────────────
    def generate(self, n: int = 20) -> np.ndarray:
        """
        Generate `n` synthetic Braille images.

        Returns:
            NumPy array of shape (n, 28, 28, 1) in [-1, 1].
        """
        noise = tf.random.normal([n, self.latent_dim])
        return self.generator.predict(noise)

    def generate_uint8(self, n: int = 20) -> np.ndarray:
        """
        Generate `n` synthetic Braille images scaled to [0, 255] uint8.

        Returns:
            NumPy array of shape (n, 28, 28, 1) dtype uint8.
        """
        imgs = self.generate(n)
        imgs = ((imgs + 1) * 127.5).astype(np.uint8)
        return imgs

    # ── Display ───────────────────────────────────────────────
    def display_samples(self, n: int = 20, cols: int = 5) -> None:
        """Display a grid of `n` generated Braille samples using matplotlib."""
        imgs = self.generate(n)
        rows = (n + cols - 1) // cols

        plt.figure(figsize=(cols * 3, rows * 3))
        plt.suptitle("DCGAN — Generated Braille Samples", fontsize=14, y=1.02)
        for i in range(n):
            plt.subplot(rows, cols, i + 1)
            plt.imshow(imgs[i, :, :, 0], cmap="gray", vmin=-1, vmax=1)
            plt.axis("off")
        plt.tight_layout()
        plt.show()

    # ── Save to disk ──────────────────────────────────────────
    def save_samples(
        self,
        n:          int = 50,
        output_dir: str = "experiments/dcgan/generated/",
    ) -> None:
        """
        Save `n` generated images as PNG files in `output_dir`.

        File naming: generated_0000.png, generated_0001.png, ...
        """
        import cv2
        os.makedirs(output_dir, exist_ok=True)
        imgs = self.generate_uint8(n)

        for i, img in enumerate(imgs):
            filename = os.path.join(output_dir, f"generated_{i:04d}.png")
            cv2.imwrite(filename, img[:, :, 0])   # save as grayscale PNG

        print(f"Saved {n} generated images to: {output_dir}")


# ── Standalone quick-test ─────────────────────────────────────
def generate_random_grid(n: int = 20, latent_dim: int = LATENT_DIM) -> None:
    """
    PLACEHOLDER — demo the generation pipeline with a randomly
    initialised (untrained) Generator.

    Useful for CI or demo environments without a trained model file.
    The output will look like random noise — that is expected.
    """
    from experiments.dcgan.model import build_generator

    print("[WARNING] Using an UNTRAINED generator — output is random noise.")
    model  = build_generator(latent_dim)
    noise  = tf.random.normal([n, latent_dim])
    imgs   = model.predict(noise)
    cols   = 5
    rows   = (n + cols - 1) // cols

    plt.figure(figsize=(cols * 3, rows * 3))
    plt.suptitle("DCGAN — Untrained Generator Output (Random Noise)", fontsize=12)
    for i in range(n):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(imgs[i, :, :, 0], cmap="gray", vmin=-1, vmax=1)
        plt.axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
        gen = BrailleGenerator(model_path)
        gen.display_samples(n=20)
    else:
        generate_random_grid(n=20)
