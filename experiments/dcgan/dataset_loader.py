"""
============================================================
BrailleVisionAI — DCGAN Experiment  |  Dataset Loader
experiments/dcgan/dataset_loader.py
============================================================

PURPOSE
-------
Loads and preprocesses Braille character images for DCGAN training.

DATASET
-------
Uses the "Braille Character Dataset" from Kaggle:
  https://www.kaggle.com/datasets/...

  Folder structure (flat — all images in one folder):
    Braille Dataset/
    └── Braille Dataset/
        ├── a1.jpg
        ├── a2.jpg
        ├── ...
        └── z_n.jpg

  Images: 28×28 grayscale PNG/JPG of embossed Braille characters.
  Each character class (a–z) has ~250 samples ≈ 6,500 total images.

PREPROCESSING PIPELINE
----------------------
  1. Read each image file path into a DataFrame (for easy inspection)
  2. Load each image using OpenCV
  3. Convert BGR → RGB → Grayscale (single channel, 28×28×1)
  4. Flatten to 1-D, reshape to (N, 28, 28, 1)
  5. Normalise pixel values from [0, 255] → [-1, 1]
     Formula: (pixel - 127.5) / 127.5
     WHY: Generator output uses tanh activation (range [-1,1]),
          so training data must match.
  6. Wrap in a tf.data.Dataset for efficient batched training.

HOW TO USE
----------
    from experiments.dcgan.dataset_loader import BrailleDatasetLoader

    loader = BrailleDatasetLoader(path='/path/to/Braille Dataset')
    dataset = loader.get_tf_dataset(batch_size=128, buffer_size=512)
    x_train = loader.get_numpy_array()   # for inspection/display

NOTES FOR FUTURE PHASES
-----------------------
  - The same loader can feed a classification model (Phase D).
  - To adapt for colour images, change C=1 → C=3 and remove
    the grayscale conversion step.
  - Label information (character class from filename) is available
    via CreateDataFrame_Gan(); uncomment label extraction if needed.
============================================================
"""

import os
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
from tqdm import tqdm


# ── Default config ───────────────────────────────────────────
DEFAULT_WIDTH   = 28    # image width (pixels)
DEFAULT_HEIGHT  = 28    # image height (pixels)
DEFAULT_CHANNEL = 1     # 1 = grayscale; 3 = RGB


class BrailleDatasetLoader:
    """
    Loads the Braille Character Dataset and returns either a
    tf.data.Dataset (for training) or a NumPy array (for display).

    Args:
        path (str): Absolute or relative path to the Braille image folder.
    """

    def __init__(self, path: str) -> None:
        self.path = path

    # ── Step 1: Collect file paths ───────────────────────────
    def _build_dataframe(self) -> pd.DataFrame:
        """Return a DataFrame with one column 'Path' for every image file."""
        rows = []
        for filename in tqdm(os.listdir(self.path), desc="Scanning files"):
            full_path = os.path.join(self.path, filename)
            rows.append(full_path)
        return pd.DataFrame(rows, columns=["Path"])

    # ── Step 2: Load & preprocess images ─────────────────────
    def _load_images(
        self,
        width:   int = DEFAULT_WIDTH,
        height:  int = DEFAULT_HEIGHT,
        channel: int = DEFAULT_CHANNEL,
    ) -> np.ndarray:
        """
        Load images, convert to grayscale (if channel==1),
        normalise to [-1, 1], and return a NumPy array of shape
        (N, height, width, channel).
        """
        df = self._build_dataframe()
        imgs = []

        for path in tqdm(df["Path"], desc="Loading images"):
            img = cv2.imread(path)                        # BGR uint8
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   # → RGB

            if channel == 1:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)   # → gray 2-D
                img = img.flatten()                            # 1-D vector
            else:
                img = img.flatten()

            imgs.append(img)

        x = np.array(imgs)
        x = x.reshape(x.shape[0], height, width, channel).astype("float32")

        # Normalise: map [0, 255] → [-1, 1]
        # The Generator uses tanh activation, so data must be in [-1, 1]
        x = (x - 127.5) / 127.5
        return x

    # ── Public API ────────────────────────────────────────────
    def get_numpy_array(
        self,
        width:   int = DEFAULT_WIDTH,
        height:  int = DEFAULT_HEIGHT,
        channel: int = DEFAULT_CHANNEL,
    ) -> np.ndarray:
        """
        Return all training images as a NumPy array of shape
        (N, height, width, channel) normalised to [-1, 1].

        Use this for sample display or quick experiments.
        """
        return self._load_images(width, height, channel)

    def get_tf_dataset(
        self,
        batch_size:  int = 128,
        buffer_size: int = 512,
        width:       int = DEFAULT_WIDTH,
        height:      int = DEFAULT_HEIGHT,
        channel:     int = DEFAULT_CHANNEL,
    ) -> tf.data.Dataset:
        """
        Return a shuffled & batched tf.data.Dataset ready for
        DCGAN training.

        Args:
            batch_size:   Number of images per training step.
            buffer_size:  Shuffle buffer size (larger = better mixing).

        Returns:
            tf.data.Dataset of float32 tensors in [-1, 1].
        """
        x = self._load_images(width, height, channel)
        dataset = tf.data.Dataset.from_tensor_slices(x)
        dataset = dataset.shuffle(buffer_size=buffer_size).batch(batch_size)
        return dataset
