"""
============================================================
BrailleVisionAI — DCGAN Experiment  |  Model Architecture
experiments/dcgan/model.py
============================================================

PURPOSE
-------
Defines the Generator and Discriminator neural networks used in the
Deep Convolutional Generative Adversarial Network (DCGAN) for
Braille character image synthesis.

ARCHITECTURE OVERVIEW
---------------------

Generator (Noise → 28×28 Braille image)
  Input:  1-D random noise vector of length `latent_dim` (default 100)
  ┌─────────────────────────────────────────────────────────────┐
  │ Dense(7×7×256)  → BatchNorm → LeakyReLU                    │
  │ Reshape → (7, 7, 256)                                       │
  │ Conv2DTranspose(128, 5×5, stride 1×1) → BN → LeakyReLU     │  7→ 7
  │ Conv2DTranspose( 64, 5×5, stride 2×2) → BN → LeakyReLU     │  7→14
  │ Conv2DTranspose( 32, 5×5, stride 2×2) → BN → LeakyReLU     │ 14→28
  │ Conv2DTranspose(  1, 5×5, stride 1×1, activation=tanh)     │ 28→28
  └─────────────────────────────────────────────────────────────┘
  Output: (28, 28, 1) grayscale image in [-1, 1]

Discriminator (28×28 image → Real/Fake probability)
  Input:  (28, 28, 1) image tensor
  ┌─────────────────────────────────────────────────────────────┐
  │ Conv2D(64, 3×3) → BN → MaxPool(2) → LeakyReLU → Dropout    │
  │ Conv2D(32, 3×3) → BN → MaxPool(2) → LeakyReLU → Dropout    │
  │ Conv2D(16, 3×3) → BN → MaxPool(2) → LeakyReLU → Dropout    │
  │ Flatten                                                     │
  │ Dense(128) → LeakyReLU → Dropout(0.2)                       │
  │ Dense(1, activation=sigmoid)                                │
  └─────────────────────────────────────────────────────────────┘
  Output: scalar probability (0 = fake, 1 = real)

WHY THESE DESIGN CHOICES?
--------------------------
  - Conv2DTranspose (deconvolution): upsamples feature maps so the
    generator gradually builds spatial structure from a tiny 7×7 seed.
  - BatchNormalization: stabilises training by normalising layer inputs,
    accelerates convergence.
  - LeakyReLU (α=0.2): avoids the "dying ReLU" problem in GANs; lets
    small gradients flow for negative activations.
  - tanh output on generator: maps outputs to [-1, 1] matching the
    normalised training data range.
  - Dropout in discriminator: prevents discriminator from becoming
    too powerful relative to the generator (balance is key in GANs).
  - BinaryCrossentropy(from_logits=True): numerically stable; the
    logits are processed internally before applying sigmoid.

HOW TO USE
----------
    from experiments.dcgan.model import build_generator, build_discriminator

    G = build_generator(latent_dim=100)
    D = build_discriminator()
    G.summary()
    D.summary()

============================================================
"""

import tensorflow as tf
from keras.models import Sequential
from keras.layers import (
    Dense, Conv2DTranspose, Conv2D, Flatten, Reshape,
    BatchNormalization, MaxPool2D, LeakyReLU, Dropout,
)


# ── Generator ────────────────────────────────────────────────
def build_generator(latent_dim: int = 100) -> Sequential:
    """
    Build the Generator model.

    The generator takes a random latent vector and produces a
    28×28×1 grayscale image using transpose convolutions.

    Args:
        latent_dim: Length of the input noise vector (default 100).

    Returns:
        Keras Sequential model (not yet compiled).
    """
    model = Sequential(name="Generator")

    # ── Seed: map noise → 7×7×256 feature map ───────────────
    model.add(Dense(7 * 7 * 256, use_bias=False, input_shape=(latent_dim,)))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(Reshape((7, 7, 256)))
    assert model.output_shape == (None, 7, 7, 256)

    # ── Upsample: 7×7 → 7×7 (more filters, same spatial size) ─
    model.add(Conv2DTranspose(128, (5, 5), strides=(1, 1), padding="same", use_bias=False))
    assert model.output_shape == (None, 7, 7, 128)
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    # ── Upsample: 7×7 → 14×14 ───────────────────────────────
    model.add(Conv2DTranspose(64, (5, 5), strides=(2, 2), padding="same", use_bias=False))
    assert model.output_shape == (None, 14, 14, 64)
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    # ── Upsample: 14×14 → 28×28 ─────────────────────────────
    model.add(Conv2DTranspose(32, (5, 5), strides=(2, 2), padding="same", use_bias=False))
    assert model.output_shape == (None, 28, 28, 32)
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    # ── Output: 28×28 → 28×28×1 (tanh maps to [-1, 1]) ──────
    model.add(Conv2DTranspose(1, (5, 5), strides=(1, 1), padding="same",
                              use_bias=False, activation="tanh"))
    assert model.output_shape == (None, 28, 28, 1)

    return model


# ── Discriminator ────────────────────────────────────────────
def build_discriminator() -> Sequential:
    """
    Build the Discriminator model.

    The discriminator takes a 28×28×1 image and outputs a scalar
    probability of it being real (1) vs. fake (0).

    Returns:
        Keras Sequential model (not yet compiled).
    """
    model = Sequential(name="Discriminator")

    # ── Conv block 1 ─────────────────────────────────────────
    model.add(Conv2D(64, (3, 3), padding="same", input_shape=[28, 28, 1]))
    model.add(BatchNormalization())
    model.add(MaxPool2D(2))
    model.add(LeakyReLU())
    model.add(Dropout(0.3))

    # ── Conv block 2 ─────────────────────────────────────────
    model.add(Conv2D(32, (3, 3)))
    model.add(BatchNormalization())
    model.add(MaxPool2D(2))
    model.add(LeakyReLU())
    model.add(Dropout(0.3))

    # ── Conv block 3 ─────────────────────────────────────────
    model.add(Conv2D(16, (3, 3)))
    model.add(BatchNormalization())
    model.add(MaxPool2D(2))
    model.add(LeakyReLU())
    model.add(Dropout(0.3))

    # ── Dense classifier ─────────────────────────────────────
    model.add(Flatten())
    model.add(Dense(128))
    model.add(LeakyReLU())
    model.add(Dropout(0.2))
    model.add(Dense(1, activation="sigmoid"))   # 0 = fake, 1 = real

    return model
