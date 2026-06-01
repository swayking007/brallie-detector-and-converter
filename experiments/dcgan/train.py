"""
============================================================
BrailleVisionAI — DCGAN Experiment  |  Training Loop
experiments/dcgan/train.py
============================================================

PURPOSE
-------
Implements the DCGAN training loop.  Trains Generator + Discriminator
simultaneously using the adversarial min-max game.

TRAINING STRATEGY
-----------------
  For each epoch:
    For each mini-batch of real images:
      1. Sample random noise z ~ N(0,1) of shape (batch_size, latent_dim)
      2. Forward pass Generator: fake_imgs = G(z)
      3. Forward pass Discriminator on real images → real_output
      4. Forward pass Discriminator on fake images → fake_output
      5. Compute losses:
           Generator loss:     G wants D to classify fakes as Real → log(D(G(z)))
           Discriminator loss: D wants to correctly classify all    → log(D(x)) + log(1-D(G(z)))
      6. Compute gradients using GradientTape (automatic differentiation)
      7. Apply gradients to update G and D weights independently
  End of epoch:
    - Record average generator loss
    - Save best generator weights (by lowest generator loss)

LOSS FUNCTIONS
--------------
  BinaryCrossentropy(from_logits=True) — numerically stable

  Generator loss:
    cross_entropy(ones_like(fake_output), fake_output)
    → G succeeds when D thinks fake images are real (label=1)

  Discriminator real loss:
    cross_entropy(ones_like(real_output), real_output)
    → D must predict real images as real (label=1)

  Discriminator fake loss:
    cross_entropy(zeros_like(fake_output), fake_output)
    → D must predict fake images as fake (label=0)

  Total discriminator loss = real_loss + fake_loss

CHECKPOINTING
-------------
  Generator weights saved to `save_path` whenever a new
  lowest average generator loss is observed.

HOW TO USE
----------
    from experiments.dcgan.dataset_loader import BrailleDatasetLoader
    from experiments.dcgan.train import DCGANTrainer

    loader  = BrailleDatasetLoader('/path/to/Braille Dataset')
    dataset = loader.get_tf_dataset(batch_size=128)

    trainer = DCGANTrainer(latent_dim=100, batch_size=128, lr=0.000097)
    trainer.train(dataset, epochs=700, save_path='experiments/dcgan/checkpoints/best_generator.keras')

RESOURCES NOTE
--------------
  The original notebook trained for 700 epochs on Kaggle (limited GPU).
  More epochs (1000+) with sufficient GPU time typically yield sharper
  Braille dot patterns.  Expect ~1-3 minutes per epoch on a modern GPU.
============================================================
"""

import tensorflow as tf
from keras.models import Sequential
from keras.losses import BinaryCrossentropy
from keras.optimizers import Adam
from tqdm import tqdm

from experiments.dcgan.model import build_generator, build_discriminator


class DCGANTrainer:
    """
    Encapsulates the DCGAN training loop.

    Args:
        latent_dim (int):   Length of the noise input vector (default 100).
        batch_size (int):   Training batch size (default 128).
        lr (float):         Learning rate for both Adam optimisers (default 0.000097).
    """

    def __init__(
        self,
        latent_dim: int   = 100,
        batch_size: int   = 128,
        lr:         float = 0.000097,
    ) -> None:
        self.latent_dim = latent_dim
        self.batch_size = batch_size

        # ── Build networks ───────────────────────────────────
        self.generator:     Sequential = build_generator(latent_dim)
        self.discriminator: Sequential = build_discriminator()

        # ── Loss function ────────────────────────────────────
        # from_logits=True → numerically stable (handles the sigmoid internally)
        self.cross_entropy = BinaryCrossentropy(from_logits=True)

        # ── Optimisers ───────────────────────────────────────
        self.gen_optimizer  = Adam(lr)
        self.disc_optimizer = Adam(lr)

        # ── Loss history (for plotting / monitoring) ─────────
        self.gen_loss_history  = []
        self.disc_loss_history = []

    # ── Loss helpers ─────────────────────────────────────────
    def _generator_loss(self, fake_output: tf.Tensor) -> tf.Tensor:
        """
        Generator wants discriminator to classify ALL fakes as real.
        Loss = cross_entropy(all-ones, fake_output)
        """
        return self.cross_entropy(tf.ones_like(fake_output), fake_output)

    def _discriminator_loss(
        self,
        real_output: tf.Tensor,
        fake_output: tf.Tensor,
    ):
        """
        Discriminator wants:
          - real images → classified as real (label 1)
          - fake images → classified as fake (label 0)
        Returns (real_loss, fake_loss) as separate tensors.
        """
        real_loss = self.cross_entropy(tf.ones_like(real_output),  real_output)
        fake_loss = self.cross_entropy(tf.zeros_like(fake_output), fake_output)
        return real_loss, fake_loss

    # ── Single training step (one mini-batch) ────────────────
    @tf.function   # compile to a TensorFlow graph for speed
    def _train_step(self, real_images: tf.Tensor) -> None:
        """Run one adversarial training step on a single mini-batch."""
        noise = tf.random.normal([self.batch_size, self.latent_dim])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            fake_images  = self.generator(noise, training=True)

            real_output  = self.discriminator(real_images,  training=True)
            fake_output  = self.discriminator(fake_images,  training=True)

            gen_loss              = self._generator_loss(fake_output)
            real_loss, fake_loss  = self._discriminator_loss(real_output, fake_output)
            disc_loss             = real_loss + fake_loss

        # ── Update Generator ─────────────────────────────────
        gen_grads = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        self.gen_optimizer.apply_gradients(
            zip(gen_grads, self.generator.trainable_variables)
        )

        # ── Update Discriminator ─────────────────────────────
        disc_grads = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
        self.disc_optimizer.apply_gradients(
            zip(disc_grads, self.discriminator.trainable_variables)
        )

        self.gen_loss_history.append(gen_loss)
        self.disc_loss_history.append([real_loss, fake_loss])

    # ── Full training loop ───────────────────────────────────
    def train(
        self,
        dataset:    tf.data.Dataset,
        epochs:     int = 700,
        save_path:  str = "experiments/dcgan/checkpoints/best_generator.keras",
    ) -> None:
        """
        Train the DCGAN for `epochs` epochs.

        The Generator model is saved to `save_path` whenever a new
        minimum average generator loss is achieved.

        Args:
            dataset:   Shuffled + batched tf.data.Dataset from BrailleDatasetLoader.
            epochs:    Number of full passes over the training data.
            save_path: File path where the best generator model is saved.
        """
        # Compile (needed for .save() compatibility)
        self.generator.compile(
            optimizer=self.gen_optimizer,
            loss=self._generator_loss,
        )
        self.discriminator.compile(
            optimizer=self.disc_optimizer,
            loss=self._discriminator_loss,
        )

        best_loss = float("inf")

        for epoch in tqdm(range(epochs), desc="Epochs"):
            # ── Iterate over all batches ──────────────────────
            for batch in dataset:
                self._train_step(batch)

            # ── Epoch-level metrics ───────────────────────────
            avg_gen_loss = tf.reduce_mean(self.gen_loss_history).numpy()

            if avg_gen_loss < best_loss:
                best_loss = avg_gen_loss
                self.generator.save(save_path)
                print(f"  [Checkpoint] saved  avg_gen_loss={avg_gen_loss:.4f}")

            print(
                f"Epoch {epoch + 1}/{epochs}  |  "
                f"avg_gen_loss={avg_gen_loss:.4f}"
            )
            print("=" * 60)

        print("Training complete.")
