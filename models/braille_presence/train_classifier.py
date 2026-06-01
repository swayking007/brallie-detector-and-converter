"""
============================================================
BrailleVisionAI — Phase D  |  AI Model Training Script
models/braille_presence/train_classifier.py
============================================================

PURPOSE
-------
Trains a YOLOv8 classification model OR a lightweight MobileNetV2
binary classifier to distinguish Braille from non-Braille images.

DATASET EXPECTED STRUCTURE
--------------------------
  datasets/
  ├── braille/
  │   ├── img001.jpg
  │   ├── img002.png
  │   └── ...
  └── non_braille/
      ├── bubble_wrap.jpg
      ├── fabric.jpg
      ├── rocks.jpg
      └── ...

  Each subfolder = one class.  Minimum recommended: 100 images per class.
  More images → better model.  500+ per class is ideal.

OPTION A — YOLOv8 Classification (RECOMMENDED for real-time)
-------------------------------------------------------------
  yolo classify train \
      data=datasets/ \
      model=yolov8n-cls.pt \
      epochs=50 \
      imgsz=224 \
      project=models/braille_presence \
      name=braille_yolov8_cls

  This creates:
      models/braille_presence/braille_yolov8_cls/weights/best.pt

  Rename to:
      models/braille_presence/braille_yolov8_cls.pt

OPTION B — MobileNetV2 Keras Classifier (pure Python, no YOLO CLI)
--------------------------------------------------------------------
  Run this script directly:
      python models/braille_presence/train_classifier.py

  This trains a MobileNetV2 fine-tuned on your Braille/non-Braille
  images and saves:
      models/braille_presence/braille_classifier.keras

HOW THE TRAINED MODEL IS USED
------------------------------
  BraillePresenceDetector auto-loads whichever file it finds:
    1. models/braille_presence/braille_yolov8_cls.pt   (YOLO preferred)
    2. models/braille_presence/braille_classifier.keras (Keras fallback)

  If neither exists, heuristics-only mode is used automatically.
============================================================
"""

import os
import sys
import numpy as np

# ── Paths ─────────────────────────────────────────────────────
ROOT_DIR    = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATASET_DIR = os.path.join(ROOT_DIR, "datasets/braille_presence")
MODEL_DIR   = os.path.join(ROOT_DIR, "models", "braille_presence")
SAVE_PATH   = os.path.join(MODEL_DIR, "braille_classifier.keras")

# Training config
IMG_SIZE   = 224
BATCH_SIZE = 16
EPOCHS     = 30
LR         = 1e-4


def build_mobilenet_classifier(img_size: int = 224) -> "tf.keras.Model":
    """
    Build a MobileNetV2 binary classifier.

    Architecture:
        MobileNetV2 (pretrained ImageNet, frozen initially)
        GlobalAveragePooling2D
        Dense(128, relu) + Dropout(0.4)
        Dense(1, sigmoid)   ← 1 = braille, 0 = non_braille
    """
    import tensorflow as tf
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input
    from tensorflow.keras.models import Model

    base = MobileNetV2(
        input_shape=(img_size, img_size, 3),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False   # freeze pretrained layers initially

    inputs = Input(shape=(img_size, img_size, 3))
    x      = base(inputs, training=False)
    x      = GlobalAveragePooling2D()(x)
    x      = Dense(128, activation="relu")(x)
    x      = Dropout(0.4)(x)
    outputs = Dense(1, activation="sigmoid")(x)

    model = Model(inputs, outputs, name="BrailleClassifier")
    return model


def build_tf_dataset(
    data_dir:   str,
    img_size:   int = IMG_SIZE,
    batch_size: int = BATCH_SIZE,
    split:      str = "training",
    val_split:  float = 0.2,
    seed:       int = 42,
):
    """
    Build a tf.data.Dataset from class-per-folder image directory.

    Expects data_dir to contain subfolders 'braille/' and 'non_braille/'.
    """
    import tensorflow as tf
    ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=val_split,
        subset=split,
        seed=seed,
        image_size=(img_size, img_size),
        batch_size=batch_size,
        label_mode="binary",
        class_names=["non_braille", "braille"],   # 0=non, 1=braille
    )
    # Normalise [0,255] → [0,1]
    normalization_layer = tf.keras.layers.Rescaling(1.0 / 255)
    ds = ds.map(lambda x, y: (normalization_layer(x), y))
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def train(
    data_dir:   str = DATASET_DIR,
    save_path:  str = SAVE_PATH,
    img_size:   int = IMG_SIZE,
    batch_size: int = BATCH_SIZE,
    epochs:     int = EPOCHS,
    lr:         float = LR,
) -> None:
    """
    Full training pipeline:
        1. Load datasets/braille and datasets/non_braille
        2. Build MobileNetV2 classifier
        3. Train with frozen base (initial pass)
        4. Unfreeze top layers, fine-tune at lower LR
        5. Save model to models/braille_presence/braille_classifier.keras

    Args:
        data_dir:   Root of the dataset folder (contains braille/ and non_braille/).
        save_path:  Where to save the trained model.
        img_size:   Input image size (square, default 224).
        batch_size: Training batch size.
        epochs:     Total number of training epochs.
        lr:         Initial learning rate.
    """
    import tensorflow as tf
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

    print(f"[Training] Dataset dir: {data_dir}")
    print(f"[Training] Save path:   {save_path}")

    # ── Load datasets ────────────────────────────────────────
    train_ds = build_tf_dataset(data_dir, img_size, batch_size, split="training")
    val_ds   = build_tf_dataset(data_dir, img_size, batch_size, split="validation")

    class_names = ["non_braille", "braille"]
    print(f"[Training] Classes: {class_names}")

    # ── Build model ──────────────────────────────────────────
    model = build_mobilenet_classifier(img_size)
    model.compile(
        optimizer=Adam(lr),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    # ── Callbacks ────────────────────────────────────────────
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    callbacks = [
        ModelCheckpoint(save_path, monitor="val_accuracy", save_best_only=True, verbose=1),
        EarlyStopping(monitor="val_accuracy", patience=6, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1),
    ]

    # ── Phase 1: Train with frozen base ──────────────────────
    print("\n=== Phase 1: Frozen base — training classifier head ===")
    phase1_epochs = min(10, epochs // 3)
    model.fit(train_ds, validation_data=val_ds, epochs=phase1_epochs, callbacks=callbacks)

    # ── Phase 2: Skip fine-tuning for hackathon build ─────────

    print("\n=== Skipping Phase 2 fine-tuning ===")

# Keep MobileNet frozen
    base_model = model.layers[1]   # MobileNetV2 layer
    base_model.trainable = False

    print("[Training] Using saved classifier without extra fine-tuning")
    print(f"\n[Training] ✅ Model saved to: {save_path}")

    # ── Quick evaluation ─────────────────────────────────────
    loss, acc = model.evaluate(val_ds, verbose=0)
    print(f"[Training] Final val accuracy: {acc:.3f}  |  val loss: {loss:.4f}")


# ── CLI ─────────────────────────────────────────────────────
if __name__ == "__main__":
    """
    Run:
        python models/braille_presence/train_classifier.py

    Options (command line):
        --data_dir   Path to dataset folder (default: datasets/)
        --epochs     Number of epochs (default: 30)
        --batch_size Batch size (default: 16)
    """
    import argparse

    parser = argparse.ArgumentParser(description="Train Braille presence classifier")
    parser.add_argument("--data_dir",   default=DATASET_DIR, help="Dataset root folder")
    parser.add_argument("--save_path",  default=SAVE_PATH,   help="Output model path")
    parser.add_argument("--epochs",     type=int, default=EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr",         type=float, default=LR)
    args = parser.parse_args()

    train(
        data_dir   = args.data_dir,
        save_path  = args.save_path,
        epochs     = args.epochs,
        batch_size = args.batch_size,
        lr         = args.lr,
    )
