# 🧠 DCGAN Experiment — BrailleVisionAI

> Experimental module for generating **synthetic Braille character images** using a **Deep Convolutional Generative Adversarial Network (DCGAN)**.

---

## 📌 What This Notebook Does

The Kaggle notebook `dcgan-on-braille-images.ipynb` (stored at the project root) implements a **DCGAN trained on the Braille Character Dataset** to synthesise new 28×28 grayscale Braille character images.

### Workflow at a Glance

```
Random noise z ~ N(0,1)
        │
        ▼
┌────────────────┐
│  Generator G   │  ← Learns to produce realistic Braille dots
└───────┬────────┘
        │ fake images
        ▼
┌────────────────┐    ┌─────────────────┐
│ Discriminator D│◄───│  Real Braille   │
└───────┬────────┘    │  Training Images│
        │             └─────────────────┘
        ▼
   Real / Fake?
   (adversarial loss signals fed back to update both G & D)
```

---

## 📂 Dataset Used

| Property | Value |
|---|---|
| **Name** | Braille Character Dataset |
| **Source** | Kaggle |
| **Format** | Flat folder of 28×28 grayscale PNG/JPG images |
| **Classes** | 26 (a–z), one Braille cell per image |
| **Est. Total Images** | ~6,500 (≈250 per class) |
| **Colour** | Grayscale (converted from RGB during loading) |
| **Pixel Range after normalisation** | [-1, 1] |

### Preprocessing Steps

1. Scan folder → collect all image paths
2. Load with OpenCV → convert BGR → RGB → Grayscale
3. Flatten to 1-D, reshape to `(N, 28, 28, 1)`
4. Normalise: `(pixel - 127.5) / 127.5` → range **[-1, 1]**
5. Wrap in `tf.data.Dataset`, shuffle, batch

---

## 🏗️ Architecture

### Generator

Converts a **100-dimensional noise vector** into a 28×28 grayscale image.

```
Input:  noise (100,)
  Dense(7×7×256) → BatchNorm → LeakyReLU
  Reshape → (7, 7, 256)
  Conv2DTranspose(128, 5×5, stride 1) → BN → LeakyReLU   [7→7]
  Conv2DTranspose( 64, 5×5, stride 2) → BN → LeakyReLU   [7→14]
  Conv2DTranspose( 32, 5×5, stride 2) → BN → LeakyReLU   [14→28]
  Conv2DTranspose(  1, 5×5, stride 1, activation=tanh)    [28→28]
Output: (28, 28, 1) ← values in [-1, 1]
```

### Discriminator

Classifies an image as **real** or **fake**.

```
Input:  (28, 28, 1)
  Conv2D(64, 3×3) → BN → MaxPool → LeakyReLU → Dropout(0.3)
  Conv2D(32, 3×3) → BN → MaxPool → LeakyReLU → Dropout(0.3)
  Conv2D(16, 3×3) → BN → MaxPool → LeakyReLU → Dropout(0.3)
  Flatten → Dense(128) → LeakyReLU → Dropout(0.2)
  Dense(1, sigmoid)
Output: probability (0 = fake, 1 = real)
```

---

## ⚙️ Training Hyperparameters

| Parameter | Value | Notes |
|---|---|---|
| `latent_dim` | 100 | Random noise vector length |
| `batch_size` | 128 | Images per training step |
| `lr` | 0.000097 | Adam learning rate (same for G & D) |
| `epochs` | 700 | Original notebook (Kaggle limit) |
| Loss | `BinaryCrossentropy(from_logits=True)` | Numerically stable |
| Optimizer | `Adam` | Separate for G and D |

---

## 📁 Modular Files in This Folder

| File | Purpose |
|---|---|
| `dataset_loader.py` | Loads & preprocesses Braille images, returns tf.data.Dataset |
| `model.py` | `build_generator()` and `build_discriminator()` functions |
| `train.py` | `DCGANTrainer` class — full training loop with checkpointing |
| `generate.py` | `BrailleGenerator` class — load model, generate & save images |
| `visualise.py` | Grid display and loss curve plotting helpers |
| `checkpoints/` | (created at runtime) Saved generator `.keras` model weights |
| `generated/` | (created at runtime) Saved synthetic Braille PNG images |
| `README.md` | This file |

---

## 🚀 How to Use

### 1. Run Full Training

```python
from experiments.dcgan.dataset_loader import BrailleDatasetLoader
from experiments.dcgan.train import DCGANTrainer

# Load data
loader  = BrailleDatasetLoader('/path/to/Braille Dataset')
dataset = loader.get_tf_dataset(batch_size=128)

# Train
trainer = DCGANTrainer(latent_dim=100, batch_size=128, lr=0.000097)
trainer.train(
    dataset,
    epochs    = 700,
    save_path = 'experiments/dcgan/checkpoints/best_generator.keras',
)
```

### 2. Generate Images from Trained Model

```python
from experiments.dcgan.generate import BrailleGenerator

gen = BrailleGenerator('experiments/dcgan/checkpoints/best_generator.keras')
gen.display_samples(n=20)           # matplotlib grid
gen.save_samples(n=100, output_dir='experiments/dcgan/generated/')  # PNG files
```

### 3. Demo without Training (Random Noise)

```python
from experiments.dcgan.generate import generate_random_grid
generate_random_grid(n=20)   # output looks like noise — expected
```

---

## 🔍 Is This DCGAN Useful for BrailleVisionAI?

### ✅ Useful For

| Use Case | Priority | Notes |
|---|---|---|
| **Data Augmentation** | 🟡 Medium | Supplement real Braille images to prevent overfitting in Phase D YOLO detector |
| **Synthetic Dataset Generation** | 🟡 Medium | Generate thousands of diverse Braille samples across all 26 characters |
| **Improving Robustness** | 🟡 Medium | Adds blur/dot variation the camera never captures |
| **Future Training** | 🟡 Medium | If real dataset is small, GAN data can fill the gap |
| **Hackathon Demo** | 🔴 Low | Showing GAN-generated images as a "bonus" capability |

### ❌ Not Necessary For

| Phase | Reason |
|---|---|
| **Phase D (YOLO detection)** | Needs real annotated images with bounding boxes; GAN output has no labels |
| **Phase E (Translation)** | No impact on rule-based Braille → English mapping |
| **Phase F (TTS)** | Completely independent |

---

## ⚠️ Limitations of This Dataset & GAN

| Limitation | Detail |
|---|---|
| **Small dataset** | ~6,500 images — GANs typically need 10k+ for sharp results |
| **28×28 resolution** | Very low res; real camera frames are much larger |
| **Grayscale only** | No colour/depth variation; real Braille has shadows, texture |
| **Flat folder structure** | No label hierarchy by character class |
| **Training cap (700 epochs)** | Original author was GPU-limited on Kaggle; more epochs = better |
| **No label conditioning** | Standard DCGAN cannot control which character is generated (would need cGAN) |
| **Dot size mismatch** | Generated dots may not match real embossed Braille cell geometry |

---

## 🗺️ Recommendations

| Action | Priority | When |
|---|---|---|
| ✅ Keep GAN code modular & documented | Done | Already done |
| ✅ Do NOT integrate into `app.py` yet | Done | Maintained as separate experiment |
| 🔜 Train locally if GPU is available | **Later (Phase D)** | After YOLO model exists |
| 🔜 Switch to **conditional GAN (cGAN)** | **Later** | To generate per-character images with labels |
| 🔜 Upgrade resolution (64×64 or 128×128) | **Later** | Would require architecture changes |
| ⏭️ Use for augmentation in Phase D | **Optional** | Only if real dataset proves too small |
| ❌ Train during hackathon (GPU cost) | **Skip for MVP** | Too slow; no training time during demo |

---

## 🏆 MVP Verdict

> **This GAN is NOT necessary for the hackathon MVP.**
>
> Phases A, B, C, D (YOLO detection) and E (translation) can be
> completed with real images only.  Keep this experiment as a
> **"future capability"** you can showcase as an advanced roadmap item.

---

## 📦 Dependencies

```
tensorflow >= 2.10
keras >= 2.10
opencv-python >= 4.9
numpy >= 1.26
pandas >= 1.4
matplotlib >= 3.6
tqdm >= 4.64
```

Install:
```bash
pip install tensorflow opencv-python numpy pandas matplotlib tqdm
```
