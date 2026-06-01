# Braille Detection Dataset

## Folder Structure

```
datasets/
├── braille/          ← Images of real Braille text (any angle, lighting)
└── non_braille/      ← Images of dot patterns that are NOT Braille
```

## What to put in `braille/`

- Photographs of embossed Braille books/pages
- Scanned Braille documents
- Close-up webcam shots of Braille
- Multiple lighting conditions (bright, dim, shadow)
- Multiple angles (straight-on, slight tilt)
- Multiple Braille types (Grade 1, Grade 2, numbers, punctuation)

**Minimum recommended:** 100 images  
**Ideal:** 500+ images  

## What to put in `non_braille/`

These are the common **false-positive traps** the model must learn to reject:

| Category | Examples |
|---|---|
| Bubble wrap | Close-up of air bubbles |
| Fabric / textile | Dotted patterns on cloth |
| Rocks / gravel | Pebble surfaces |
| Dotted paper | Perforated paper, hole-punched sheets |
| Electronic boards | PCB with solder dots |
| Skin texture | Pores, goosebumps |
| Food | Blueberries, grapes, seeds |
| Natural textures | Bark, coral, sponge |

**Minimum recommended:** 100 images  
**Should roughly match `braille/` in count**  

## Free Dataset Sources

1. **Kaggle Braille Character Dataset**  
   https://www.kaggle.com/datasets/...  
   (the same dataset used in the DCGAN experiment)

2. **Roboflow Braille Universe**  
   https://universe.roboflow.com/search?q=braille

3. **Non-Braille textures**  
   - DTD (Describable Textures Dataset): https://www.robots.ox.ac.uk/~vgg/data/dtd/
   - Search Kaggle for "texture dataset", "dotted patterns"

## After Collecting Images

Run the training script:

```bash
# Option A: YOLOv8 (recommended for real-time)
pip install ultralytics
yolo classify train data=datasets/ model=yolov8n-cls.pt epochs=50 imgsz=224 \
    project=models/braille_presence name=braille_yolov8_cls
# Then rename best.pt:
# models/braille_presence/braille_yolov8_cls/weights/best.pt
# → models/braille_presence/braille_yolov8_cls.pt

# Option B: MobileNetV2 Keras classifier
pip install tensorflow
python models/braille_presence/train_classifier.py --epochs 30
```

The BraillePresenceDetector will **automatically load** whichever model file it finds.  
If neither model exists, it runs in **heuristics-only mode** (still functional).
