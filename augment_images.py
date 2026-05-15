"""
Data Augmentation Script
------------------------
Takes images from INPUT_FOLDER and saves ONLY augmented versions to OUTPUT_FOLDER.
Does NOT copy originals to output.

Install dependencies:
    pip install albumentations opencv-python
"""

import os
import cv2
import random
import albumentations as A
from pathlib import Path

# ── CONFIG ──────────────────────────────────────────────────────────────────
INPUT_FOLDER  = "input_images"   # folder with your original 150 images
OUTPUT_FOLDER = "output_images"  # only augmented images saved here
TARGET_COUNT  = 150              # how many augmented images to generate
# ────────────────────────────────────────────────────────────────────────────

augment = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.2),
    A.Rotate(limit=25, p=0.7),
    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.8),
    A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=30, val_shift_limit=20, p=0.5),
    A.GaussNoise(var_limit=(5.0, 30.0), p=0.4),
    A.Blur(blur_limit=3, p=0.3),
    A.RandomScale(scale_limit=0.2, p=0.5),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5),
    A.RandomShadow(p=0.3),
    A.CLAHE(p=0.3),
])

def load_images(folder):
    supported = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images = []
    for f in Path(folder).iterdir():
        if f.suffix.lower() in supported:
            img = cv2.imread(str(f))
            if img is not None:
                images.append((f.stem, img))
    return images

def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    print(f"Loading images from '{INPUT_FOLDER}'...")
    originals = load_images(INPUT_FOLDER)

    if not originals:
        print("No images found! Check your INPUT_FOLDER path.")
        return

    print(f"Found {len(originals)} original images.")
    print(f"Generating {TARGET_COUNT} augmented images...")

    for i in range(TARGET_COUNT):
        name, img = random.choice(originals)
        augmented = augment(image=img)["image"]
        out_path = os.path.join(OUTPUT_FOLDER, f"{name}_aug_{i:04d}.jpg")
        cv2.imwrite(out_path, augmented)

    print(f"\nDone! {TARGET_COUNT} augmented images saved to '{OUTPUT_FOLDER}/'")

if __name__ == "__main__":
    main()
