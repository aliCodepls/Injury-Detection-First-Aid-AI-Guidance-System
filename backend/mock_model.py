"""
mock_model.py
-------------
Placeholder for your real CNN model.
Right now it returns a random wound type so you can test
the full camera → backend → phone flow without needing the trained model.

When your model is ready, ONLY replace the predict() function body.
Nothing else in the project changes.
"""

import random

# Must match your training class labels exactly
CLASSES = [
    "abrasion",
    "bruise",
    "burn",
    "cut",
    "ingrown_nail",
    "laceration",
    "stab_wound",
]

SEVERITY_MAP = {
    "abrasion":     "mild",
    "bruise":       "mild",
    "burn":         "moderate",
    "cut":          "mild",
    "ingrown_nail": "mild",
    "laceration":   "moderate",
    "stab_wound":   "severe",
}

EMERGENCY_CLASSES = {"stab_wound", "burn", "laceration"}


def predict(image_bytes: bytes, severity: str, bleeding: bool, swelling: bool) -> dict:
    """
    MOCK — returns a random wound class with fake confidence.

    ── REPLACE THIS BODY WHEN YOUR MODEL IS READY ──────────────────────────
    from PIL import Image
    import numpy as np
    import tensorflow as tf
    import io

    model = tf.keras.models.load_model("firstsight_model.h5")

    img = Image.open(io.BytesIO(image_bytes)).resize((224, 224))
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)
    preds = model.predict(arr)[0]
    class_idx = int(np.argmax(preds))
    confidence = float(preds[class_idx])
    wound_type = CLASSES[class_idx]
    ─────────────────────────────────────────────────────────────────────────
    """

    # Random pick for now
    wound_type = random.choice(CLASSES)
    confidence = round(random.uniform(0.72, 0.97), 2)

    # Adjust severity based on metadata
    base_severity = SEVERITY_MAP[wound_type]
    if bleeding and base_severity == "mild":
        base_severity = "moderate"
    if severity == "severe" or (bleeding and swelling):
        base_severity = "severe"

    seek_emergency = wound_type in EMERGENCY_CLASSES or base_severity == "severe"

    return {
        "wound_type":    wound_type,
        "confidence":    confidence,
        "severity_level": base_severity,
        "seek_emergency": seek_emergency,
    }
