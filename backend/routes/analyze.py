from flask import Blueprint, request, jsonify
from mock_model import predict

analyze_bp = Blueprint("analyze", __name__)


@analyze_bp.route("/analyze", methods=["POST"])
def analyze():

    # Validate image
    if "image" not in request.files:
        return jsonify({"error": "No image provided."}), 400

    image_file = request.files["image"]
    if image_file.filename == "":
        return jsonify({"error": "Empty filename."}), 400

    image_bytes = image_file.read()

    # Parse metadata
    severity = request.form.get("severity", "mild").lower()
    if severity not in ("mild", "moderate", "severe"):
        severity = "mild"

    bleeding = request.form.get("bleeding", "false").lower() in ("true", "1", "yes")
    swelling = request.form.get("swelling", "false").lower() in ("true", "1", "yes")

    # Run model
    try:
        result = predict(image_bytes, severity, bleeding, swelling)
    except Exception as e:
        return jsonify({"error": f"Model error: {str(e)}"}), 500

    wound_type     = result["wound_type"]
    confidence     = result["confidence"]
    severity_level = result["severity_level"]
    seek_emergency = result["seek_emergency"]

    # Placeholder first aid — Gemini will replace this later
    first_aid = {
        "steps": [
            "Clean the wound with water.",
            "Apply antiseptic.",
            "Cover with a sterile bandage.",
            "Monitor for signs of infection.",
        ],
        "do_not": [
            "Do not touch the wound with dirty hands.",
            "Do not ignore signs of infection.",
        ],
    }

    return jsonify({
        "success":          True,
        "wound_type":       wound_type,
        "wound_label":      wound_type.replace("_", " ").title(),
        "confidence":       confidence,
        "confidence_percent": f"{int(confidence * 100)}%",
        "severity_level":   severity_level,
        "seek_emergency":   seek_emergency,
        "first_aid":        first_aid,
    }), 200
