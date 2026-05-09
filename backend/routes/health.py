from flask import Blueprint, jsonify

health_bp = Blueprint("health", __name__)


@health_bp.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status":       "ok",
        "service":      "FirstSight AI Backend",
        "model_status": "mock",
        "version":      "1.0.0",
    }), 200
