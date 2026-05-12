"""
FIRSTSIGHT AI — full wound pipeline in one module:

1) Classify with CNN weights (best_wound_model.pth; optional WOUND_MODEL_PATH).
2) Run WoundWatch vision (OpenAI-compatible /predict-style JSON).
3) Persist both + merged clinical view (vision prioritized over CNN for triage fields).
4) Optional HTTP server: POST /predict (multipart image).

Usage:
  python final_pipeline.py path/to/wound.jpg
  python final_pipeline.py path/to/wound.jpg --no-vision   # CNN only
  python final_pipeline.py --serve --port 5050
  python final_pipeline.py --serve --no-vision   # API/UI without llama-server (CNN only)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import traceback
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "pipeline_outputs"

# Lazy imports for vision / server to allow CNN-only runs without requests/flask
def _run_cnn(image_bytes: bytes) -> dict[str, Any]:
    from wound_model_infer import predict_wound_class_from_bytes

    return predict_wound_class_from_bytes(image_bytes)


def _run_vision(image_bytes: bytes, mime: str) -> dict[str, Any]:
    from woundwatch import analyze_wound_with_vision, normalize_vision_output

    raw = analyze_wound_with_vision(image_bytes, mime=mime)
    normalized = normalize_vision_output(raw)
    return {"raw": raw, "normalized": normalized}


def _injury_agreement(cnn_label: str, vision_label: str) -> bool:
    def norm(s: str) -> str:
        s = s.lower().replace("_", " ").replace("-", " ")
        s = re.sub(r"[^a-z0-9\s]", "", s)
        return " ".join(s.split())

    a, b = norm(cnn_label), norm(vision_label)
    if a == b:
        return True
    if a in b or b in a:
        return True
    tokens_a, tokens_b = set(a.split()), set(b.split())
    return bool(tokens_a & tokens_b)


def merge_results(cnn: dict[str, Any], vision_norm: dict[str, Any]) -> dict[str, Any]:
    """
    Clinical fields follow WoundWatch (vision). CNN is advisory and shown alongside.
    """
    return {
        "primary_source": "woundwatch_vision",
        "injury_type": vision_norm["injury_type"],
        "severity": vision_norm["severity"],
        "bleeding": vision_norm["has_bleeding"],
        "swelling": vision_norm["has_swelling"],
        "emergency_needed": vision_norm["emergency_needed"],
        "first_aid_steps": vision_norm["first_aid_steps"],
        "clinical_notes": vision_norm.get("clinical_notes", ""),
        "best_model_cnn": {
            "injury_type": cnn["injury_type"],
            "confidence": cnn["confidence"],
            "class_probabilities": cnn["class_probabilities"],
        },
        "models_agree_on_category": _injury_agreement(cnn["injury_type"], vision_norm["injury_type"]),
    }


def save_session(
    image_bytes: bytes,
    cnn: dict[str, Any],
    vision_bundle: dict[str, Any] | None,
    merged: dict[str, Any],
) -> dict[str, str]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    uid = uuid.uuid4().hex[:8]
    base = OUTPUT_DIR / f"scan_{stamp}_{uid}"

    paths: dict[str, str] = {}
    img_path = base.with_suffix(".bin")
    with open(img_path, "wb") as f:
        f.write(image_bytes)
    paths["image_saved"] = str(img_path)

    with open(f"{base}_best_model.json", "w", encoding="utf-8") as f:
        json.dump(cnn, f, indent=2)
    paths["best_model_json"] = str(Path(f"{base}_best_model.json"))

    if vision_bundle is not None:
        with open(f"{base}_woundwatch.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "normalized": vision_bundle["normalized"],
                    "raw": {k: v for k, v in vision_bundle["raw"].items() if k != "_raw_model_content"},
                    "raw_model_content": vision_bundle["raw"].get("_raw_model_content", ""),
                },
                f,
                indent=2,
            )
        paths["woundwatch_json"] = str(Path(f"{base}_woundwatch.json"))

    with open(f"{base}_merged.json", "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2)
    paths["merged_json"] = str(Path(f"{base}_merged.json"))

    return paths


def process_image_bytes(
    image_bytes: bytes,
    mime: str = "image/jpeg",
    *,
    use_vision: bool = True,
) -> dict[str, Any]:
    """
    Dual path: (1) image -> PyTorch CNN (best_wound_model.pth) in-process.
    (2) Same image -> WoundWatch HTTP POST (OpenAI-style chat+vision) to a local server —
    Ollama (e.g. Gemma on :11434) or llama-server (:11435), controlled by WOUNDWATCH_* env.

    CNN probabilities are not sent inside the vision prompt; vision sees image + fixed text
    prompts only. Outputs are merged afterward (vision drives triage fields; CNN is advisory).
    """
    cnn = _run_cnn(image_bytes)
    if not use_vision:
        merged = {
            "primary_source": "best_model_cnn_only",
            "injury_type": cnn["injury_type"],
            "severity": min(10, max(1, int(round((1.0 - cnn["confidence"]) * 6 + 3)))),
            "bleeding": False,
            "swelling": False,
            "emergency_needed": cnn["injury_type"] in ("Burns", "Cuts_lacerations"),
            "first_aid_steps": [
                "Keep the area clean and dry; avoid rubbing.",
                "If bleeding, apply gentle pressure with sterile gauze.",
                "Monitor for signs of infection (increasing pain, pus, fever).",
                "Use cool running water for minor thermal burns (not ice).",
                "Seek professional care if symptoms worsen or emergency flags apply.",
            ],
            "clinical_notes": "Vision model disabled; severity and first aid are heuristic placeholders from CNN confidence only.",
            "best_model_cnn": {
                "injury_type": cnn["injury_type"],
                "confidence": cnn["confidence"],
                "class_probabilities": cnn["class_probabilities"],
            },
            "models_agree_on_category": None,
        }
        paths = save_session(image_bytes, cnn, None, merged)
        return {"merged": merged, "cnn": cnn, "vision": None, "saved_paths": paths}

    vision_bundle = _run_vision(image_bytes, mime)
    merged = merge_results(cnn, vision_bundle["normalized"])
    paths = save_session(image_bytes, cnn, vision_bundle, merged)
    return {
        "merged": merged,
        "cnn": cnn,
        "vision": vision_bundle,
        "saved_paths": paths,
    }


def _guess_mime(filename: str) -> str:
    lower = filename.lower()
    if lower.endswith(".png"):
        return "image/png"
    if lower.endswith(".webp"):
        return "image/webp"
    return "image/jpeg"


def _dist_safe_file(dist: Path, rel: str) -> Path | None:
    """Return path only if it is a file under dist (no path traversal)."""
    try:
        base = dist.resolve()
        target = (dist / rel).resolve()
        target.relative_to(base)
    except (ValueError, OSError):
        return None
    return target if target.is_file() else None


def create_app():
    from flask import Flask, abort, jsonify, request, send_from_directory

    app = Flask(__name__)
    app.config.setdefault("USE_VISION", True)
    dist = SCRIPT_DIR / "frontend" / "dist"
    dist_assets = dist / "assets"

    @app.after_request
    def cors(resp):
        resp.headers["Access-Control-Allow-Origin"] = "*"
        resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
        return resp

    @app.route("/predict", methods=["OPTIONS"])
    def predict_options():
        return "", 204

    @app.route("/predict", methods=["POST"])
    def predict():
        if not request.files.get("image"):
            return jsonify({"error": "Missing multipart field 'image'"}), 400
        f = request.files["image"]
        data = f.read()
        if not data:
            return jsonify({"error": "Empty file"}), 400
        mime = f.mimetype or _guess_mime(f.filename or "")
        use_vision = bool(app.config.get("USE_VISION", True))
        try:
            bundle = process_image_bytes(data, mime=mime, use_vision=use_vision)
        except Exception as e:
            traceback.print_exc()
            return jsonify({"error": str(e), "exception": type(e).__name__}), 500
        m = bundle["merged"]
        body = {
            "injury_type": m["injury_type"],
            "severity": m["severity"],
            "bleeding": m["bleeding"],
            "swelling": m["swelling"],
            "emergency_needed": m["emergency_needed"],
            "first_aid_steps": m["first_aid_steps"],
            "clinical_notes": m.get("clinical_notes", ""),
            "best_model": bundle["cnn"],
            "woundwatch": bundle["vision"]["normalized"] if bundle["vision"] else None,
            "merged_meta": {
                "primary_source": m["primary_source"],
                "models_agree_on_category": m.get("models_agree_on_category"),
            },
            "saved_paths": bundle["saved_paths"],
        }
        return jsonify(body)

    # --- Static UI (Vite build): fetch("/predict") requires same origin as this server ---
    @app.route("/")
    def ui_index():
        if not (dist / "index.html").is_file():
            return jsonify({"error": f"Missing frontend build: {dist / 'index.html'}"}), 503
        return send_from_directory(dist, "index.html")

    @app.route("/assets/<path:filename>")
    def ui_assets(filename: str):
        if not dist_assets.is_dir():
            abort(503)
        return send_from_directory(dist_assets, filename)

    @app.route("/favicon.svg")
    def ui_favicon():
        if _dist_safe_file(dist, "favicon.svg"):
            return send_from_directory(dist, "favicon.svg")
        abort(404)

    @app.route("/icons.svg")
    def ui_icons():
        if _dist_safe_file(dist, "icons.svg"):
            return send_from_directory(dist, "icons.svg")
        abort(404)

    @app.route("/<path:path>")
    def ui_spa_or_file(path: str):
        if path == "predict":
            abort(404)
        hit = _dist_safe_file(dist, path)
        if hit:
            return send_from_directory(dist, path)
        if (dist / "index.html").is_file():
            return send_from_directory(dist, "index.html")
        abort(503)

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="FIRSTSIGHT AI wound pipeline")
    parser.add_argument("image", nargs="?", help="Image path to analyze")
    parser.add_argument(
        "--no-vision",
        action="store_true",
        help="CNN only: no WoundWatch / llama-server call (CLI and --serve)",
    )
    parser.add_argument("--serve", action="store_true", help="Run Flask /predict server")
    parser.add_argument("--port", type=int, default=5050)
    args = parser.parse_args()

    if args.serve:
        from woundwatch import resolved_woundwatch_endpoints

        app = create_app()
        app.config["USE_VISION"] = not args.no_vision
        print(f"FIRSTSIGHT AI UI + API -> http://127.0.0.1:{args.port}/")
        print(f"  POST /predict (multipart field 'image')")
        if app.config["USE_VISION"]:
            raw_u = os.environ.get("WOUNDWATCH_VISION_URL", "").strip()
            url, model = resolved_woundwatch_endpoints()
            print(f"  WoundWatch vision -> {url!r} (model={model!r})")
            if raw_u and ":11434" in raw_u and ":11435" in url:
                print(
                    "  (Ignored WOUNDWATCH_VISION_URL on :11434 — using llama-server :11435. "
                    "Set WOUNDWATCH_FORCE_OLLAMA_11434=1 if you really want Ollama on 11434.)"
                )
            elif raw_u:
                print("  (WOUNDWATCH_VISION_URL is set.)")
        else:
            print("  WoundWatch: OFF (--no-vision); POST /predict uses CNN only.")
        app.run(host="0.0.0.0", port=args.port, debug=False)
        return

    if not args.image:
        print("Provide an image path or use --serve", file=sys.stderr)
        sys.exit(1)

    path = Path(args.image)
    if not path.is_file():
        print(f"Not found: {path}", file=sys.stderr)
        sys.exit(1)

    with open(path, "rb") as f:
        raw = f.read()
    mime = _guess_mime(path.name)
    bundle = process_image_bytes(raw, mime=mime, use_vision=not args.no_vision)

    print("\n" + "=" * 60)
    print("FIRSTSIGHT AI — dual-model wound analysis")
    print("=" * 60)
    print("\n--- Best model (CNN) ---")
    print(json.dumps(bundle["cnn"], indent=2))
    if bundle["vision"]:
        print("\n--- WoundWatch (vision, prioritized for triage) ---")
        print(json.dumps(bundle["vision"]["normalized"], indent=2))
    print("\n--- Merged output (shown to user / API) ---")
    print(json.dumps(bundle["merged"], indent=2))
    print("\n--- Saved files ---")
    for k, v in bundle["saved_paths"].items():
        print(f"  {k}: {v}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
