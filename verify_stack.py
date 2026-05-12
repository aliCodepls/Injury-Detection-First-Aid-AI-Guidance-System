"""One-shot health check: CNN weights, vision HTTP server, Flask /predict contract."""
from __future__ import annotations

import io
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)


def main() -> int:
    report: dict = {"project": str(ROOT)}

    # 1) CNN
    try:
        from wound_model_infer import predict_wound_class_from_bytes, _resolve_weights_path
        from PIL import Image
        import numpy as np

        pth = _resolve_weights_path()
        buf = io.BytesIO()
        Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8)).save(buf, format="JPEG")
        out = predict_wound_class_from_bytes(buf.getvalue())
        need = {"injury_type", "confidence", "class_probabilities"}
        report["cnn"] = {
            "ok": need <= set(out),
            "weights_file": str(pth),
            "injury_type": out.get("injury_type"),
            "confidence": round(float(out.get("confidence", 0)), 4),
        }
    except Exception as e:
        report["cnn"] = {"ok": False, "error": f"{type(e).__name__}: {e}"}

    # 2) Optional: OpenAI-style /v1/models (Ollama exposes this; llama-server may or may not)
    from woundwatch import resolved_woundwatch_endpoints

    base = resolved_woundwatch_endpoints()[0]
    try:
        root = base.split("/v1/")[0] + "/v1/models"
        import requests

        r = requests.get(root, timeout=4)
        mids = []
        if r.ok:
            j = r.json()
            rows = j.get("data") or j.get("models") or []
            mids = [m.get("id") or m.get("name") for m in rows if isinstance(m, dict)]
        report["GET_v1_models"] = {
            "ok": r.status_code == 200,
            "status": r.status_code,
            "url": root,
            "model_ids_sample": mids[:12],
        }
    except Exception as e:
        report["GET_v1_models"] = {"ok": False, "error": f"{type(e).__name__}: {e}"}

    # 3) Vision chat completions
    try:
        from woundwatch import analyze_wound_with_vision, normalize_vision_output
        from PIL import Image
        import numpy as np

        buf = io.BytesIO()
        Image.fromarray(np.zeros((48, 48, 3), dtype=np.uint8)).save(buf, format="JPEG")
        raw = analyze_wound_with_vision(buf.getvalue(), mime="image/jpeg")
        norm = normalize_vision_output(raw)
        report["vision_llm"] = {"ok": True, "injury_type": norm.get("injury_type")}
    except Exception as e:
        report["vision_llm"] = {"ok": False, "error": f"{type(e).__name__}: {e}"}

    # 4) Flask contract (in-process; no need for server process)
    try:
        from final_pipeline import create_app
        from PIL import Image
        import numpy as np

        app = create_app()
        c = app.test_client()
        r_index = c.get("/")
        r_bad = c.post("/predict")
        buf = io.BytesIO()
        Image.fromarray(np.zeros((48, 48, 3), dtype=np.uint8)).save(buf, format="JPEG")
        r_ok = c.post("/predict", data={"image": (io.BytesIO(buf.getvalue()), "t.jpg")}, content_type="multipart/form-data")

        body_ok = r_ok.get_json() if r_ok.is_json else None
        expected_keys = {
            "injury_type",
            "severity",
            "bleeding",
            "swelling",
            "emergency_needed",
            "first_aid_steps",
            "clinical_notes",
            "best_model",
            "woundwatch",
            "merged_meta",
            "saved_paths",
        }
        post_ok_shape = bool(body_ok) and expected_keys <= set(body_ok)
        report["flask"] = {
            "GET_/": r_index.status_code,
            "POST_/predict_no_file": (r_bad.status_code, (r_bad.get_json() or {}).get("error")),
            "POST_/predict_status": r_ok.status_code,
            "POST_success_shape_ok": post_ok_shape,
            "POST_error": (body_ok or {}).get("error") if r_ok.status_code >= 400 else None,
            "note": "POST /predict runs CNN then vision; 500 means exception (often vision/Ollama).",
        }
    except Exception as e:
        report["flask"] = {"error": f"{type(e).__name__}: {e}"}

    print(json.dumps(report, indent=2))
    cnn_ok = bool((report.get("cnn") or {}).get("ok"))
    flask_index_ok = (report.get("flask") or {}).get("GET_/") == 200
    predict_no_file_ok = (report.get("flask") or {}).get("POST_/predict_no_file", (None,))[0] == 400
    return 0 if cnn_ok and flask_index_ok and predict_no_file_ok else 1


if __name__ == "__main__":
    sys.exit(main())
