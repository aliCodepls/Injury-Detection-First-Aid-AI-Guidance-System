"""
WoundWatch vision client: POST image bytes to a local OpenAI-style `/v1/chat/completions` server.

Typical Windows layout: `C:\\woundmodel\\` holds the GGUF + mmproj + `run.bat`. That batch starts
**llama-server.exe** on **port 11435** (see `--port 11435`). FIRSTSIGHT defaults match that — start
that server before the pipeline or `final_pipeline.py --serve`.

If you use **Ollama** on port 11434 instead, set:

  WOUNDWATCH_VISION_URL=http://127.0.0.1:11434/v1/chat/completions
  WOUNDWATCH_MODEL=gemma4-vision:latest

If `WOUNDWATCH_VISION_URL` points at **:11434** but you did **not** set `WOUNDWATCH_FORCE_OLLAMA_11434=1`,
this project **ignores** that URL and uses the llama-server defaults above (your `run.bat` stack).
Set `WOUNDWATCH_FORCE_OLLAMA_11434=1` when you truly want Ollama on 11434.

`Modelfile` in `C:\\woundmodel` is for `ollama create`; it does not change how `llama-server.exe`
is started from `run.bat`.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import re
from typing import Any

import requests

# Defaults match C:\woundmodel\run.bat: llama-server --port 11435 (OpenAI-compatible API).
# llama.cpp accepts a placeholder "model" string in JSON (see llama-server docs).
DEFAULT_VISION_URL = "http://127.0.0.1:11435/v1/chat/completions"
DEFAULT_MODEL = "gpt-3.5-turbo"


def resolved_woundwatch_endpoints() -> tuple[str, str]:
    """
    Return (vision_api_url, model_name) used for HTTP calls.

    A stale Windows env often sets WOUNDWATCH_VISION_URL to Ollama (:11434) while the app is
    meant to use llama-server from C:\\woundmodel\\run.bat (:11434 -> :11435 remap unless forced).
    """
    raw_url = os.environ.get("WOUNDWATCH_VISION_URL", "").strip()
    raw_model = os.environ.get("WOUNDWATCH_MODEL", "").strip()
    force_ollama = os.environ.get("WOUNDWATCH_FORCE_OLLAMA_11434", "").strip()
    if raw_url and ":11434" in raw_url and not force_ollama:
        return DEFAULT_VISION_URL, DEFAULT_MODEL
    return (raw_url or DEFAULT_VISION_URL), (raw_model or DEFAULT_MODEL)


def _vision_url_from_env() -> str:
    return resolved_woundwatch_endpoints()[0]


def _vision_model_from_env() -> str:
    return resolved_woundwatch_endpoints()[1]

SYSTEM_PROMPT = """You are FIRSTSIGHT AI, a clinical wound triage assistant. Analyze the image carefully.
You must respond with ONLY a single JSON object (no markdown fences, no commentary).
Use conservative triage: if unsure about emergency, bias toward emergency_needed true.
First aid must be actionable, ordered steps for a layperson (not a diagnosis)."""


def build_user_prompt() -> str:
    return """Analyze this wound image. Return ONLY valid JSON with exactly these keys:
{
  "injury_type": string (one concise label, e.g. Burn, Laceration, Abrasion, Bruise, Insect bite, Ulcer, Unknown),
  "severity": integer from 1 to 10 (10 = most severe / highest concern),
  "has_bleeding": boolean,
  "has_swelling": boolean,
  "emergency_needed": boolean,
  "first_aid_steps": array of 5 to 10 short strings, each one clear imperative step (e.g. "Apply steady pressure with clean gauze."),
  "clinical_notes": string (1-3 sentences: what you observe and why severity/emergency were chosen)
}

Rules:
- first_aid_steps must have between 5 and 10 items inclusive.
- Booleans must be true or false (lowercase JSON).
- severity must be an integer 1-10.
- Do not include any text outside the JSON object."""


def image_bytes_to_data_url(image_bytes: bytes, mime: str = "image/jpeg") -> str:
    b64 = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{mime};base64,{b64}"


def _strip_json_fences(text: str) -> str:
    t = text.strip()
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", t, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return t


def parse_vision_json(content: str) -> dict[str, Any]:
    raw = _strip_json_fences(content)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end > start:
            return json.loads(raw[start : end + 1])
        raise


def analyze_wound_with_vision(
    image_bytes: bytes,
    *,
    mime: str = "image/jpeg",
    api_url: str | None = None,
    model: str | None = None,
    timeout: float = 120.0,
) -> dict[str, Any]:
    """
    Call vision LLM with raw image bytes. Returns parsed dict (may need normalization).

    If api_url or model is omitted, values come from WOUNDWATCH_VISION_URL and
    WOUNDWATCH_MODEL, then the module defaults.
    """
    if api_url is None:
        api_url = _vision_url_from_env()
    if model is None:
        model = _vision_model_from_env()
    data_url = image_bytes_to_data_url(image_bytes, mime=mime)
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": build_user_prompt()},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            },
        ],
    }
    response = requests.post(api_url, json=payload, timeout=timeout)
    if not response.ok:
        snippet = (response.text or "")[:1500].strip()
        hint = ""
        if "11434" in api_url:
            hint = (
                " If you meant llama-server (C:\\woundmodel\\run.bat), URL should be :11435. "
                "A global env WOUNDWATCH_VISION_URL=:11434 is ignored unless WOUNDWATCH_FORCE_OLLAMA_11434=1."
            )
        raise RuntimeError(
            f"Vision HTTP {response.status_code} from {api_url!r}. "
            f"If you use llama-server (run.bat), ensure it is listening on that port.{hint} "
            f"Body (truncated): {snippet}"
        )
    body = response.json()
    content = str(body["choices"][0]["message"]["content"])
    data = parse_vision_json(content)
    data["_raw_model_content"] = content
    return data


def normalize_vision_output(data: dict[str, Any]) -> dict[str, Any]:
    """Map vision keys to a stable schema for API / storage."""
    steps = data.get("first_aid_steps") or data.get("first_aid")
    if isinstance(steps, str):
        steps = [s.strip() for s in re.split(r"\n|(?<=[.!?])\s+", steps) if s.strip()]
    if not isinstance(steps, list):
        steps = []
    steps = [str(s).strip() for s in steps if str(s).strip()]
    if len(steps) < 5:
        pad = "Follow local emergency guidance if symptoms worsen."
        while len(steps) < 5:
            steps.append(pad)
    if len(steps) > 10:
        steps = steps[:10]

    sev = data.get("severity")
    try:
        severity = int(float(sev))
    except (TypeError, ValueError):
        severity = 5
    severity = max(1, min(10, severity))

    def as_bool(v: Any) -> bool:
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            return v.strip().lower() in ("true", "yes", "1", "y")
        return bool(v)

    return {
        "injury_type": str(data.get("injury_type") or "Unknown"),
        "severity": severity,
        "has_bleeding": as_bool(data.get("has_bleeding", data.get("bleeding", False))),
        "has_swelling": as_bool(data.get("has_swelling", data.get("swelling", False))),
        "emergency_needed": as_bool(data.get("emergency_needed", data.get("emergency", False))),
        "first_aid_steps": steps,
        "clinical_notes": str(data.get("clinical_notes") or ""),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="WoundWatch vision analysis (image path → JSON)")
    parser.add_argument("image", nargs="?", help="Path to image (jpeg/png). If omitted, use --stdin-bytes not supported; path required.")
    parser.add_argument(
        "--url",
        default=None,
        help=f"OpenAI-compatible chat completions URL (default: env WOUNDWATCH_VISION_URL or {DEFAULT_VISION_URL})",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=f"Vision model name (default: env WOUNDWATCH_MODEL or {DEFAULT_MODEL})",
    )
    parser.add_argument("--out", help="Write vision JSON to this file")
    args = parser.parse_args()
    if not args.image:
        parser.error("Please pass an image path, e.g. python woundwatch.py photo.jpg")
    with open(args.image, "rb") as f:
        raw = f.read()
    mime = "image/png" if args.image.lower().endswith(".png") else "image/jpeg"
    vision_raw = analyze_wound_with_vision(
        raw,
        mime=mime,
        api_url=args.url if args.url is not None else None,
        model=args.model if args.model is not None else None,
    )
    normalized = normalize_vision_output(vision_raw)
    out = {**normalized, "_vision_raw_keys": list(vision_raw.keys())}
    text = json.dumps(out, indent=2)
    print(text)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as wf:
            wf.write(text)


if __name__ == "__main__":
    main()
