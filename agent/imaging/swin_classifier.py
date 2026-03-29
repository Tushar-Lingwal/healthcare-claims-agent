"""
swin_classifier.py — Calls your local Swin Transformer model via ngrok tunnel.

Setup:
  1. Run app.py locally:  python app.py          (starts on port 7860)
  2. Run ngrok tunnel:    ngrok http 7860
  3. Set in Railway:      LOCAL_MODEL_URL=https://your-tunnel.ngrok-free.app

No HuggingFace Space dependency.
"""

import asyncio
import base64
import json
import logging
import os

import httpx

logger = logging.getLogger(__name__)

TIMEOUT = httpx.Timeout(120.0, connect=15.0)

CLASS_LABELS = {
    0: "Mild Dementia",
    1: "Moderate Dementia",
    2: "Non Demented",
    3: "Very Mild Dementia",
    4: "Glioma",
    5: "Healthy",
    6: "Meningioma",
    7: "Pituitary",
}

CLASS_LABEL_NORM = {
    "mild dementia":      "Mild Dementia",
    "moderate dementia":  "Moderate Dementia",
    "non demented":       "Non Demented",
    "very mild dementia": "Very Mild Dementia",
    "very_mild_dementia": "Very Mild Dementia",
    "glioma":             "Glioma",
    "healthy":            "Healthy",
    "meningioma":         "Meningioma",
    "pituitary":          "Pituitary",
    "pituitary tumor":    "Pituitary",
    "pituitary_tumor":    "Pituitary",
    "no tumor":           "Healthy",
    "no_tumor":           "Healthy",
    "non_demented":       "Non Demented",
    "mild_dementia":      "Mild Dementia",
    "moderate_dementia":  "Moderate Dementia",
}

ICD10_MAP = {
    "Mild Dementia":      {"code": "G30.9",  "desc": "Alzheimer's disease, unspecified"},
    "Moderate Dementia":  {"code": "G30.9",  "desc": "Alzheimer's disease, unspecified"},
    "Non Demented":       {"code": "Z03.89", "desc": "No abnormality detected"},
    "Very Mild Dementia": {"code": "G30.0",  "desc": "Alzheimer's disease with early onset"},
    "Glioma":             {"code": "C71.9",  "desc": "Malignant neoplasm of brain, unspecified"},
    "Healthy":            {"code": "Z03.89", "desc": "No abnormality detected"},
    "Meningioma":         {"code": "D32.9",  "desc": "Benign neoplasm of meninges, unspecified"},
    "Pituitary":          {"code": "D35.2",  "desc": "Benign neoplasm of pituitary gland"},
}


def _get_local_url() -> str:
    return os.environ.get("LOCAL_MODEL_URL", "").strip().rstrip("/")


async def classify_mri_image(
    image_bytes: bytes,
    filename: str = "scan.jpg",
) -> dict:
    """Send MRI image to local model and return classification result."""
    url = _get_local_url()

    if not url:
        logger.warning("LOCAL_MODEL_URL not set — model unavailable")
        return _fallback_result("LOCAL_MODEL_URL is not configured. Set it to your ngrok tunnel URL in Railway variables.")

    try:
        logger.info(f"Calling local model at: {url}")
        result = await _call_gradio_api(image_bytes, filename, url)
        result["source"] = "local"
        return result
    except Exception as e:
        logger.warning(f"Local model call failed ({url}): {e}")
        return _fallback_result(f"Local model unreachable at {url}: {e}. Make sure app.py and ngrok are running.")


async def _call_gradio_api(image_bytes: bytes, filename: str, base_url: str) -> dict:
    """Call the Gradio /run/predict endpoint with a base64-encoded image."""
    ext      = filename.rsplit(".", 1)[-1].lower() if "." in filename else "jpg"
    mime     = {"jpg": "image/jpeg", "jpeg": "image/jpeg",
                "png": "image/png", "gif": "image/gif"}.get(ext, "image/jpeg")
    b64      = base64.b64encode(image_bytes).decode()
    data_url = f"data:{mime};base64,{b64}"

    async with httpx.AsyncClient(timeout=TIMEOUT) as client:

        # Try /run/predict
        resp = await client.post(
            f"{base_url}/run/predict",
            json={"data": [data_url]},
            headers={"Content-Type": "application/json"},
        )

        # Try with fn_index if 404
        if resp.status_code == 404:
            resp = await client.post(
                f"{base_url}/run/predict",
                json={"data": [data_url], "fn_index": 0},
                headers={"Content-Type": "application/json"},
            )

        # Gradio streaming /call/ API
        if resp.status_code == 405:
            call_resp = await client.post(
                f"{base_url}/call/predict",
                json={"data": [data_url]},
                headers={"Content-Type": "application/json"},
            )
            if call_resp.status_code == 200:
                event_id = call_resp.json().get("event_id")
                if event_id:
                    for _ in range(40):
                        await asyncio.sleep(0.5)
                        poll = await client.get(f"{base_url}/call/predict/{event_id}")
                        if poll.status_code == 200 and "data:" in poll.text:
                            for line in poll.text.splitlines():
                                if line.startswith("data:"):
                                    raw = json.loads(line[5:].strip())
                                    if isinstance(raw, list) and raw:
                                        r = json.loads(raw[0]) if isinstance(raw[0], str) else raw[0]
                                        if "predicted_class" in r:
                                            return _normalise(r)
            raise ValueError(f"Gradio /call/ API failed: {call_resp.status_code}")

        resp.raise_for_status()
        body = resp.json()

    raw = body.get("data", [None])[0]
    if raw is None:
        raise ValueError(f"Empty Gradio response: {body}")

    result = json.loads(raw) if isinstance(raw, str) else raw

    if not isinstance(result, dict) or "predicted_class" not in result:
        raise ValueError(f"Unexpected result format: {str(result)[:200]}")

    if "error" in result:
        raise ValueError(result["error"])

    return _normalise(result)


def _normalise(result: dict) -> dict:
    """Normalise class label casing to canonical names."""
    raw_label = result.get("predicted_class", "")
    result["predicted_class"] = CLASS_LABEL_NORM.get(raw_label.lower().strip(), raw_label)

    if "all_probabilities" in result and isinstance(result["all_probabilities"], dict):
        result["all_probabilities"] = {
            CLASS_LABEL_NORM.get(k.lower().strip(), k): v
            for k, v in result["all_probabilities"].items()
        }

    logger.info(f"Model result: {result.get('predicted_class')} ({result.get('confidence', 0):.1%})")
    return result


def _fallback_result(error_msg: str) -> dict:
    return {
        "predicted_class":   "Unknown",
        "class_index":       -1,
        "confidence":        0.0,
        "category":          "unknown",
        "icd10_code":        "Z03.89",
        "icd10_description": "Observation — model unavailable",
        "all_probabilities": {label: 0.0 for label in CLASS_LABELS.values()},
        "source":            "fallback",
        "error":             error_msg,
        "model":             "swin_base_patch4_window7_224",
    }


def get_space_status() -> dict:
    """Health check for /imaging/status endpoint."""
    import httpx as _httpx
    url = _get_local_url()
    if not url:
        return {"status": "offline", "error": "LOCAL_MODEL_URL not configured", "source": "local"}
    try:
        resp = _httpx.get(f"{url}/", timeout=10.0)
        if resp.status_code == 200:
            return {"status": "online", "url": url, "source": "local"}
        return {"status": "degraded", "code": resp.status_code, "url": url}
    except Exception as e:
        return {"status": "offline", "error": str(e), "url": url}