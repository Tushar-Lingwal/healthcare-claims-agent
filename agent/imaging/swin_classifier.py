"""
swin_classifier.py — Client for the Hugging Face Space inference API.

Calls Raven004/brain-mri-classifier Space to run Swin Transformer inference
on brain MRI images. Falls back gracefully if the Space is unavailable
(cold start, quota exceeded, etc.)

Usage:
    from agent.imaging.swin_classifier import classify_mri_image
    result = await classify_mri_image(image_bytes, filename)
"""

import asyncio
import base64
import io
import json
import logging
import os
import tempfile
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

HF_SPACE_URL = os.environ.get(
    "HF_SPACE_URL",
    "https://raven004-brain-mri-classifier.hf.space",
)

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

ICD10_MAP = {
    "Mild Dementia":       {"code": "G30.9",  "desc": "Alzheimer's disease, unspecified"},
    "Moderate Dementia":   {"code": "G30.9",  "desc": "Alzheimer's disease, unspecified"},
    "Non Demented":        {"code": "Z03.89", "desc": "No abnormality detected"},
    "Very Mild Dementia":  {"code": "G30.0",  "desc": "Alzheimer's disease with early onset"},
    "Glioma":              {"code": "C71.9",  "desc": "Malignant neoplasm of brain, unspecified"},
    "Healthy":             {"code": "Z03.89", "desc": "No abnormality detected"},
    "Meningioma":          {"code": "D32.9",  "desc": "Benign neoplasm of meninges, unspecified"},
    "Pituitary":           {"code": "D35.2",  "desc": "Benign neoplasm of pituitary gland"},
}

TIMEOUT = httpx.Timeout(120.0, connect=15.0)  # HF Spaces cold start can be 90s+


async def classify_mri_image(
    image_bytes: bytes,
    filename: str = "scan.jpg",
) -> dict:
    """
    Send an MRI image to the HF Space and return structured classification result.

    Returns dict with keys:
        predicted_class, class_index, confidence, category,
        icd10_code, icd10_description, all_probabilities,
        source (hf_space | fallback), error (optional)
    """
    try:
        result = await _call_gradio_api(image_bytes, filename)
        result["source"] = "hf_space"
        return result
    except Exception as e:
        logger.warning(f"HF Space call failed: {e} — returning fallback")
        return _fallback_result(str(e))


async def _call_gradio_api(image_bytes: bytes, filename: str) -> dict:
    """
    Calls the HuggingFace Space Gradio 6.x API.
    The Space app.py defines: predict(image: Image.Image) -> str
    Gradio 6 /run/predict expects {"data": [<base64_data_url>]} for Image inputs.
    """
    ext  = filename.rsplit(".", 1)[-1].lower() if "." in filename else "jpg"
    mime = {"jpg":"image/jpeg","jpeg":"image/jpeg",
            "png":"image/png","gif":"image/gif"}.get(ext,"image/jpeg")
    b64      = base64.b64encode(image_bytes).decode()
    data_url = f"data:{mime};base64,{b64}"

    url = f"{HF_SPACE_URL}/run/predict"
    logger.info(f"Calling HF Space: {url}")

    async with httpx.AsyncClient(timeout=TIMEOUT) as client:

        # Gradio 6 with Image(type="pil") input accepts plain base64 data URL
        payload = {"data": [data_url]}
        resp    = await client.post(url, json=payload,
                                    headers={"Content-Type": "application/json"})

        if resp.status_code == 404:
            # Try with fn_index explicitly
            payload = {"data": [data_url], "fn_index": 0}
            resp    = await client.post(url, json=payload,
                                        headers={"Content-Type": "application/json"})

        if resp.status_code == 405:
            # Gradio 6 /call/ streaming API
            call_url  = f"{HF_SPACE_URL}/call/predict"
            call_resp = await client.post(call_url, json={"data": [data_url]},
                                          headers={"Content-Type": "application/json"})
            if call_resp.status_code == 200:
                event_id = call_resp.json().get("event_id")
                if event_id:
                    for _ in range(40):
                        await asyncio.sleep(0.5)
                        poll = await client.get(f"{HF_SPACE_URL}/call/predict/{event_id}")
                        if poll.status_code == 200 and "data:" in poll.text:
                            for line in poll.text.splitlines():
                                if line.startswith("data:"):
                                    raw = json.loads(line[5:].strip())
                                    if isinstance(raw, list) and raw:
                                        result = json.loads(raw[0]) if isinstance(raw[0], str) else raw[0]
                                        if "predicted_class" in result:
                                            return result
            raise ValueError(f"Gradio /call/ API failed: {call_resp.status_code}")

        resp.raise_for_status()
        body = resp.json()

    # Parse response — Gradio wraps in {"data": ["json_string"]}
    raw = body.get("data", [None])[0]
    if raw is None:
        raise ValueError(f"Empty Gradio response: {body}")

    result = json.loads(raw) if isinstance(raw, str) else raw

    if not isinstance(result, dict) or "predicted_class" not in result:
        raise ValueError(f"Unexpected result format: {str(result)[:200]}")

    if "error" in result:
        raise ValueError(result["error"])

    logger.info(f"HF Space result: {result.get('predicted_class')} ({result.get('confidence',0):.1%})")
    return result


def _fallback_result(error_msg: str) -> dict:
    """Returns a structured fallback when the Space is unavailable."""
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
    """Synchronous health check for the HF Space (used in /health endpoint)."""
    import httpx as _httpx
    try:
        resp = _httpx.get(f"{HF_SPACE_URL}/", timeout=12.0)
        if resp.status_code == 200:
            return {"status": "online", "url": HF_SPACE_URL}
        return {"status": "degraded", "code": resp.status_code}
    except Exception as e:
        return {"status": "offline", "error": str(e)}