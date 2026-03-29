"""
swin_classifier.py — Client for the Swin Transformer inference API.

Priority order:
  1. LOCAL_MODEL_URL  — your local ngrok tunnel (development)
  2. HF_SPACE_URL     — HuggingFace Space (production)
  3. fallback result  — if both unavailable

Set env vars:
  HF_SPACE_URL=https://raven004-brain-mri-classifier.hf.space   (default)
  LOCAL_MODEL_URL=https://abc123.ngrok-free.app                  (optional, overrides HF)

Usage:
    from agent.imaging.swin_classifier import classify_mri_image
    result = await classify_mri_image(image_bytes, filename)
"""

import asyncio
import base64
import json
import logging
import os

import httpx

logger = logging.getLogger(__name__)

# ── URL priority: local ngrok > HF Space ──────────────────────────────────
# NOTE: read at call time (inside functions) so env var changes take effect
# after redeploy without needing to change code.
def _get_active_url() -> str:
    local = os.environ.get("LOCAL_MODEL_URL", "").strip()
    hf    = os.environ.get("HF_SPACE_URL", "https://raven004-brain-mri-classifier.hf.space").strip()
    return local if local else hf

def _get_source_label() -> str:
    return "local" if os.environ.get("LOCAL_MODEL_URL", "").strip() else "hf_space"

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

# Normalised lookup — maps any model output casing → canonical label
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


async def classify_mri_image(
    image_bytes: bytes,
    filename: str = "scan.jpg",
) -> dict:
    """
    Classify a brain MRI image.
    Tries LOCAL_MODEL_URL first (if set), then HF_SPACE_URL, then returns fallback.
    """
    active_url   = _get_active_url()
    source_label = _get_source_label()
    local_url    = os.environ.get("LOCAL_MODEL_URL", "").strip()
    hf_url       = os.environ.get("HF_SPACE_URL", "https://raven004-brain-mri-classifier.hf.space").strip()

    # ── Try primary URL ───────────────────────────────────────────────────
    try:
        logger.info(f"Calling model at: {active_url} (source={source_label})")
        result = await _call_gradio_api(image_bytes, filename, active_url)
        result["source"] = source_label
        return result
    except Exception as e:
        logger.warning(f"Primary model call failed ({active_url}): {e}")

    # ── If local failed, try HF Space as fallback ─────────────────────────
    if local_url and hf_url and hf_url != active_url:
        try:
            logger.info(f"Falling back to HF Space: {hf_url}")
            result = await _call_gradio_api(image_bytes, filename, hf_url)
            result["source"] = "hf_space_fallback"
            return result
        except Exception as e2:
            logger.warning(f"HF Space fallback also failed: {e2}")

    # ── Both failed ───────────────────────────────────────────────────────
    return _fallback_result("Model unavailable — both local and HF Space unreachable")


async def _call_gradio_api(image_bytes: bytes, filename: str, base_url: str) -> dict:
    """
    Call a Gradio app's /run/predict endpoint with a base64 image.
    Handles Gradio 3.x, 4.x, and 6.x API formats.
    """
    ext      = filename.rsplit(".", 1)[-1].lower() if "." in filename else "jpg"
    mime     = {"jpg": "image/jpeg", "jpeg": "image/jpeg",
                "png": "image/png", "gif": "image/gif"}.get(ext, "image/jpeg")
    b64      = base64.b64encode(image_bytes).decode()
    data_url = f"data:{mime};base64,{b64}"

    url = f"{base_url}/run/predict"

    async with httpx.AsyncClient(timeout=TIMEOUT) as client:

        # ── Try /run/predict (Gradio 3/4/6 default) ───────────────────────
        resp = await client.post(
            url,
            json={"data": [data_url]},
            headers={"Content-Type": "application/json"},
        )

        if resp.status_code == 404:
            # Try with explicit fn_index
            resp = await client.post(
                url,
                json={"data": [data_url], "fn_index": 0},
                headers={"Content-Type": "application/json"},
            )

        # ── Gradio 6 streaming /call/ API ─────────────────────────────────
        if resp.status_code == 405:
            call_url  = f"{base_url}/call/predict"
            call_resp = await client.post(
                call_url,
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
                                        result = json.loads(raw[0]) if isinstance(raw[0], str) else raw[0]
                                        if "predicted_class" in result:
                                            return result
            raise ValueError(f"Gradio /call/ API failed: {call_resp.status_code}")

        resp.raise_for_status()
        body = resp.json()

    # ── Parse response ────────────────────────────────────────────────────
    raw = body.get("data", [None])[0]
    if raw is None:
        raise ValueError(f"Empty Gradio response: {body}")

    result = json.loads(raw) if isinstance(raw, str) else raw

    if not isinstance(result, dict) or "predicted_class" not in result:
        raise ValueError(f"Unexpected result format: {str(result)[:200]}")

    if "error" in result:
        raise ValueError(result["error"])

    # Normalise predicted_class casing to match CLASS_LABEL_NORM
    raw_label = result.get("predicted_class", "")
    canonical = CLASS_LABEL_NORM.get(raw_label.lower().strip(), raw_label)
    result["predicted_class"] = canonical

    # Also normalise all_probabilities keys if present
    if "all_probabilities" in result and isinstance(result["all_probabilities"], dict):
        result["all_probabilities"] = {
            CLASS_LABEL_NORM.get(k.lower().strip(), k): v
            for k, v in result["all_probabilities"].items()
        }

    logger.info(
        f"Model result: {result.get('predicted_class')} "
        f"({result.get('confidence', 0):.1%}) from {base_url}"
    )
    return result


def _fallback_result(error_msg: str) -> dict:
    """Structured fallback when no model is reachable."""
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
    """Synchronous health check — used in /health endpoint."""
    import httpx as _httpx
    url    = _get_active_url()
    source = _get_source_label()
    try:
        resp = _httpx.get(f"{url}/", timeout=12.0)
        if resp.status_code == 200:
            return {"status": "online", "url": url, "source": source}
        return {"status": "degraded", "code": resp.status_code, "url": url}
    except Exception as e:
        return {"status": "offline", "error": str(e), "url": url}