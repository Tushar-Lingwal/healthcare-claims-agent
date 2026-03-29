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
import json
import logging
import os
import uuid
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

HF_SPACE_URL = os.environ.get(
    "HF_SPACE_URL",
    "https://raven004-brain-mri-classifier.hf.space",
).rstrip("/")

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


def _make_data_url(image_bytes: bytes, filename: str) -> tuple[str, str]:
    """Returns (data_url, mime_type)."""
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else "jpg"
    mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg",
            "png": "image/png", "gif": "image/gif"}.get(ext, "image/jpeg")
    b64 = base64.b64encode(image_bytes).decode()
    return f"data:{mime};base64,{b64}", mime


def _extract_result(raw) -> Optional[dict]:
    """Parse raw response value into a result dict if it contains predicted_class."""
    if not raw:
        return None
    result = json.loads(raw) if isinstance(raw, str) else raw
    if isinstance(result, dict) and "predicted_class" in result:
        return result
    return None


async def _call_gradio_api(image_bytes: bytes, filename: str) -> dict:
    """
    Calls the HuggingFace Space Gradio API.
    Tries multiple endpoint formats for compatibility with Gradio 4.x and 5.x/6.x.

    TIP: Check what endpoints are available at:
        {HF_SPACE_URL}/?view=api
    and update fn_index / endpoint name if needed.
    """
    data_url, mime = _make_data_url(image_bytes, filename)
    session_hash = uuid.uuid4().hex  # required by Gradio 4+ for routing

    async with httpx.AsyncClient(timeout=TIMEOUT) as client:

        # ── Attempt 1: Gradio 5/6 /run/predict with session_hash ──────────────
        # fn_index=0 targets the first registered event. Check /?view=api to confirm.
        try:
            url = f"{HF_SPACE_URL}/run/predict"
            payload = {"data": [data_url], "fn_index": 0, "session_hash": session_hash}
            logger.info(f"Attempt 1 (fn_index=0 + session_hash): POST {url}")
            resp = await client.post(url, json=payload)
            logger.debug(f"Attempt 1 status: {resp.status_code}")
            if resp.status_code == 200:
                result = _extract_result(resp.json().get("data", [None])[0])
                if result:
                    return result
        except Exception as e:
            logger.debug(f"Attempt 1 failed: {e}")

        # ── Attempt 2: Gradio 6 /call/ SSE API ────────────────────────────────
        # This is the preferred format for Gradio >=4.0 Spaces.
        # Endpoint name must match what the Space exposes — check /?view=api.
        try:
            url = f"{HF_SPACE_URL}/call/predict"
            payload = {"data": [data_url], "session_hash": session_hash}
            logger.info(f"Attempt 2 (Gradio /call/ SSE): POST {url}")
            resp = await client.post(url, json=payload)
            logger.debug(f"Attempt 2 status: {resp.status_code}")
            if resp.status_code == 200:
                event_id = resp.json().get("event_id")
                if event_id:
                    result = await _poll_sse(client, event_id)
                    if result:
                        return result
        except Exception as e:
            logger.debug(f"Attempt 2 failed: {e}")

        # ── Attempt 3: Gradio FileData wrapper ────────────────────────────────
        try:
            url = f"{HF_SPACE_URL}/run/predict"
            payload = {
                "data": [{"path": data_url, "meta": {"_type": "gradio.FileData"}}],
                "fn_index": 0,
                "session_hash": session_hash,
            }
            logger.info(f"Attempt 3 (FileData): POST {url}")
            resp = await client.post(url, json=payload)
            logger.debug(f"Attempt 3 status: {resp.status_code}")
            if resp.status_code == 200:
                result = _extract_result(resp.json().get("data", [None])[0])
                if result:
                    return result
        except Exception as e:
            logger.debug(f"Attempt 3 failed: {e}")

        # ── Attempt 4: Upload file then predict ───────────────────────────────
        try:
            upload_url = f"{HF_SPACE_URL}/upload"
            files = {"files": (filename, image_bytes, mime)}
            logger.info(f"Attempt 4 (upload+predict): POST {upload_url}")
            upload_resp = await client.post(upload_url, files=files)
            if upload_resp.status_code == 200:
                file_paths = upload_resp.json()
                file_path = file_paths[0] if isinstance(file_paths, list) else file_paths
                payload = {
                    "data": [{"path": file_path, "orig_name": filename, "meta": {"_type": "gradio.FileData"}}],
                    "fn_index": 0,
                    "session_hash": session_hash,
                }
                resp = await client.post(f"{HF_SPACE_URL}/run/predict", json=payload)
                logger.debug(f"Attempt 4 predict status: {resp.status_code}")
                if resp.status_code == 200:
                    result = _extract_result(resp.json().get("data", [None])[0])
                    if result:
                        return result
        except Exception as e:
            logger.debug(f"Attempt 4 failed: {e}")

    raise ValueError(
        f"All Gradio API formats failed for {HF_SPACE_URL}. "
        f"Check {HF_SPACE_URL}/?view=api for the correct endpoint name and fn_index."
    )


async def _poll_sse(client: httpx.AsyncClient, event_id: str, max_polls: int = 40) -> Optional[dict]:
    """
    Poll the Gradio SSE stream until a 'complete' event is received.

    SSE lines look like:
        event: complete
        data: [<json>]
    We wait for 'event: complete' before reading the data line.
    """
    result_url = f"{HF_SPACE_URL}/call/predict/{event_id}"
    found_complete = False

    for attempt in range(max_polls):
        await asyncio.sleep(0.5)
        try:
            poll = await client.get(result_url)
            if poll.status_code != 200:
                continue

            lines = poll.text.splitlines()
            for i, line in enumerate(lines):
                if line.strip() == "event: complete":
                    found_complete = True
                elif line.strip() == "event: error":
                    logger.warning("SSE stream returned error event")
                    return None
                elif found_complete and line.startswith("data:"):
                    data_str = line[5:].strip()
                    try:
                        parsed = json.loads(data_str)
                        raw = parsed[0] if isinstance(parsed, list) and parsed else parsed
                        return _extract_result(raw)
                    except json.JSONDecodeError:
                        logger.debug(f"SSE data parse failed: {data_str!r}")
                        return None
        except Exception as e:
            logger.debug(f"SSE poll attempt {attempt} failed: {e}")

    logger.warning(f"SSE polling timed out after {max_polls} attempts")
    return None


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


def get_space_api_info() -> dict:
    """
    Fetches the Space's API info to discover available endpoints and fn_index values.
    Call this during debugging to confirm endpoint names.

    Returns the parsed JSON from /?view=api or /info endpoint.
    """
    import httpx as _httpx
    for path in ["/info", "/api/predict", "/?view=api"]:
        try:
            resp = _httpx.get(f"{HF_SPACE_URL}{path}", timeout=12.0)
            if resp.status_code == 200 and resp.headers.get("content-type", "").startswith("application/json"):
                return resp.json()
        except Exception:
            pass
    return {"error": "Could not retrieve API info — visit the Space URL manually"}