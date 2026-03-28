"""
api.py — FastAPI application for the Healthcare Claims Adjudication Agent.

Endpoints:
  POST /adjudicate          — process a single claim through the full pipeline
  POST /adjudicate/batch    — process multiple claims in one request
  GET  /audit/{trace_id}    — retrieve full audit trail for a claim
  GET  /health              — service health check

Security:
  PHI tokenizer is called BEFORE the pipeline on every request.
  The pipeline never receives raw PHI — only vault tokens.

Usage:
  uvicorn agent.api:app --reload --host 0.0.0.0 --port 8000
  Swagger UI: http://localhost:8000/docs
"""

import logging
import os
import time

# Load .env file before anything else
from dotenv import load_dotenv
load_dotenv()
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException, Request, Depends
from agent.auth import auth_router, get_current_user, ensure_default_admin
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from agent.models.enums import ImagingMode, InsurancePlan
from agent.models.schemas import ClaimInput, PatientInfo, StructuredClinicalData
from agent.security.phi_tokenizer import PHITokenizer
from agent.security.token_vault import get_vault, reset_vault
from agent.engine.pipeline import ClaimsAdjudicationPipeline
from agent.logging.audit_logger import get_audit_logger, reset_audit_logger
from agent.logging.audit_exporter import AuditExporter
from agent.reporting.report_generator import ReportGenerator
from agent.imaging.swin_classifier import classify_mri_image, get_space_status
from agent.reporting.llm_report_writer import (
    generate_llm_narrative, _rule_based_narrative, build_full_report_html
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# LIFESPAN — startup / shutdown
# ─────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialises shared resources at startup, closes them at shutdown."""
    logger.info("Starting Healthcare Claims Adjudication Agent...")

    # Initialise vault and audit logger singletons
    vault        = get_vault()
    audit_logger = get_audit_logger()
    tokenizer    = PHITokenizer(vault=vault)
    pipeline     = ClaimsAdjudicationPipeline(audit_logger=audit_logger)
    exporter     = AuditExporter(audit_logger)

    # Store on app state for use in route handlers
    app.state.tokenizer  = tokenizer
    app.state.pipeline   = pipeline
    app.state.exporter   = exporter
    app.state.audit      = audit_logger
    app.state.report_gen = ReportGenerator()

    ensure_default_admin()
    logger.info("Agent ready. Swagger UI: http://localhost:8000/docs")
    yield

    logger.info("Shutting down agent...")
    reset_audit_logger()
    reset_vault()


# ─────────────────────────────────────────────
# APP
# ─────────────────────────────────────────────

app = FastAPI(
    title       = "Healthcare Claims Adjudication Agent",
    description = (
        "Domain-specific AI agent for automated healthcare claims processing "
        "with auditable reasoning at every step. PHI is tokenized at the gateway — "
        "the pipeline never processes raw patient identifiers."
    ),
    version     = "1.0.0",
    lifespan    = lifespan,
)

app.include_router(auth_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],   # Restrict in production
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


# ─────────────────────────────────────────────
# ROOT ENDPOINT
# ─────────────────────────────────────────────

@app.get("/", tags=["Info"], summary="API root")
async def root():
    return {
        "name":        "ClaimIQ — Healthcare Claims Adjudication Agent",
        "version":     "1.0.0",
        "status":      "running",
        "docs":        "/docs",
        "health":      "/health",
        "adjudicate":  "POST /adjudicate",
    }


# ─────────────────────────────────────────────
# REQUEST / RESPONSE MODELS (Pydantic)
# ─────────────────────────────────────────────

class PatientInfoRequest(BaseModel):
    patient_id:     str             = Field(..., example="MRN-98234")
    name:           str             = Field(..., example="Jane Smith")
    date_of_birth:  str             = Field(..., example="1964-03-12")
    age:            int             = Field(..., example=60)
    sex:            str             = Field(..., example="F")
    insurance_plan: str             = Field(..., example="premium")
    policy_number:  str             = Field(..., example="POL-99234")
    phone:          Optional[str]   = Field(None, example="555-123-4567")
    email:          Optional[str]   = Field(None, example="patient@email.com")
    address:        Optional[str]   = Field(None, example="123 Main St")


class StructuredDataRequest(BaseModel):
    diagnoses:   list[str] = Field(default_factory=list)
    procedures:  list[str] = Field(default_factory=list)
    symptoms:    list[str] = Field(default_factory=list)
    medications: list[str] = Field(default_factory=list)


class AdjudicateRequest(BaseModel):
    patient_info:           PatientInfoRequest
    clinical_notes:         str             = Field(..., example="Patient presents with stage 2 glioma.")
    structured_data:        Optional[StructuredDataRequest] = None
    imaging_mode:           str             = Field(default="skipped", example="skipped")
    precomputed_class:      Optional[str]   = Field(None, example="glioma_stage_2")
    precomputed_confidence: Optional[float] = Field(None, example=0.94)


class BatchAdjudicateRequest(BaseModel):
    claims: list[AdjudicateRequest] = Field(..., min_length=1, max_length=50)


class DecisionResponse(BaseModel):
    claim_id:         str
    audit_trace_id:   str
    decision:         str
    confidence_score: float
    icd10_codes:      list[dict]
    cpt_codes:        list[dict]
    reasons:          list[str]
    edge_cases:       list[dict]
    rule_evaluations: list[dict]
    explainability:   Optional[dict]
    decided_at:       str


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def _build_claim_input(req: AdjudicateRequest) -> ClaimInput:
    """Converts a Pydantic request model into a ClaimInput dataclass."""
    try:
        plan = InsurancePlan(req.patient_info.insurance_plan.lower())
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid insurance_plan '{req.patient_info.insurance_plan}'. "
                   f"Valid values: basic, standard, premium"
        )

    try:
        imaging = ImagingMode(req.imaging_mode.lower())
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid imaging_mode '{req.imaging_mode}'. "
                   f"Valid values: skipped, builtin, custom, precomputed"
        )

    patient = PatientInfo(
        patient_id    = req.patient_info.patient_id,
        name          = req.patient_info.name,
        date_of_birth = req.patient_info.date_of_birth,
        age           = req.patient_info.age,
        sex           = req.patient_info.sex,
        insurance_plan= plan,
        policy_number = req.patient_info.policy_number,
        phone         = req.patient_info.phone,
        email         = req.patient_info.email,
        address       = req.patient_info.address,
    )

    structured = None
    if req.structured_data:
        structured = StructuredClinicalData(
            diagnoses   = req.structured_data.diagnoses,
            procedures  = req.structured_data.procedures,
            symptoms    = req.structured_data.symptoms,
            medications = req.structured_data.medications,
        )

    return ClaimInput(
        patient_info           = patient,
        clinical_notes         = req.clinical_notes,
        structured_data        = structured,
        imaging_mode           = imaging,
        precomputed_class      = req.precomputed_class,
        precomputed_confidence = req.precomputed_confidence,
    )


def _result_to_response(result) -> DecisionResponse:
    """Converts AdjudicationResult to a Pydantic response model."""
    return DecisionResponse(
        claim_id         = result.claim_id,
        audit_trace_id   = result.audit_trace_id,
        decision         = result.decision.value,
        confidence_score = result.confidence_score,
        icd10_codes      = [
            {"code": c.code, "description": c.description, "confidence": c.confidence}
            for c in result.icd10_codes
        ],
        cpt_codes        = [
            {"code": c.code, "description": c.description, "confidence": c.confidence}
            for c in result.cpt_codes
        ],
        reasons          = result.reasons,
        edge_cases       = [
            {
                "type":           ec.edge_case_type.value,
                "severity":       ec.severity.value,
                "description":    ec.description,
                "recommendation": ec.recommendation,
                "image_signal":   ec.image_signal,
                "text_signal":    ec.text_signal,
                "mismatch_score": ec.mismatch_score,
            }
            for ec in result.edge_cases
        ],
        rule_evaluations = [
            {
                "rule_id":  ev.rule_id,
                "passed":   ev.passed,
                "action":   ev.action.value,
                "reason":   ev.reason,
            }
            for ev in result.rule_evaluations
        ],
        explainability   = (
            {
                "summary":     result.explainability.summary,
                "key_factors": result.explainability.key_factors,
                "risk_flags":  result.explainability.risk_flags,
                "chain_steps": len(result.explainability.reasoning_chain),
            }
            if result.explainability else None
        ),
        decided_at       = result.decided_at.isoformat(),
    )


# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────

@app.post(
    "/adjudicate",
    response_model = DecisionResponse,
    summary        = "Adjudicate a single claim",
    description    = (
        "Processes a single healthcare claim through the full 9-stage pipeline. "
        "PHI is tokenized at this endpoint before entering the pipeline. "
        "Returns decision, confidence score, medical codes, and full reasoning chain."
    ),
    tags=["Claims"],
)
async def adjudicate(req: AdjudicateRequest, request: Request, current_user=Depends(get_current_user)):
    start = time.monotonic()

    try:
        raw_claim = _build_claim_input(req)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Request validation error: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid request: {e}")

    # ── PHI tokenization (gateway — before pipeline) ──────────────────────
    try:
        tokenized = request.app.state.tokenizer.tokenize(raw_claim)
    except Exception as e:
        logger.error(f"PHI tokenization failed for {raw_claim.claim_id}: {e}")
        raise HTTPException(status_code=500, detail="PHI tokenization failed")

    # ── Pipeline ──────────────────────────────────────────────────────────
    result = request.app.state.pipeline.process(tokenized)

    elapsed = int((time.monotonic() - start) * 1000)
    logger.info(
        f"POST /adjudicate → {result.decision.value} "
        f"conf={result.confidence_score:.3f} ({elapsed}ms) "
        f"claim={result.claim_id}"
    )

    return _result_to_response(result)


@app.post(
    "/adjudicate/batch",
    response_model = list[DecisionResponse],
    summary        = "Adjudicate multiple claims",
    description    = "Processes up to 50 claims in a single request. Each claim is tokenized and processed independently.",
    tags=["Claims"],
)
async def adjudicate_batch(req: BatchAdjudicateRequest, request: Request, current_user=Depends(get_current_user)):
    results = []
    for i, claim_req in enumerate(req.claims):
        try:
            raw_claim = _build_claim_input(claim_req)
            tokenized = request.app.state.tokenizer.tokenize(raw_claim)
            result    = request.app.state.pipeline.process(tokenized)
            results.append(_result_to_response(result))
        except Exception as e:
            logger.error(f"Batch claim {i} failed: {e}")
            # Return a partial error response for this claim rather than failing the whole batch
            results.append(DecisionResponse(
                claim_id         = f"CLM-BATCH-ERR-{i:03d}",
                audit_trace_id   = "AUD-ERROR",
                decision         = "needs_review",
                confidence_score = 0.0,
                icd10_codes      = [],
                cpt_codes        = [],
                reasons          = [f"Processing error: {e}"],
                edge_cases       = [],
                rule_evaluations = [],
                explainability   = None,
                decided_at       = datetime.utcnow().isoformat(),
            ))

    logger.info(f"POST /adjudicate/batch → {len(results)} claims processed")
    return results


@app.get(
    "/audit/{trace_id}",
    summary     = "Retrieve full audit trail",
    description = (
        "Returns the complete audit trail for a claim identified by its audit_trace_id. "
        "Includes all 9 stage entries with input/output snapshots and timing data. "
        "No PHI is present in audit records — only token IDs and medical codes."
    ),
    tags=["Audit"],
)
async def get_audit(trace_id: str, request: Request, current_user=Depends(get_current_user)):
    try:
        entries = request.app.state.audit.get_trace(trace_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Audit retrieval failed: {e}")

    if not entries:
        raise HTTPException(
            status_code=404,
            detail=f"No audit entries found for trace_id: {trace_id}"
        )

    return {
        "audit_trace_id": trace_id,
        "entry_count":    len(entries),
        "entries": [
            {
                "stage":           e.stage.value,
                "status":          e.status.value,
                "timestamp":       e.timestamp.isoformat(),
                "duration_ms":     e.duration_ms,
                "input_snapshot":  e.input_snapshot,
                "output_snapshot": e.output_snapshot,
                "error_message":   e.error_message,
            }
            for e in entries
        ],
    }


@app.get(
    "/report/{trace_id}",
    summary     = "Generate HTML report for a claim",
    description = "Generates a full print-ready HTML adjudication report for a claim. Pass the audit_trace_id from the adjudicate response.",
    tags=["Reports"],
    response_class=None,
)
async def get_report_html(trace_id: str, request: Request):
    from fastapi.responses import HTMLResponse
    try:
        entries = request.app.state.audit.get_trace(trace_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Audit retrieval failed: {e}")
    if not entries:
        raise HTTPException(status_code=404, detail=f"No audit entries found for trace_id: {trace_id}")
    # Reconstruct a minimal result for report generation
    raise HTTPException(status_code=501, detail="Use POST /report to generate report from full adjudication result.")


@app.post(
    "/imaging/classify",
    summary     = "Classify a brain MRI scan using the Swin Transformer model",
    description = (
        "Accepts a brain MRI image (JPG/PNG) and returns classification results "
        "from the Swin Transformer model hosted on Hugging Face Spaces. "
        "Returns predicted class, confidence, ICD-10 code, and full probability distribution."
    ),
    tags=["Imaging"],
)
async def imaging_classify(request: Request):
    """
    Accepts multipart/form-data with an 'image' file field.
    Returns structured JSON with classification result.
    """
    try:
        form    = await request.form()
        imgfile = form.get("image")
        if imgfile is None:
            raise HTTPException(status_code=422, detail="No 'image' field in form data")

        image_bytes = await imgfile.read()
        filename    = getattr(imgfile, "filename", "scan.jpg") or "scan.jpg"

        if len(image_bytes) == 0:
            raise HTTPException(status_code=422, detail="Empty image file")

        logger.info(f"Imaging classify: {filename} ({len(image_bytes):,} bytes)")
        result = await classify_mri_image(image_bytes, filename)
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Imaging classify error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get(
    "/imaging/status",
    summary="Check Hugging Face Space availability",
    tags=["Imaging"],
)
async def imaging_status():
    """Returns the current status of the HF Space inference endpoint."""
    return get_space_status()


@app.post(
    "/report",
    summary     = "Generate full LLM-written clinical adjudication report",
    description = (
        "Generates a complete narrative clinical report from an adjudication result. "
        "The LLM writes full prose explaining the agent reasoning, evidence, and recommendations. "
        "Returns HTML by default. Add ?format=text for plain text or ?format=json for structured data."
    ),
    tags=["Reports"],
)
async def generate_report(
    req:     dict,
    request: Request,
    format:  str = "html",
):
    from fastapi.responses import HTMLResponse, PlainTextResponse
    from datetime import datetime as dt

    try:
        # Try LLM narrative first
        narrative = generate_llm_narrative(req)

        # Fall back to rule-based narrative if LLM unavailable
        if not narrative:
            narrative = _rule_based_narrative(req)

        generated_at = dt.utcnow()

        if format == "text":
            return PlainTextResponse(content=narrative, status_code=200)

        if format == "json":
            return {
                "claim_id":      req.get("claim_id"),
                "audit_trace_id":req.get("audit_trace_id"),
                "decision":      req.get("decision"),
                "generated_at":  generated_at.isoformat(),
                "narrative":     narrative,
                "provider":      os.environ.get("LLM_PROVIDER","rules"),
            }

        # Default: full HTML document
        html = build_full_report_html(req, narrative, generated_at)
        return HTMLResponse(content=html, status_code=200)

    except Exception as e:
        import traceback
        logger.error(f"Report generation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Report generation failed: {e}")



@app.get(
    "/health",
    summary     = "Health check",
    description = "Returns service status, version, and loaded policy rule count.",
    tags=["System"],
)
async def health(request: Request):
    try:
        from agent.rules.policy_store import PolicyStore
        rule_count = PolicyStore().count()
    except Exception:
        rule_count = 0

    return {
        "status":       "healthy",
        "version":      "1.0.0",
        "timestamp":    datetime.utcnow().isoformat(),
        "rules_loaded": rule_count,
        "vault_backend": os.environ.get("VAULT_BACKEND", "sqlite"),
        "audit_backend": os.environ.get("AUDIT_BACKEND", "sqlite"),
        "rag_backend":   os.environ.get("RAG_BACKEND",   "local"),
    }


# ─────────────────────────────────────────────
# EXCEPTION HANDLERS
# ─────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error on {request.method} {request.url}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. The incident has been logged."},
    )