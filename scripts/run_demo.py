"""
run_demo.py — CLI demo runner for the Healthcare Claims Adjudication Agent.

Runs all 8 sample claims through the full pipeline and prints a formatted
summary report showing decisions, confidence scores, and reasoning.

Usage:
    python scripts/run_demo.py
    python scripts/run_demo.py --claim 3        # run only claim #3
    python scripts/run_demo.py --verbose        # show full reasoning chain
    python scripts/run_demo.py --export         # save audit reports to /tmp/
"""

import argparse
import json
import logging
import os
import secrets
import sys
import tempfile
import time
from pathlib import Path

# ── Setup path so we can import agent/ from project root ──────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Configure environment for local demo (if not already set) ─────────────────
if not os.environ.get("PHI_VAULT_KEY"):
    os.environ["PHI_VAULT_KEY"]         = secrets.token_hex(32)
    print("  [demo] PHI_VAULT_KEY not set — generated ephemeral key for this session")

if not os.environ.get("VAULT_BACKEND"):
    os.environ["VAULT_BACKEND"]         = "sqlite"
    os.environ["PHI_VAULT_SQLITE_PATH"] = tempfile.mktemp(suffix="_vault_demo.sqlite")

if not os.environ.get("AUDIT_BACKEND"):
    os.environ["AUDIT_BACKEND"]         = "sqlite"
    os.environ["AUDIT_SQLITE_PATH"]     = tempfile.mktemp(suffix="_audit_demo.sqlite")

if not os.environ.get("RAG_BACKEND"):
    os.environ["RAG_BACKEND"] = "local"

# Suppress noisy logs during demo
logging.basicConfig(level=logging.WARNING)

# ── Now safe to import agent modules ──────────────────────────────────────────
from agent.models.enums import ClaimDecision, ImagingMode, InsurancePlan
from agent.models.schemas import ClaimInput, PatientInfo, StructuredClinicalData
from agent.security.phi_tokenizer import PHITokenizer
from agent.security.token_vault import get_vault, reset_vault
from agent.engine.pipeline import ClaimsAdjudicationPipeline


class _StubExtractor:
    """Fallback extractor used when anthropic package is not installed."""
    def extract(self, claim):
        from agent.models.schemas import ExtractionResult, ClinicalEntity
        from agent.models.enums import EntityCategory
        # Parse structured_data from claim if available (no NER needed)
        diags = []
        procs = []
        if claim.structured_data:
            diags = [ClinicalEntity(text=t, category=EntityCategory.DIAGNOSIS, confidence=0.99, normalized=t) for t in claim.structured_data.diagnoses]
            procs = [ClinicalEntity(text=t, category=EntityCategory.PROCEDURE,  confidence=0.99, normalized=t) for t in claim.structured_data.procedures]
        return ExtractionResult(
            diagnoses=diags, procedures=procs, symptoms=[], medications=[],
            normalized_text=claim.clinical_notes, overall_confidence=0.99 if diags else 0.0,
        )
from agent.logging.audit_logger import get_audit_logger, reset_audit_logger
from agent.logging.audit_exporter import AuditExporter

# ── Colour helpers ─────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
BLUE   = "\033[94m"
BOLD   = "\033[1m"
RESET  = "\033[0m"
DIM    = "\033[2m"

DECISION_COLOURS = {
    ClaimDecision.APPROVED:      GREEN,
    ClaimDecision.REJECTED:      RED,
    ClaimDecision.NEEDS_REVIEW:  YELLOW,
}


def _colour(text: str, colour: str) -> str:
    return f"{colour}{text}{RESET}"


def _decision_colour(decision: ClaimDecision, text: str) -> str:
    return _colour(text, DECISION_COLOURS.get(decision, ""))


# ── Claim builder ──────────────────────────────────────────────────────────────

def build_claim(raw: dict) -> ClaimInput:
    pi   = raw["patient_info"]
    plan = InsurancePlan(pi["insurance_plan"].lower())
    mode = ImagingMode(raw.get("imaging_mode", "skipped").lower())

    patient = PatientInfo(
        patient_id    = pi["patient_id"],
        name          = pi["name"],
        date_of_birth = pi["date_of_birth"],
        age           = pi["age"],
        sex           = pi["sex"],
        insurance_plan= plan,
        policy_number = pi["policy_number"],
    )

    structured = None
    if raw.get("structured_data"):
        sd = raw["structured_data"]
        structured = StructuredClinicalData(
            diagnoses   = sd.get("diagnoses",   []),
            procedures  = sd.get("procedures",  []),
            symptoms    = sd.get("symptoms",    []),
            medications = sd.get("medications", []),
        )

    return ClaimInput(
        patient_info           = patient,
        clinical_notes         = raw["clinical_notes"],
        structured_data        = structured,
        imaging_mode           = mode,
        precomputed_class      = raw.get("precomputed_class"),
        precomputed_confidence = raw.get("precomputed_confidence"),
    )


# ── Printing ────────────────────────────────────────────────────────────────────

def print_claim_result(
    idx:      int,
    raw:      dict,
    result,
    verbose:  bool = False,
    elapsed:  int  = 0,
) -> None:
    decision    = result.decision
    conf_pct    = f"{result.confidence_score:.0%}"
    scenario    = raw.get("scenario", f"Claim #{idx}")
    expected    = raw.get("expected_decision", "?")
    match_sym   = "✓" if result.decision.value == expected else "≈"
    decision_str = _decision_colour(decision, decision.value.upper().replace("_", " "))

    print(f"\n  {'─'*60}")
    print(f"  {BOLD}#{idx:02d}{RESET}  {scenario}")
    print(f"  {'─'*60}")
    print(f"  Decision   : {decision_str}  ({conf_pct} confidence)  [{elapsed}ms]")
    print(f"  Expected   : {expected.upper().replace('_',' ')}  {match_sym}")

    if result.icd10_codes:
        codes = ", ".join(f"{c.code}" for c in result.icd10_codes)
        print(f"  ICD-10     : {codes}")
    if result.cpt_codes:
        codes = ", ".join(f"{c.code}" for c in result.cpt_codes)
        print(f"  CPT        : {codes}")

    if result.reasons:
        print(f"  Reason     : {result.reasons[0][:80]}")

    if result.edge_cases:
        for ec in result.edge_cases:
            colour = RED if ec.severity.is_blocking() else YELLOW
            print(f"  Edge case  : {_colour(ec.edge_case_type.value, colour)} — {ec.description[:60]}")

    if result.explainability:
        exp = result.explainability
        print(f"  Summary    : {DIM}{exp.summary[:80]}{RESET}")

        if verbose and exp.reasoning_chain:
            print(f"\n  {BLUE}Reasoning chain:{RESET}")
            for step in exp.reasoning_chain:
                print(f"    Step {step.step} [{step.stage.value:20s}] {step.confidence:.0%}")
                print(f"           {DIM}{step.outcome[:80]}{RESET}")

        if exp.risk_flags:
            for flag in exp.risk_flags[:2]:
                print(f"  {RED}Risk flag  : {flag[:80]}{RESET}")

    print(f"  Trace ID   : {DIM}{result.audit_trace_id}{RESET}")


def print_summary(results: list, total_ms: int) -> None:
    approved = sum(1 for r in results if r.decision == ClaimDecision.APPROVED)
    rejected = sum(1 for r in results if r.decision == ClaimDecision.REJECTED)
    review   = sum(1 for r in results if r.decision == ClaimDecision.NEEDS_REVIEW)
    avg_conf = sum(r.confidence_score for r in results) / len(results) if results else 0

    print(f"\n  {'═'*60}")
    print(f"  {BOLD}DEMO SUMMARY — {len(results)} claims processed in {total_ms}ms{RESET}")
    print(f"  {'═'*60}")
    print(f"  {_colour(f'Approved    : {approved}', GREEN)}")
    print(f"  {_colour(f'Rejected    : {rejected}', RED)}")
    print(f"  {_colour(f'Needs Review: {review}', YELLOW)}")
    print(f"  Avg confidence : {avg_conf:.0%}")
    print(f"  {'═'*60}\n")


# ── Main ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Healthcare Claims Agent — Demo Runner")
    parser.add_argument("--claim",   type=int, default=None, help="Run only claim N (1-indexed)")
    parser.add_argument("--verbose", action="store_true",    help="Show full reasoning chain")
    parser.add_argument("--export",  action="store_true",    help="Export audit reports to /tmp/")
    args = parser.parse_args()

    # ── Load sample claims ────────────────────────────────────────────────
    claims_path = Path(__file__).parent.parent / "data" / "sample_claims" / "sample_claims.json"
    if not claims_path.exists():
        print(f"  Error: sample_claims.json not found at {claims_path}")
        sys.exit(1)

    with open(claims_path) as f:
        all_claims = json.load(f)

    if args.claim:
        idx = args.claim - 1
        if idx < 0 or idx >= len(all_claims):
            print(f"  Error: claim #{args.claim} not found (1–{len(all_claims)} available)")
            sys.exit(1)
        claims_to_run = [(args.claim, all_claims[idx])]
    else:
        claims_to_run = list(enumerate(all_claims, start=1))

    # ── Initialise pipeline ───────────────────────────────────────────────
    print(f"\n  {BOLD}Healthcare Claims Adjudication Agent — Demo{RESET}")
    print(f"  {'─'*60}")
    print(f"  Vault    : {os.environ.get('VAULT_BACKEND', 'sqlite')}")
    print(f"  Audit    : {os.environ.get('AUDIT_BACKEND', 'sqlite')}")
    print(f"  RAG      : {os.environ.get('RAG_BACKEND', 'local')}")
    print(f"  API key  : {'SET' if os.environ.get('ANTHROPIC_API_KEY') else 'NOT SET (NER will use stub)'}")
    print(f"  Claims   : {len(claims_to_run)} to process")

    vault        = get_vault()
    audit_logger = get_audit_logger()
    tokenizer    = PHITokenizer(vault=vault)
    # Use stub extractor if anthropic package not installed
    try:
        from agent.extractors.clinical_extractor import ClinicalExtractor
        extractor = ClinicalExtractor() if os.environ.get('ANTHROPIC_API_KEY') else _StubExtractor()
    except ImportError:
        extractor = _StubExtractor()
        print(f'  {YELLOW}[demo] anthropic not installed — using stub extractor (structured_data only){RESET}')

    pipeline     = ClaimsAdjudicationPipeline(audit_logger=audit_logger, clinical_extractor=extractor)
    exporter     = AuditExporter(audit_logger)

    # ── Process each claim ────────────────────────────────────────────────
    results     = []
    total_start = time.monotonic()

    for idx, raw in claims_to_run:
        start = time.monotonic()
        try:
            claim     = build_claim(raw)
            tokenized = tokenizer.tokenize(claim)
            result    = pipeline.process(tokenized)
            elapsed   = int((time.monotonic() - start) * 1000)
            results.append(result)
            print_claim_result(idx, raw, result, verbose=args.verbose, elapsed=elapsed)

            if args.export:
                report_path = f"/tmp/audit_claim_{idx:02d}_{result.claim_id}.txt"
                with open(report_path, "w") as f:
                    f.write(exporter.export_text(result))
                print(f"  Exported   : {report_path}")

        except Exception as e:
            print(f"\n  #{idx:02d} ERROR: {e}")
            import traceback
            traceback.print_exc()

    total_ms = int((time.monotonic() - total_start) * 1000)
    print_summary(results, total_ms)

    # ── Cleanup ───────────────────────────────────────────────────────────
    reset_audit_logger()
    reset_vault()


if __name__ == "__main__":
    main()