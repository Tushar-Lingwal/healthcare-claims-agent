"""
audit_exporter.py — Exports a full claim audit trail as a structured report.

Produces a human-readable audit report for a single claim containing:
  - Claim metadata (ID, trace ID, decision, confidence)
  - Per-stage reasoning steps
  - Rule evaluation results
  - Edge cases detected
  - Risk flags
  - Guideline citations
  - Full stage-by-stage audit log

Output formats:
  - dict  — structured Python dict (for API responses)
  - JSON  — serialised string (for storage / transmission)
  - text  — human-readable plain text (for email / PDF input)

The text format is what feeds audit_pdf generation (future sprint).
No PHI appears in any export — only token IDs and medical codes.
"""

import json
import logging
from datetime import datetime
from typing import Optional

from agent.models.enums import AuditStatus, ClaimDecision
from agent.models.schemas import AdjudicationResult, AuditEntry
from agent.logging.audit_logger import AuditLogger

logger = logging.getLogger(__name__)


class AuditExporter:
    """
    Exports a complete claim audit report in multiple formats.

    Usage:
        exporter = AuditExporter(audit_logger)
        report   = exporter.export_dict(result)
        text     = exporter.export_text(result)
        json_str = exporter.export_json(result)
    """

    def __init__(self, audit_logger: AuditLogger):
        self._logger = audit_logger

    # ── Primary export methods ───────────────────────────────────────────────

    def export_dict(self, result: AdjudicationResult) -> dict:
        """
        Exports the full audit report as a structured Python dict.
        Used by the API GET /audit/{trace_id} endpoint.
        """
        audit_entries = self._logger.get_trace(result.audit_trace_id)

        return {
            "report_generated_at": datetime.utcnow().isoformat(),
            "claim": {
                "claim_id":       result.claim_id,
                "audit_trace_id": result.audit_trace_id,
                "decided_at":     result.decided_at.isoformat(),
                "pipeline_version": result.pipeline_version,
            },
            "decision": {
                "outcome":          result.decision.value,
                "confidence_score": result.confidence_score,
                "reasons":          result.reasons,
            },
            "medical_codes": {
                "icd10": [
                    {
                        "code":        c.code,
                        "description": c.description,
                        "confidence":  c.confidence,
                        "exact_match": c.is_exact_match,
                    }
                    for c in result.icd10_codes
                ],
                "cpt": [
                    {
                        "code":        c.code,
                        "description": c.description,
                        "confidence":  c.confidence,
                        "exact_match": c.is_exact_match,
                    }
                    for c in result.cpt_codes
                ],
            },
            "rule_evaluations": [
                {
                    "rule_id":   ev.rule_id,
                    "rule_name": ev.rule_name,
                    "passed":    ev.passed,
                    "action":    ev.action.value,
                    "reason":    ev.reason,
                    "severity":  ev.severity.value,
                }
                for ev in result.rule_evaluations
            ],
            "edge_cases": [
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
            "imaging": self._imaging_dict(result),
            "explainability": self._explainability_dict(result),
            "guideline_citations": (
                result.explainability.guideline_citations
                if result.explainability else []
            ),
            "audit_trail": [self._entry_dict(e) for e in audit_entries],
            "audit_trail_count": len(audit_entries),
        }

    def export_json(self, result: AdjudicationResult, indent: int = 2) -> str:
        """Returns the audit report as a formatted JSON string."""
        return json.dumps(self.export_dict(result), indent=indent, default=str)

    def export_text(self, result: AdjudicationResult) -> str:
        """
        Returns the audit report as human-readable plain text.
        Suitable for email, PDF input, or terminal display.
        """
        lines = []
        sep   = "=" * 60

        lines.append(sep)
        lines.append("  HEALTHCARE CLAIMS ADJUDICATION — AUDIT REPORT")
        lines.append(sep)
        lines.append(f"  Generated : {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")
        lines.append(f"  Claim ID  : {result.claim_id}")
        lines.append(f"  Trace ID  : {result.audit_trace_id}")
        lines.append(f"  Decided   : {result.decided_at.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        lines.append(sep)

        # Decision
        decision_label = result.decision.value.upper().replace("_", " ")
        lines.append(f"\n  DECISION: {decision_label}")
        lines.append(f"  Confidence: {result.confidence_score:.0%}")
        lines.append("")
        for reason in result.reasons:
            lines.append(f"    • {reason}")

        # Medical codes
        if result.icd10_codes or result.cpt_codes:
            lines.append(f"\n  MEDICAL CODES")
            lines.append("  " + "-" * 40)
            for c in result.icd10_codes:
                match = "exact" if c.is_exact_match else "fuzzy"
                lines.append(f"    ICD-10  {c.code:12s} {c.description} ({match}, {c.confidence:.0%})")
            for c in result.cpt_codes:
                match = "exact" if c.is_exact_match else "fuzzy"
                lines.append(f"    CPT     {c.code:12s} {c.description} ({match}, {c.confidence:.0%})")

        # Rule evaluations
        if result.rule_evaluations:
            lines.append(f"\n  RULE EVALUATIONS ({len(result.rule_evaluations)} rules)")
            lines.append("  " + "-" * 40)
            for ev in result.rule_evaluations:
                status = "PASS" if ev.passed else f"FAIL [{ev.action.value.upper()}]"
                lines.append(f"    [{status:16s}] {ev.rule_id}: {ev.reason[:80]}")

        # Edge cases
        if result.edge_cases:
            lines.append(f"\n  EDGE CASES ({len(result.edge_cases)} detected)")
            lines.append("  " + "-" * 40)
            for ec in result.edge_cases:
                lines.append(f"    [{ec.severity.value.upper():8s}] {ec.edge_case_type.value}")
                lines.append(f"             {ec.description}")
                if ec.image_signal:
                    lines.append(f"             Image: {ec.image_signal} | Text: {ec.text_signal} | Score: {ec.mismatch_score:.2f}")
                lines.append(f"             → {ec.recommendation}")

        # Imaging
        if result.imaging_result:
            img = result.imaging_result
            lines.append(f"\n  IMAGING ANALYSIS")
            lines.append("  " + "-" * 40)
            lines.append(f"    Mode       : {img.mode_used.value}")
            lines.append(f"    Class      : {img.class_label}")
            lines.append(f"    Confidence : {img.confidence:.0%}")
            if img.icd10_suggestion:
                lines.append(f"    ICD-10 hint: {img.icd10_suggestion}")

        # Explainability
        if result.explainability:
            exp = result.explainability
            lines.append(f"\n  REASONING CHAIN")
            lines.append("  " + "-" * 40)
            lines.append(f"    Summary: {exp.summary}")
            lines.append("")
            for step in exp.reasoning_chain:
                lines.append(f"    Step {step.step} [{step.stage.value:20s}] conf={step.confidence:.0%}")
                lines.append(f"           {step.description}")
                lines.append(f"           → {step.outcome}")

            if exp.risk_flags:
                lines.append(f"\n  RISK FLAGS")
                lines.append("  " + "-" * 40)
                for flag in exp.risk_flags:
                    lines.append(f"    ⚠  {flag}")

            if exp.guideline_citations:
                lines.append(f"\n  GUIDELINE CITATIONS")
                lines.append("  " + "-" * 40)
                for cite in exp.guideline_citations:
                    lines.append(f"    {cite[:100]}")

        # Audit trail
        audit_entries = self._logger.get_trace(result.audit_trace_id)
        if audit_entries:
            lines.append(f"\n  AUDIT TRAIL ({len(audit_entries)} entries)")
            lines.append("  " + "-" * 40)
            for entry in audit_entries:
                status = entry.status.value.upper()
                ts     = entry.timestamp.strftime("%H:%M:%S.%f")[:-3]
                lines.append(
                    f"    [{ts}] {entry.stage.value:20s} [{status:7s}] {entry.duration_ms}ms"
                    + (f" ERROR: {entry.error_message}" if entry.error_message else "")
                )

        lines.append(f"\n{sep}")
        lines.append("  END OF AUDIT REPORT")
        lines.append(sep)

        return "\n".join(lines)

    # ── Helper formatters ────────────────────────────────────────────────────

    def _imaging_dict(self, result: AdjudicationResult) -> Optional[dict]:
        if not result.imaging_result:
            return None
        img = result.imaging_result
        return {
            "mode":             img.mode_used.value,
            "class_label":      img.class_label,
            "confidence":       img.confidence,
            "icd10_suggestion": img.icd10_suggestion,
            "heatmap_path":     img.heatmap_path,
        }

    def _explainability_dict(self, result: AdjudicationResult) -> Optional[dict]:
        if not result.explainability:
            return None
        exp = result.explainability
        return {
            "summary":     exp.summary,
            "key_factors": exp.key_factors,
            "risk_flags":  exp.risk_flags,
            "reasoning_chain": [
                {
                    "step":        s.step,
                    "stage":       s.stage.value,
                    "description": s.description,
                    "outcome":     s.outcome,
                    "confidence":  s.confidence,
                }
                for s in exp.reasoning_chain
            ],
        }

    def _entry_dict(self, entry: AuditEntry) -> dict:
        return {
            "audit_trace_id":  entry.audit_trace_id,
            "claim_id":        entry.claim_id,
            "stage":           entry.stage.value,
            "status":          entry.status.value,
            "timestamp":       entry.timestamp.isoformat(),
            "duration_ms":     entry.duration_ms,
            "input_snapshot":  entry.input_snapshot,
            "output_snapshot": entry.output_snapshot,
            "error_message":   entry.error_message,
        }
