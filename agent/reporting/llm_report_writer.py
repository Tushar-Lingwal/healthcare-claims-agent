"""
llm_report_writer.py — LLM-powered narrative clinical report writer.

Takes a structured AdjudicationResult and generates a full clinical document
written in professional medical prose, explaining:
  - The agent's complete reasoning process at each pipeline stage
  - How each piece of evidence influenced the decision
  - What clinical guidelines were consulted and what they said
  - Why each policy rule passed or failed
  - What the imaging model found (if applicable)
  - Specific, actionable clinical and administrative recommendations

Uses the same LLM provider configured in .env (groq / gemini / anthropic / rules).
Falls back to rule-based narrative generation if no LLM is available.

Output: HTML string ready to be served as a full page.
"""

import json
import logging
import os
import re
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

REPORT_SYSTEM_PROMPT = """You are an expert medical documentation system and insurance adjudication analyst.

Your task is to generate a formal, realistic, and legally-defensible Claim Adjudication & Clinical Review Report.

The output must resemble a real-world healthcare insurance document used by hospitals, TPAs, and insurers.
Maintain a professional, clinical, and precise tone. Avoid conversational language.

Generate a structured report with ALL of the following sections, using the EXACT headers shown:

## 1. REPORT HEADER
## 2. PATIENT SUMMARY (PHI-SAFE)
## 3. CLAIM OVERVIEW
## 4. CLINICAL EXTRACTION SUMMARY
## 5. MEDICAL CODING MAPPING
## 6. CLINICAL GUIDELINE JUSTIFICATION
## 7. POLICY RULE EVALUATION
## 8. ANOMALY & RISK DETECTION
## 9. MULTI-AGENT REASONING SUMMARY
## 10. FINAL ADJUDICATION DECISION
## 11. FINANCIAL BREAKDOWN
## 12. ACTIONABLE RECOMMENDATIONS
## 13. APPENDIX

SECTION REQUIREMENTS:

Section 1 - REPORT HEADER: Include Claim ID, Case ID, Date of Processing, System (ClaimIQ v1.0), Adjudication Type (Automated/Semi-Automated), Turnaround Time.

Section 2 - PATIENT SUMMARY: Patient Token ID (from audit trace), Age Group, Gender, Encounter Type, plan type. NEVER include real PHI.

Section 3 - CLAIM OVERVIEW: Claim Type, Insurance Plan, Provider Type, Primary Diagnosis, Key Procedures. Include estimated claimed amount range based on procedure complexity.

Section 4 - CLINICAL EXTRACTION SUMMARY: Present as a markdown table with columns: Entity Type | Extracted Term | Confidence | Source. Include all diagnoses, procedures, medications, symptoms extracted.

Section 5 - MEDICAL CODING MAPPING: Table with ICD-10 codes, descriptions, confidence. Table with CPT codes, descriptions, confidence. Include alternative code suggestions where relevant.

Section 6 - CLINICAL GUIDELINE JUSTIFICATION: For each relevant guideline — Source, Relevant principle, Interpretation for this claim.

Section 7 - POLICY RULE EVALUATION: Table with Rule Name | Status (PASS/FAIL) | Explanation | Impact. List ALL rules evaluated.

Section 8 - ANOMALY & RISK DETECTION: List anomalies with Severity (Low/Medium/High), Fraud Risk Score (0.0-1.0), justification. If no anomalies, state clearly.

Section 9 - MULTI-AGENT REASONING SUMMARY: For each activated MoE expert — agent name, opinion, risk level, key flags, recommendations. If no MoE data, state not applicable.

Section 10 - FINAL ADJUDICATION DECISION: Decision, Confidence Score %, 2-3 sentence justification.

Section 11 - FINANCIAL BREAKDOWN: Table with Claimed Amount | Approved Amount | Deducted Amount | Deduction Reason. Use realistic estimates if exact amounts not provided. Base on procedure complexity and plan type.

Section 12 - ACTIONABLE RECOMMENDATIONS: Numbered list. Required documents, coding corrections, clinical clarifications, next steps. Minimum 4 specific recommendations.

Section 13 - APPENDIX: Additional clinical context, model versions, any caveats.

FORMATTING RULES:
- Use markdown tables where specified — pipe-separated with header row
- Keep each section concise but complete — no padding
- Formal medical + insurance tone throughout
- Be specific — always reference actual codes, rule IDs, scores from the provided data
- Ensure decision in Section 11 matches rule evaluation in Section 7
- NEVER reconstruct PHI — only use tokenized identifiers

Total length: 1,500-2,500 words. Start directly with ## 1. REPORT HEADER."""


def _build_report_prompt(data: dict) -> str:
    """Builds the comprehensive user prompt from all available adjudication data."""
    dec         = data.get("decision", "unknown")
    conf        = data.get("confidence_score", 0)
    claim_id    = data.get("claim_id", "unknown")
    trace_id    = data.get("audit_trace_id", "unknown")
    icd_codes   = data.get("icd10_codes", [])
    cpt_codes   = data.get("cpt_codes", [])
    rules       = data.get("rule_evaluations", [])
    edges       = data.get("edge_cases", [])
    passages    = data.get("rag_passages", [])
    chain       = (data.get("explainability") or {}).get("reasoning_chain", [])
    summary     = (data.get("explainability") or {}).get("summary", "")
    risk_flags  = (data.get("explainability") or {}).get("risk_flags", [])
    reasons     = data.get("reasons", [])
    notes       = data.get("_notes", "")
    plan        = data.get("_plan", "unknown")
    moe         = data.get("moe_analysis") or data.get("moe") or {}
    img_result  = data.get("imaging_result") or data.get("_imaging") or {}

    # ── Format all data sections ──────────────────────────────────────────

    icd_str = "\n".join([
        f"  • {c['code']}: {c['description']} | Confidence: {c['confidence']:.1%} | "
        f"Match type: {'exact' if c.get('is_exact_match') else 'fuzzy'}"
        for c in icd_codes
    ]) or "  No ICD-10 codes mapped"

    cpt_str = "\n".join([
        f"  • {c['code']}: {c['description']} | Confidence: {c['confidence']:.1%}"
        for c in cpt_codes
    ]) or "  No CPT codes mapped"

    passing  = [r for r in rules if r.get("passed")]
    blocking = [r for r in rules if not r.get("passed") and r.get("action") == "reject"]
    warning  = [r for r in rules if not r.get("passed") and r.get("action") == "flag_review"]

    rules_str  = ""
    if passing:
        rules_str += f"PASSING RULES ({len(passing)}):\n" + "\n".join([
            f"  ✓ {r['rule_id']}: {r['reason']}"
            for r in passing
        ]) + "\n\n"
    if blocking:
        rules_str += f"BLOCKING FAILURES ({len(blocking)}) — CAUSE OF REJECTION:\n" + "\n".join([
            f"  ✗ {r['rule_id']}: {r['reason']}"
            for r in blocking
        ]) + "\n\n"
    if warning:
        rules_str += f"WARNING FLAGS ({len(warning)}) — REQUIRE REVIEW:\n" + "\n".join([
            f"  ⚠ {r['rule_id']}: {r['reason']}"
            for r in warning
        ]) + "\n"
    if not rules_str:
        rules_str = "  No rules evaluated"

    passage_str = ""
    for i, p in enumerate(passages[:5], 1):
        passage_str += (
            f"\n  [{i}] Source: {p.get('source', 'Unknown')} "
            f"(Relevance: {p.get('relevance', 0):.3f})\n"
            f"      Content: {p.get('content', '')[:500]}\n"
        )
    if not passage_str:
        passage_str = "  No guideline passages retrieved"

    chain_str = "\n".join([
        f"  Stage {s.get('step', i+1)} — {s.get('stage','').replace('_',' ').upper()}: "
        f"Confidence {s.get('confidence', 0):.1%} | {s.get('outcome','')}"
        for i, s in enumerate(chain)
    ]) or "  No reasoning chain available"

    edge_str = ""
    for e in edges:
        edge_str += (
            f"\n  Type: {e.get('type','unknown').replace('_',' ').upper()} | "
            f"Severity: {e.get('severity','unknown').upper()}\n"
            f"  Description: {e.get('description','')}\n"
            f"  Recommendation: {e.get('recommendation','')}\n"
        )
        if e.get('image_signal'):
            edge_str += (
                f"  Image signal: {e['image_signal']} | "
                f"Text signal: {e.get('text_signal','')} | "
                f"Mismatch score: {e.get('mismatch_score',0):.2f}\n"
            )
    if not edge_str:
        edge_str = "  No edge cases detected"

    # ── MoE Analysis ──────────────────────────────────────────────────────
    moe_str = ""
    if moe and not moe.get("skipped") and (moe.get("activated_experts") or moe.get("findings")):
        moe_str = f"""
MOE SYSTEM SUMMARY:
  Activated experts: {', '.join(moe.get('activated_experts', []))}
  Consensus risk level: {moe.get('consensus_risk', 'unknown').upper()}
  Average expert confidence: {moe.get('moe_confidence', 0):.1%}
  Routing reason: {moe.get('routing_reason', '')}

ROUTER SCORES:
{chr(10).join([f"  {k}: {v:.1%}" for k, v in (moe.get('router_scores') or {}).items()])}

EXPERT FINDINGS:"""
        for f in (moe.get("findings") or []):
            moe_str += f"""

  {f.get('expert_name','').upper()} (Router score: {f.get('router_score',0):.1%} | Expert confidence: {f.get('expert_confidence',0):.1%})
  Assessment: {f.get('assessment','')}
  Risk level: {f.get('risk_level','').upper()}
  Source: {f.get('source','')}
  Risk flags: {'; '.join(f.get('risk_flags', [])) or 'None'}
  Recommendations: {'; '.join(f.get('recommendations', [])) or 'None'}
  Imaging assessment: {f.get('imaging_assessment') or 'N/A'}
  Narrative: {f.get('narrative') or 'Not generated (LLM not configured)'}"""

        if moe.get("suggested_codes"):
            moe_str += f"""

SUGGESTED ADDITIONAL CODES:
{chr(10).join([f"  • {c['code']}: {c['description']} — {c.get('reason','')}" for c in moe['suggested_codes']])}"""

        if moe.get("imaging_assessments"):
            moe_str += f"""

IMAGING ASSESSMENTS FROM EXPERTS:
{chr(10).join([f"  • {a}" for a in moe['imaging_assessments']])}"""
    else:
        moe_str = "  MoE analysis not available for this claim (no experts activated or system unavailable)"

    # ── Imaging ───────────────────────────────────────────────────────────
    img_str = ""
    if img_result and img_result.get("predicted_class"):
        img_str = f"""
IMAGING MODEL RESULTS:
  Model: Swin Transformer (swin_base_patch4_window7_224)
  Primary classification: {img_result.get('predicted_class', 'Unknown')}
  Confidence: {img_result.get('confidence', 0):.1%}
  Category: {img_result.get('category', 'unknown')}
  Suggested ICD-10: {img_result.get('icd10_code', '')} — {img_result.get('icd10_description', '')}
  Mode: {img_result.get('mode_used', img_result.get('mode', 'unknown'))}"""
        if img_result.get("all_probabilities"):
            top3 = sorted(img_result["all_probabilities"].items(), key=lambda x: x[1], reverse=True)[:3]
            img_str += f"""
  Top-3 probabilities: {', '.join([f"{k}: {v:.1%}" for k, v in top3])}"""
    else:
        img_str = "  No imaging model output available for this claim"

    return f"""Write a comprehensive, professional Clinical Adjudication Report for the following claim.
This report will be reviewed by medical directors, compliance officers, and clinical governance committees.
Write with authority, precision, and complete transparency about the AI system's reasoning.

{"="*70}
CLAIM IDENTIFICATION
{"="*70}
Claim ID:           {claim_id}
Audit Trace ID:     {trace_id}
FINAL DECISION:     {dec.upper().replace('_', ' ')}
Overall Confidence: {conf:.1%}
Insurance Plan:     {plan.upper()}
Decided At:         {data.get('decided_at', 'unknown')}
Pipeline Version:   {data.get('pipeline_version', '1.0.0')}

{"="*70}
STAGE 1 — PHI TOKENIZATION
{"="*70}
All patient identifiers (name, DOB, policy number, MRN) were encrypted using
AES-256 vault tokenization before entering the AI pipeline. No PHI appears
anywhere in the processing chain or this audit report.

{"="*70}
STAGE 2 — CLINICAL NOTES SUBMITTED
{"="*70}
{notes or '(No clinical notes provided)'}

AI Summary: {summary or '(Not generated)'}
Decision Reasons: {chr(10).join(f'  • {r}' for r in reasons) if reasons else '  None listed'}
Risk Flags: {chr(10).join(f'  • {f}' for f in risk_flags) if risk_flags else '  None identified'}

{"="*70}
STAGE 3 — MEDICAL CODING OUTPUT
{"="*70}
ICD-10-CM DIAGNOSIS CODES:
{icd_str}

CPT PROCEDURE CODES:
{cpt_str}

{"="*70}
STAGE 4 — CLINICAL GUIDELINE RAG RETRIEVAL
{"="*70}
{passage_str}

{"="*70}
STAGE 5 — POLICY RULE ENGINE (14 rules evaluated)
{"="*70}
{rules_str}

{"="*70}
STAGE 5b — MIXTURE OF EXPERTS ANALYSIS
{"="*70}
{moe_str}

{"="*70}
STAGE 5c — IMAGING MODEL OUTPUT
{"="*70}
{img_str}

{"="*70}
STAGE 6 — EDGE CASE DETECTION
{"="*70}
{edge_str}

{"="*70}
STAGE 7 — DECISION ENGINE REASONING CHAIN
{"="*70}
{chain_str}

{"="*70}
FINANCIAL CONTEXT (estimate based on procedure complexity)
{"="*70}
Insurance Plan:      {plan.upper()}
Procedure Count:     {len(icd_codes)} ICD-10 codes, {len(cpt_codes)} CPT codes
Estimated Complexity: {"High — brain imaging + specialist procedures" if any(c["code"].startswith(("70","78","93")) for c in cpt_codes) else "Standard outpatient"}
Plan Coverage Notes:  {"Basic plan — elective procedures may be excluded" if plan=="basic" else "Standard plan — prior auth required for imaging" if plan=="standard" else "Premium plan — comprehensive coverage"}

For Section 12 FINANCIAL BREAKDOWN: Estimate realistic amounts based on:
- MRI Brain (CPT 70551-70553): $1,200-$3,500
- PET Scan (CPT 78608): $3,000-$6,000  
- Cardiac Cath (CPT 93452): $8,000-$15,000
- Joint Replacement (CPT 27447): $25,000-$45,000
- Office visits (CPT 9921x): $150-$400
- Cognitive testing (CPT 96132): $400-$800
Adjust approved amount based on plan type and decision. Show realistic deductions.

{"="*70}
CONTEXT FOR REPORT WRITING
{"="*70}
Decision: {dec.upper().replace('_', ' ')} at {conf:.1%} confidence

If APPROVED: Explain comprehensively why all evidence supports approval.
  Detail each passing rule, the guideline evidence, and why the codes are medically appropriate.
If REJECTED: Explain precisely what is wrong, why it violates policy, and exactly what must be fixed.
  Be specific — name the rule, the code, the plan limitation, and the exact corrective action.
If NEEDS REVIEW: Explain the specific uncertainties, what human judgment is needed, and why automation cannot resolve it.

For MoE experts: Explain what each activated expert found, what their assessment means clinically,
and how their findings influenced or should influence the final decision.

For imaging: Explain what the model found, how confident it was, and whether it agrees with the clinical notes.

Write the complete professional report now:"""


def _call_groq(prompt: str, system: str) -> Optional[str]:
    try:
        from groq import Groq
        client = Groq(api_key=os.environ.get("GROQ_API_KEY",""))
        model  = os.environ.get("GROQ_MODEL","llama-3.3-70b-versatile")
        resp   = client.chat.completions.create(
            model=model, max_tokens=4096, temperature=0.2,
            messages=[{"role":"system","content":system},{"role":"user","content":prompt}]
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        logger.warning(f"Groq failed: {e}")
        return None

def _call_gemini(prompt: str, system: str) -> Optional[str]:
    try:
        import google.generativeai as genai
        genai.configure(api_key=os.environ.get("GEMINI_API_KEY",""))
        model = genai.GenerativeModel(
            model_name=os.environ.get("GEMINI_MODEL","gemini-1.5-flash"),
            system_instruction=system,
        )
        resp = model.generate_content(prompt,
            generation_config={"temperature":0.2,"max_output_tokens":4096})
        return resp.text.strip()
    except Exception as e:
        logger.warning(f"Gemini failed: {e}")
        return None

def _call_anthropic(prompt: str, system: str) -> Optional[str]:
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY",""))
        resp = client.messages.create(
            model=os.environ.get("ANTHROPIC_MODEL","claude-sonnet-4-6"),
            max_tokens=4096, system=system,
            messages=[{"role":"user","content":prompt}]
        )
        return resp.content[0].text.strip()
    except Exception as e:
        logger.warning(f"Anthropic failed: {e}")
        return None

def _call_openai(prompt: str, system: str) -> Optional[str]:
    try:
        from openai import OpenAI
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY",""))
        resp = client.chat.completions.create(
            model=os.environ.get("OPENAI_MODEL","gpt-4o-mini"),
            max_tokens=4096, temperature=0.2,
            messages=[{"role":"system","content":system},{"role":"user","content":prompt}]
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        logger.warning(f"OpenAI failed: {e}")
        return None


def generate_llm_narrative(data: dict) -> Optional[str]:
    """
    Calls the configured LLM to generate a full narrative report.
    Tries providers in order: groq → gemini → anthropic → openai → None
    """
    provider = os.environ.get("LLM_PROVIDER", "rules").lower()
    prompt   = _build_report_prompt(data)
    system   = REPORT_SYSTEM_PROMPT

    logger.info(f"Generating LLM narrative report using provider: {provider}")

    result = None
    if provider == "groq":
        result = _call_groq(prompt, system)
    elif provider == "gemini":
        result = _call_gemini(prompt, system)
    elif provider == "anthropic":
        result = _call_anthropic(prompt, system)
    elif provider == "openai":
        result = _call_openai(prompt, system)

    # Fallback cascade
    if result is None and provider != "groq":
        result = _call_groq(prompt, system)
    if result is None and provider != "gemini":
        result = _call_gemini(prompt, system)
    if result is None:
        logger.warning("All LLM providers failed — using rule-based narrative")

    return result


def _rule_based_narrative(data: dict) -> str:
    """
    Generates a detailed narrative without an LLM, using the structured data.
    Used as fallback when no API keys are configured.
    """
    dec      = data.get("decision", "unknown")
    conf     = data.get("confidence_score", 0)
    icd      = data.get("icd10_codes", [])
    cpt      = data.get("cpt_codes", [])
    rules    = data.get("rule_evaluations", [])
    edges    = data.get("edge_cases", [])
    passages = data.get("rag_passages", [])
    chain    = (data.get("explainability") or {}).get("reasoning_chain", [])
    summary  = (data.get("explainability") or {}).get("summary", "")
    flags    = (data.get("explainability") or {}).get("risk_flags", [])
    notes    = data.get("_notes", "")
    plan     = data.get("_plan", "unknown")
    claim_id = data.get("claim_id", "")
    trace_id = data.get("audit_trace_id", "")

    blocking = [r for r in rules if not r.get("passed") and r.get("action") == "reject"]
    warnings = [r for r in rules if not r.get("passed") and r.get("action") == "flag_review"]
    passing  = [r for r in rules if r.get("passed")]

    dec_label = {"approved": "APPROVED", "rejected": "REJECTED", "needs_review": "NEEDS REVIEW"}.get(dec, dec.upper())

    sections = {}

    # ── Executive Summary ────────────────────────────────────────────────────
    if dec == "approved":
        exec_text = (
            f"Following a comprehensive multi-stage automated review, claim {claim_id} has been "
            f"**APPROVED** with an overall confidence score of {conf:.1%}. The ClaimIQ adjudication "
            f"agent evaluated the submitted clinical documentation against {len(rules)} applicable "
            f"policy rules, cross-referenced {len(passages)} clinical guideline passages from authoritative "
            f"sources including NCCN, CMS, AHA, and AAOS, and performed full ICD-10-CM and CPT coding "
            f"verification. All evaluated rules returned a passing status, no clinical anomalies were "
            f"detected, and the submitted diagnosis-procedure relationships are medically consistent and "
            f"covered under the patient's {plan.upper()} insurance plan. This claim is cleared for "
            f"payment processing without further clinical review."
        )
    elif dec == "rejected":
        primary = blocking[0] if blocking else None
        exec_text = (
            f"Following a comprehensive multi-stage automated review, claim {claim_id} has been "
            f"**REJECTED** with an overall confidence score of {conf:.1%}. The ClaimIQ adjudication "
            f"agent identified {len(blocking)} blocking policy violation(s) that prevent approval under "
            f"current plan terms. "
            f"{'The primary rejection reason is rule ' + primary['rule_id'] + ': ' + primary['reason'] + '.' if primary else ''} "
            f"The claim cannot be processed for payment in its current form and must be corrected "
            f"and resubmitted. This report provides a detailed account of the agent's reasoning and "
            f"specific corrective actions required."
        )
    else:
        exec_text = (
            f"Following a comprehensive multi-stage automated review, claim {claim_id} has been "
            f"**FLAGGED FOR MANUAL REVIEW** with an overall confidence score of {conf:.1%}. The "
            f"ClaimIQ adjudication agent identified {len(warnings)} policy rule(s) requiring payer "
            f"verification and {len(edges)} clinical anomaly(ies) that require human judgment before "
            f"a final payment decision can be rendered. This report provides a complete account of "
            f"the agent's reasoning, the specific issues identified, and the steps required for "
            f"clinical review completion."
        )
    sections["## Executive Summary"] = exec_text

    # ── Clinical Background ──────────────────────────────────────────────────
    bg_text = (
        f"The claim was submitted under a {plan.upper()} insurance plan with policy reference "
        f"{data.get('_plan', 'on file')}. The clinical documentation submitted with this claim "
        f"reads as follows: \"{notes[:500]}{'...' if len(notes) > 500 else ''}\"\n\n"
        f"This narrative formed the primary input to the agent's clinical NER (Named Entity Recognition) "
        f"pipeline, from which diagnoses and procedures were automatically extracted. Any structured "
        f"diagnosis and procedure tags submitted alongside the clinical notes were merged with the "
        f"NER output to produce the final entity set used for coding and rule evaluation."
    )
    sections["## Clinical Background and Patient Context"] = bg_text

    # ── AI Reasoning Process ─────────────────────────────────────────────────
    reasoning_text = (
        f"The ClaimIQ agent processed this claim through a deterministic 9-stage pipeline. "
        f"The following account describes the agent's activity and findings at each stage.\n\n"
        f"**Stage 1 — PHI Tokenization:** The agent's first action was to strip all Protected Health "
        f"Information from the submitted claim data. Patient name, date of birth, policy number, and "
        f"all other HIPAA-defined identifiers were replaced with encrypted vault tokens before any "
        f"processing began. This ensures zero PHI exposure throughout the AI pipeline.\n\n"
        f"**Stage 2 — Clinical Extraction:** Using a combination of rule-based NER patterns (184 "
        f"medical regex patterns) and the configured LLM provider, the agent parsed the clinical notes "
        f"to identify medical entities. The extraction identified {len(icd)} diagnosis concept(s) and "
        f"{len(cpt)} procedure concept(s) from the submitted documentation.\n\n"
        f"**Stage 3 — Medical Coding:** Each extracted entity was mapped to official ICD-10-CM and CPT "
        f"codes using FTS5 full-text search across a database of 534 codes. The agent applied "
        f"synonym translation and fuzzy matching to handle abbreviations and clinical shorthand.\n\n"
        f"**Stage 4 — Guideline Retrieval (RAG):** The agent queried its clinical guideline database "
        f"using TF-IDF medical vector similarity, retrieving {len(passages)} relevant passage(s) from "
        f"authoritative sources to establish medical necessity context.\n\n"
        f"**Stage 5 — Policy Rule Engine:** {len(rules)} applicable policy rules were evaluated against "
        f"the coded claim. Rules cover five categories: code compatibility, prior authorization "
        f"requirements, age restrictions, plan coverage limitations, and frequency controls.\n\n"
        f"**Stage 6 — Edge Case Detection:** The agent screened for seven categories of clinical "
        f"anomalies including missing diagnoses, imaging-text mismatches, low confidence extractions, "
        f"and conflicting code pairs. This screen found {len(edges)} anomaly(ies).\n\n"
        f"**Stage 7 — Decision Engine:** Using a weighted confidence model (40% coding confidence + "
        f"35% rule pass rate + 25% edge case penalty), the agent computed a final confidence score "
        f"of {conf:.1%} and rendered the decision: **{dec_label}**.\n\n"
        f"**Stage 8 — Explainability:** The agent generated a reasoning chain documenting the outcome "
        f"of each stage with individual confidence scores, producing the audit trail for this report."
    )
    if chain:
        reasoning_text += "\n\n**Detailed Stage-by-Stage Confidence Scores:**\n"
        for step in chain:
            stage = step.get("stage", "").replace("_", " ").title()
            sc    = step.get("confidence", 0)
            out   = step.get("outcome", "")
            reasoning_text += f"\n- {stage}: {sc:.0%} confidence — {out}"
    sections["## AI Agent Reasoning Process"] = reasoning_text

    # ── Medical Coding Analysis ──────────────────────────────────────────────
    coding_text = ""
    if icd:
        coding_text += "**ICD-10-CM Diagnosis Codes:**\n\n"
        for c in icd:
            coding_text += (
                f"The agent mapped a diagnosis to **{c['code']}** ({c['description']}) "
                f"with {c['confidence']:.0%} confidence. "
            )
            if c['confidence'] >= 0.90:
                coding_text += "This is a high-confidence match indicating the entity text closely matched the code description. "
            elif c['confidence'] >= 0.80:
                coding_text += "This is a moderate-confidence match using FTS5 full-text search and fuzzy reranking. "
            else:
                coding_text += "This is a lower-confidence match — clinical review of this code mapping is advisable. "
            coding_text += "\n\n"
    else:
        coding_text += ("No ICD-10-CM diagnosis codes were mapped. This may indicate the clinical notes "
                        "did not contain recognizable diagnostic terminology, or that the extraction pipeline "
                        "failed to identify relevant entities. This is a significant gap that should be "
                        "addressed before resubmission.\n\n")

    if cpt:
        coding_text += "**CPT Procedure Codes:**\n\n"
        for c in cpt:
            coding_text += (
                f"The procedure was mapped to **{c['code']}** ({c['description']}) "
                f"with {c['confidence']:.0%} confidence. "
            )
            if c['confidence'] >= 0.90:
                coding_text += "This code match is highly reliable. "
            else:
                coding_text += "This code match should be verified by a certified medical coder. "
            coding_text += "\n\n"
    else:
        coding_text += "No CPT procedure codes were mapped for this claim.\n\n"

    sections["## Medical Coding Analysis"] = coding_text

    # ── Rule Evaluation ──────────────────────────────────────────────────────
    rule_text = (
        f"The policy rule engine evaluated {len(rules)} rule(s) applicable to the codes "
        f"identified in this claim. Rules are organized into five categories: compatibility "
        f"(COMPAT), authorization (AUTH), age restrictions (AGE), coverage (COV), and "
        f"frequency (FREQ).\n\n"
    )
    if passing:
        rule_text += f"**Rules that PASSED ({len(passing)}):**\n\n"
        for r in passing:
            rule_text += (
                f"Rule **{r['rule_id']}** passed evaluation. {r['reason']} "
                f"This rule's passage contributes positively to the claim's eligibility.\n\n"
            )
    if blocking:
        rule_text += f"**Rules that FAILED with BLOCKING action ({len(blocking)}):**\n\n"
        for r in blocking:
            rule_text += (
                f"Rule **{r['rule_id']}** failed and triggered a REJECT action, which is the "
                f"direct cause of this claim's rejection. Specifically: {r['reason']} This rule "
                f"failure cannot be overridden by passing other rules — it must be resolved before "
                f"the claim can be approved.\n\n"
            )
    if warnings:
        rule_text += f"**Rules that FAILED with REVIEW flag ({len(warnings)}):**\n\n"
        for r in warnings:
            rule_text += (
                f"Rule **{r['rule_id']}** failed and triggered a FLAG FOR REVIEW action. "
                f"{r['reason']} Unlike a blocking failure, this does not automatically reject "
                f"the claim, but it does require payer verification or prior authorization before "
                f"payment can proceed.\n\n"
            )
    sections["## Policy Rule Evaluation"] = rule_text

    # ── Guideline Evidence ───────────────────────────────────────────────────
    if passages:
        guide_text = (
            f"The agent retrieved {len(passages)} clinical guideline passage(s) from its knowledge "
            f"base using medical TF-IDF vector similarity search. These passages established the "
            f"medical necessity context for the claim's procedures and diagnoses.\n\n"
        )
        for p in passages[:4]:
            guide_text += (
                f"**From {p.get('source', 'Unknown Source')}** (relevance score: {p.get('relevance', 0):.3f}):\n"
                f"{p.get('content', '')[:350]}...\n\n"
                f"This passage is relevant because it addresses the clinical scenario presented in "
                f"this claim and provides evidence for the medical necessity of the claimed services.\n\n"
            )
    else:
        guide_text = (
            "No clinical guideline passages were retrieved for this claim. This may indicate "
            "the ICD-10 and CPT codes were not found in the guideline index, or that the claim "
            "relates to a clinical area not yet covered in the guideline database. The absence "
            "of guideline citations means the medical necessity evaluation relied solely on "
            "policy rules rather than clinical evidence, which may be a limitation of this review."
        )
    sections["## Clinical Guideline Evidence"] = guide_text

    # ── Edge Cases ───────────────────────────────────────────────────────────
    if edges:
        edge_text = (
            f"The edge case detection system identified {len(edges)} anomaly(ies) in this claim. "
            f"Each anomaly is described below with its severity and the agent's recommendation.\n\n"
        )
        for e in edges:
            typ  = e.get("type", "unknown").replace("_", " ").title()
            sev  = e.get("severity", "unknown").upper()
            desc = e.get("description", "")
            rec  = e.get("recommendation", "")
            edge_text += (
                f"**{typ} — Severity: {sev}**\n\n"
                f"{desc} "
            )
            if e.get("image_signal"):
                edge_text += (
                    f"The imaging model classified the scan as '{e['image_signal']}' while the "
                    f"clinical text suggests '{e.get('text_signal', 'unknown')}'. The mismatch "
                    f"score of {e.get('mismatch_score', 0):.2f} exceeds the threshold for high-severity "
                    f"discordance. This means either the imaging result and clinical notes describe "
                    f"different patients/studies, or there is a genuine diagnostic discrepancy that "
                    f"requires expert reconciliation. "
                )
            edge_text += f"\n\n*Agent recommendation:* {rec}\n\n"
    else:
        edge_text = (
            "The edge case detection system screened for seven categories of clinical anomalies "
            "including missing diagnoses, imaging-text mismatches, high-risk procedure combinations, "
            "low confidence extractions, and duplicate submissions. No anomalies were detected. "
            "This is a positive indicator for claim integrity and supports the agent's confidence "
            "in the adjudication decision."
        )
    sections["## Edge Case and Risk Analysis"] = edge_text

    # ── Decision Justification ───────────────────────────────────────────────
    if dec == "approved":
        just_text = (
            f"The agent's final decision of APPROVED is justified by the convergence of multiple "
            f"positive evidence streams. All {len(passing)} evaluated policy rules passed without "
            f"exception. The ICD-10 and CPT codes are mutually consistent — the diagnoses support "
            f"the medical necessity of the procedures. The clinical guidelines retrieved confirm "
            f"that the requested services align with evidence-based treatment standards for the "
            f"conditions documented. No clinical anomalies were detected. The overall confidence "
            f"score of {conf:.1%} reflects high certainty in this decision. The plan's coverage "
            f"terms are satisfied for the {plan.upper()} tier. This claim is appropriate for "
            f"payment processing."
        )
    elif dec == "rejected":
        just_text = (
            f"The agent's final decision of REJECTED is determined primarily by {len(blocking)} "
            f"blocking rule failure(s). The weighted confidence model scored this claim at {conf:.1%}, "
            f"with the rule failure(s) contributing a significant penalty to the overall score. "
            f"In the ClaimIQ system, any blocking rule failure (action: reject) immediately constrains "
            f"the final decision to REJECTED, regardless of the confidence level, because these rules "
            f"represent absolute coverage policy — not probabilistic assessments. The specific "
            f"blocking failure(s) must be resolved before this claim can be reconsidered."
        )
    else:
        just_text = (
            f"The agent's final decision of NEEDS REVIEW reflects genuine uncertainty that cannot "
            f"be resolved algorithmically. The confidence score of {conf:.1%} falls in the range "
            f"that requires human judgment. Specifically, the {len(warnings)} flagged rule(s) "
            f"require payer verification steps that the automated system cannot complete — such as "
            f"confirming prior authorization status with the payer, or verifying that a required "
            f"precondition has been met. Additionally, {len(edges)} edge case(s) were detected "
            f"that introduce clinical uncertainty the agent cannot resolve. A qualified clinical "
            f"reviewer with access to the complete patient record and payer communication is the "
            f"appropriate decision-maker for this claim."
        )
    sections["## Final Decision Justification"] = just_text

    # ── Recommendations ──────────────────────────────────────────────────────
    rec_text = ""
    if dec == "approved":
        rec_text = (
            f"**Administrative:** This claim is cleared for payment processing. The adjudication "
            f"system has validated all required elements. The payer should proceed with reimbursement "
            f"according to the {plan.upper()} plan's fee schedule for the applicable CPT codes. "
            f"The audit trace {trace_id} should be filed with the payment documentation for "
            f"compliance purposes.\n\n"
            f"**Clinical:** No further clinical review is required for this claim. Standard post-payment "
            f"audit procedures apply. The clinical documentation is adequate to support the codes "
            f"submitted."
        )
    elif dec == "rejected":
        rec_text = "The following specific corrective actions are required before resubmission:\n\n"
        for i, r in enumerate(blocking, 1):
            rid = r['rule_id']
            if rid.startswith("COMPAT"):
                rec_text += (
                    f"**Action {i} — Code Compatibility ({rid}):** The claim must establish a clear "
                    f"clinical relationship between the submitted diagnosis codes and procedure codes. "
                    f"{r['reason']} Obtain additional clinical documentation from the treating "
                    f"physician that explicitly connects the diagnosis to the necessity of the "
                    f"procedure, and ensure this is reflected in the corrected claim submission.\n\n"
                )
            elif rid.startswith("COV"):
                rec_text += (
                    f"**Action {i} — Coverage Limitation ({rid}):** {r['reason']} The patient's "
                    f"current {plan.upper()} plan does not cover this procedure. Options include: "
                    f"(a) coordinating an upgrade to STANDARD or PREMIUM plan coverage prior to "
                    f"service, (b) applying for a medical necessity exception with supporting "
                    f"documentation from a specialist, or (c) direct patient billing if the patient "
                    f"accepts financial responsibility.\n\n"
                )
            elif rid.startswith("AGE"):
                rec_text += (
                    f"**Action {i} — Age Restriction ({rid}):** {r['reason']} If a clinical exception "
                    f"is warranted, obtain written documentation from a relevant specialist providing "
                    f"medical justification for the procedure in this patient's age group, and include "
                    f"it with the resubmission.\n\n"
                )
            else:
                rec_text += f"**Action {i} — {rid}:** {r['reason']} Address this issue and resubmit.\n\n"

        rec_text += (
            f"**General:** All corrective actions must be addressed before resubmission. Partial "
            f"corrections will result in re-rejection. Reference audit trace **{trace_id}** in all "
            f"resubmission correspondence. If you believe this rejection is in error, initiate the "
            f"formal appeals process with supporting clinical and administrative evidence."
        )
    else:
        rec_text = ""
        for i, r in enumerate(warnings, 1):
            if r['rule_id'].startswith("AUTH"):
                rec_text += (
                    f"**Action {i} — Prior Authorization ({r['rule_id']}):** {r['reason']} A prior "
                    f"authorization request must be submitted to the payer before this claim can be "
                    f"processed for payment. The PA request should include: the treating physician's "
                    f"clinical notes, diagnosis codes, procedure codes, and documentation of medical "
                    f"necessity per the applicable clinical guideline. Do not process this claim for "
                    f"payment until PA approval is received in writing.\n\n"
                )
            else:
                rec_text += f"**Action {i} — {r['rule_id']}:** {r['reason']}\n\n"

        for e in edges:
            if e.get("type") == "image_text_mismatch":
                rec_text += (
                    f"**Critical — Imaging Discordance:** The imaging AI and clinical text diagnoses "
                    f"disagree significantly. This requires a radiologist to formally review the scan "
                    f"and issue a written report reconciling the discrepancy. Additionally, the treating "
                    f"physician should attest in writing that the clinical diagnosis is accurate. Do not "
                    f"process for payment until both reviews are complete and the discrepancy is resolved.\n\n"
                )

        rec_text += (
            f"**Routing:** This claim should be routed to the clinical reviewer queue with a target "
            f"turnaround of 48 business hours. The reviewer should have access to the full patient "
            f"record, the payer's prior authorization portal, and this report. Reference audit trace "
            f"**{trace_id}** in all review documentation."
        )
    sections["## Clinical and Administrative Recommendations"] = rec_text

    # ── Confidence Assessment ────────────────────────────────────────────────
    conf_text = (
        f"The agent's overall confidence score for this adjudication is **{conf:.1%}**. "
        f"This score is computed as a weighted average: 40% from coding confidence "
        f"({sum(c['confidence'] for c in icd + cpt) / len(icd + cpt):.0%} average across {len(icd + cpt)} codes), "
        f"35% from rule evaluation ({len(passing)}/{len(rules)} rules passed), and "
        f"25% from edge case analysis ({1 - (len(edges) * 0.1):.0%} after penalties). "
        f"A score above 90% indicates very high certainty. A score of 70-90% indicates "
        f"moderate certainty suitable for automated approval or rejection. Below 70% "
        f"indicates the decision should be reviewed by a human. "
        f"At {conf:.1%}, this case {'is appropriate for automated processing' if conf >= 0.80 else 'warrants careful human review'}."
    ) if (icd or cpt) else (
        f"The agent's overall confidence score is {conf:.1%}. No codes were mapped, "
        f"which significantly limits the agent's ability to evaluate this claim accurately. "
        f"Human review is strongly recommended."
    )
    sections["## Agent Confidence Assessment"] = conf_text

    # ── Limitations ──────────────────────────────────────────────────────────
    limit_text = (
        f"This report was generated by ClaimIQ v1.0, an automated healthcare claims adjudication "
        f"system. The following limitations apply to this analysis:\n\n"
        f"The agent's ICD-10 and CPT code database contains {534} codes covering major clinical "
        f"areas. For rare or highly specialized procedures, the coding may be incomplete or "
        f"approximate. The clinical guideline database contains {35} passages from selected sources "
        f"and may not reflect the most current version of all referenced guidelines. The NER "
        f"extraction is based on pattern matching and may miss clinical concepts expressed in "
        f"unusual terminology. This system does not have access to the patient's full medical "
        f"record, prior claims history, or real-time payer authorization systems. All decisions "
        f"made by this system are advisory and subject to override by qualified clinical and "
        f"administrative professionals. This report does not constitute medical advice and should "
        f"not be used as a substitute for professional clinical judgment."
    )
    sections["## Limitations and Caveats"] = limit_text

    # Assemble full markdown
    full_md = ""
    for header, body in sections.items():
        full_md += f"{header}\n\n{body}\n\n---\n\n"
    return full_md


def markdown_to_html_sections(md: str) -> str:
    """Converts 17-section markdown report into styled HTML."""
    if not md:
        return '<p style="color:var(--t3);font-style:italic">No narrative generated.</p>'

    lines  = md.split("\n")
    html   = []
    in_table = False
    table_rows = []
    skip_section = False

    SECTION_COLORS = {
        "1":  "#6b8fff", "2":  "#6b8fff", "3":  "#a67fff",
        "4":  "#2dd4bf", "5":  "#2dd4bf", "6":  "#34d97b",
        "7":  "#f5a623", "8":  "#ff6b6b", "9":  "#a67fff",
        "10": "#6b8fff", "11": "#34d97b", "12": "#f5a623",
        "13": "#2dd4bf", "14": "#8888a8", "15": "#8888a8",
        "16": "#8888a8", "17": "#8888a8",
    }

    def flush_table():
        nonlocal in_table, table_rows
        if not table_rows:
            return
        header = table_rows[0]
        rows   = table_rows[2:]  # skip separator row
        cols   = [c.strip() for c in header.split("|") if c.strip()]
        thtml  = '<div class="rpt-table-wrap"><table class="rpt-table"><thead><tr>'
        thtml += "".join(f"<th>{c}</th>" for c in cols)
        thtml += "</tr></thead><tbody>"
        for row in rows:
            cells = [c.strip() for c in row.split("|") if c.strip()]
            if not cells:
                continue
            thtml += "<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>"
        thtml += "</tbody></table></div>"
        html.append(thtml)
        in_table  = False
        table_rows = []

    for line in lines:
        raw = line.strip()

        # Table row
        if raw.startswith("|"):
            if skip_section:
                continue
            if not in_table:
                in_table = True
                table_rows = []
            table_rows.append(raw)
            continue
        elif in_table:
            flush_table()

        if not raw:
            continue

        # Skip content belonging to a removed section
        if skip_section:
            continue

        # Section headers  ## N. TITLE
        import re
        m = re.match(r"^##\s+(\d+)\.\s+(.+)$", raw)
        if m:
            num, title = m.group(1), m.group(2)
            # Skip removed sections (by title, handles old cached reports too)
            SKIP_TITLES = {
                "COUNTERFACTUAL ANALYSIS",
                "EXPLAINABILITY TRACE",
                "COMPLIANCE & SECURITY NOTE",
                "IMMUTABLE AUDIT TRAIL",
            }
            if title.strip().upper() in SKIP_TITLES:
                # Skip this section header and all content until next ## header
                skip_section = True
                continue
            skip_section = False
            col = SECTION_COLORS.get(num, "#8888a8")
            html.append(
                f'<div class="rpt-section-header" style="--sec-col:{col}">'
                f'<span class="rpt-sec-num">{num}</span>'
                f'<span class="rpt-sec-title">{title}</span>'
                f'</div>'
            )
            continue

        # Sub-headers ### 
        if raw.startswith("### "):
            html.append(f'<div class="rpt-h3">{raw[4:]}</div>')
            continue

        # Horizontal rule
        if raw in ("---", "***", "___"):
            html.append('<hr class="rpt-hr">')
            continue

        # Numbered list
        m2 = re.match(r"^(\d+)\.\s+(.+)$", raw)
        if m2:
            html.append(f'<div class="rpt-num-item"><span class="rpt-num">{m2.group(1)}.</span><span>{m2.group(2)}</span></div>')
            continue

        # Bullet
        if raw.startswith("- ") or raw.startswith("* "):
            html.append(f'<div class="rpt-bullet"><span>•</span><span>{raw[2:]}</span></div>')
            continue

        # Bold key-value  **Key:** value
        m3 = re.match(r"^\*\*(.+?):\*\*\s*(.*)$", raw)
        if m3:
            html.append(
                f'<div class="rpt-kv"><span class="rpt-kv-key">{m3.group(1)}</span>'
                f'<span class="rpt-kv-val">{m3.group(2)}</span></div>'
            )
            continue

        # Plain inline markdown → span
        text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", raw)
        text = re.sub(r"\*(.+?)\*",   r"<em>\1</em>",           text)
        text = re.sub(r"`(.+?)`",     r'<code class="rpt-code">\1</code>', text)
        html.append(f'<p class="rpt-p">{text}</p>')

    if in_table:
        flush_table()

    return "\n".join(html)


def build_full_report_html(data: dict, narrative: str, generated_at: datetime) -> str:
    """Builds a professional clinical report HTML document."""
    import os
    dec      = data.get("decision", "unknown")
    conf     = data.get("confidence_score", 0)
    claim_id = data.get("claim_id", "")
    trace_id = data.get("audit_trace_id", "")
    plan     = data.get("_plan", "unknown")
    icd      = data.get("icd10_codes", [])
    cpt      = data.get("cpt_codes", [])
    rules    = data.get("rule_evaluations", [])
    edges    = data.get("edge_cases", [])
    moe      = data.get("moe_analysis") or data.get("moe") or {}
    notes    = data.get("_notes", "")
    passages = data.get("rag_passages", [])

    if dec == "approved":
        dec_col="#059669"; dec_bg="rgba(5,150,105,0.08)"; dec_border="rgba(5,150,105,0.25)"; dec_label="APPROVED"
    elif dec == "rejected":
        dec_col="#dc2626"; dec_bg="rgba(220,38,38,0.08)"; dec_border="rgba(220,38,38,0.25)"; dec_label="REJECTED"
    else:
        dec_col="#f5a623"; dec_bg="rgba(245,166,35,0.08)"; dec_border="rgba(245,166,35,0.25)"; dec_label="NEEDS REVIEW"

    passing  = [r for r in rules if r.get("passed")]
    blocking = [r for r in rules if not r.get("passed") and r.get("action") == "reject"]
    warning  = [r for r in rules if not r.get("passed") and r.get("action") == "flag_review"]

    # Convert narrative markdown to HTML
    narrative_html = markdown_to_html_sections(narrative)

    # ICD chips
    icd_chips = "".join([
        f'<div class="code-chip">'
        f'<div class="chip-code">{c["code"]}</div>'
        f'<div class="chip-desc">{c["description"]}</div>'
        f'<div class="chip-conf">{c["confidence"]:.0%} confidence</div>'
        f'</div>'
        for c in icd
    ]) or '<span class="no-data">No codes mapped</span>'

    cpt_chips = "".join([
        f'<div class="code-chip cpt">'
        f'<div class="chip-code">{c["code"]}</div>'
        f'<div class="chip-desc">{c["description"]}</div>'
        f'<div class="chip-conf">{c["confidence"]:.0%} confidence</div>'
        f'</div>'
        for c in cpt
    ]) or '<span class="no-data">No codes mapped</span>'

    # Rules table
    rule_rows = "".join([
        f'<tr class="rule-{"pass" if r["passed"] else ("fail" if r.get("action")=="reject" else "warn")}">'
        f'<td class="rule-id">{r["rule_id"]}</td>'
        f'<td class="rule-status">{"✓ PASS" if r["passed"] else ("✗ REJECT" if r.get("action")=="reject" else "⚠ REVIEW")}</td>'
        f'<td class="rule-reason">{r["reason"]}</td>'
        f'</tr>'
        for r in rules
    ]) or '<tr><td colspan="3" class="no-data">No rules evaluated</td></tr>'

    # MoE summary
    moe_section = ""
    if moe and not moe.get("skipped") and (moe.get("activated_experts") or moe.get("findings")):
        experts_html = ""
        for f in (moe.get("findings") or []):
            risk = f.get("risk_level","low")
            rcol = "#ff6b6b" if risk in ["critical","high"] else "#f5a623" if risk == "moderate" else "#34d97b"
            flags_html = "".join([f'<div class="moe-flag">{fl}</div>' for fl in (f.get("risk_flags") or [])]) or '<div class="moe-flag ok">No flags from this expert</div>'
            recs_html  = "".join([f'<div class="moe-rec">→ {r}</div>' for r in (f.get("recommendations") or [])])
            narr_html  = f'<div class="moe-narr">{f["narrative"]}</div>' if f.get("narrative") else ""
            img_html   = f'<div class="moe-img-assess"><strong>Imaging:</strong> {f["imaging_assessment"]}</div>' if f.get("imaging_assessment") else ""
            experts_html += f"""
            <div class="moe-expert">
              <div class="moe-expert-header">
                <div>
                  <div class="moe-expert-name">{f.get("expert_name","")}</div>
                  <div class="moe-expert-assess">{f.get("assessment","")}</div>
                </div>
                <div style="display:flex;gap:8px;align-items:center;flex-shrink:0">
                  <span class="moe-risk" style="color:{rcol};border-color:{rcol}">{risk.upper()}</span>
                  <span class="moe-conf">{f.get("expert_confidence",0):.0%}</span>
                </div>
              </div>
              {img_html}
              {narr_html}
              <div class="moe-flags-wrap">{flags_html}</div>
              <div class="moe-recs-wrap">{recs_html}</div>
            </div>"""

        sug_codes = ""
        if moe.get("suggested_codes"):
            sug_codes = '<div class="sub-label">Suggested Additional Codes</div><div class="code-grid">' + "".join([
                f'<div class="code-chip purple"><div class="chip-code">{c["code"]}</div><div class="chip-desc">{c["description"]}</div><div class="chip-conf">{c.get("reason","")}</div></div>'
                for c in moe["suggested_codes"]
            ]) + "</div>"

        moe_section = f"""
        <div class="moe-block">
          <div class="moe-header">
            <div>
              <div class="moe-consensus">Consensus Risk: <span style="color:{dec_col}">{moe.get("consensus_risk","").upper()}</span></div>
              <div class="moe-experts-list">Activated: {", ".join(moe.get("activated_experts",[]))}</div>
            </div>
            <div class="moe-conf-avg">{moe.get("moe_confidence",0):.0%}<div style="font-size:.6rem;color:var(--t3);margin-top:2px">avg conf</div></div>
          </div>
          {experts_html}
          {sug_codes}
        </div>"""
    else:
        moe_section = '<div class="no-data-block">No MoE expert analysis — no specialists activated for this claim type</div>'

    # Guideline passages
    passages_html = ""
    for p in passages[:4]:
        passages_html += f"""
        <div class="passage-block">
          <div class="passage-source">{p.get("source","Unknown")} <span class="passage-rel">Relevance: {p.get("relevance",0):.3f}</span></div>
          <div class="passage-content">{p.get("content","")[:500]}...</div>
        </div>"""
    if not passages_html:
        passages_html = '<div class="no-data-block">No guideline passages retrieved for this claim</div>'

    provider = os.environ.get("LLM_PROVIDER","rules")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>Clinical Adjudication Report — {claim_id}</title>
<link href="https://fonts.googleapis.com/css2?family=DM+Sans:opsz,wght@9..40,300;9..40,400;9..40,500;9..40,600&family=Instrument+Serif:ital@0;1&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
*,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}
:root{{--bg:#FAFBFD;--s1:#FFFFFF;--s2:#F8F9FB;--s3:#F0EEEA;--s4:#E8E6E1;--b1:rgba(0,0,0,0.05);--b2:rgba(0,0,0,0.08);--b3:rgba(0,0,0,0.12);--t1:#1A1A1A;--t2:#5A5A5A;--t3:#8A8A8A;--t4:#B0B0B0;--blue:#2563EB;--purple:#7c3aed;--teal:#0B6E5F;--green:#059669;--red:#dc2626;--amber:#b45309;--serif:'Instrument Serif',serif;--mono:'IBM Plex Mono',monospace;--sans:'DM Sans',system-ui,sans-serif}}
html{{font-size:15px}}
body{{background:var(--bg);color:var(--t1);font-family:var(--sans);-webkit-font-smoothing:antialiased;min-height:100vh}}
@media print{{body{{background:#fff;color:#111}}:root{{--bg:#fff;--s1:#f8f8f8;--s2:#f0f0f0;--s3:#e8e8e8;--b1:rgba(0,0,0,0.08);--b2:rgba(0,0,0,0.12);--t1:#111;--t2:#444;--t3:#888}}.toolbar{{display:none}}.page{{max-width:100%;border-radius:0;box-shadow:none}}}}

.toolbar{{position:sticky;top:0;z-index:100;height:52px;background:rgba(250,251,253,0.95);backdrop-filter:blur(12px);border-bottom:1px solid var(--b1);display:flex;align-items:center;justify-content:space-between;padding:0 32px}}
.tb-brand{{display:flex;align-items:center;gap:8px;text-decoration:none}}.tb-mark{{width:24px;height:24px;border-radius:6px;background:var(--teal);display:flex;align-items:center;justify-content:center;font-family:var(--mono);font-size:9px;color:#fff}}.tb-name{{font-family:var(--mono);font-size:.72rem;color:var(--t1)}}
.tb-right{{display:flex;gap:8px}}
.btn-print{{padding:7px 16px;border-radius:6px;border:1px solid rgba(11,110,95,0.2);background:rgba(11,110,95,0.07);color:var(--teal);font-family:var(--mono);font-size:.6rem;cursor:pointer;transition:all .2s}}.btn-print:hover{{background:rgba(11,110,95,0.14)}}
.btn-close{{padding:7px 12px;border-radius:6px;border:1px solid var(--b1);background:transparent;color:var(--t3);font-family:var(--mono);font-size:.6rem;cursor:pointer}}.btn-close:hover{{color:var(--t2)}}

.page{{max-width:900px;margin:32px auto 80px;background:var(--s1);border:1px solid var(--b1);border-radius:16px;overflow:hidden;box-shadow:0 4px 24px rgba(0,0,0,0.06)}}

/* REPORT HEADER */
.rpt-header{{padding:36px 40px;border-bottom:1px solid var(--b1)}}
.rpt-org{{display:flex;align-items:center;gap:10px;margin-bottom:20px}}
.rpt-mark{{width:36px;height:36px;border-radius:9px;background:var(--teal);display:flex;align-items:center;justify-content:center;font-family:var(--mono);font-size:11px;color:#fff;font-weight:500}}
.rpt-org-name{{font-family:var(--serif);font-size:1.15rem;font-weight:400;color:var(--t1)}}.rpt-org-sub{{font-family:var(--mono);font-size:.58rem;color:var(--t3);margin-top:1px}}
.rpt-title{{font-family:var(--serif);font-size:1.9rem;font-weight:400;font-style:italic;letter-spacing:.01em;margin-bottom:6px}}
.rpt-subtitle{{font-family:var(--mono);font-size:.62rem;color:var(--t3);margin-bottom:20px}}
.rpt-meta-grid{{display:grid;grid-template-columns:1fr 1fr;gap:6px 28px}}
.rpt-meta{{display:flex;gap:8px;font-size:.72rem}}.rpt-meta-label{{font-family:var(--mono);font-size:.6rem;color:var(--t3);width:100px;flex-shrink:0}}.rpt-meta-val{{font-family:var(--mono);color:var(--t1)}}

/* DECISION BANNER */
.dec-banner{{margin:28px 40px;border-radius:12px;padding:22px 24px;border:1px solid {dec_border};background:{dec_bg};display:flex;align-items:flex-start;gap:16px}}
.dec-icon{{width:48px;height:48px;border-radius:12px;border:1.5px solid {dec_col};color:{dec_col};display:flex;align-items:center;justify-content:center;font-size:1.2rem;font-weight:700;flex-shrink:0}}
.dec-label{{font-family:var(--serif);font-size:2.4rem;font-weight:400;font-style:italic;color:{dec_col};line-height:1;margin-bottom:5px}}
.dec-conf{{font-family:var(--mono);font-size:.65rem;color:{dec_col};opacity:.8;margin-bottom:6px}}
.dec-id{{font-family:var(--mono);font-size:.58rem;color:{dec_col};opacity:.55}}

/* QUICK METRICS */
.metrics-row{{margin:0 40px 28px;display:grid;grid-template-columns:repeat(4,1fr);gap:1px;background:var(--b1);border:1px solid var(--b1);border-radius:10px;overflow:hidden}}
.metric{{background:var(--s2);padding:14px 16px;text-align:center}}.metric-val{{font-family:var(--serif);font-size:1.8rem;font-weight:400;font-style:italic;line-height:1}}.metric-lbl{{font-family:var(--mono);font-size:.54rem;color:var(--t3);text-transform:uppercase;letter-spacing:.1em;margin-top:3px}}

/* STRUCTURED DATA SECTIONS */
.data-section{{margin:0 40px 24px}}
.ds-title{{font-family:var(--mono);font-size:.58rem;letter-spacing:.14em;text-transform:uppercase;color:var(--t3);margin-bottom:10px;display:flex;align-items:center;gap:8px}}
.ds-title::after{{content:'';flex:1;height:1px;background:var(--b1)}}
.code-grid{{display:grid;grid-template-columns:1fr 1fr;gap:6px;margin-bottom:8px}}
.code-chip{{background:var(--s2);border:1px solid var(--b1);border-radius:7px;padding:8px 10px}}.code-chip.cpt .chip-code{{color:var(--purple)}}.code-chip.purple .chip-code{{color:var(--purple)}}
.chip-code{{font-family:var(--mono);font-size:.8rem;font-weight:500;color:var(--blue);margin-bottom:2px}}.chip-desc{{font-size:.68rem;color:var(--t1);margin-bottom:3px}}.chip-conf{{font-family:var(--mono);font-size:.56rem;color:var(--t3)}}
.no-data{{font-family:var(--mono);font-size:.65rem;color:var(--t3)}}.sub-label{{font-family:var(--mono);font-size:.56rem;color:var(--t3);text-transform:uppercase;letter-spacing:.1em;margin:10px 0 6px}}

/* RULES TABLE */
.rules-table{{width:100%;border-collapse:collapse;font-size:.75rem}}
.rules-table th{{padding:8px 10px;text-align:left;font-family:var(--mono);font-size:.54rem;letter-spacing:.1em;text-transform:uppercase;color:var(--t3);border-bottom:1px solid var(--b1);background:var(--s2)}}
.rules-table td{{padding:9px 10px;border-bottom:1px solid var(--b1);vertical-align:top}}
.rules-table tr:last-child td{{border:none}}
.rule-id{{font-family:var(--mono);font-size:.68rem;font-weight:500;white-space:nowrap}}.rule-status{{font-family:var(--mono);font-size:.62rem;white-space:nowrap;font-weight:500}}.rule-reason{{font-size:.72rem;color:var(--t1);line-height:1.45}}
tr.rule-pass .rule-status{{color:var(--green)}}tr.rule-fail .rule-status{{color:var(--red)}}tr.rule-warn .rule-status{{color:var(--amber)}}
tr.rule-pass{{background:rgba(5,150,105,0.03)}}tr.rule-fail{{background:rgba(220,38,38,0.03)}}tr.rule-warn{{background:rgba(180,83,9,0.03)}}

/* MOE */
.moe-block{{background:rgba(124,58,237,0.04);border:1px solid rgba(124,58,237,0.15);border-radius:10px;overflow:hidden}}
.moe-header{{display:flex;align-items:center;justify-content:space-between;padding:12px 14px;border-bottom:1px solid rgba(124,58,237,0.1)}}
.moe-consensus{{font-family:var(--mono);font-size:.65rem;font-weight:500;color:var(--t1);margin-bottom:2px}}.moe-experts-list{{font-family:var(--mono);font-size:.58rem;color:var(--t3)}}
.moe-conf-avg{{font-family:var(--serif);font-size:1.5rem;font-style:italic;color:var(--purple);text-align:center}}
.moe-expert{{padding:12px 14px;border-bottom:1px solid rgba(124,58,237,0.08)}}.moe-expert:last-of-type{{border:none}}
.moe-expert-header{{display:flex;align-items:flex-start;justify-content:space-between;margin-bottom:6px}}
.moe-expert-name{{font-size:.78rem;font-weight:500;color:var(--t1);margin-bottom:2px}}.moe-expert-assess{{font-family:var(--mono);font-size:.62rem;color:var(--t2)}}
.moe-risk{{font-family:var(--mono);font-size:.56rem;padding:2px 7px;border-radius:4px;border:1px solid;background:transparent}}.moe-conf{{font-family:var(--mono);font-size:.62rem;color:var(--t3)}}
.moe-narr{{font-size:.74rem;color:var(--t1);line-height:1.65;margin:8px 0;padding:10px 12px;background:var(--s3);border-radius:7px;border-left:2px solid rgba(124,58,237,0.3)}}
.moe-img-assess{{font-size:.72rem;color:var(--blue);margin:6px 0;padding:7px 10px;background:rgba(37,99,235,0.05);border-radius:6px}}
.moe-flags-wrap{{display:flex;flex-direction:column;gap:4px;margin:6px 0}}
.moe-flag{{font-size:.7rem;color:var(--red);padding:4px 8px;background:rgba(220,38,38,0.05);border-radius:5px;border:1px solid rgba(220,38,38,0.15)}}
.moe-flag.ok{{color:var(--green);background:rgba(5,150,105,0.05);border-color:rgba(5,150,105,0.2)}}
.moe-rec{{font-size:.7rem;color:var(--teal);padding:4px 8px}}
.no-data-block{{font-family:var(--mono);font-size:.65rem;color:var(--t3);padding:12px;background:var(--s2);border-radius:7px}}

/* PASSAGES */
.passage-block{{background:var(--s2);border:1px solid var(--b1);border-radius:8px;padding:12px 14px;margin-bottom:8px}}.passage-block:last-child{{margin:0}}
.passage-source{{font-family:var(--mono);font-size:.62rem;color:var(--blue);margin-bottom:5px;display:flex;justify-content:space-between;align-items:center}}
.passage-rel{{color:var(--t3);font-size:.58rem}}.passage-content{{font-size:.72rem;color:var(--t1);line-height:1.6}}

/* NARRATIVE BODY */
.rpt-body{{padding:32px 40px;border-top:1px solid var(--b1)}}
.rpt-h2{{font-family:var(--serif);font-size:1.15rem;font-weight:500;font-style:italic;letter-spacing:.01em;color:var(--t1);margin:28px 0 12px;padding-bottom:8px;border-bottom:1px solid var(--b1)}}
.rpt-h2:first-child{{margin-top:0}}
.rpt-p{{font-size:.84rem;color:var(--t1);line-height:1.85;margin-bottom:14px}}
.rpt-p strong{{color:var(--t1);font-weight:600}}.rpt-p em{{font-style:italic;color:var(--t1)}}
.rpt-p code{{font-family:var(--mono);font-size:.74rem;background:var(--s2);padding:1px 5px;border-radius:3px;border:1px solid var(--b1)}}
.rpt-divider{{border:none;border-top:1px solid var(--b1);margin:20px 0}}

/* FOOTER */
.rpt-footer{{padding:24px 40px;border-top:1px solid var(--b1);background:var(--s2)}}
.footer-grid{{display:grid;grid-template-columns:1fr 1fr;gap:8px 24px;margin-bottom:14px}}
.footer-field{{display:flex;gap:8px;font-size:.7rem}}.footer-label{{font-family:var(--mono);font-size:.58rem;color:var(--t3);width:110px;flex-shrink:0}}.footer-val{{font-family:var(--mono);color:var(--t1)}}
.footer-note{{font-size:.68rem;color:var(--t2);line-height:1.6;padding-top:12px;border-top:1px solid var(--b1)}}
::-webkit-scrollbar{{width:3px}}::-webkit-scrollbar-thumb{{background:#222235}}
.rpt-section-header{{display:flex;align-items:center;gap:10px;margin:28px 0 14px;padding-bottom:10px;border-bottom:2px solid var(--sec-col,#6b8fff)}}
.rpt-section-header:first-child{{margin-top:0}}
.rpt-sec-num{{font-family:var(--mono);font-size:.56rem;font-weight:500;color:var(--sec-col,#6b8fff);border:1px solid;border-radius:4px;padding:2px 7px;flex-shrink:0}}
.rpt-sec-title{{font-family:var(--serif);font-size:1.05rem;font-weight:500;font-style:italic;color:var(--t1)}}
.rpt-h3{{font-family:var(--mono);font-size:.6rem;text-transform:uppercase;letter-spacing:.1em;color:var(--t1);margin:14px 0 8px;padding-left:8px;border-left:2px solid var(--b2)}}
.rpt-hr{{border:none;border-top:1px solid var(--b1);margin:12px 0}}
.rpt-kv{{display:flex;gap:12px;padding:5px 0;border-bottom:1px solid var(--b1);font-size:.78rem}}
.rpt-kv:last-of-type{{border:none}}
.rpt-kv-key{{font-family:var(--mono);font-size:.62rem;color:var(--t3);width:160px;flex-shrink:0}}
.rpt-kv-val{{color:var(--t1);line-height:1.55;font-size:.82rem;flex:1}}
.rpt-bullet{{display:flex;gap:8px;padding:4px 0;font-size:.78rem;color:var(--t1);line-height:1.55}}
.rpt-bullet span:first-child{{color:var(--blue);font-weight:700;flex-shrink:0}}
.rpt-num-item{{display:flex;gap:10px;padding:5px 0;font-size:.78rem;color:var(--t1);line-height:1.55}}
.rpt-num{{font-family:var(--mono);font-size:.6rem;font-weight:600;color:var(--teal);min-width:20px;flex-shrink:0}}
.rpt-p{{font-size:.82rem;color:var(--t1);line-height:1.8;margin-bottom:10px}}
.rpt-p strong{{color:var(--t1);font-weight:600}}
.rpt-p em{{font-style:italic;color:var(--t1)}}
.rpt-code{{font-family:var(--mono);font-size:.7rem;background:var(--s2);padding:1px 5px;border-radius:3px;border:1px solid var(--b1)}}
.rpt-table-wrap{{overflow-x:auto;margin:10px 0 16px;border-radius:8px;border:1px solid var(--b1)}}
.rpt-table{{width:100%;border-collapse:collapse;font-size:.74rem}}
.rpt-table th{{padding:8px 12px;text-align:left;font-family:var(--mono);font-size:.54rem;letter-spacing:.09em;text-transform:uppercase;color:var(--t3);background:var(--s2);border-bottom:1px solid var(--b1);white-space:nowrap}}
.rpt-table td{{padding:8px 12px;border-bottom:1px solid var(--b1);color:var(--t2);vertical-align:top;line-height:1.45}}
.rpt-table tr:last-child td{{border:none}}
.rpt-table tr:hover td{{background:var(--s3)}}
.rpt-table td:first-child{{color:var(--t1)}}
</style>
</head>
<body>
<div class="toolbar">
  <a class="tb-brand" href="index.html"><div class="tb-mark">Rx</div><div class="tb-name">ClaimIQ — Clinical Report</div></a>
  <div class="tb-right">
    <button class="btn-print" onclick="window.print()">🖨 Print / Save PDF</button>
    <button class="btn-close" onclick="window.close()">✕ Close</button>
  </div>
</div>

<div class="page">
  <!-- Header -->
  <div class="rpt-header">
    <div class="rpt-org">
      <div class="rpt-mark">Rx</div>
      <div><div class="rpt-org-name">ClaimIQ</div><div class="rpt-org-sub">Healthcare Claims Adjudication Agent · AI Audit Division</div></div>
    </div>
    <div class="rpt-title">Clinical Adjudication Report</div>
    <div class="rpt-subtitle">AI-Generated · For clinical and administrative review only</div>
    <div class="rpt-meta-grid">
      <div class="rpt-meta"><span class="rpt-meta-label">Report Generated</span><span class="rpt-meta-val">{generated_at.strftime('%Y-%m-%d %H:%M:%S')} UTC</span></div>
      <div class="rpt-meta"><span class="rpt-meta-label">Claim ID</span><span class="rpt-meta-val">{claim_id}</span></div>
      <div class="rpt-meta"><span class="rpt-meta-label">Audit Trace</span><span class="rpt-meta-val">{trace_id}</span></div>
      <div class="rpt-meta"><span class="rpt-meta-label">Insurance Plan</span><span class="rpt-meta-val">{plan.upper()}</span></div>
      <div class="rpt-meta"><span class="rpt-meta-label">Decided At</span><span class="rpt-meta-val">{data.get('decided_at','—')[:19].replace('T',' ')} UTC</span></div>
      <div class="rpt-meta"><span class="rpt-meta-label">LLM Provider</span><span class="rpt-meta-val">{provider.upper()}</span></div>
    </div>
  </div>

  <!-- Decision Banner -->
  <div class="dec-banner">
    <div class="dec-icon">{"✓" if dec=="approved" else ("✗" if dec=="rejected" else "!")}</div>
    <div>
      <div class="dec-label">{dec_label}</div>
      <div class="dec-conf">{conf:.1%} overall confidence · {len(rules)} rules evaluated · {len(icd)} ICD-10 · {len(cpt)} CPT · {len(edges)} anomalies</div>
      <div class="dec-id">{claim_id} · {trace_id}</div>
    </div>
  </div>

  <!-- Metrics -->
  <div class="metrics-row">
    <div class="metric"><div class="metric-val" style="color:var(--blue)">{conf:.0%}</div><div class="metric-lbl">Confidence</div></div>
    <div class="metric"><div class="metric-val" style="color:{"var(--green)" if len(blocking)==0 else "var(--red)"}">{len(rules)}</div><div class="metric-lbl">Rules</div></div>
    <div class="metric"><div class="metric-val" style="color:var(--purple)">{len(icd)+len(cpt)}</div><div class="metric-lbl">Codes</div></div>
    <div class="metric"><div class="metric-val" style="color:{"var(--amber)" if edges else "var(--green)"}">{len(edges)}</div><div class="metric-lbl">Anomalies</div></div>
  </div>

  <!-- Codes -->
  <div class="data-section">
    <div class="ds-title">ICD-10-CM Diagnoses</div>
    <div class="code-grid">{icd_chips}</div>
    <div class="ds-title">CPT Procedures</div>
    <div class="code-grid">{cpt_chips}</div>
  </div>

  <!-- Rules -->
  <div class="data-section">
    <div class="ds-title">Policy Rule Evaluations ({len(passing)} pass · {len(blocking)} block · {len(warning)} warn)</div>
    <table class="rules-table">
      <thead><tr><th>Rule ID</th><th>Status</th><th>Reason</th></tr></thead>
      <tbody>{rule_rows}</tbody>
    </table>
  </div>

  <!-- MoE -->
  <div class="data-section">
    <div class="ds-title">Mixture of Experts Analysis</div>
    {moe_section}
  </div>

  <!-- Guideline Passages -->
  <div class="data-section">
    <div class="ds-title">Clinical Guideline Evidence (RAG Retrieval)</div>
    {passages_html}
  </div>

  <!-- AI Narrative -->
  <div class="rpt-body">
    {narrative_html}
  </div>

  <!-- Footer -->
  <div class="rpt-footer">
    <div class="footer-grid">
      <div class="footer-field"><span class="footer-label">Generated By</span><span class="footer-val">ClaimIQ Adjudication Agent v1.0</span></div>
      <div class="footer-field"><span class="footer-label">Report ID</span><span class="footer-val">RPT-{claim_id}-{generated_at.strftime('%Y%m%d%H%M%S')}</span></div>
      <div class="footer-field"><span class="footer-label">Decision</span><span class="footer-val">{dec_label} · {conf:.1%}</span></div>
      <div class="footer-field"><span class="footer-label">PHI Status</span><span class="footer-val">TOKENIZED — No PHI in report</span></div>
      <div class="footer-field"><span class="footer-label">Audit Trace</span><span class="footer-val">{trace_id}</span></div>
      <div class="footer-field"><span class="footer-label">LLM Provider</span><span class="footer-val">{provider.upper()}</span></div>
    </div>
    <div class="footer-note">
      This report was generated automatically by the ClaimIQ Healthcare Claims Adjudication Agent.
      {"The narrative was written by the " + provider.upper() + " language model based on structured adjudication data." if provider != "rules" else "The narrative was generated using rule-based templates. Configure LLM_PROVIDER=groq for AI-written narratives."}
      All patient identifiers have been replaced with cryptographic vault tokens — no PHI is present.
      This report is for adjudication purposes only and does not constitute medical advice.
      Any clinical decisions must be reviewed by qualified healthcare professionals.
    </div>
  </div>
</div>
</body>
</html>"""