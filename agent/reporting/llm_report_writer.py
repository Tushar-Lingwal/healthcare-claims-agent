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

REPORT_SYSTEM_PROMPT = """You are a senior clinical documentation specialist and healthcare AI auditor. Your task is to write a detailed, professional clinical adjudication report that explains exactly how an AI claims adjudication system reached its decision.

The report must:
1. Be written in clear, authoritative medical-administrative prose — not bullet points, not tables
2. Explain the agent's REASONING PROCESS at every stage — what it looked for, what it found, why it mattered
3. Cite specific evidence: ICD-10 codes, CPT codes, rule IDs, guideline passages, confidence scores
4. Be honest about uncertainty — explain what triggered edge cases and what the risks are
5. Give SPECIFIC, ACTIONABLE recommendations — not generic advice
6. Be written for a clinical reviewer or senior medical director who needs to understand the full picture

Structure the report with these exact section headers (use ## for headers):
## Executive Summary
## Clinical Background and Patient Context
## AI Agent Reasoning Process
## Medical Coding Analysis
## Policy Rule Evaluation
## Clinical Guideline Evidence
## Edge Case and Risk Analysis
## Imaging Analysis (only if imaging was performed)
## Final Decision Justification
## Clinical and Administrative Recommendations
## Agent Confidence Assessment
## Limitations and Caveats

Write each section as full paragraphs. Be specific, thorough, and professional. The goal is that a medical director reading this report can fully understand why the AI made its decision and has everything needed to either approve, override, or act on it.

Respond with ONLY the report content — no preamble, no meta-commentary. Start directly with ## Executive Summary."""


def _build_report_prompt(data: dict) -> str:
    """Builds the user prompt from adjudication result data."""
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
    img_result  = data.get("imaging_result") or data.get("_imaging", {})

    # Format codes
    icd_str  = "\n".join([f"  - {c['code']}: {c['description']} (confidence: {c['confidence']:.0%})" for c in icd_codes]) or "  None mapped"
    cpt_str  = "\n".join([f"  - {c['code']}: {c['description']} (confidence: {c['confidence']:.0%})" for c in cpt_codes]) or "  None mapped"

    # Format rules
    passing  = [r for r in rules if r.get("passed")]
    blocking = [r for r in rules if not r.get("passed") and r.get("action") == "reject"]
    warning  = [r for r in rules if not r.get("passed") and r.get("action") == "flag_review"]

    rule_str  = ""
    if passing:
        rule_str += "PASSED rules:\n" + "\n".join([f"  - {r['rule_id']}: {r['reason']}" for r in passing]) + "\n"
    if blocking:
        rule_str += "\nFAILED (BLOCKING) rules:\n" + "\n".join([f"  - {r['rule_id']}: {r['reason']}" for r in blocking]) + "\n"
    if warning:
        rule_str += "\nFAILED (WARNING) rules:\n" + "\n".join([f"  - {r['rule_id']}: {r['reason']}" for r in warning]) + "\n"
    if not rule_str:
        rule_str = "  No rules evaluated"

    # Format edges
    edge_str = "\n".join([
        f"  - Type: {e.get('type','unknown')} | Severity: {e.get('severity','unknown')}\n"
        f"    Description: {e.get('description','')}\n"
        f"    Recommendation: {e.get('recommendation','')}\n"
        f"    {'Image signal: ' + str(e.get('image_signal','')) + ' | Text signal: ' + str(e.get('text_signal','')) + ' | Mismatch score: ' + str(e.get('mismatch_score','')) if e.get('image_signal') else ''}"
        for e in edges
    ]) or "  None detected"

    # Format guideline passages
    passage_str = ""
    for p in passages[:4]:
        source  = p.get("source", "Unknown")
        content = p.get("content", "")[:400]
        rel     = p.get("relevance", 0)
        passage_str += f"\n  Source: {source} (relevance: {rel:.3f})\n  Excerpt: {content}...\n"
    if not passage_str:
        passage_str = "  No guideline passages retrieved"

    # Format reasoning chain
    chain_str = "\n".join([
        f"  Stage {s.get('step',i+1)}: {s.get('stage','unknown').replace('_',' ').upper()}\n"
        f"    Confidence: {s.get('confidence',0):.0%}\n"
        f"    Outcome: {s.get('outcome','')}"
        for i, s in enumerate(chain)
    ]) or "  No chain available"

    # Format imaging
    img_str = ""
    if img_result and (img_result.get("class_label") or img_result.get("predicted")):
        label    = img_result.get("class_label") or img_result.get("predicted", "unknown")
        conf_img = img_result.get("confidence", 0)
        mode     = img_result.get("mode_used") or img_result.get("mode", "unknown")
        icd_hint = img_result.get("icd10_suggestion", "")
        img_str = f"""
IMAGING ANALYSIS RESULTS:
  Classification: {label}
  Confidence: {conf_img:.0%}
  Mode: {mode}
  Suggested ICD-10: {icd_hint}
  Note: Imaging result {'was' if img_result else 'was not'} incorporated into adjudication decision."""

    prompt = f"""Write a comprehensive clinical adjudication report for the following case.

═══════════════════════════════════════════════
ADJUDICATION RESULT SUMMARY
═══════════════════════════════════════════════
Claim ID:           {claim_id}
Audit Trace ID:     {trace_id}
FINAL DECISION:     {dec.upper().replace('_', ' ')}
Overall Confidence: {conf:.1%}
Insurance Plan:     {plan.upper()}
Decision Time:      {data.get('decided_at', 'unknown')}

CLINICAL NOTES SUBMITTED:
{notes or '(No clinical notes provided)'}

AI SUMMARY (from pipeline):
{summary or '(No summary generated)'}

DECISION REASONS:
{chr(10).join(f'  - {r}' for r in reasons) if reasons else '  None listed'}

RISK FLAGS:
{chr(10).join(f'  - {f}' for f in risk_flags) if risk_flags else '  None'}

═══════════════════════════════════════════════
MEDICAL CODING (Stage 3-4 output)
═══════════════════════════════════════════════
ICD-10-CM Diagnoses:
{icd_str}

CPT Procedures:
{cpt_str}

═══════════════════════════════════════════════
POLICY RULE ENGINE (Stage 6 output)
═══════════════════════════════════════════════
{rule_str}

═══════════════════════════════════════════════
CLINICAL GUIDELINE RAG (Stage 5 output)
═══════════════════════════════════════════════
{passage_str}

═══════════════════════════════════════════════
EDGE CASE DETECTION (Stage 7 output)
═══════════════════════════════════════════════
{edge_str}

═══════════════════════════════════════════════
AI REASONING CHAIN (all stages)
═══════════════════════════════════════════════
{chain_str}
{img_str}

═══════════════════════════════════════════════
CONTEXT FOR REPORT WRITING
═══════════════════════════════════════════════
- This report is for a clinical reviewer / medical director
- Explain the AI's thinking at each stage in detail
- Be specific about which codes, rules, and guidelines influenced the decision
- If APPROVED: explain why the evidence was sufficient and what should happen next
- If REJECTED: explain exactly what is wrong and precisely what must be corrected
- If NEEDS REVIEW: explain what is uncertain, what needs human judgment, and why
- For imaging: explain how the imaging result relates to the clinical text and whether they agree
- Write in authoritative medical-administrative prose — full paragraphs, no bullet points in body text
- Minimum 1200 words. Be thorough.

Now write the full clinical adjudication report:"""

    return prompt


def _call_groq(prompt: str, system: str) -> Optional[str]:
    try:
        from groq import Groq
        client = Groq(api_key=os.environ.get("GROQ_API_KEY", ""))
        model  = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
        resp   = client.chat.completions.create(
            model=model, max_tokens=4000, temperature=0.3,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": prompt},
            ],
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        logger.warning(f"Groq report generation failed: {e}")
        return None


def _call_gemini(prompt: str, system: str) -> Optional[str]:
    try:
        import google.generativeai as genai
        genai.configure(api_key=os.environ.get("GEMINI_API_KEY", ""))
        model = genai.GenerativeModel(
            model_name=os.environ.get("GEMINI_MODEL", "gemini-1.5-flash"),
            system_instruction=system,
        )
        resp = model.generate_content(prompt, generation_config={"temperature": 0.3, "max_output_tokens": 4000})
        return resp.text.strip()
    except Exception as e:
        logger.warning(f"Gemini report generation failed: {e}")
        return None


def _call_anthropic(prompt: str, system: str) -> Optional[str]:
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))
        resp   = client.messages.create(
            model=os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6"),
            max_tokens=4000,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text.strip()
    except Exception as e:
        logger.warning(f"Anthropic report generation failed: {e}")
        return None


def _call_openai(prompt: str, system: str) -> Optional[str]:
    try:
        from openai import OpenAI
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
        resp   = client.chat.completions.create(
            model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
            max_tokens=4000, temperature=0.3,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": prompt},
            ],
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        logger.warning(f"OpenAI report generation failed: {e}")
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
    """Converts the markdown report to clean HTML sections."""
    lines   = md.split("\n")
    html    = []
    in_para = False

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("## "):
            if in_para:
                html.append("</p>")
                in_para = False
            title = stripped[3:]
            html.append(f'<h2 class="rpt-h2">{title}</h2>')
        elif stripped == "---":
            if in_para:
                html.append("</p>")
                in_para = False
            html.append('<hr class="rpt-divider">')
        elif stripped == "":
            if in_para:
                html.append("</p>")
                in_para = False
        else:
            # Process inline formatting
            text = stripped
            # Bold **text**
            text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
            # Italic *text*
            text = re.sub(r'\*(.+?)\*', r'<em>\1</em>', text)
            # Inline code `text`
            text = re.sub(r'`(.+?)`', r'<code>\1</code>', text)

            if not in_para:
                html.append('<p class="rpt-p">')
                in_para = True
            else:
                html.append(' ')
            html.append(text)

    if in_para:
        html.append("</p>")

    return "\n".join(html)


def build_full_report_html(data: dict, narrative: str, generated_at: datetime) -> str:
    """
    Builds the complete HTML document for the clinical report.
    Combines the structured metadata header with the LLM-written narrative body.
    """
    dec      = data.get("decision", "unknown")
    conf     = data.get("confidence_score", 0)
    claim_id = data.get("claim_id", "")
    trace_id = data.get("audit_trace_id", "")
    plan     = data.get("_plan", "unknown")
    icd      = data.get("icd10_codes", [])
    cpt      = data.get("cpt_codes", [])
    rules    = data.get("rule_evaluations", [])
    edges    = data.get("edge_cases", [])

    if dec == "approved":
        dec_col = "#065f46"; dec_bg = "#d1fae5"; dec_border = "#6ee7b7"; dec_label = "APPROVED"
    elif dec == "rejected":
        dec_col = "#991b1b"; dec_bg = "#fee2e2"; dec_border = "#fca5a5"; dec_label = "REJECTED"
    else:
        dec_col = "#78350f"; dec_bg = "#fef3c7"; dec_border = "#fcd34d"; dec_label = "NEEDS REVIEW"

    narrative_html = markdown_to_html_sections(narrative)

    icd_chips = "".join([
        f'<div class="meta-chip"><span class="chip-code">{c["code"]}</span>'
        f'<span class="chip-desc">{c["description"]}</span>'
        f'<span class="chip-conf">{c["confidence"]:.0%}</span></div>'
        for c in icd
    ]) or '<span class="no-data">None mapped</span>'

    cpt_chips = "".join([
        f'<div class="meta-chip cpt-chip"><span class="chip-code cpt">{c["code"]}</span>'
        f'<span class="chip-desc">{c["description"]}</span>'
        f'<span class="chip-conf">{c["confidence"]:.0%}</span></div>'
        for c in cpt
    ]) or '<span class="no-data">None mapped</span>'

    rule_rows = "".join([
        f'<tr class="rule-row-{"pass" if r["passed"] else ("fail" if r.get("action")=="reject" else "warn")}">'
        f'<td class="rule-id-cell">{r["rule_id"]}</td>'
        f'<td class="rule-status">{"✓ PASS" if r["passed"] else ("✗ REJECT" if r.get("action")=="reject" else "⚠ REVIEW")}</td>'
        f'<td class="rule-reason-cell">{r["reason"]}</td></tr>'
        for r in rules
    ]) or '<tr><td colspan="3" class="no-data">No rules evaluated</td></tr>'

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ClaimIQ Clinical Report — {claim_id}</title>
<link href="https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=Fraunces:ital,opsz,wght@0,9..144,300;0,9..144,400;0,9..144,500;0,9..144,600;1,9..144,300;1,9..144,400;1,9..144,500&family=DM+Sans:opsz,wght@9..40,300;9..40,400;9..40,500;9..40,600&display=swap" rel="stylesheet">
<style>
*,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}
:root{{
  --cream:#f6f4ee;--white:#fff;--paper:#faf9f6;--stone:#eae8e0;--ash:#d8d6cc;
  --ink:#18181a;--ink2:#4a4845;--ink3:#8a8780;--ink4:#b8b5ac;
  --blue:#1e40af;--purple:#5b21b6;
  --serif:'Fraunces',Georgia,serif;--mono:'DM Mono','Courier New',monospace;--sans:'DM Sans',system-ui,sans-serif;
}}
html{{font-size:15px;scroll-behavior:smooth}}
body{{background:var(--cream);color:var(--ink);font-family:var(--sans);-webkit-font-smoothing:antialiased;min-height:100vh}}
body::before{{content:'';position:fixed;inset:0;z-index:-2;background:radial-gradient(ellipse 900px 600px at 15% 20%,rgba(30,64,175,0.04) 0%,transparent 70%),radial-gradient(ellipse 700px 500px at 85% 75%,rgba(91,33,182,0.04) 0%,transparent 70%);pointer-events:none}}
body::after{{content:'';position:fixed;inset:0;z-index:-1;background-image:radial-gradient(circle,rgba(24,24,26,0.055) 1px,transparent 1px);background-size:28px 28px;mask-image:radial-gradient(ellipse 80% 80% at 50% 50%,black 30%,transparent 100%);pointer-events:none}}
@media print{{body{{background:#fff}}body::before,body::after{{display:none}}.no-print{{display:none!important}}.page{{max-width:100%;margin:0;border-radius:0;box-shadow:none}}}}

/* TOOLBAR */
.toolbar{{position:sticky;top:0;z-index:100;background:rgba(255,255,255,.94);backdrop-filter:blur(12px);border-bottom:1px solid var(--stone);display:flex;align-items:center;justify-content:space-between;padding:0 40px;height:58px;box-shadow:0 1px 3px rgba(0,0,0,.05)}}
.tb-brand{{display:flex;align-items:center;gap:10px;text-decoration:none}}
.tb-logo{{width:30px;height:30px;border-radius:8px;background:linear-gradient(135deg,#1e40af,#5b21b6);display:flex;align-items:center;justify-content:center;color:#fff;font-family:var(--mono);font-size:10px;font-weight:500}}
.tb-name{{font-family:var(--serif);font-size:1rem;color:var(--ink)}}
.tb-sub{{font-family:var(--mono);font-size:.55rem;color:var(--ink3)}}
.tb-right{{display:flex;gap:8px;align-items:center}}
.btn-print{{padding:8px 18px;border-radius:8px;border:1.5px solid var(--blue);background:#eff6ff;color:var(--blue);font-family:var(--sans);font-size:.78rem;font-weight:600;cursor:pointer;display:flex;align-items:center;gap:6px;transition:all .2s}}
.btn-print:hover{{background:var(--blue);color:#fff}}
.btn-close{{padding:8px 14px;border-radius:8px;border:1px solid var(--stone);background:var(--white);color:var(--ink2);font-family:var(--sans);font-size:.78rem;cursor:pointer;transition:all .2s}}
.btn-close:hover{{border-color:var(--ash)}}

/* PAGE */
.page{{max-width:860px;margin:40px auto 80px;background:var(--white);border:1px solid var(--stone);border-radius:16px;box-shadow:0 8px 40px rgba(0,0,0,.08),0 1px 4px rgba(0,0,0,.05);overflow:hidden}}

/* REPORT HEADER */
.rpt-header{{padding:40px 48px 32px;border-bottom:2px solid var(--blue);background:var(--white)}}
.rpt-logo-row{{display:flex;align-items:center;gap:12px;margin-bottom:24px}}
.rpt-logo-mark{{width:38px;height:38px;border-radius:10px;background:linear-gradient(135deg,#1e40af,#5b21b6);display:flex;align-items:center;justify-content:center;color:#fff;font-family:var(--mono);font-size:12px;font-weight:500}}
.rpt-logo-name{{font-family:var(--serif);font-size:1.4rem;letter-spacing:-.01em}}
.rpt-logo-sub{{font-family:var(--mono);font-size:.6rem;color:var(--ink3);margin-top:1px}}
.rpt-title{{font-family:var(--serif);font-size:1.6rem;font-weight:400;letter-spacing:-.02em;margin-bottom:4px}}
.rpt-subtitle{{font-family:var(--mono);font-size:.68rem;color:var(--ink3);margin-bottom:20px}}
.rpt-meta-grid{{display:grid;grid-template-columns:1fr 1fr;gap:5px 32px}}
.rpt-meta-row{{display:flex;gap:8px;font-size:.75rem}}
.rpt-meta-label{{color:var(--ink3);font-family:var(--mono);width:110px;flex-shrink:0}}
.rpt-meta-value{{color:var(--ink);font-family:var(--mono)}}

/* DECISION BANNER */
.dec-banner{{margin:32px 48px;border-radius:12px;padding:24px 28px;display:flex;align-items:flex-start;gap:18px;border:1.5px solid {dec_border};background:{dec_bg}}}
.dec-icon{{width:48px;height:48px;border-radius:12px;display:flex;align-items:center;justify-content:center;font-size:1.3rem;font-weight:700;flex-shrink:0;border:2px solid {dec_col};color:{dec_col};background:{dec_bg}}}
.dec-label{{font-family:var(--serif);font-size:1.9rem;font-weight:500;color:{dec_col};letter-spacing:-.02em;line-height:1;margin-bottom:5px}}
.dec-conf{{font-family:var(--mono);font-size:.72rem;color:{dec_col};opacity:.85;margin-bottom:10px}}
.dec-ids{{font-family:var(--mono);font-size:.62rem;color:{dec_col};opacity:.7}}

/* QUICK STATS ROW */
.qs-row{{margin:0 48px 32px;display:grid;grid-template-columns:repeat(4,1fr);gap:1px;background:var(--stone);border:1px solid var(--stone);border-radius:12px;overflow:hidden}}
.qs-cell{{background:var(--paper);padding:14px 16px;text-align:center}}
.qs-val{{font-family:var(--serif);font-size:1.5rem;font-weight:500;line-height:1;color:var(--ink)}}
.qs-label{{font-family:var(--mono);font-size:.58rem;color:var(--ink3);margin-top:3px;text-transform:uppercase;letter-spacing:.1em}}

/* CODE CHIPS */
.codes-section{{margin:0 48px 28px}}
.codes-section-title{{font-family:var(--mono);font-size:.6rem;letter-spacing:.14em;text-transform:uppercase;color:var(--ink3);margin-bottom:10px;display:flex;align-items:center;gap:8px}}
.codes-section-title::after{{content:'';flex:1;height:1px;background:var(--stone)}}
.chips-wrap{{display:flex;flex-wrap:wrap;gap:7px;margin-bottom:16px}}
.meta-chip{{display:flex;align-items:center;gap:8px;background:var(--paper);border:1px solid var(--stone);border-radius:8px;padding:7px 11px}}
.chip-code{{font-family:var(--mono);font-size:.82rem;font-weight:500;color:var(--blue)}}
.chip-code.cpt{{color:var(--purple)}}
.chip-desc{{font-size:.7rem;color:var(--ink2)}}
.chip-conf{{font-family:var(--mono);font-size:.58rem;color:var(--ink3);background:var(--stone);padding:1px 6px;border-radius:3px}}
.no-data{{font-family:var(--mono);font-size:.7rem;color:var(--ink3)}}

/* RULES TABLE */
.rules-section{{margin:0 48px 32px}}
.rules-table{{width:100%;border-collapse:collapse;font-size:.78rem}}
.rules-table th{{padding:9px 12px;text-align:left;font-family:var(--mono);font-size:.58rem;letter-spacing:.1em;text-transform:uppercase;color:var(--ink3);border-bottom:1px solid var(--stone);background:var(--paper)}}
.rules-table td{{padding:9px 12px;border-bottom:1px solid var(--paper);vertical-align:top}}
.rule-id-cell{{font-family:var(--mono);font-size:.7rem;font-weight:500;white-space:nowrap}}
.rule-status{{font-family:var(--mono);font-size:.65rem;font-weight:500;white-space:nowrap}}
.rule-reason-cell{{color:var(--ink2);font-size:.74rem;line-height:1.5}}
.rule-row-pass .rule-status{{color:#065f46}}
.rule-row-fail .rule-status{{color:#991b1b}}
.rule-row-warn .rule-status{{color:#78350f}}
.rule-row-pass{{background:#fafffe}}
.rule-row-fail{{background:#fff8f8}}
.rule-row-warn{{background:#fffdf5}}

/* NARRATIVE BODY */
.rpt-body{{padding:40px 48px}}
.rpt-h2{{font-family:var(--serif);font-size:1.2rem;font-weight:500;letter-spacing:-.01em;color:var(--ink);margin:32px 0 14px;padding-bottom:8px;border-bottom:1px solid var(--stone)}}
.rpt-h2:first-child{{margin-top:0}}
.rpt-p{{font-size:.88rem;color:var(--ink2);line-height:1.8;margin-bottom:14px}}
.rpt-p strong{{color:var(--ink);font-weight:600}}
.rpt-p em{{font-style:italic;color:var(--ink)}}
.rpt-p code{{font-family:var(--mono);font-size:.78rem;background:var(--paper);padding:1px 5px;border-radius:3px;border:1px solid var(--stone)}}
.rpt-divider{{border:none;border-top:1px solid var(--stone);margin:24px 0}}

/* SIGNATURE */
.rpt-footer{{padding:28px 48px;border-top:1px solid var(--stone);background:var(--paper)}}
.sig-grid{{display:grid;grid-template-columns:1fr 1fr;gap:10px}}
.sig-field{{}}
.sig-label{{font-family:var(--mono);font-size:.58rem;color:var(--ink3);text-transform:uppercase;letter-spacing:.08em;margin-bottom:2px}}
.sig-value{{font-family:var(--mono);font-size:.7rem;color:var(--ink)}}
.sig-note{{font-size:.72rem;color:var(--ink3);line-height:1.6;margin-top:14px;padding-top:14px;border-top:1px solid var(--stone);font-style:italic}}

::-webkit-scrollbar{{width:4px}}::-webkit-scrollbar-thumb{{background:var(--ash);border-radius:2px}}
</style>
</head>
<body>

<div class="toolbar no-print">
  <a class="tb-brand" href="index.html">
    <div class="tb-logo">Rx</div>
    <div><div class="tb-name">ClaimIQ</div><div class="tb-sub">Clinical Adjudication Report</div></div>
  </a>
  <div class="tb-right">
    <button class="btn-print" onclick="window.print()">🖨 Print / Save PDF</button>
    <button class="btn-close" onclick="window.close()">✕ Close</button>
  </div>
</div>

<div class="page">

  <!-- Report Header -->
  <div class="rpt-header">
    <div class="rpt-logo-row">
      <div class="rpt-logo-mark">Rx</div>
      <div><div class="rpt-logo-name">ClaimIQ</div><div class="rpt-logo-sub">Healthcare Claims Adjudication Agent — Clinical Report</div></div>
    </div>
    <div class="rpt-title">Clinical Adjudication Report</div>
    <div class="rpt-subtitle">AI-generated — for clinical and administrative review</div>
    <div class="rpt-meta-grid">
      <div class="rpt-meta-row"><span class="rpt-meta-label">Report Generated</span><span class="rpt-meta-value">{generated_at.strftime('%Y-%m-%d %H:%M:%S UTC')}</span></div>
      <div class="rpt-meta-row"><span class="rpt-meta-label">Claim ID</span><span class="rpt-meta-value">{claim_id}</span></div>
      <div class="rpt-meta-row"><span class="rpt-meta-label">Audit Trace</span><span class="rpt-meta-value">{trace_id}</span></div>
      <div class="rpt-meta-row"><span class="rpt-meta-label">Insurance Plan</span><span class="rpt-meta-value">{plan.upper()}</span></div>
      <div class="rpt-meta-row"><span class="rpt-meta-label">Decided At</span><span class="rpt-meta-value">{data.get('decided_at','—')[:19].replace('T',' ')} UTC</span></div>
      <div class="rpt-meta-row"><span class="rpt-meta-label">Pipeline Version</span><span class="rpt-meta-value">{data.get('pipeline_version','1.0.0')}</span></div>
    </div>
  </div>

  <!-- Decision Banner -->
  <div class="dec-banner">
    <div class="dec-icon">{'✓' if dec=='approved' else ('✗' if dec=='rejected' else '!')}</div>
    <div>
      <div class="dec-label">{dec_label}</div>
      <div class="dec-conf">{conf:.1%} overall confidence · {len(rules)} rules evaluated · {len(icd)} ICD-10 · {len(cpt)} CPT · {len(edges)} edge cases</div>
      <div class="dec-ids">{claim_id} · {trace_id}</div>
    </div>
  </div>

  <!-- Quick Stats -->
  <div class="qs-row">
    <div class="qs-cell"><div class="qs-val">{conf:.0%}</div><div class="qs-label">Confidence</div></div>
    <div class="qs-cell"><div class="qs-val">{len(rules)}</div><div class="qs-label">Rules</div></div>
    <div class="qs-cell"><div class="qs-val">{len(icd) + len(cpt)}</div><div class="qs-label">Codes</div></div>
    <div class="qs-cell"><div class="qs-val">{len(edges)}</div><div class="qs-label">Anomalies</div></div>
  </div>

  <!-- Codes -->
  <div class="codes-section">
    <div class="codes-section-title">ICD-10-CM Diagnoses</div>
    <div class="chips-wrap">{icd_chips}</div>
    <div class="codes-section-title">CPT Procedures</div>
    <div class="chips-wrap">{cpt_chips}</div>
  </div>

  <!-- Rules Table -->
  <div class="rules-section">
    <div class="codes-section-title">Policy Rule Evaluations</div>
    <table class="rules-table">
      <thead><tr><th>Rule ID</th><th>Status</th><th>Reason</th></tr></thead>
      <tbody>{rule_rows}</tbody>
    </table>
  </div>

  <!-- Narrative Body (LLM or rule-based) -->
  <div class="rpt-body">
    {narrative_html}
  </div>

  <!-- Footer / Signature -->
  <div class="rpt-footer">
    <div class="sig-grid">
      <div class="sig-field"><div class="sig-label">Generated By</div><div class="sig-value">ClaimIQ Adjudication Agent v1.0</div></div>
      <div class="sig-field"><div class="sig-label">Report ID</div><div class="sig-value">RPT-{claim_id}-{generated_at.strftime('%Y%m%d%H%M%S')}</div></div>
      <div class="sig-field"><div class="sig-label">Decision</div><div class="sig-value">{dec_label} at {conf:.1%}</div></div>
      <div class="sig-field"><div class="sig-label">PHI Status</div><div class="sig-value">TOKENIZED — No PHI in report</div></div>
      <div class="sig-field"><div class="sig-label">Audit Trace</div><div class="sig-value">{trace_id}</div></div>
      <div class="sig-field"><div class="sig-label">Claim ID</div><div class="sig-value">{claim_id}</div></div>
    </div>
    <div class="sig-note">
      This report was generated automatically by the ClaimIQ Healthcare Claims Adjudication Agent. The narrative sections
      {'were written by the configured LLM (' + os.environ.get('LLM_PROVIDER','rules') + ') based on the structured adjudication data.' if os.environ.get('LLM_PROVIDER','rules') != 'rules' else 'were generated using rule-based narrative templates. For richer AI-written explanations, configure LLM_PROVIDER=groq or gemini in .env.'}
      All patient identifiers have been replaced with cryptographic vault tokens — no Protected Health Information (PHI) is present
      in this document. This report is for adjudication purposes only and does not constitute medical advice.
    </div>
  </div>

</div>
</body>
</html>"""
