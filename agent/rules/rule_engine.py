"""
rule_engine.py — Stage 6: Evaluates policy rules against a tokenized claim.

Takes CodingResult + TokenizedClaimInput and evaluates every applicable rule.
Returns a RuleEngineResult with pass/fail per rule and aggregated flags.

Five rule categories — each has its own evaluation strategy:

  COMPATIBILITY   — does the CPT code have a medically appropriate ICD-10 diagnosis?
  PRIOR_AUTH      — does this procedure require payer pre-approval?
  AGE_RESTRICTION — is the patient within the allowed age range?
  COVERAGE        — does the patient's plan cover this procedure?
  FREQUENCY       — has this procedure been claimed too recently? (stub — needs claims DB)

Each rule is only evaluated if the claim contains a CPT code the rule applies to.
Non-applicable rules are skipped entirely — not logged, not penalised.

Result classification:
  blocking_failures — rules that REJECT (block approval)
  warnings          — rules that FLAG_REVIEW (route to human)
  all_passed        — True only if zero blocking failures AND zero warnings
"""

import logging
from typing import Optional

from agent.models.enums import RuleAction, RuleType, Severity
from agent.models.schemas import (
    CodingResult,
    RuleEngineResult,
    RuleEvaluation,
    TokenizedClaimInput,
)
from agent.rules.policy_store import PolicyStore

logger = logging.getLogger(__name__)


class RuleEngine:
    """
    Stage 6 — Evaluates policy rules against a claim.

    Usage (called by pipeline.py):
        engine = RuleEngine()
        result = engine.evaluate(claim, coding)
    """

    def __init__(self, store: Optional[PolicyStore] = None):
        self._store = store or PolicyStore()
        logger.info(f"RuleEngine initialised with {self._store.count()} rules")

    # ── Primary entry point ──────────────────────────────────────────────────

    def evaluate(
        self,
        claim:  TokenizedClaimInput,
        coding: CodingResult,
    ) -> RuleEngineResult:
        """
        Evaluates all applicable rules against the claim.

        Args:
            claim:  TokenizedClaimInput — provides patient info (age, plan)
            coding: CodingResult — provides ICD-10 and CPT codes

        Returns:
            RuleEngineResult with per-rule evaluations and aggregated flags
        """
        cpt_codes_on_claim = {c.code for c in coding.cpt_codes}
        icd_codes_on_claim = {c.code for c in coding.icd10_codes}

        # Collect all rules applicable to this claim's CPT codes
        applicable_rules = self._get_applicable_rules(cpt_codes_on_claim)

        if not applicable_rules:
            logger.debug(f"[{claim.claim_id}] No applicable rules for CPT codes: {cpt_codes_on_claim}")
            return RuleEngineResult(
                evaluations       = [],
                blocking_failures = [],
                warnings          = [],
                all_passed        = True,
                rules_evaluated   = 0,
            )

        evaluations: list[RuleEvaluation] = []

        for rule in applicable_rules:
            try:
                ev = self._evaluate_rule(rule, claim, icd_codes_on_claim, cpt_codes_on_claim)
                evaluations.append(ev)
                if not ev.passed:
                    logger.info(
                        f"[{claim.claim_id}] Rule {rule.rule_id} FAILED "
                        f"action={ev.action.value}: {ev.reason}"
                    )
            except Exception as e:
                logger.error(f"[{claim.claim_id}] Rule {rule.rule_id} evaluation error: {e}")
                # Treat evaluation errors as warnings — don't silently pass
                evaluations.append(RuleEvaluation(
                    rule_id   = rule.rule_id,
                    rule_name = rule.rule_name,
                    rule_type = rule.rule_type,
                    passed    = False,
                    action    = RuleAction.FLAG_REVIEW,
                    reason    = f"Rule evaluation error: {e}",
                    severity  = Severity.MEDIUM,
                ))

        blocking = [ev for ev in evaluations if not ev.passed and ev.action == RuleAction.REJECT]
        warnings = [ev for ev in evaluations if not ev.passed and ev.action == RuleAction.FLAG_REVIEW]

        return RuleEngineResult(
            evaluations       = evaluations,
            blocking_failures = blocking,
            warnings          = warnings,
            all_passed        = len(blocking) == 0 and len(warnings) == 0,
            rules_evaluated   = len(evaluations),
        )

    # ── Applicability check ───────────────────────────────────────────────────

    def _get_applicable_rules(self, cpt_codes: set[str]) -> list:
        """
        Returns rules that apply to at least one CPT code on the claim.
        Deduplicates — a rule covering multiple CPT codes is returned once.
        """
        seen:  set[str]  = set()
        rules: list      = []

        for cpt in cpt_codes:
            for rule in self._store.get_rules_for_cpt(cpt):
                if rule.rule_id not in seen:
                    seen.add(rule.rule_id)
                    rules.append(rule)

        return rules

    # ── Rule dispatch ─────────────────────────────────────────────────────────

    def _evaluate_rule(
        self,
        rule:             object,   # PolicyRule
        claim:            TokenizedClaimInput,
        icd_codes:        set[str],
        cpt_codes:        set[str],
    ) -> RuleEvaluation:
        """Dispatches to the correct evaluation strategy by rule type."""
        dispatch = {
            RuleType.COMPATIBILITY:   self._eval_compatibility,
            RuleType.PRIOR_AUTH:      self._eval_prior_auth,
            RuleType.AGE_RESTRICTION: self._eval_age_restriction,
            RuleType.COVERAGE:        self._eval_coverage,
            RuleType.FREQUENCY:       self._eval_frequency,
        }

        fn = dispatch.get(rule.rule_type)
        if fn is None:
            raise ValueError(f"Unknown rule type: {rule.rule_type}")

        return fn(rule, claim, icd_codes, cpt_codes)

    # ── COMPATIBILITY evaluation ──────────────────────────────────────────────

    def _eval_compatibility(self, rule, claim, icd_codes, cpt_codes) -> RuleEvaluation:
        """
        Checks that at least one ICD-10 code on the claim starts with
        one of the required prefixes defined in the rule conditions.

        PASS: any ICD code matches a required prefix
        FAIL: no ICD code matches any required prefix → REJECT
        """
        required_prefixes: list[str] = rule.conditions.get("required_icd_prefixes", [])
        rule_cpts:         list[str] = rule.conditions.get("cpt_codes", [])

        # Only evaluate if claim has the trigger CPT code
        trigger_cpts = cpt_codes & set(rule_cpts)
        if not trigger_cpts:
            return self._pass(rule, "Rule not triggered (CPT not on claim)")

        # Check ICD prefixes
        for icd in icd_codes:
            for prefix in required_prefixes:
                if icd.upper().startswith(prefix.upper()):
                    return self._pass(
                        rule,
                        f"Diagnosis {icd} satisfies required prefix '{prefix}' for CPT {list(trigger_cpts)[0]}"
                    )

        return self._fail(
            rule,
            f"CPT {list(trigger_cpts)} requires a diagnosis with prefix "
            f"{required_prefixes} — none found in claim codes {list(icd_codes)}"
        )

    # ── PRIOR AUTH evaluation ─────────────────────────────────────────────────

    def _eval_prior_auth(self, rule, claim, icd_codes, cpt_codes) -> RuleEvaluation:
        """
        Prior authorization rules always FLAG_REVIEW because authorization
        status is not present in the claim input — it requires external
        verification with the payer.

        PRODUCTION: Add a prior_auth_verified field to ClaimInput and check it here.
        """
        rule_cpts     = rule.conditions.get("cpt_codes", [])
        trigger_cpts  = list(cpt_codes & set(rule_cpts))

        if not trigger_cpts:
            return self._pass(rule, "Rule not triggered (CPT not on claim)")

        return self._fail(
            rule,
            f"CPT {trigger_cpts} requires prior authorization — "
            f"payer pre-approval must be verified before adjudication"
        )

    # ── AGE RESTRICTION evaluation ────────────────────────────────────────────

    def _eval_age_restriction(self, rule, claim, icd_codes, cpt_codes) -> RuleEvaluation:
        """
        Checks patient age against minimum_age in rule conditions.
        Age is extracted from the age_band string (e.g. "60s" → decade midpoint).

        PASS: patient age meets the minimum requirement
        FAIL: patient is too young → REJECT or FLAG_REVIEW per rule action
        """
        rule_cpts    = rule.conditions.get("cpt_codes", [])
        trigger_cpts = list(cpt_codes & set(rule_cpts))

        if not trigger_cpts:
            return self._pass(rule, "Rule not triggered (CPT not on claim)")

        minimum_age: int = rule.conditions.get("minimum_age", 0)
        age_band         = claim.patient_info.age_band  # e.g. "60s", "10s", "90s+"

        patient_age = self._parse_age_band(age_band)

        if patient_age is None:
            # Cannot determine age — flag for review
            return RuleEvaluation(
                rule_id   = rule.rule_id,
                rule_name = rule.rule_name,
                rule_type = rule.rule_type,
                passed    = False,
                action    = RuleAction.FLAG_REVIEW,
                reason    = f"Cannot determine patient age from band '{age_band}' — manual review required",
                severity  = Severity.MEDIUM,
            )

        if rule.rule_id == "AGE-002":
            # Under-18 restriction — fail if below minimum
            if patient_age < minimum_age:
                return self._fail(
                    rule,
                    f"Patient age band '{age_band}' (est. {patient_age}) is under {minimum_age} — "
                    f"CPT {trigger_cpts} not covered for pediatric patients"
                )
        else:
            # Screening age minimum — fail if below minimum
            if patient_age < minimum_age:
                return self._fail(
                    rule,
                    f"Patient age band '{age_band}' (est. {patient_age}) is below minimum age "
                    f"{minimum_age} for CPT {trigger_cpts}"
                )

        return self._pass(
            rule,
            f"Patient age band '{age_band}' meets minimum age {minimum_age} for CPT {trigger_cpts}"
        )

    # ── COVERAGE evaluation ───────────────────────────────────────────────────

    def _eval_coverage(self, rule, claim, icd_codes, cpt_codes) -> RuleEvaluation:
        """
        Checks if the patient's insurance plan is in the excluded_plans list.

        PASS: patient's plan is not excluded
        FAIL: patient's plan is excluded → REJECT
        """
        rule_cpts      = rule.conditions.get("cpt_codes", [])
        trigger_cpts   = list(cpt_codes & set(rule_cpts))

        if not trigger_cpts:
            return self._pass(rule, "Rule not triggered (CPT not on claim)")

        excluded_plans : list[str] = [p.lower() for p in rule.conditions.get("excluded_plans", [])]
        patient_plan               = claim.patient_info.insurance_plan.value.lower()

        if patient_plan in excluded_plans:
            return self._fail(
                rule,
                f"Insurance plan '{patient_plan}' does not cover CPT {trigger_cpts} — "
                f"excluded plans: {excluded_plans}"
            )

        return self._pass(
            rule,
            f"Insurance plan '{patient_plan}' covers CPT {trigger_cpts}"
        )

    # ── FREQUENCY evaluation ──────────────────────────────────────────────────

    def _eval_frequency(self, rule, claim, icd_codes, cpt_codes) -> RuleEvaluation:
        """
        Frequency limit check — e.g. colonoscopy once every 10 years.

        STUB: always passes with an informational note.
        PRODUCTION: query claims history DB for patient's prior claims
        within the frequency window using claim.patient_info.token_patient_id.
        """
        rule_cpts    = rule.conditions.get("cpt_codes", [])
        trigger_cpts = list(cpt_codes & set(rule_cpts))

        if not trigger_cpts:
            return self._pass(rule, "Rule not triggered (CPT not on claim)")

        logger.debug(
            f"FREQ rule {rule.rule_id}: claims history check not yet implemented — passing. "
            f"Production: query DB for token_patient_id={claim.patient_info.token_patient_id}"
        )

        return self._pass(
            rule,
            f"Frequency check for CPT {trigger_cpts} passed (claims history query pending in production)"
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _pass(self, rule, reason: str) -> RuleEvaluation:
        return RuleEvaluation(
            rule_id   = rule.rule_id,
            rule_name = rule.rule_name,
            rule_type = rule.rule_type,
            passed    = True,
            action    = rule.action,
            reason    = reason,
            severity  = rule.severity,
        )

    def _fail(self, rule, reason: str) -> RuleEvaluation:
        return RuleEvaluation(
            rule_id   = rule.rule_id,
            rule_name = rule.rule_name,
            rule_type = rule.rule_type,
            passed    = False,
            action    = rule.action,
            reason    = reason,
            severity  = rule.severity,
        )

    @staticmethod
    def _parse_age_band(age_band: str) -> Optional[int]:
        """
        Converts age_band string to estimated age integer.
        "60s" → 65 (midpoint), "90s+" → 95, "10s" → 15.
        Returns None if unparseable.
        """
        if not age_band:
            return None

        band = age_band.strip().lower()

        if band == "90s+":
            return 95

        if band.endswith("s") and band[:-1].isdigit():
            decade = int(band[:-1])
            return decade + 5   # midpoint of decade

        # Try plain integer
        try:
            return int(band)
        except ValueError:
            return None
