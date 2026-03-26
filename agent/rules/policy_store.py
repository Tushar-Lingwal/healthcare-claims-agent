"""
policy_store.py — Loads, validates, and provides policy rules to the RuleEngine.

Rules are loaded from default_rules.json at startup.
Custom rules can be added at runtime or loaded from an alternate JSON file.

Design:
  - Rules are immutable after loading — no runtime modification.
  - Each rule is validated against the PolicyRule schema on load.
  - Rules indexed by rule_id for O(1) lookup.
  - Rules indexed by CPT code for fast applicability checks.
  - Adding new rules requires only appending to the JSON file — zero code change.
"""

import json
import logging
import os
from pathlib import Path
from typing import Optional

from agent.models.enums import RuleAction, RuleType, Severity
from agent.models.schemas import PolicyRule

logger = logging.getLogger(__name__)

# Default path relative to project root
_DEFAULT_RULES_PATH = Path(__file__).parent.parent.parent / "data" / "rules" / "default_rules.json"


class PolicyStore:
    """
    Loads and indexes policy rules from JSON.

    Usage:
        store = PolicyStore()                          # loads default rules
        store = PolicyStore(path="custom_rules.json")  # loads custom rules
        rules = store.get_rules_for_cpt("27447")       # fast CPT lookup
    """

    def __init__(self, path: Optional[str] = None):
        self._rules:       list[PolicyRule]             = []
        self._by_id:       dict[str, PolicyRule]        = {}
        self._by_cpt:      dict[str, list[PolicyRule]]  = {}

        rules_path = Path(path) if path else _DEFAULT_RULES_PATH
        self._load(rules_path)

    # ── Loading ──────────────────────────────────────────────────────────────

    def _load(self, path: Path) -> None:
        """Loads and validates rules from a JSON file."""
        if not path.exists():
            logger.error(f"Rules file not found: {path}")
            raise FileNotFoundError(f"Policy rules file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        if not isinstance(raw, list):
            raise ValueError(f"Rules file must contain a JSON array, got {type(raw)}")

        loaded = 0
        for item in raw:
            try:
                rule = self._parse_rule(item)
                self._register(rule)
                loaded += 1
            except Exception as e:
                logger.error(f"Skipping invalid rule {item.get('rule_id', '?')}: {e}")

        logger.info(f"PolicyStore loaded {loaded} rules from {path}")

    def _parse_rule(self, item: dict) -> PolicyRule:
        """Parses and validates a single rule dict into a PolicyRule."""
        required = ["rule_id", "rule_name", "description", "rule_type", "conditions", "action", "severity"]
        for field in required:
            if field not in item:
                raise ValueError(f"Missing required field: '{field}'")

        try:
            rule_type = RuleType(item["rule_type"])
        except ValueError:
            raise ValueError(f"Unknown rule_type: '{item['rule_type']}'. Valid: {[e.value for e in RuleType]}")

        try:
            action = RuleAction(item["action"])
        except ValueError:
            raise ValueError(f"Unknown action: '{item['action']}'. Valid: {[e.value for e in RuleAction]}")

        try:
            severity = Severity(item["severity"])
        except ValueError:
            raise ValueError(f"Unknown severity: '{item['severity']}'. Valid: {[e.value for e in Severity]}")

        conditions = item["conditions"]
        if not isinstance(conditions, dict):
            raise ValueError(f"'conditions' must be a dict, got {type(conditions)}")

        return PolicyRule(
            rule_id     = item["rule_id"],
            rule_name   = item["rule_name"],
            description = item["description"],
            rule_type   = rule_type,
            conditions  = conditions,
            action      = action,
            severity    = severity,
        )

    def _register(self, rule: PolicyRule) -> None:
        """Adds a rule to all indexes."""
        if rule.rule_id in self._by_id:
            logger.warning(f"Duplicate rule_id '{rule.rule_id}' — overwriting")

        self._rules.append(rule)
        self._by_id[rule.rule_id] = rule

        # Index by every CPT code in conditions
        for cpt in rule.conditions.get("cpt_codes", []):
            self._by_cpt.setdefault(cpt, []).append(rule)

    # ── Query interface ───────────────────────────────────────────────────────

    def get_all_rules(self) -> list[PolicyRule]:
        """Returns all loaded rules."""
        return list(self._rules)

    def get_rule(self, rule_id: str) -> Optional[PolicyRule]:
        """Returns a rule by ID, or None if not found."""
        return self._by_id.get(rule_id)

    def get_rules_for_cpt(self, cpt_code: str) -> list[PolicyRule]:
        """
        Returns all rules that apply to a given CPT code.
        Used by the rule engine to skip non-applicable rules fast.
        """
        return self._by_cpt.get(cpt_code, [])

    def get_rules_by_type(self, rule_type: RuleType) -> list[PolicyRule]:
        """Returns all rules of a given type."""
        return [r for r in self._rules if r.rule_type == rule_type]

    def count(self) -> int:
        """Returns total number of loaded rules."""
        return len(self._rules)

    def add_rule(self, rule: PolicyRule) -> None:
        """
        Adds a rule at runtime (e.g. payer-specific overrides).
        Does not persist to the JSON file.
        """
        self._register(rule)
        logger.info(f"Runtime rule added: {rule.rule_id}")
