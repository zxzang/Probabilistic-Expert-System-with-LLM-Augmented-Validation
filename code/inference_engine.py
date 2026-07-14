# inference_engine.py
"""Pattern Matching with Scoring (PMS) inference engine.

Implements the PMS algorithm described in Section 3.2 of the paper:
  S_match(r_i) = (matched_conditions / total_conditions) × 100
  S_conf(r_i)  = θ_i × S_match × (1 + α·|C_i|) × (1 + β / P_i)
  P(r_i | O)   = S_conf(r_i) / Σ S_conf(r_k)        (Eq. 3)

Multi-key sort: S_match (desc) → Priority (asc) → Specificity (desc)
"""

from typing import List, Tuple, Dict, Any

from knowledge_base import KnowledgeBase, Rule


class ProbabilisticInferenceEngine:
    """Engine that infers faults using Pattern Matching with Scoring.

    Parameters
    ----------
    kb : KnowledgeBase, optional
        Knowledge base instance. If not provided, a new one is created.
    alpha : float
        Specificity reward coefficient (default 0.1).
    beta : float
        Priority boost coefficient (default 0.2).
    min_match : float
        Minimum S_match percentage required to activate a rule (default 75.0).
    """

    def __init__(self, kb: KnowledgeBase = None,
                 alpha: float = 0.1, beta: float = 0.2,
                 min_match: float = 75.0,
                 require_abnormal_anchor: bool = False,
                 evidence_damping: bool = False) -> None:
        self.kb = kb if kb is not None else KnowledgeBase()
        self.alpha = alpha
        self.beta = beta
        self.min_match = min_match  # minimum S_match to activate a rule
        self.require_abnormal_anchor = require_abnormal_anchor
        self.evidence_damping = evidence_damping

    # -----------------------------------------------------------------
    # PMS scoring (Section 3.2)
    # -----------------------------------------------------------------
    @staticmethod
    def _match_score(rule: Rule, params: Dict[str, Any]) -> float:
        """Compute S_match: percentage of conditions satisfied (Eq. 1)."""
        n_conditions = len(rule.conditions)
        if n_conditions == 0:
            return 0.0
        matched = rule.count_matched(params)
        return (matched / n_conditions) * 100.0

    @staticmethod
    def _is_normal_fault(fault: Any) -> bool:
        return str(fault).strip().lower() in {"normal", "none", "no fault", "null", ""}

    @staticmethod
    def _is_abnormal_condition(condition: Tuple[str, Any]) -> bool:
        op, value = condition
        return not (op == "==" and value == 0)

    @staticmethod
    def _abnormal_condition_count(rule: Rule) -> int:
        return sum(1 for condition in rule.conditions.values()
                   if ProbabilisticInferenceEngine._is_abnormal_condition(condition))

    def _has_abnormal_anchor(self, rule: Rule, params: Dict[str, Any]) -> bool:
        """Require non-normal rules to match at least one abnormal predicate.

        Conditions such as ``predicate == 0`` are useful as exclusions, but a
        non-normal rule should not activate only because normal-state predicates
        matched. This keeps the original S_match denominator unchanged while
        preventing purely negative partial matches.
        """
        if self._is_normal_fault(rule.fault):
            return True
        abnormal_conditions = [
            param for param, condition in rule.conditions.items()
            if self._is_abnormal_condition(condition)
        ]
        if not abnormal_conditions:
            return True
        return any(rule._eval_condition(param, params) for param in abnormal_conditions)

    @staticmethod
    def _evidence_damp(rule: Rule, params: Dict[str, Any], max_match: float) -> float:
        """Calibrate confidence downward for sparse public-predicate evidence."""
        if ProbabilisticInferenceEngine._is_normal_fault(rule.fault):
            return 1.0
        active_values = [
            float(value) for value in params.values()
            if isinstance(value, (int, float)) and value > 0
        ]
        if not active_values:
            return 0.0
        nonzero_count = len(active_values)
        severe_bonus = 0.5 * sum(max(0.0, value - 1.0) for value in active_values)
        evidence_units = nonzero_count + severe_bonus
        expected = max(3.0, float(ProbabilisticInferenceEngine._abnormal_condition_count(rule) or 1))
        evidence_ratio = min(1.0, evidence_units / expected)
        return max(0.0, min(1.0, evidence_ratio, max_match / 100.0))

    def _confidence_score(self, rule: Rule, s_match: float) -> float:
        """Compute S_conf incorporating prior, specificity, and priority."""
        specificity = len(rule.conditions)
        return rule.confidence * s_match * (1 + self.alpha * specificity) * (1 + self.beta / rule.priority)

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------
    def infer(self, equipment: str, params: Dict[str, Any]) -> List[Tuple[str, float]]:
        """Return a list of (fault, confidence) tuples for the given input.

        Only rules with at least one matching condition (S_match > 0) are
        activated.  Confidences are normalised via Eq. 3.  The result list
        is sorted by the multi-key protocol:
            S_match (desc) → Priority (asc) → Specificity |C_i| (desc)
        """
        if not self.kb.is_feasible(equipment, params):
            return []

        # --- Step 1: score every rule (partial matching) ---------------
        activated: List[Tuple[Rule, float, float]] = []  # (rule, s_match, s_conf)
        for rule in self.kb.get_rules(equipment):
            if self.require_abnormal_anchor and not self._has_abnormal_anchor(rule, params):
                continue
            s_match = self._match_score(rule, params)
            if s_match >= self.min_match:              # meet minimum match threshold
                s_conf = self._confidence_score(rule, s_match)
                activated.append((rule, s_match, s_conf))

        if not activated:
            return []

        # --- Step 2: normalise (Eq. 3) ---------------------------------
        total_conf = sum(s_conf for _, _, s_conf in activated)

        # --- Step 3: multi-key sort ------------------------------------
        # S_match desc → Priority asc → Specificity desc
        activated.sort(key=lambda t: (-t[1], t[0].priority, -len(t[0].conditions)))

        results: List[Tuple[str, float]] = []
        for rule, _s_match, s_conf in activated:
            norm_conf = s_conf / total_conf if total_conf > 0 else 0.0
            if self.evidence_damping:
                norm_conf *= self._evidence_damp(rule, params, _s_match)
            results.append((rule.fault, norm_conf))

        return results

    def infer_with_details(self, equipment: str, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Return ranked PMS outputs with rule-level scoring details.

        This optional diagnostic API preserves the public `infer()` contract
        while exposing the intermediate values used for interpretability.
        """
        if not self.kb.is_feasible(equipment, params):
            return []

        activated: List[Tuple[Rule, float, float]] = []
        for rule in self.kb.get_rules(equipment):
            if self.require_abnormal_anchor and not self._has_abnormal_anchor(rule, params):
                continue
            s_match = self._match_score(rule, params)
            if s_match >= self.min_match:
                activated.append((rule, s_match, self._confidence_score(rule, s_match)))

        if not activated:
            return []

        total_conf = sum(s_conf for _, _, s_conf in activated)
        activated.sort(key=lambda t: (-t[1], t[0].priority, -len(t[0].conditions)))

        details: List[Dict[str, Any]] = []
        for rule, s_match, s_conf in activated:
            damp = self._evidence_damp(rule, params, s_match) if self.evidence_damping else 1.0
            details.append(
                {
                    "fault": rule.fault,
                    "confidence": (s_conf / total_conf) * damp if total_conf > 0 else 0.0,
                    "s_match": s_match,
                    "s_conf": s_conf,
                    "evidence_damp": damp,
                    "priority": rule.priority,
                    "specificity": len(rule.conditions),
                    "conditions": dict(rule.conditions),
                    "matched_predicates": {
                        param: params.get(param)
                        for param in rule.conditions
                        if rule._eval_condition(param, params)
                    },
                    "unmatched_predicates": {
                        param: params.get(param)
                        for param in rule.conditions
                        if not rule._eval_condition(param, params)
                    },
                    "matched_conditions": rule.count_matched(params),
                    "total_conditions": len(rule.conditions),
                    "analysis": rule.analysis,
                }
            )
        return details
