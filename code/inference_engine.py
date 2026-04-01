# inference_engine.py
"""Pattern Matching with Scoring (PMS) inference engine.

Implements the PMS algorithm described in Section 3.2 of the paper:
  S_match(r_i) = (matched_conditions / total_conditions) × 100
  S_conf(r_i)  = S_match × (1 + α·|C_i|) × (1 + β / P_i)
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
                 min_match: float = 75.0) -> None:
        self.kb = kb if kb is not None else KnowledgeBase()
        self.alpha = alpha
        self.beta = beta
        self.min_match = min_match  # minimum S_match to activate a rule

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

    def _confidence_score(self, rule: Rule, s_match: float) -> float:
        """Compute S_conf incorporating specificity and priority (Eq. 2)."""
        specificity = len(rule.conditions)
        return s_match * (1 + self.alpha * specificity) * (1 + self.beta / rule.priority)

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
            results.append((rule.fault, norm_conf))

        return results
