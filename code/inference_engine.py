# inference_engine.py
"""Probabilistic inference engine for the expert system.

Implements Bayesian‑style confidence updating for matched rules.
"""

from typing import List, Tuple, Dict, Any

from knowledge_base import KnowledgeBase, Rule


class ProbabilisticInferenceEngine:
    """Engine that infers faults with probabilistic confidence.

    Parameters
    ----------
    kb : KnowledgeBase, optional
        Knowledge base instance. If not provided, a new one is created.
    """

    def __init__(self, kb: KnowledgeBase = None) -> None:
        self.kb = kb if kb is not None else KnowledgeBase()

    def _bayesian_update(self, rule: Rule, match_count: int) -> float:
        """Simple Bayesian‑style update.

        The prior is the rule's base confidence. The likelihood is modelled as
        ``1 + 0.1 * match_count`` (more conditions matched → higher likelihood).
        The posterior is clipped to the range [0, 1].
        """
        prior = rule.confidence
        likelihood = 1.0 + 0.1 * match_count
        # Normalize likelihood to a probability-like value (capped at 1.0)
        likelihood = min(likelihood, 1.0)
        # Bayesian update (assuming binary hypothesis)
        posterior = (prior * likelihood) / (prior * likelihood + (1 - prior) * (1 - likelihood))
        return min(max(posterior, 0.0), 1.0)

    def infer(self, equipment: str, params: Dict[str, Any]) -> List[Tuple[str, float]]:
        """Return a list of (fault, confidence) tuples for the given input.

        The list is sorted by confidence descending.
        """
        if not self.kb.is_feasible(equipment, params):
            return []
        matches: List[Tuple[Rule, float]] = []
        for rule in self.kb.get_rules(equipment):
            if rule.matches(params):
                match_count = len(rule.conditions)
                posterior = self._bayesian_update(rule, match_count)
                matches.append((rule, posterior))
        # Normalize confidences so they sum to 1 (optional but nice for ranking)
        total_conf = sum(conf for _, conf in matches)
        results: List[Tuple[str, float]] = []
        for rule, conf in matches:
            norm_conf = conf / total_conf if total_conf else conf
            results.append((rule.fault, norm_conf))
        # Sort by confidence descending
        results.sort(key=lambda x: x[1], reverse=True)
        return results
