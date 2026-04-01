# llm_baseline.py
"""Simulated LLM baseline for equipment fault diagnosis.

The real implementation would call an LLM API (e.g., OpenAI, Claude) with a
prompt describing the equipment parameters and ask it to infer the fault. For
research purposes we provide a lightweight mock that mimics LLM behaviour:

* It has a configurable *accuracy* (probability of returning the correct fault).
* When the LLM is uncertain it may return a generic "unknown" result.
* The module also offers a *hybrid* decision function that falls back to the
  expert system when the LLM confidence is below a user‑defined threshold.
"""

import random
from typing import Dict, Any, Tuple, Optional

from knowledge_base import KnowledgeBase, Rule
from inference_engine import ProbabilisticInferenceEngine


class SimulatedLLM:
    """A mock LLM that can be tuned for accuracy and hallucination.

    Parameters
    ----------
    accuracy : float, default 0.85
        Probability that the LLM returns the *correct* fault label when the
        ground‑truth fault is known.
    hallucination_rate : float, default 0.15
        Probability that the LLM returns an unrelated fault (simulating a
        hallucination).
    """

    def __init__(self, accuracy: float = 0.85, hallucination_rate: float = 0.15):
        self.accuracy = max(0.0, min(1.0, accuracy))
        self.hallucination_rate = max(0.0, min(1.0, hallucination_rate))
        # A small pool of plausible but generic fault strings for hallucinations
        self.generic_faults = [
            "General performance degradation",
            "Sensor calibration issue",
            "Unexpected power fluctuation",
            "Control system instability",
        ]

    def diagnose(self, equipment: str, params: Dict[str, Any], ground_truth: Optional[str] = None) -> Tuple[Optional[str], float]:
        """Return a fault label and a confidence score.

        If *ground_truth* is provided (including None for normal operation) the
        mock decides whether to be correct based on ``self.accuracy``. When
        ground_truth is the sentinel ``_NO_GT`` (not passed), the LLM behaves
        like a black-box and returns a random generic fault with low confidence.
        """
        # Simulate confidence as a random value centred around the expected
        # reliability of the model.
        base_confidence = random.uniform(0.6, 0.95)

        # Decide outcome
        roll = random.random()
        if roll < self.accuracy:
            # Correct answer (may be None for normal operation)
            fault = ground_truth
            confidence = base_confidence
        else:
            # Hallucination – pick unrelated fault
            fault = random.choice(self.generic_faults)
            confidence = base_confidence * 0.5
        return fault, round(confidence, 3)


class LLMBaseline:
    """Facade exposing a simple API compatible with the experiment runner.

    The class wraps ``SimulatedLLM`` and provides a ``diagnose`` method that
    mirrors the signature of the expert‑system inference engine.
    """

    def __init__(self, accuracy: float = 0.85, hallucination_rate: float = 0.15):
        self.llm = SimulatedLLM(accuracy, hallucination_rate)
        self.kb = KnowledgeBase()

    def diagnose(self, equipment: str, params: Dict[str, Any]) -> Tuple[Optional[str], float]:
        """Run the mock LLM on *params*.

        The ground‑truth fault is derived from the knowledge base (if any) so
        that the simulated accuracy reflects realistic behaviour.
        """
        # Find the true fault using the expert system (may be None)
        true_rule = None
        for rule in self.kb.get_rules(equipment):
            if rule.matches(params):
                true_rule = rule
                break
        true_fault = true_rule.fault if true_rule else None
        return self.llm.diagnose(equipment, params, ground_truth=true_fault)

    def hybrid_decision(self, equipment: str, params: Dict[str, Any], llm_conf_threshold: float = 0.7) -> Tuple[str, float]:
        """Combine expert system and LLM based on confidence.

        Returns a tuple ``(fault, confidence)`` where the fault comes from the
        expert system if the LLM confidence is below *llm_conf_threshold*;
        otherwise the LLM result is used.
        """
        llm_fault, llm_conf = self.diagnose(equipment, params)
        if llm_conf >= llm_conf_threshold and llm_fault is not None:
            return llm_fault, llm_conf
        # Fall back to expert system inference
        engine = ProbabilisticInferenceEngine(self.kb)
        expert_matches = engine.infer(equipment, params)
        if expert_matches:
            # Return the top‑ranked expert rule
            fault, conf = expert_matches[0]
            return fault, conf
        # If no expert rule matches, return whatever the LLM gave (even if low)
        return llm_fault or "Unknown", llm_conf


# Simple demo when run as a script
if __name__ == "__main__":
    baseline = LLMBaseline()
    sample = {"pressure": 25, "temperature": 360}
    fault, conf = baseline.diagnose("Boiler", sample)
    print(f"LLM prediction: {fault} (confidence={conf})")
    hybrid_fault, hybrid_conf = baseline.hybrid_decision("Boiler", sample)
    print(f"Hybrid decision: {hybrid_fault} (confidence={hybrid_conf})")
