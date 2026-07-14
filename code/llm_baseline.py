# llm_baseline.py
"""Simulated LLM baseline for equipment fault diagnosis.

The real implementation would call an LLM API (e.g., OpenAI, Claude) with a
prompt describing the equipment parameters and ask it to infer the fault. For
research purposes we provide a lightweight mock that mimics LLM behaviour:

* It has a configurable *accuracy* (probability of returning the correct fault).
* When the LLM is uncertain it may return a generic "unknown" result.
* The module also offers a *hybrid* decision function that falls back to the
  expert system when the LLM confidence is below a user‑defined threshold.

The mock receives ground-truth labels in the synthetic experiments only to
sample from a calibrated accuracy/hallucination process. This is an oracle
simulation of model quality, not a deployable diagnostic procedure.
"""

import random
from typing import Dict, Any, Tuple, Optional

from knowledge_base import KnowledgeBase
from inference_engine import ProbabilisticInferenceEngine


_GROUND_TRUTH_FROM_KB = object()


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

    def __init__(self, accuracy: float = 0.85, hallucination_rate: float = 0.15, seed: Optional[int] = None):
        self.accuracy = max(0.0, min(1.0, accuracy))
        self.hallucination_rate = max(0.0, min(1.0, hallucination_rate))
        self.rng = random.Random(seed) if seed is not None else random
        # A small pool of plausible but generic fault strings for hallucinations
        self.generic_faults = [
            "General performance degradation",
            "Sensor calibration issue",
            "Unexpected power fluctuation",
            "Control system instability",
        ]

    def diagnose(self, equipment: str, params: Dict[str, Any], ground_truth: Optional[str] = None) -> Tuple[Optional[str], float]:
        """Return a fault label and a confidence score.

        The synthetic experiments pass *ground_truth* so the mock can simulate
        an LLM with a specified accuracy. The label is used only to decide
        whether the sampled output is correct or hallucinated.
        """
        # Simulate confidence as a random value centred around the expected
        # reliability of the model.
        base_confidence = self.rng.uniform(0.6, 0.95)

        # Decide outcome
        roll = self.rng.random()
        if roll < self.accuracy:
            # Correct answer (may be None for normal operation)
            fault = ground_truth
            confidence = base_confidence
        else:
            # Hallucination – pick unrelated fault
            fault = self.rng.choice(self.generic_faults)
            confidence = base_confidence * 0.5
        return fault, round(confidence, 3)


class LLMBaseline:
    """Facade exposing a simple API compatible with the experiment runner.

    The class wraps ``SimulatedLLM`` and provides a ``diagnose`` method that
    mirrors the signature of the expert‑system inference engine.
    """

    def __init__(self, accuracy: float = 0.85, hallucination_rate: float = 0.15, seed: Optional[int] = None):
        self.llm = SimulatedLLM(accuracy, hallucination_rate, seed=seed)
        self.kb = KnowledgeBase()

    def diagnose(
        self,
        equipment: str,
        params: Dict[str, Any],
        ground_truth: Any = _GROUND_TRUTH_FROM_KB,
    ) -> Tuple[Optional[str], float]:
        """Run the mock LLM on *params*.

        Pass an explicit ``ground_truth`` in experiments that already have a
        dataset label, including ``None`` for normal samples. If omitted, the
        method derives a best-effort label from full KB rule matches for demos
        and backward compatibility.
        """
        if ground_truth is _GROUND_TRUTH_FROM_KB:
            true_rule = None
            for rule in self.kb.get_rules(equipment):
                if rule.matches(params):
                    true_rule = rule
                    break
            true_fault = true_rule.fault if true_rule else None
        else:
            true_fault = ground_truth
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
