# data_generator.py
"""Synthetic data generator for equipment fault diagnosis.

Generates samples with configurable noise levels. Each sample includes:
- equipment type
- noisy parameters dict
- ground‑truth fault label (or None for normal operation)
"""

import random
from typing import List, Dict, Any, Tuple

from knowledge_base import KnowledgeBase, Rule


class SyntheticDataGenerator:
    """Generate synthetic fault and normal operation data.

    Parameters
    ----------
    samples_per_eq: int, default 500
        Number of samples to generate per equipment type per noise level.
    noise_levels: Tuple[float, ...], default (0.0, 0.05, 0.10)
        Multiplicative noise factors applied to each parameter.
    """

    def __init__(self, samples_per_eq: int = 500, noise_levels: Tuple[float, ...] = (0.0, 0.05, 0.10)) -> None:
        self.kb = KnowledgeBase()
        self.samples_per_eq = samples_per_eq
        self.noise_levels = noise_levels
        # Reuse the same parameter ranges as CoverageAnalyzer for consistency
        self.param_ranges: Dict[str, Dict[str, Tuple[float, float]]] = {
            "Boiler": {"pressure": (10, 150), "temperature": (100, 450)},
            "Chiller": {"evap_temp": (0, 25), "compressor_current": (5, 25), "coolant_flow": (5, 30)},
            "HVAC": {"indoor_temp": (15, 35), "outdoor_temp": (-30, 45), "indoor_humidity": (20, 80)},
            "Air Compressor": {"inlet_pressure": (30, 120), "discharge_pressure": (80, 200), "temperature": (20, 100)},
            "Vacuum Machine": {"vacuum_level": (-800, -100), "motor_current": (2, 15)},
            "Power Distribution": {"voltage": (190, 250), "current": (10, 200), "frequency": (45, 55)},
            "Water Supply": {"pressure": (20, 120), "flow_rate": (1, 20), "temperature": (5, 60)},
        }

    def _sample_parameters(self, equipment: str) -> Dict[str, float]:
        """Draw a random feasible parameter set for the given equipment."""
        ranges = self.param_ranges.get(equipment, {})
        return {param: random.uniform(low, high) for param, (low, high) in ranges.items()}

    def _apply_noise(self, params: Dict[str, float], noise_factor: float) -> Dict[str, float]:
        """Apply Gaussian‑like multiplicative noise to each parameter.

        noise_factor is a proportion of the parameter value (e.g., 0.05 = 5%).
        """
        if noise_factor == 0.0:
            return params.copy()
        noisy = {}
        for k, v in params.items():
            sigma = abs(v) * noise_factor
            noisy[k] = random.gauss(v, sigma)
        return noisy

    def _find_matching_rule(self, equipment: str, params: Dict[str, Any]) -> Rule | None:
        """Return the first rule that matches the given parameters, or None."""
        for rule in self.kb.get_rules(equipment):
            if rule.matches(params):
                return rule
        return None

    def generate(self) -> List[Dict[str, Any]]:
        """Generate the full dataset.

        Returns a list of dictionaries with keys:
        - "equipment"
        - "noise_level"
        - "params" (noisy parameters)
        - "fault" (ground‑truth fault label or None)
        """
        dataset: List[Dict[str, Any]] = []
        for eq in self.kb.equipment_types:
            for noise in self.noise_levels:
                for _ in range(self.samples_per_eq):
                    raw_params = self._sample_parameters(eq)
                    if not self.kb.is_feasible(eq, raw_params):
                        continue
                    rule = self._find_matching_rule(eq, raw_params)
                    fault_label = rule.fault if rule else None
                    noisy_params = self._apply_noise(raw_params, noise)
                    dataset.append({
                        "equipment": eq,
                        "noise_level": noise,
                        "params": noisy_params,
                        "fault": fault_label,
                    })
        return dataset
