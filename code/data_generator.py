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
    partial_ratio: float, default 0.2
        Proportion of samples per noise level generated as partial faults.
    """

    def __init__(self, samples_per_eq: int = 500,
                 noise_levels: Tuple[float, ...] = (0.0, 0.05, 0.10),
                 partial_ratio: float = 0.2) -> None:
        self.kb = KnowledgeBase()
        self.samples_per_eq = samples_per_eq
        self.noise_levels = noise_levels
        self.partial_ratio = partial_ratio
        # Reuse the same parameter ranges as CoverageAnalyzer for consistency
        self.param_ranges: Dict[str, Dict[str, Tuple[float, float]]] = {
            "Boiler": {
                "pressure": (10, 150),
                "temperature": (100, 450),
                "gas_flow": (50, 500),
                "steam_flow": (100, 800),
                "water_level": (30, 90),
            },
            "Chiller": {
                "compressor_current": (5, 25),
                "coolant_flow": (5, 30),
                "chilled_water_out_temp": (4, 15),
                "cooling_water_in_temp": (20, 40),
            },
            "HVAC": {
                "indoor_temp": (15, 35),
                "indoor_humidity": (20, 80),
                "supply_temp": (10, 30),
                "fan_power": (0.5, 5.0),
                "coil_delta_t": (2, 12),
            },
            "Air Compressor": {
                "inlet_pressure": (30, 120),
                "discharge_pressure": (80, 200),
                "temperature": (20, 100),
                "motor_current": (5, 50),
                "gas_production": (2, 20),
            },
            "Vacuum Machine": {
                "vacuum_level": (-800, -100),
                "motor_current": (2, 15),
                "power_consumption": (5, 40),
                "production_rate": (10, 100),
            },
            "Power Distribution": {
                "voltage": (190, 250),
                "current": (10, 200),
                "power_factor": (0.6, 1.0),
                "phase_imbalance": (0, 30),
            },
            "Water Supply": {
                "pressure": (20, 120),
                "flow_rate": (1, 20),
                "temperature": (5, 60),
                "pump_current": (3, 30),
                "water_elec_ratio": (0.5, 5.0),
            },
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

    def _generate_partial_fault_params(self, rule: Rule) -> Dict[str, float] | None:
        """Generate parameters that satisfy most but NOT all conditions of *rule*.

        Returns a parameter dict where one condition is deliberately set to a
        normal (non-triggering) value, simulating an incipient fault.
        Returns None if the rule has fewer than 2 conditions (cannot drop one).
        """
        conds = list(rule.conditions.items())
        if len(conds) < 2:
            return None  # need at least 2 conditions to drop one

        equipment = rule.equipment
        ranges = self.param_ranges.get(equipment, {})

        # Pick one condition to suppress (make normal)
        drop_idx = random.randint(0, len(conds) - 1)

        params = self._sample_parameters(equipment)
        for i, (param, (op, threshold)) in enumerate(conds):
            if param not in ranges:
                continue
            low, high = ranges[param]
            if i == drop_idx:
                # Set to SAFE / non-triggering value
                if op in ('>', '>='):
                    # Condition requires param > threshold; set below
                    safe_high = min(threshold * 0.8, threshold - 1) if threshold > 0 else threshold - 1
                    params[param] = random.uniform(low, max(low, safe_high))
                elif op in ('<', '<='):
                    # Condition requires param < threshold; set above
                    safe_low = max(threshold * 1.2, threshold + 1) if threshold > 0 else threshold + 1
                    params[param] = random.uniform(min(high, safe_low), high)
                # For == / !=, skip (rare in our rules)
            else:
                # Set to TRIGGERING value (satisfy the condition)
                if op in ('>', '>='):
                    params[param] = random.uniform(threshold * 1.05, high) if threshold < high else threshold * 1.05
                elif op in ('<', '<='):
                    params[param] = random.uniform(low, threshold * 0.95) if threshold > low else threshold * 0.95

        return params

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
            rules = self.kb.get_rules(eq)
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
                        "sample_type": "fault" if fault_label else "normal",
                    })

                # Generate partial fault samples
                n_partial = int(self.samples_per_eq * self.partial_ratio)
                for _ in range(n_partial):
                    rule = random.choice(rules)
                    partial_params = self._generate_partial_fault_params(rule)
                    if partial_params is None:
                        continue
                    noisy_params = self._apply_noise(partial_params, noise)
                    dataset.append({
                        "equipment": eq,
                        "noise_level": noise,
                        "params": noisy_params,
                        "fault": rule.fault,
                        "sample_type": "partial_fault",
                    })
        return dataset
