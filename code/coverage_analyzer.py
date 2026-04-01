# coverage_analyzer.py
"""Coverage analysis for the expert‑system rules.

The goal is to estimate how much of the *feasible* parameter space for each
equipment type is covered by at least one diagnostic rule. Because enumerating
all possible continuous values is infeasible, we use a Monte‑Carlo sampling
approach:

1. For each equipment we define a reasonable range for each relevant sensor
   parameter (derived from the knowledge base constraints).
2. Randomly sample a large number of parameter vectors within those ranges.
3. Discard samples that violate physical constraints (using
   ``KnowledgeBase.is_feasible``).
4. Count the fraction of feasible samples that trigger at least one rule.

The ``CoverageAnalyzer`` class provides ``estimate_coverage`` which returns a
dictionary mapping equipment names to coverage percentages.
"""

import random
from typing import Dict, List, Tuple, Any

from knowledge_base import KnowledgeBase, Rule


class CoverageAnalyzer:
    """Monte‑Carlo coverage estimator.

    Attributes
    ----------
    kb: KnowledgeBase
        The knowledge base containing rules and constraints.
    samples_per_eq: int
        Number of random samples to draw for each equipment type.
    param_ranges: Dict[str, Dict[str, Tuple[float, float]]]
        Mapping from equipment → parameter → (min, max) range. The ranges are
        heuristically chosen based on typical operating limits.
    """

    def __init__(self, samples_per_eq: int = 2000) -> None:
        self.kb = KnowledgeBase()
        self.samples_per_eq = samples_per_eq
        # Define generic ranges; these can be refined later.
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

    # ---------------------------------------------------------------------
    # Helper methods
    # ---------------------------------------------------------------------
    def _sample_parameters(self, equipment: str) -> Dict[str, float]:
        """Draw a random parameter vector for *equipment* within defined ranges.
        """
        ranges = self.param_ranges.get(equipment, {})
        sample: Dict[str, float] = {}
        for param, (low, high) in ranges.items():
            sample[param] = random.uniform(low, high)
        return sample

    def _has_matching_rule(self, equipment: str, params: Dict[str, Any]) -> bool:
        """Return True if at least one rule matches the supplied *params*.
        """
        for rule in self.kb.get_rules(equipment):
            if rule.matches(params):
                return True
        return False

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def estimate_coverage(self) -> Dict[str, float]:
        """Estimate coverage for each equipment type.

        Returns
        -------
        dict
            Mapping ``equipment -> coverage_percent`` where the percentage is the
            proportion of feasible random samples that trigger at least one rule.
        """
        coverage: Dict[str, float] = {}
        for equipment in self.kb.equipment_types:
            total_feasible = 0
            covered = 0
            for _ in range(self.samples_per_eq):
                params = self._sample_parameters(equipment)
                if not self.kb.is_feasible(equipment, params):
                    continue
                total_feasible += 1
                if self._has_matching_rule(equipment, params):
                    covered += 1
            # Guard against division by zero (unlikely with reasonable ranges)
            coverage_pct = (covered / total_feasible * 100) if total_feasible else 0.0
            coverage[equipment] = round(coverage_pct, 2)
        return coverage


# Simple demo when run as a script
if __name__ == "__main__":
    analyzer = CoverageAnalyzer(samples_per_eq=2000)
    cov = analyzer.estimate_coverage()
    for eq, pct in cov.items():
        print(f"{eq}: {pct}% of feasible parameter space covered by rules")
