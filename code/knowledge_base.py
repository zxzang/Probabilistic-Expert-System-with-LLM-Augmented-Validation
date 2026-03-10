# knowledge_base.py
"""Knowledge base for equipment fault diagnosis.

Defines:
- Equipment types (7 major categories)
- Physical constraints for parameter combinations (infeasible combos)
- Diagnostic rules mapping abnormal parameter patterns to fault descriptions

The module provides a simple API:
- `KnowledgeBase()` loads all rules and constraints
- `get_rules(equipment)` returns applicable rules for a given equipment type
- `is_feasible(equipment, params)` checks if a parameter set respects physical constraints
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Tuple


@dataclass
class Rule:
    """A diagnostic rule.

    Attributes:
        equipment: Equipment type the rule applies to.
        conditions: Mapping of parameter name to a tuple (operator, threshold).
            Supported operators: '>', '>=', '<', '<=', '==', '!='.
        fault: Human‑readable fault description.
        confidence: Base confidence (0‑1) before Bayesian update.
        priority: Integer priority, higher wins on conflict.
    """
    equipment: str
    conditions: Dict[str, Tuple[str, Any]]
    fault: str
    confidence: float = 0.9
    priority: int = 1

    def matches(self, params: Dict[str, Any]) -> bool:
        """Return True if *params* satisfy all conditions."""
        for param, (op, value) in self.conditions.items():
            if param not in params:
                return False
            p = params[param]
            if op == '>':
                if not (p > value):
                    return False
            elif op == '>=':
                if not (p >= value):
                    return False
            elif op == '<':
                if not (p < value):
                    return False
            elif op == '<=':
                if not (p <= value):
                    return False
            elif op == '==':
                if not (p == value):
                    return False
            elif op == '!=':
                if not (p != value):
                    return False
            else:
                raise ValueError(f"Unsupported operator {op}")
        return True


class KnowledgeBase:
    """Container for all rules and constraints.

    The knowledge base is deliberately lightweight – it can be extended by loading
    JSON/YAML files, but for the purpose of this research project the rules are
    hard‑coded to keep the repository self‑contained.
    """

    def __init__(self) -> None:
        self.equipment_types = [
            "Boiler",
            "Chiller",
            "HVAC",
            "Air Compressor",
            "Vacuum Machine",
            "Power Distribution",
            "Water Supply",
        ]
        self.constraints = self._build_constraints()
        self.rules = self._build_rules()

    # ---------------------------------------------------------------------
    # Physical constraints
    # ---------------------------------------------------------------------
    def _build_constraints(self) -> Dict[str, List[Tuple[str, Any]]]:
        """Return a dict mapping equipment -> list of infeasible (param, value) pairs.

        Each entry describes a *forbidden* exact value or range. The helper
        ``is_feasible`` uses these to filter out impossible parameter combos.
        """
        return {
            "Boiler": [
                ("pressure", lambda v: v < 0),  # negative pressure impossible
                ("temperature", lambda v: v > 500),  # beyond design limit
            ],
            "Chiller": [
                ("coolant_flow", lambda v: v <= 0),
                ("evap_temp", lambda v: v > 30),
            ],
            "HVAC": [
                ("outdoor_temp", lambda v: v < -40),
                ("indoor_humidity", lambda v: v > 100),
            ],
            "Air Compressor": [
                ("inlet_pressure", lambda v: v <= 0),
                ("discharge_pressure", lambda v: v < 0),
            ],
            "Vacuum Machine": [
                ("vacuum_level", lambda v: v > 0),  # vacuum level should be negative
            ],
            "Power Distribution": [
                ("voltage", lambda v: v < 0),
                ("current", lambda v: v < 0),
            ],
            "Water Supply": [
                ("flow_rate", lambda v: v <= 0),
                ("pressure", lambda v: v < 0),
            ],
        }

    def is_feasible(self, equipment: str, params: Dict[str, Any]) -> bool:
        """Check whether *params* respect the physical constraints for *equipment*.
        """
        if equipment not in self.constraints:
            return True
        for param, test in self.constraints[equipment]:
            if param in params and test(params[param]):
                return False
        return True

    # ---------------------------------------------------------------------
    # Rule definitions
    # ---------------------------------------------------------------------
    def _build_rules(self) -> List[Rule]:
        """Create a list of diagnostic rules.

        The rules are illustrative – they cover typical fault patterns for the
        seven equipment categories. Each rule contains a confidence score that
        will later be refined by Bayesian updating.
        """
        rules: List[Rule] = []

        # Boiler rules
        rules.append(
            Rule(
                equipment="Boiler",
                conditions={"pressure": ("<", 30), "temperature": (">", 350)},
                fault="Boiler pressure drop leading to low efficiency",
                confidence=0.85,
                priority=2,
            )
        )
        rules.append(
            Rule(
                equipment="Boiler",
                conditions={"temperature": ("<", 150)},
                fault="Insufficient heating – possible burner failure",
                confidence=0.9,
                priority=1,
            )
        )

        # Chiller rules
        rules.append(
            Rule(
                equipment="Chiller",
                conditions={"evap_temp": (">", 20), "compressor_current": (">", 15)},
                fault="Evaporator fouling causing high evaporator temperature",
                confidence=0.88,
                priority=2,
            )
        )
        rules.append(
            Rule(
                equipment="Chiller",
                conditions={"coolant_flow": ("<", 10)},
                fault="Low coolant flow – possible pump blockage",
                confidence=0.9,
                priority=1,
            )
        )

        # HVAC rules
        rules.append(
            Rule(
                equipment="HVAC",
                conditions={"indoor_temp": (">", 28), "outdoor_temp": ("<", 5)},
                fault="HVAC unable to meet cooling demand – possible refrigerant leak",
                confidence=0.87,
                priority=2,
            )
        )
        rules.append(
            Rule(
                equipment="HVAC",
                conditions={"indoor_humidity": (">", 70)},
                fault="Excess humidity – possible condensate drain blockage",
                confidence=0.85,
                priority=1,
            )
        )

        # Air Compressor rules
        rules.append(
            Rule(
                equipment="Air Compressor",
                conditions={"discharge_pressure": ("<", 80), "inlet_pressure": ("<", 30)},
                fault="Compressor performance drop – possible valve wear",
                confidence=0.86,
                priority=2,
            )
        )
        rules.append(
            Rule(
                equipment="Air Compressor",
                conditions={"temperature": (">", 90)},
                fault="Overheating – possible lubrication issue",
                confidence=0.9,
                priority=1,
            )
        )

        # Vacuum Machine rules
        rules.append(
            Rule(
                equipment="Vacuum Machine",
                conditions={"vacuum_level": (">", -200)},
                fault="Insufficient vacuum – possible leak",
                confidence=0.88,
                priority=2,
            )
        )
        rules.append(
            Rule(
                equipment="Vacuum Machine",
                conditions={"motor_current": (">", 12)},
                fault="Motor overload – possible bearing wear",
                confidence=0.9,
                priority=1,
            )
        )

        # Power Distribution rules
        rules.append(
            Rule(
                equipment="Power Distribution",
                conditions={"voltage": ("<", 210), "current": (">", 150)},
                fault="Undervoltage under high load – possible transformer degradation",
                confidence=0.87,
                priority=2,
            )
        )
        rules.append(
            Rule(
                equipment="Power Distribution",
                conditions={"frequency": ("!=", 50)},
                fault="Frequency deviation – possible generator instability",
                confidence=0.85,
                priority=1,
            )
        )

        # Water Supply rules
        rules.append(
            Rule(
                equipment="Water Supply",
                conditions={"pressure": ("<", 30), "flow_rate": ("<", 5)},
                fault="Low pressure and flow – possible pipe blockage",
                confidence=0.88,
                priority=2,
            )
        )
        rules.append(
            Rule(
                equipment="Water Supply",
                conditions={"temperature": (">", 45)},
                fault="High water temperature – possible heat exchanger fouling",
                confidence=0.9,
                priority=1,
            )
        )

        return rules

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def get_rules(self, equipment: str) -> List[Rule]:
        """Return all rules applicable to *equipment*.
        """
        return [r for r in self.rules if r.equipment == equipment]

    def diagnose(self, equipment: str, params: Dict[str, Any]) -> List[Tuple[Rule, float]]:
        """Run the knowledge base against *params*.

        Returns a list of (Rule, confidence) tuples for all matching rules,
        sorted by priority then confidence.
        """
        if not self.is_feasible(equipment, params):
            return []
        matches = []
        for rule in self.get_rules(equipment):
            if rule.matches(params):
                matches.append((rule, rule.confidence))
        matches.sort(key=lambda x: (-x[0].priority, -x[1]))
        return matches


# Simple demo when run as script
if __name__ == "__main__":
    kb = KnowledgeBase()
    sample_params = {
        "pressure": 25,
        "temperature": 360,
    }
    results = kb.diagnose("Boiler", sample_params)
    for rule, conf in results:
        print(f"[Boiler] Fault: {rule.fault} (confidence={conf})")
