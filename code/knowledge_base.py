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
        analysis: Explanatory analytical text to guide human operators.
        confidence: Base confidence (0‑1) — the prior θ_i.
        priority: Integer priority (1 = high, 2 = middle, 3 = low).
    """
    equipment: str
    conditions: Dict[str, Tuple[str, Any]]
    fault: str
    analysis: str = ""
    confidence: float = 0.9
    priority: int = 1

    def _eval_condition(self, param: str, params: Dict[str, Any]) -> bool:
        """Evaluate a single condition against *params*."""
        if param not in params or param not in self.conditions:
            return False
        op, value = self.conditions[param]
        p = params[param]
        if op == '>':
            return p > value
        elif op == '>=':
            return p >= value
        elif op == '<':
            return p < value
        elif op == '<=':
            return p <= value
        elif op == '==':
            return p == value
        elif op == '!=':
            return p != value
        else:
            raise ValueError(f"Unsupported operator {op}")

    def matches(self, params: Dict[str, Any]) -> bool:
        """Return True if *params* satisfy ALL conditions (full match)."""
        for param in self.conditions:
            if not self._eval_condition(param, params):
                return False
        return True

    def count_matched(self, params: Dict[str, Any]) -> int:
        """Return the number of conditions satisfied by *params* (partial match)."""
        return sum(1 for param in self.conditions if self._eval_condition(param, params))


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
                ("pressure", lambda v: v < 0),
                ("temperature", lambda v: v > 500),
                ("gas_flow", lambda v: v < 0),
                ("steam_flow", lambda v: v < 0),
                ("water_level", lambda v: v < 0 or v > 100),
            ],
            "Chiller": [
                ("coolant_flow", lambda v: v <= 0),
                ("compressor_current", lambda v: v < 0),
                ("chilled_water_out_temp", lambda v: v < 0),
                ("cooling_water_in_temp", lambda v: v < 0),
            ],
            "HVAC": [
                ("indoor_humidity", lambda v: v > 100 or v < 0),
                ("fan_power", lambda v: v < 0),
                ("coil_delta_t", lambda v: v < 0),
            ],
            "Air Compressor": [
                ("inlet_pressure", lambda v: v <= 0),
                ("discharge_pressure", lambda v: v < 0),
                ("motor_current", lambda v: v < 0),
                ("gas_production", lambda v: v < 0),
            ],
            "Vacuum Machine": [
                ("vacuum_level", lambda v: v > 0),
                ("motor_current", lambda v: v < 0),
                ("power_consumption", lambda v: v < 0),
                ("production_rate", lambda v: v < 0),
            ],
            "Power Distribution": [
                ("voltage", lambda v: v < 0),
                ("current", lambda v: v < 0),
                ("power_factor", lambda v: v < 0 or v > 1.0),
                ("phase_imbalance", lambda v: v < 0 or v > 100),
            ],
            "Water Supply": [
                ("flow_rate", lambda v: v <= 0),
                ("pressure", lambda v: v < 0),
                ("pump_current", lambda v: v < 0),
                ("water_elec_ratio", lambda v: v < 0),
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

        The rules cover typical fault patterns for the seven equipment
        categories, derived from domain expert knowledge. Each rule contains
        a confidence score that will later be refined by Bayesian updating.
        """
        rules: List[Rule] = []

        # ---- Boiler rules (5) ----
        # B-F001: Combustion inefficiency – high gas consumption, low steam output
        rules.append(
            Rule(
                equipment="Boiler",
                conditions={
                    "gas_flow": (">", 350),
                    "steam_flow": ("<", 250),
                    "pressure": ("<", 40),
                },
                fault="Combustion inefficiency – abnormal air-fuel ratio or burner degradation",
                analysis="Fuel is not being efficiently converted to thermal energy. The system requires more gas to maintain setpoint steam temperature and pressure, indicating combustion efficiency deterioration typically linked to improper air-fuel ratio or burner performance decline.",
                confidence=0.85,
                priority=2,
            )
        )
        # B-F002: Heat exchanger fouling – normal gas flow but low outlet temperature
        rules.append(
            Rule(
                equipment="Boiler",
                conditions={
                    "gas_flow": (">", 200),
                    "temperature": ("<", 160),
                },
                fault="Heat exchanger fouling – scale or ash deposit on tube surfaces",
                analysis="Scale, fouling, or ash deposits on heat exchange surfaces form an insulating layer that severely impedes heat transfer from hot flue gas to boiler water, significantly reducing heat exchange efficiency.",
                confidence=0.88,
                priority=2,
            )
        )
        # B-F003: Leakage – water level dropping, steam output abnormally low
        rules.append(
            Rule(
                equipment="Boiler",
                conditions={
                    "water_level": ("<", 35),
                    "steam_flow": ("<", 200),
                },
                fault="Boiler or pipeline leakage – water-steam imbalance",
                analysis="Typical mass imbalance: water input significantly exceeds steam output, indicating unmetered fluid loss through abnormal pathways. The most common cause is leakage in the boiler body, valves, or piping.",
                confidence=0.90,
                priority=1,
            )
        )
        # B-F004: Steam quality degradation – pressure ok but temperature much lower
        rules.append(
            Rule(
                equipment="Boiler",
                conditions={
                    "pressure": ("<", 50),
                    "temperature": ("<", 150),
                },
                fault="Poor steam quality – possible priming or carryover",
                analysis="Incomplete steam-water separation causes liquid water droplets to be carried out by high-velocity steam. These droplets absorb latent heat during subsequent evaporation, causing a sharp drop in superheat.",
                confidence=0.82,
                priority=2,
            )
        )
        # B-S001: Sensor fault – contradictory readings
        rules.append(
            Rule(
                equipment="Boiler",
                conditions={
                    "steam_flow": (">", 400),
                    "water_level": (">", 75),
                    "temperature": ("<", 120),
                },
                fault="Sensor or instrument fault – physically contradictory readings",
                analysis="The parameter combination violates fundamental physics (mass conservation). Continuous steam production without water input while maintaining water level is impossible, strongly indicating at least one instrument has failed.",
                confidence=0.92,
                priority=1,
            )
        )

        # ---- Chiller rules (4) ----
        # CH-R001: Condenser fouling – high compressor current, low cooling water delta-T
        rules.append(
            Rule(
                equipment="Chiller",
                conditions={
                    "compressor_current": (">", 18),
                    "coolant_flow": (">", 12),
                    "cooling_water_in_temp": ("<", 32),
                },
                fault="Condenser fouling – scale or biofilm reducing heat transfer",
                analysis="High probability of degraded condenser heat exchange efficiency due to scale or biofilm accumulation on condenser tubes.",
                confidence=0.88,
                priority=1,
            )
        )
        # CH-R002: Cooling tower fault – high cooling water inlet temperature
        rules.append(
            Rule(
                equipment="Chiller",
                conditions={
                    "compressor_current": (">", 20),
                    "cooling_water_in_temp": (">", 35),
                },
                fault="Cooling tower malfunction – fan failure or fill fouling",
                analysis="The cooling system cannot effectively dissipate heat due to cooling tower fan failure, fill fouling, or water distributor blockage.",
                confidence=0.87,
                priority=1,
            )
        )
        # CH-R006: Refrigerant leak – low compressor current, high chilled water outlet
        rules.append(
            Rule(
                equipment="Chiller",
                conditions={
                    "compressor_current": ("<", 8),
                    "chilled_water_out_temp": (">", 12),
                },
                fault="Refrigerant leak or undercharge – reduced cooling capacity",
                analysis="Insufficient refrigerant charge results in reduced cooling capacity and lower compressor load, indicated by low current draw combined with elevated chilled water outlet temperature.",
                confidence=0.90,
                priority=2,
            )
        )
        # Low coolant flow – possible pump blockage
        rules.append(
            Rule(
                equipment="Chiller",
                conditions={"coolant_flow": ("<", 8)},
                fault="Low coolant flow – possible pump blockage or valve closure",
                analysis="Coolant flow rate has dropped below operational threshold, likely caused by chilled water pump blockage, valve malfunction, or pipe obstruction.",
                confidence=0.90,
                priority=1,
            )
        )

        # ---- HVAC rules (5) ----
        # AC-F001: Filter blockage – high fan power, high supply temperature
        rules.append(
            Rule(
                equipment="HVAC",
                conditions={
                    "fan_power": (">", 3.5),
                    "supply_temp": (">", 22),
                },
                fault="Air filter blockage – increased airway resistance",
                analysis="Fan power consistently above baseline, typically accompanied by elevated supply temperature due to severely blocked air filters increasing duct resistance.",
                confidence=0.87,
                priority=1,
            )
        )
        # AC-F002: Fan belt slip – low fan power, low airflow (modeled via temp)
        rules.append(
            Rule(
                equipment="HVAC",
                conditions={
                    "fan_power": ("<", 1.0),
                    "supply_temp": (">", 25),
                },
                fault="Fan belt slip or breakage – loss of air delivery",
                analysis="Fan power far below baseline with extremely low airflow, typically caused by severe belt slippage or breakage preventing effective impeller drive.",
                confidence=0.90,
                priority=1,
            )
        )
        # AC-A001: Cooling coil fouling – high supply temp, low water delta-T
        rules.append(
            Rule(
                equipment="HVAC",
                conditions={
                    "supply_temp": (">", 24),
                    "coil_delta_t": ("<", 3),
                },
                fault="Cooling coil fouling – reduced heat exchange efficiency",
                analysis="Post-coil air temperature is elevated while chilled water temperature differential is below design value, indicating poor air-to-coil heat exchange due to fin surface contamination.",
                confidence=0.86,
                priority=1,
            )
        )
        # AC-W001: Cold source failure – indoor temp rising
        rules.append(
            Rule(
                equipment="HVAC",
                conditions={
                    "indoor_temp": (">", 28),
                    "supply_temp": (">", 26),
                },
                fault="Cold source failure – chiller or chilled water system fault",
                analysis="Chilled water supply temperature remains above setpoint, indicating a fundamental upstream problem with the chiller plant or chilled water distribution system.",
                confidence=0.88,
                priority=1,
            )
        )
        # Excess humidity – condensate drain blockage
        rules.append(
            Rule(
                equipment="HVAC",
                conditions={"indoor_humidity": (">", 70)},
                fault="Excess humidity – possible condensate drain blockage",
                analysis="Indoor humidity exceeds acceptable range, potentially caused by blocked condensate drain lines preventing proper moisture removal from the air handling unit.",
                confidence=0.85,
                priority=2,
            )
        )

        # ---- Air Compressor rules (4) ----
        # COMP-F01: Discharge overheating – cooler fouling or lubrication issue
        rules.append(
            Rule(
                equipment="Air Compressor",
                conditions={
                    "temperature": (">", 90),
                    "discharge_pressure": (">", 100),
                },
                fault="Discharge overheating – cooler fouling or lubrication degradation",
                analysis="Possible causes: cooler fin blockage, insufficient coolant flow, degraded lubricant or blocked oil lines, main bearing wear causing excessive friction, or poor ventilation in the compressor room.",
                confidence=0.88,
                priority=1,
            )
        )
        # COMP-F03: Leakage or internal wear – low gas output despite high current
        rules.append(
            Rule(
                equipment="Air Compressor",
                conditions={
                    "gas_production": ("<", 5),
                    "discharge_pressure": ("<", 90),
                    "motor_current": (">", 35),
                },
                fault="Internal or pipeline leakage – low gas-to-electricity ratio",
                analysis="Possible causes: downstream piping leakage, internal valve or minimum-pressure valve leakage, or compressor element wear-induced internal leakage.",
                confidence=0.86,
                priority=2,
            )
        )
        # Valve wear – low pressures on both sides
        rules.append(
            Rule(
                equipment="Air Compressor",
                conditions={
                    "discharge_pressure": ("<", 85),
                    "inlet_pressure": ("<", 35),
                },
                fault="Compressor valve wear – insufficient compression ratio",
                analysis="Inlet and discharge pressures both below normal, indicating compressor valve deterioration resulting in inability to achieve the required compression ratio.",
                confidence=0.84,
                priority=2,
            )
        )
        # Motor overload – high current with normal pressure
        rules.append(
            Rule(
                equipment="Air Compressor",
                conditions={
                    "motor_current": (">", 40),
                    "temperature": (">", 85),
                },
                fault="Motor overload – possible bearing wear or mechanical friction",
                analysis="Motor current significantly above rated value accompanied by elevated temperature, likely caused by bearing wear, mechanical friction, or overloading conditions.",
                confidence=0.90,
                priority=1,
            )
        )

        # ---- Vacuum Machine rules (4) ----
        # VM_001: Power unit anomaly – high power consumption, normal production
        rules.append(
            Rule(
                equipment="Vacuum Machine",
                conditions={
                    "power_consumption": (">", 30),
                    "production_rate": (">", 40),
                },
                fault="Power unit energy anomaly – motor or impeller degradation",
                analysis="Power consumption of one unit significantly exceeds normal baseline. Possible causes: motor bearing wear, winding aging, impeller damage, or local pipe blockage/leakage in the associated ductwork.",
                confidence=0.87,
                priority=1,
            )
        )
        # VM_005: Idle running – low production but power still consuming
        rules.append(
            Rule(
                equipment="Vacuum Machine",
                conditions={
                    "production_rate": ("<", 15),
                    "power_consumption": (">", 15),
                },
                fault="Idle running or production sensor fault – interlock logic issue",
                analysis="Production count growth rate has dropped significantly or is zero while power consumption continues. The vacuum machine may not have stopped in sync with production line shutdown (interlock logic issue), or the production counter sensor has failed.",
                confidence=0.85,
                priority=2,
            )
        )
        # Vacuum leak – insufficient vacuum level
        rules.append(
            Rule(
                equipment="Vacuum Machine",
                conditions={
                    "vacuum_level": (">", -200),
                    "motor_current": (">", 8),
                },
                fault="Insufficient vacuum – possible system leak",
                analysis="Vacuum level is insufficient despite normal motor current, indicating a possible air leak in the vacuum system, filter blockage, or seal degradation.",
                confidence=0.88,
                priority=2,
            )
        )
        # Motor overload – high current, possible bearing wear
        rules.append(
            Rule(
                equipment="Vacuum Machine",
                conditions={"motor_current": (">", 12)},
                fault="Motor overload – possible bearing wear",
                analysis="Motor current exceeds rated value, likely caused by bearing wear, shaft misalignment, or excessive mechanical load.",
                confidence=0.90,
                priority=1,
            )
        )

        # ---- Power Distribution rules (4) ----
        # PD_001: Three-phase current imbalance
        rules.append(
            Rule(
                equipment="Power Distribution",
                conditions={
                    "phase_imbalance": (">", 15),
                    "current": (">", 50),
                },
                fault="Severe three-phase current imbalance – possible winding fault",
                analysis="Uneven load distribution across phases, or a downstream device fault such as motor winding inter-turn short circuit or insulation aging on the heavily loaded phase.",
                confidence=0.87,
                priority=2,
            )
        )
        # PD_002: Phase loss – extremely low current (near zero on one phase)
        rules.append(
            Rule(
                equipment="Power Distribution",
                conditions={
                    "current": ("<", 5),
                    "voltage": (">", 200),
                },
                fault="Phase loss or open circuit – fuse blown or breaker tripped",
                analysis="The affected phase has a blown fuse, tripped breaker, or contactor failure. Alternatively, a cable or terminal may be disconnected, or the upstream supply is missing a phase.",
                confidence=0.92,
                priority=1,
            )
        )
        # Undervoltage under high load
        rules.append(
            Rule(
                equipment="Power Distribution",
                conditions={
                    "voltage": ("<", 210),
                    "current": (">", 150),
                },
                fault="Undervoltage under high load – possible transformer degradation",
                analysis="Voltage drop under high-load conditions suggests transformer capacity limitations, cable resistance increase, or upstream power supply quality issues.",
                confidence=0.87,
                priority=2,
            )
        )
        # Low power factor – reactive power issue
        rules.append(
            Rule(
                equipment="Power Distribution",
                conditions={"power_factor": ("<", 0.75)},
                fault="Low power factor – excessive reactive power consumption",
                analysis="Low power factor indicates excessive reactive power draw, typically caused by inductive loads (motors, transformers) without adequate capacitor compensation.",
                confidence=0.84,
                priority=2,
            )
        )

        # ---- Water Supply rules (4) ----
        # WS_001: Pump efficiency drop – low flow, high pump current
        rules.append(
            Rule(
                equipment="Water Supply",
                conditions={
                    "flow_rate": ("<", 5),
                    "pump_current": (">", 20),
                    "water_elec_ratio": ("<", 1.5),
                },
                fault="Pump efficiency drop – impeller wear or cavitation",
                analysis="Water output is declining while power consumption remains unchanged or increases. Possible causes: pump impeller wear, cavitation, pipe blockage, upstream leakage, or flow meter inaccuracy.",
                confidence=0.88,
                priority=1,
            )
        )
        # Pipe blockage or leak – low pressure and flow
        rules.append(
            Rule(
                equipment="Water Supply",
                conditions={
                    "pressure": ("<", 30),
                    "flow_rate": ("<", 5),
                },
                fault="Pipe blockage or upstream leak – low pressure and flow",
                analysis="Both pressure and flow rate are below normal, indicating pipe system obstruction (filter blockage, valve not fully open) or a leak upstream of the flow meter.",
                confidence=0.86,
                priority=2,
            )
        )
        # WS_003: Power supply three-phase imbalance
        rules.append(
            Rule(
                equipment="Water Supply",
                conditions={
                    "pump_current": (">", 22),
                    "water_elec_ratio": ("<", 1.0),
                },
                fault="Power supply phase imbalance – motor efficiency degradation",
                analysis="Motor efficiency drops sharply with increased heating and losses due to phase imbalance. Possible causes: power grid quality issues, switchgear faults, motor winding inter-turn short circuit, or loose cable terminals.",
                confidence=0.85,
                priority=1,
            )
        )
        # High water temperature
        rules.append(
            Rule(
                equipment="Water Supply",
                conditions={"temperature": (">", 45)},
                fault="High water temperature – possible heat exchanger fouling",
                analysis="Water temperature exceeds operational limits, potentially caused by heat exchanger fouling, insufficient cooling capacity, or external heat source contamination.",
                confidence=0.90,
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
    print(f"Total rules: {len(kb.rules)}")
    for eq in kb.equipment_types:
        print(f"  {eq}: {len(kb.get_rules(eq))} rules")
    print()
    sample_params = {
        "pressure": 25,
        "temperature": 140,
        "gas_flow": 380,
        "steam_flow": 210,
        "water_level": 55,
    }
    results = kb.diagnose("Boiler", sample_params)
    for rule, conf in results:
        print(f"[Boiler] Fault: {rule.fault} (confidence={conf})")

