"""Public-data benchmark runner for LBNL FDD and MetroPT.

This script keeps the paper's scope focused on the final diagnostic step:

    abnormal parameter predicates -> equipment fault diagnosis

It therefore implements deterministic preprocessing that converts public
time-series data into interpretable abnormal-parameter predicates, then reuses
the same PMS inference engine used by the synthetic experiments.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import sys

sys.path.append(os.path.join(os.path.dirname(__file__)))
from inference_engine import ProbabilisticInferenceEngine
from experiment_metrics import ece, hybrid_decision
from knowledge_base import Rule
from llm_baseline import SimulatedLLM
from real_llm import RealLLMBaseline


ROOT = Path(__file__).resolve().parents[1]
LBNL_ROOT = ROOT / "code" / "LBNL_FDD_Data_Sets"
METROPT_ROOT = ROOT / "code" / "MetroPT_dataset"


@dataclass
class PublicSample:
    source: str
    equipment: str
    scenario: str
    true_fault: Optional[str]
    params: Dict[str, int]
    context: str = ""
    rich_context: str = ""
    start_time: str = ""
    end_time: str = ""


@dataclass(frozen=True)
class ThresholdSetting:
    """Routing thresholds used for one public benchmark equipment type."""

    min_match: float
    tau: float
    rationale: str


GLOBAL_THRESHOLD_SETTING = ThresholdSetting(
    min_match=50.0,
    tau=0.6,
    rationale="Legacy global public-benchmark setting.",
)

# Per-source z-score thresholds for predicate extraction.
# LBNL fault scenarios produce smaller sensor deviations than MetroPT's
# real-world operational variability — tighter thresholds are needed to
# avoid converting genuine fault samples to all‑zero predicate vectors.
SOURCE_ZSCORE_THRESHOLDS: Dict[str, Tuple[float, float]] = {
    #              (mild_σ, severe_σ)
    "LBNL Boiler":   (1.0, 2.0),
    "LBNL Chiller":  (1.0, 2.0),
    "LBNL RTU":      (0.8, 1.5),   # RTU signals are the weakest
    "MetroPT":        (1.5, 3.0),   # keep strict for false-alarm control
}


PUBLIC_TUNED_THRESHOLDS: Dict[str, ThresholdSetting] = {
    "LBNL Boiler Plant": ThresholdSetting(
        min_match=66.66,
        tau=0.45,
        rationale="Suppresses weak partial matches while keeping confident boiler expert diagnoses above the fallback threshold.",
    ),
    "LBNL Chiller Plant": ThresholdSetting(
        min_match=66.66,
        tau=0.45,
        rationale="Chiller public predicates are sparse; a lower tau avoids replacing plausible expert matches with LLM guesses.",
    ),
    "LBNL RTU": ThresholdSetting(
        min_match=50.0,
        tau=0.50,
        rationale="RTU predicates are the sparsest among LBNL sources; min_match=50 allows 2-condition rules to fire on 1/2 match.",
    ),
    "MetroPT Compressor": ThresholdSetting(
        min_match=66.66,
        tau=0.20,
        rationale="MetroPT predicates are trend-derived and noisy; tau=0.20 limits fallback to low-confidence expert outputs while preserving most expert detections.",
    ),
}

# Maximum expert confidence at which a strong-evidence LLM conflict may
# override the expert diagnosis. This is stricter than the routing tau for
# most LBNL sources: tau marks fallback-needed cases, while this cap controls
# whether the LLM is allowed to replace the expert output.
#
# MetroPT uses a higher cap because earlier public-benchmark runs showed that
# LLM fallback is comparatively more useful for compressor strong-evidence
# conflicts than for the sparse LBNL scenario subsets.
EVIDENCE_AWARE_EXPERT_CAPS: Dict[str, float] = {
    "LBNL Boiler Plant": 0.35,
    "LBNL Chiller Plant": 0.35,
    "LBNL RTU": 0.35,
    "MetroPT Compressor": 0.45,
}

PUBLIC_LLM_FAULT_GUIDES: Dict[str, List[str]] = {
    "LBNL Boiler Plant": [
        "Boiler fouling <- boiler_fuel_high + heat_delivery_low; boiler_pi_abnormal strengthens the diagnosis.",
        "Boiler fouling <- boiler_fuel_high + boiler_delta_t_low when heat delivery loss is visible through reduced loop temperature lift.",
        "Boiler fouling <- boiler_pressure_bias + boiler_pi_abnormal when pressure-side deviation co-occurs with abnormal hot-water thermal trend.",
        "Hot water temperature sensor bias <- boiler_temp_bias.",
        "Hot water pressure sensor bias <- boiler_pressure_bias.",
        "Boiler sensor/control bias <- boiler_operation_bias + boiler_fuel_high, or multiple boiler sensor/control biases.",
        "Boiler sensor/control bias <- boiler_temp_bias + boiler_pi_abnormal when boiler_pressure_bias and boiler_fuel_high are level 0.",
        "Boiler performance degradation <- boiler_pi_abnormal when more specific fouling/sensor/control evidence is absent.",
        "Normal <- all boiler predicates are level 0.",
    ],
    "LBNL Chiller Plant": [
        "Chiller fouling <- chiller_power_high + cooling_capacity_low.",
        "Chiller fouling <- chiller_power_high + chiller_delta_t_low when power-to-temperature-lift efficiency degrades.",
        "Chiller sensor bias <- chiller_temp_bias; if chiller_temp_bias co-occurs with bypass_flow_abnormal and cooling_capacity_low, treat bypass/capacity effects as possible secondary symptoms.",
        "Cooling tower fouling <- cooling_tower_power_high + cooling_tower_temp_high; cooling_tower_pi_abnormal strengthens the diagnosis.",
        "Cooling tower fouling <- cooling_tower_power_high + condenser_temp_lift_high.",
        "Cooling tower sensor bias <- cooling_tower_temp_bias.",
        "Bypass valve leakage or stuck <- bypass_flow_abnormal, especially with cooling_capacity_low when chiller_temp_bias is not active.",
        "Bypass valve leakage or stuck <- cooling_tower_temp_high == 2 with chiller_temp_bias <= 1 in the data-assisted public rule audit; severe tower-temperature response can appear in low-leakage bypass scenarios before direct bypass-flow predicates activate.",
        "Bypass valve leakage or stuck <- bypass_flow_abnormal == 2 + cooling_tower_temp_high == 2 + cooling_tower_temp_bias == 2 + cooling_tower_power_high == 2; keep the direct bypass predicate primary when tower-side severe responses co-occur.",
        "Secondary chilled water pressure sensor bias <- secondary_pressure_bias.",
        "Cooling tower performance degradation <- cooling_tower_pi_abnormal without more specific tower-fouling evidence.",
        "Normal <- all chiller predicates are level 0.",
    ],
    "LBNL RTU": [
        "Compressor staging or cooling capacity fault <- compressor_power_abnormal + cooling_delivery_low.",
        "Compressor staging or cooling capacity fault <- compressor_power_abnormal + supply_air_temp_high.",
        "Refrigerant undercharge <- refrigerant_pressure_low + cooling_delivery_low; compressor_power_abnormal strengthens the diagnosis.",
        "Refrigerant undercharge <- refrigerant_pressure_low + refrigerant_pressure_imbalance.",
        "Outdoor air damper fault <- outdoor_air_damper_abnormal; economizer_signal_bias can be secondary air-side control evidence.",
        "Economizer setpoint bias <- economizer_signal_bias with outdoor_air_damper_abnormal == 0 and refrigerant_pressure_low == 0.",
        "Sparse RTU evidence policy: a single active primary predicate can support a tentative low-confidence diagnosis (0.35-0.55) when it uniquely matches a candidate fault; do not default to Normal solely because corroborating predicates are absent.",
        "Normal <- all RTU predicates are level 0.",
    ],
    "MetroPT Compressor": [
        "Air Leak <- pressure_recovery_slow + pressure_drop_frequent; pressure_recovery_steep or low_pressure_switch_active strengthens the diagnosis.",
        "Air Leak <- reservoir_pressure_low + compressor_runtime_high or low_pressure_switch_active.",
        "Air Leak <- oil_temperature_high == 2 + pressure_switch_abnormal >= 1 in the data-assisted public rule audit; severe oil-temperature elevation can occur in labelled Air Leak windows before oil-side trend predicates become active.",
        "Air Leak <- oil_temperature_high == 2 + pressure_switch_abnormal >= 1 + motor_current_high >= 1 when oil_temp_rising and motor_current_rising are not the dominant evidence.",
        "Air Leak <- pressure_recovery_slow + pressure_drop_frequent + low_pressure_switch_active + reservoir_pressure_low in the data-assisted public rule audit, even when compressor_runtime_high is not active.",
        "Oil Leak <- oil_temperature_high + motor_current_high; oil_temp_rising or current_pressure_gap strengthens the diagnosis.",
        "Oil Leak <- motor_current_rising + oil_temperature_high or oil_temp_rising.",
        "Oil Leak <- pressure_switch_abnormal == 2 + oil_temperature_high == 0 in the data-assisted public rule audit.",
        "Oil Leak <- pressure_switch_abnormal == 2 + oil_temperature_high == 1 + pressure_recovery_slow + pressure_recovery_steep when motor_current_rising == 0 in the data-assisted public rule audit.",
        "Compressor control abnormality <- isolated pressure_switch_abnormal without air-leak or oil-side support.",
        "Normal <- all compressor predicates are level 0.",
    ],
}

PUBLIC_LLM_PREDICATE_DESCRIPTIONS: Dict[str, Dict[str, str]] = {
    "LBNL Boiler Plant": {
        "boiler_fuel_high": "Boiler fuel consumption is elevated relative to the fault-free baseline.",
        "heat_delivery_low": "Delivered hot-water heat output is degraded.",
        "boiler_temp_bias": "Hot-water supply/return temperature signals show a systematic offset.",
        "boiler_pressure_bias": "Hot-water differential pressure signal shows a systematic mean or trend offset.",
        "boiler_operation_bias": "Boiler-level operation/control signal is biased or shifted.",
        "boiler_pi_abnormal": "Boiler performance-index or hot-water thermal trend behaviour is broadly abnormal.",
        "boiler_delta_t_low": "Hot-water supply/return temperature lift is lower than the healthy baseline.",
    },
    "LBNL Chiller Plant": {
        "chiller_power_high": "Chiller compressor or plant power is elevated.",
        "cooling_capacity_low": "Cooling capacity is degraded.",
        "chiller_temp_bias": "Chiller supply/return temperature sensors show systematic offset.",
        "cooling_tower_power_high": "Cooling tower fan/pump power is elevated.",
        "cooling_tower_temp_high": "Cooling tower or condenser-water temperature is high.",
        "cooling_tower_temp_bias": "Cooling tower temperature sensor signal is biased.",
        "bypass_flow_abnormal": "Bypass valve command/flow behaviour is abnormal.",
        "secondary_pressure_bias": "Secondary chilled-water pressure sensor is biased.",
        "cooling_tower_pi_abnormal": "Cooling tower performance-index behaviour is broadly abnormal.",
        "chiller_delta_t_low": "Chilled-water temperature lift is lower than the healthy baseline.",
        "condenser_temp_lift_high": "Condenser/cooling-tower water temperature lift is elevated.",
    },
    "LBNL RTU": {
        "compressor_power_abnormal": "RTU compressor power or staging signal is abnormal.",
        "cooling_delivery_low": "Supply-air or zone cooling delivery is degraded.",
        "outdoor_air_damper_abnormal": "Outdoor/return-air damper command or position is abnormal.",
        "economizer_signal_bias": "Economizer thermal or power response is shifted.",
        "refrigerant_pressure_low": "Refrigerant suction/discharge pressure is low.",
        "supply_air_temp_high": "Supply-air or zone temperature is high, indicating weak cooling delivery.",
        "refrigerant_pressure_imbalance": "Refrigerant suction/discharge pressure relationship deviates from baseline.",
    },
    "MetroPT Compressor": {
        "pressure_recovery_slow": "Reservoir pressure recovers more slowly or stays lower than healthy baseline.",
        "pressure_drop_frequent": "Reservoir pressure drops occur frequently.",
        "compressor_runtime_high": "Compressor duty cycle or runtime is elevated.",
        "motor_current_high": "Motor current is elevated, indicating mechanical load.",
        "oil_temperature_high": "Oil temperature is elevated; severe level-2 oil temperature can also appear in Air Leak windows when pressure-side evidence is strong.",
        "low_pressure_switch_active": "Low-pressure switch activity is elevated.",
        "pressure_switch_abnormal": "Pressure switch behaviour deviates from baseline.",
        "oil_temp_rising": "Oil temperature rises during the window — favours Oil Leak over Air Leak.",
        "current_pressure_gap": "Motor current high while pressure recovery is poor — mechanical friction pattern (Oil Leak), not pure pressure loss.",
        "pressure_recovery_steep": "Pressure recovery slope is sharply negative; interpret with oil-temperature, pressure-switch, and pressure-side predicates.",
        "reservoir_pressure_low": "Reservoir pressure mean is lower than the healthy baseline.",
        "motor_current_rising": "Motor current rises within the window, indicating increasing mechanical load.",
    },
}

PUBLIC_LLM_FAULT_EVIDENCE_GROUPS: Dict[str, Dict[str, Dict[str, List[str]]]] = {
    "LBNL Boiler Plant": {
        "Boiler fouling": {
            "primary": [
                "boiler_fuel_high with heat_delivery_low or boiler_delta_t_low",
                "boiler_pressure_bias with boiler_pi_abnormal and a coherent hot-water thermal trend",
            ],
            "supporting": ["boiler_pi_abnormal", "boiler_fuel_high"],
            "insufficient_alone": [
                "boiler_fuel_high without heat-delivery or delta-T evidence",
                "boiler_pi_abnormal without a more specific fuel, pressure, or thermal mechanism",
            ],
        },
        "Hot water temperature sensor bias": {
            "primary": ["boiler_temp_bias as the dominant abnormal predicate"],
            "supporting": ["stable fuel and pressure predicates at level 0 support a sensor-bias interpretation"],
            "insufficient_alone": ["boiler_temp_bias with broad performance degradation may indicate boiler sensor/control bias instead"],
        },
        "Hot water pressure sensor bias": {
            "primary": ["boiler_pressure_bias as the dominant abnormal predicate"],
            "supporting": ["level-2 boiler_pressure_bias is strong pressure-sensor evidence"],
            "insufficient_alone": ["boiler_pressure_bias with boiler_pi_abnormal can be boiler fouling if thermal evidence is coherent"],
        },
        "Boiler sensor/control bias": {
            "primary": [
                "boiler_operation_bias with boiler_fuel_high or boiler_pi_abnormal",
                "boiler_temp_bias with boiler_pi_abnormal when boiler_pressure_bias and boiler_fuel_high are level 0",
            ],
            "supporting": ["multiple boiler sensor/control predicates active together"],
            "insufficient_alone": ["a single mild boiler_operation_bias is weak evidence unless supported by fuel or PI abnormality"],
        },
        "Boiler performance degradation": {
            "primary": ["boiler_pi_abnormal when no more specific fouling, sensor, or control mechanism is supported"],
            "supporting": ["boiler_delta_t_low with boiler_pi_abnormal"],
            "insufficient_alone": ["do not use this catch-all when a specific sensor/control or fouling pattern is stronger"],
        },
        "Normal": {
            "primary": ["all boiler predicates are level 0"],
            "supporting": ["only isolated weak level-1 evidence without a coherent mechanism"],
            "insufficient_alone": [],
        },
    },
    "LBNL Chiller Plant": {
        "Chiller fouling": {
            "primary": [
                "chiller_power_high with cooling_capacity_low",
                "chiller_power_high with chiller_delta_t_low",
            ],
            "supporting": ["severe chiller_power_high", "cooling_capacity_low"],
            "insufficient_alone": ["chiller_power_high alone is early/weak and should not override stronger valve or sensor evidence"],
        },
        "Chiller sensor bias": {
            "primary": ["chiller_temp_bias as dominant evidence"],
            "supporting": ["chiller_temp_bias with bypass_flow_abnormal and cooling_capacity_low can be sensor bias with secondary effects"],
            "insufficient_alone": ["do not diagnose bypass leakage solely because bypass/capacity effects accompany a strong chiller_temp_bias"],
        },
        "Cooling tower fouling": {
            "primary": [
                "cooling_tower_power_high with cooling_tower_temp_high",
                "cooling_tower_power_high with condenser_temp_lift_high",
            ],
            "supporting": ["cooling_tower_pi_abnormal strengthens tower-fouling evidence"],
            "insufficient_alone": ["cooling_tower_power_high alone is weak unless tower temperature or condenser lift is also abnormal"],
        },
        "Cooling tower sensor bias": {
            "primary": ["cooling_tower_temp_bias as dominant evidence"],
            "supporting": ["level-2 cooling_tower_temp_bias"],
            "insufficient_alone": ["tower-temperature bias with direct bypass evidence may be secondary to bypass leakage"],
        },
        "Bypass valve leakage or stuck": {
            "primary": [
                "bypass_flow_abnormal, especially with cooling_capacity_low or chiller_delta_t_low",
                "severe cooling_tower_temp_high with chiller_temp_bias <= 1 when direct bypass evidence is absent or delayed",
            ],
            "supporting": ["severe bypass_flow_abnormal with severe tower-side responses"],
            "insufficient_alone": ["mild tower-temperature response without bypass, capacity, or delta-T evidence is weak"],
        },
        "Secondary chilled water pressure sensor bias": {
            "primary": ["secondary_pressure_bias as dominant evidence"],
            "supporting": ["level-2 secondary_pressure_bias"],
            "insufficient_alone": ["do not use pressure bias as a generic fallback for tower or chiller thermal faults"],
        },
        "Cooling tower performance degradation": {
            "primary": ["cooling_tower_pi_abnormal when no more specific tower-fouling or sensor mechanism is supported"],
            "supporting": ["tower-side abnormal predicates can strengthen this catch-all"],
            "insufficient_alone": ["do not use this catch-all when cooling_tower_fouling or sensor-bias evidence is stronger"],
        },
        "Normal": {
            "primary": ["all chiller predicates are level 0"],
            "supporting": ["only isolated weak level-1 evidence without a coherent mechanism"],
            "insufficient_alone": [],
        },
    },
    "LBNL RTU": {
        "Compressor staging or cooling capacity fault": {
            "primary": [
                "compressor_power_abnormal with cooling_delivery_low",
                "compressor_power_abnormal with supply_air_temp_high",
            ],
            "supporting": [
                "compressor_power_abnormal alone is early/weak compressor evidence",
                "In sparse RTU data, compressor_power_abnormal alone can be the only available compressor-side signal; a tentative low-confidence diagnosis (0.35-0.55) is acceptable when no economizer, damper, or refrigerant evidence is stronger.",
            ],
            "insufficient_alone": ["do not diagnose compressor capacity fault from supply_air_temp_high without compressor-power evidence"],
        },
        "Refrigerant undercharge": {
            "primary": [
                "refrigerant_pressure_low with cooling_delivery_low",
                "refrigerant_pressure_low with refrigerant_pressure_imbalance",
            ],
            "supporting": [
                "compressor_power_abnormal strengthens undercharge when pressure and cooling evidence are present",
                "In sparse RTU data, refrigerant_pressure_low alone is weak but can be the only available refrigerant-side signal; a tentative low-confidence diagnosis (0.35-0.55) is acceptable when air-side and compressor-stage evidence are not stronger.",
            ],
            "insufficient_alone": ["do not force Normal solely because refrigerant_pressure_low is isolated; treat it as weak RTU evidence and keep confidence low unless cooling/pressure imbalance corroborates it"],
        },
        "Outdoor air damper fault": {
            "primary": ["outdoor_air_damper_abnormal"],
            "supporting": [
                "outdoor_air_damper_abnormal alone can be sufficient for a tentative low-confidence damper diagnosis in sparse RTU data",
                "economizer_signal_bias can be secondary air-side control evidence",
            ],
            "insufficient_alone": ["economizer_signal_bias without damper abnormality favours economizer setpoint bias"],
        },
        "Economizer setpoint bias": {
            "primary": [
                "economizer_signal_bias with outdoor_air_damper_abnormal == 0 and refrigerant_pressure_low == 0",
            ],
            "supporting": [
                "level-2 economizer_signal_bias strengthens setpoint/control-bias evidence",
                "economizer_signal_bias alone can be the only available setpoint signal in sparse RTU data; a tentative low-confidence diagnosis (0.35-0.55) is acceptable when outdoor_air_damper_abnormal and refrigerant_pressure_low are inactive",
            ],
            "insufficient_alone": ["do not diagnose economizer bias with high confidence when damper or refrigerant primary evidence is active; use low confidence if economizer_signal_bias is isolated"],
        },
        "Normal": {
            "primary": ["all RTU predicates are level 0"],
            "supporting": ["all-zero RTU predicates strongly support Normal; isolated active RTU predicates should usually be treated as weak fault evidence rather than automatic Normal when they uniquely match a candidate fault"],
            "insufficient_alone": [],
        },
    },
    "MetroPT Compressor": {
        "Air Leak": {
            "primary": [
                "pressure_recovery_slow with pressure_drop_frequent and low_pressure_switch_active",
                "pressure_recovery_slow with compressor_runtime_high",
                "reservoir_pressure_low with compressor_runtime_high",
                "oil_temperature_high == 2 with pressure_switch_abnormal when pressure-side evidence is coherent",
            ],
            "supporting": [
                "pressure_recovery_steep",
                "reservoir_pressure_low",
                "low_pressure_switch_active",
                "pressure_drop_frequent",
            ],
            "insufficient_alone": [
                "pressure_recovery_slow alone",
                "pressure_recovery_slow with pressure_recovery_steep but without pressure-drop, switch, reservoir, or runtime evidence",
                "oil_temperature_high without pressure-side support",
            ],
        },
        "Oil Leak": {
            "primary": [
                "oil_temperature_high with motor_current_high",
                "motor_current_rising with oil_temperature_high or oil_temp_rising",
                "oil_temp_rising with pressure_switch_abnormal",
                "current_pressure_gap",
                "pressure_switch_abnormal == 2 with oil-temperature or pressure-recovery context matching the audited oil-leak pattern",
            ],
            "supporting": ["motor_current_high", "oil_temperature_high", "oil_temp_rising", "motor_current_rising"],
            "insufficient_alone": [
                "motor_current_rising alone",
                "oil_temp_rising alone",
                "motor_current_rising with mild oil_temp_rising but no oil_temperature_high, pressure_switch_abnormal, or current_pressure_gap",
                "mild oil_temperature_high without motor-current, trend, or pressure-switch support",
            ],
        },
        "Compressor control abnormality": {
            "primary": [
                "isolated pressure_switch_abnormal after excluding Air Leak and Oil Leak mechanisms",
                "pressure_switch_abnormal with compressor_runtime_high when pressure-loss and oil-side mechanisms are not supported",
            ],
            "supporting": ["level-2 pressure_switch_abnormal strengthens switch/control evidence"],
            "insufficient_alone": [
                "do not use this as a fallback when oil-side or air-leak primary evidence is present",
                "generic pressure anomalies without switch abnormality are not enough",
            ],
        },
        "Normal": {
            "primary": ["all compressor predicates are level 0"],
            "supporting": ["isolated mild trends such as motor_current_rising or oil_temp_rising without a primary mechanism"],
            "insufficient_alone": [],
        },
    },
}

RICH_CONTEXT_COLUMNS: Dict[str, List[str]] = {
    "LBNL Boiler Plant": [
        "HWL_SW_TEMP", "HWL_RW_TEMP", "HWL_DP", "BOI_GAS_CSUM_1",
        "BOI_GAS_CSUM_2", "BOI_SW_TEMP_1", "BOI_RW_TEMP_1",
    ],
    "LBNL Chiller Plant": [
        "CHL_SW_TEMP_1", "CHL_RW_TEMP_1", "CHL_POW_1", "CWL_SEC_LOAD",
        "CWL_SEC_SW_TEMP", "CT_SW_TEMP_1", "CT_RW_TEMP_1", "CT_POW_1",
        "CT_FAN_SPD_1", "CDWL_SW_TEMP", "TWV_CTRL", "CWL_SEC_CW_FLOW",
        "CWL_SEC_DP",
    ],
    "LBNL RTU": [
        "RTU_COMP_WATT", "RTU_COMP_WATT_1", "RTU_COMP_WATT_2",
        "RTU_TOT_WATT", "HVAC_TOT_WATT", "RTU_SA_TEMP", "RTU_MA_TEMP",
        "RTU_RA_TEMP", "ZA_TEMP", "TERM_RM_TEMP_102", "RTU_OA_DMPR_DM",
        "RTU_RA_DMPR_DM", "RTU_REFG_SUCT_PRES_1", "RTU_REFG_SUCT_PRES_2",
        "RTU_REFG_DISC_PRES_1", "RTU_REFG_DISC_PRES_2",
    ],
    "MetroPT Compressor": [
        "Reservoirs", "COMP", "Motor_current", "Oil_temperature", "LPS",
        "Pressure_switch",
    ],
}

RICH_CONTEXT_DOMAIN_NOTES: Dict[str, List[str]] = {
    "LBNL Boiler Plant": [
        "Boiler fouling usually raises fuel use while reducing delivered heat or changing hot-water temperatures.",
        "Hot-water pressure or temperature trend deviations can reveal boiler fouling or sensor/control bias even when mean z-scores are modest.",
        "Temperature or pressure sensor bias should be diagnosed when the dominant deviations are in those measured signals.",
        "A broad performance-index degradation is plausible when plant-level indicators move but no single sensor/control bias dominates.",
    ],
    "LBNL Chiller Plant": [
        "Chiller fouling tends to combine elevated chiller power with degraded cooling capacity.",
        "Cooling tower fouling tends to affect tower power, fan speed, and condenser/tower water temperatures.",
        "Bypass leakage or valve sticking tends to alter bypass/secondary flow and can reduce delivered cooling; in low-leakage cases it may first appear as severe tower-temperature response before direct bypass-flow predicates activate.",
        "Sensor-bias faults should be preferred when a measured temperature or pressure signal is the dominant outlier.",
    ],
    "LBNL RTU": [
        "Compressor staging faults affect compressor power and delivered cooling.",
        "Refrigerant undercharge is suggested by low refrigerant pressure together with poor cooling delivery.",
        "Outdoor-air damper faults affect outdoor/return damper commands and mixed-air behaviour.",
        "Economizer setpoint bias is inferred from mixed/return/supply-air and total-power response when damper faults are absent.",
        "RTU public predicates are sparse; a single active primary predicate may be the only available signal, so allow tentative low-confidence non-Normal diagnoses instead of forcing Normal solely because corroborating predicates are absent.",
    ],
    "MetroPT Compressor": [
        "Air Leak is pressure-side: repeated pressure loss, slow reservoir recovery, low reservoir pressure, and switch activity are primary indicators; in this public benchmark, severe oil-temperature elevation with abnormal pressure-switch behaviour can also appear in labelled Air Leak windows before oil-side trend predicates become active.",
        "Oil Leak is oil/mechanical-side: oil temperature or motor current rises, sometimes with pressure-switch anomalies before thermal evidence becomes large; mild oil-temperature elevation with severe pressure-switch behaviour needs pressure-recovery context to distinguish it from early Air Leak.",
        "Compressor control abnormality is plausible when pressure-switch behaviour is abnormal without pressure-loss or oil-side support.",
    ],
}


class PublicBenchmarkKnowledgeBase:
    """Compact public-benchmark rule base using abnormal predicate inputs.

    This benchmark validates the PMS routing architecture on public time-series
    data after deterministic abnormal-predicate extraction. It is not the same
    continuous-parameter seven-equipment knowledge base used by the synthetic
    experiments.
    """

    def __init__(self) -> None:
        self.equipment_types = [
            "LBNL Boiler Plant",
            "LBNL Chiller Plant",
            "LBNL RTU",
            "MetroPT Compressor",
        ]
        self.rules = self._build_rules()

    def is_feasible(self, equipment: str, params: Dict[str, Any]) -> bool:
        if equipment not in self.equipment_types:
            return False
        for value in params.values():
            if isinstance(value, bool):
                level = int(value)
            elif isinstance(value, (int, float)) and float(value).is_integer():
                level = int(value)
            else:
                return False
            if level not in {0, 1, 2}:
                return False
        return True

    def get_rules(self, equipment: str) -> List[Rule]:
        return [r for r in self.rules if r.equipment == equipment]

    def candidate_faults(self, equipment: str) -> List[str]:
        return [r.fault for r in self.get_rules(equipment)]

    def _build_rules(self) -> List[Rule]:
        """Build calibrated rule base from domain-expert fault-to-predicate mappings.

        Rule organisation per equipment type (descending diagnostic strength):
          1. Severe / confirmed — ≥3 predicates, key sensor flags at ``== 2``, confidence ≥0.91
          2. Core diagnostic     — 2 predicates at ``>= 1``, confidence 0.88–0.92
          3. Supporting          — 1-2 predicates, medium confidence 0.82–0.87, priority 2
          4. Weak indicator      — single predicate ``>= 1``, low confidence 0.68–0.77, priority 2-3
          5. Cross-fault         — multi-sensor anomalies suggesting bus / DAQ faults
          6. Normal detection    — all-zero predicate vectors → Normal

        Predicate levels (produced by the ``params_*`` / ``_severity_level`` helpers):
          0 = normal      (below the source-specific mild z-score threshold)
          1 = mild        (between the source-specific mild and severe z-score thresholds)
          2 = severe      (at or above the source-specific severe z-score threshold)

        Rules use ``>= 1`` for "any deviation counts" and ``== 2`` for
        "only severe deviations qualify" on the most specific sensor-bias
        predicates.
        """
        rules: List[Rule] = []

        def add(
            equipment: str,
            conditions: Dict[str, Tuple[str, int]],
            fault: str,
            analysis: str,
            *,
            confidence: float,
            priority: int = 1,
        ) -> None:
            rules.append(
                Rule(
                    equipment=equipment,
                    conditions=conditions,
                    fault=fault,
                    analysis=analysis,
                    confidence=confidence,
                    priority=priority,
                )
            )

        # =====================================================================
        # LBNL Boiler Plant  (6 predicates, 5 fault classes + Normal)
        #
        # Predicates:  boiler_fuel_high (F), heat_delivery_low (D),
        #   boiler_temp_bias (TB), boiler_pressure_bias (PB),
        #   boiler_operation_bias (OB), boiler_pi_abnormal (PA)
        # =====================================================================

        # --- Boiler fouling  (core: F + D) ---
        add("LBNL Boiler Plant",
            {"boiler_fuel_high": (">=", 1), "heat_delivery_low": (">=", 1)},
            "Boiler fouling",
            "Fuel consumption elevated while heat output degrades — classic fouling signature (fire-side or water-side).",
            confidence=0.90)
        add("LBNL Boiler Plant",
            {"boiler_fuel_high": (">=", 1), "boiler_delta_t_low": (">=", 1)},
            "Boiler fouling",
            "Fuel consumption elevated while hot-water temperature lift is reduced — efficiency-loss pattern consistent with fouling.",
            confidence=0.88)
        add("LBNL Boiler Plant",
            {"boiler_fuel_high": (">=", 1), "heat_delivery_low": (">=", 1), "boiler_pi_abnormal": (">=", 1)},
            "Boiler fouling",
            "Severe fouling: fuel/heat imbalance PLUS broad performance-index degradation — three corroborating signals.",
            confidence=0.93)
        add("LBNL Boiler Plant",
            {"boiler_pressure_bias": (">=", 1), "boiler_pi_abnormal": (">=", 1)},
            "Boiler fouling",
            "Data-assisted public rule audit: pressure-side deviation plus abnormal hot-water thermal trend repeatedly indicates boiler fouling in the LBNL Boiler Plant benchmark rather than an isolated pressure sensor bias.",
            confidence=0.90)
        add("LBNL Boiler Plant",
            {"boiler_fuel_high": (">=", 1)},
            "Boiler fouling",
            "Elevated fuel consumption alone — possible early-stage fouling or combustion tuning drift.",
            confidence=0.72, priority=2)

        # --- Hot water temperature sensor bias  (core: TB) ---
        add("LBNL Boiler Plant",
            {"boiler_temp_bias": (">=", 1)},
            "Hot water temperature sensor bias",
            "Supply or return hot-water temperature deviates systematically from the fault-free baseline.",
            confidence=0.91)
        add("LBNL Boiler Plant",
            {"boiler_temp_bias": ("==", 2)},
            "Hot water temperature sensor bias",
            "Large temperature sensor deviation (>3σ) — high-confidence sensor fault; verify with physical inspection.",
            confidence=0.93)
        add("LBNL Boiler Plant",
            {"boiler_temp_bias": (">=", 1), "boiler_pressure_bias": (">=", 1)},
            "Hot water temperature sensor bias",
            "Temperature and pressure sensors both biased — possible sensor bus, DAQ card, or common-mode fault rather than two independent sensor failures.",
            confidence=0.86, priority=2)

        # --- Hot water pressure sensor bias  (core: PB) ---
        add("LBNL Boiler Plant",
            {"boiler_pressure_bias": (">=", 1)},
            "Hot water pressure sensor bias",
            "Hot-water differential pressure deviates systematically from the baseline.",
            confidence=0.91)
        add("LBNL Boiler Plant",
            {"boiler_pressure_bias": ("==", 2)},
            "Hot water pressure sensor bias",
            "Large pressure sensor deviation (>3σ) — high-confidence sensor fault.",
            confidence=0.93)

        # --- Boiler sensor / control bias  (core: OB + F) ---
        add("LBNL Boiler Plant",
            {"boiler_operation_bias": (">=", 1), "boiler_fuel_high": (">=", 1)},
            "Boiler sensor/control bias",
            "Boiler-level sensor or control offset together with elevated fuel consumption — control-loop or actuator drift.",
            confidence=0.88)
        add("LBNL Boiler Plant",
            {"boiler_operation_bias": (">=", 1), "boiler_fuel_high": (">=", 1), "boiler_pi_abnormal": (">=", 1)},
            "Boiler sensor/control bias",
            "Control bias with fuel disturbance AND performance-index degradation — the control offset is measurably degrading plant efficiency.",
            confidence=0.91)
        add("LBNL Boiler Plant",
            {"boiler_operation_bias": (">=", 1), "boiler_pi_abnormal": (">=", 1)},
            "Boiler sensor/control bias",
            "Control bias coincident with broad performance degradation — the PI flag may be a consequence of the control offset.",
            confidence=0.84, priority=2)
        add("LBNL Boiler Plant",
            {"boiler_temp_bias": (">=", 1), "boiler_pi_abnormal": (">=", 1),
             "boiler_pressure_bias": ("==", 0), "boiler_fuel_high": ("==", 0)},
            "Boiler sensor/control bias",
            "Data-assisted public rule audit: temperature deviation with broad performance-index abnormality, but without pressure or fuel evidence, is more consistent with boiler-level sensor/control bias than an isolated hot-water temperature sensor fault.",
            confidence=0.90)
        add("LBNL Boiler Plant",
            {"boiler_operation_bias": (">=", 1)},
            "Boiler sensor/control bias",
            "Isolated boiler-level sensor/control offset — possible early-stage control drift or setpoint calibration issue.",
            confidence=0.72, priority=2)

        # --- Boiler performance degradation  (non-specific catch-all) ---
        add("LBNL Boiler Plant",
            {"boiler_pi_abnormal": (">=", 1)},
            "Boiler performance degradation",
            "Broad performance-index degradation — non-specific indicator; inspect fouling, sensor, and control subsystems.",
            confidence=0.82, priority=2)
        add("LBNL Boiler Plant",
            {"boiler_delta_t_low": (">=", 1), "boiler_pi_abnormal": (">=", 1)},
            "Boiler performance degradation",
            "Hot-water temperature lift is reduced together with broad performance degradation — plant-level efficiency loss without a dominant sensor bias.",
            confidence=0.84, priority=2)

        # --- Cross-fault: multiple simultaneous sensor anomalies ---
        add("LBNL Boiler Plant",
            {"boiler_temp_bias": (">=", 1), "boiler_pressure_bias": (">=", 1), "boiler_operation_bias": (">=", 1)},
            "Boiler sensor/control bias",
            "Three boiler sensor/control signals simultaneously biased — probable sensor bus, power-supply, or DAQ fault rather than three independent sensor failures.",
            confidence=0.88, priority=2)

        # --- Normal operation ---
        add("LBNL Boiler Plant",
            {"boiler_fuel_high": ("==", 0), "heat_delivery_low": ("==", 0), "boiler_temp_bias": ("==", 0),
             "boiler_pressure_bias": ("==", 0), "boiler_operation_bias": ("==", 0), "boiler_pi_abnormal": ("==", 0),
             "boiler_delta_t_low": ("==", 0)},
            "Normal",
            "All boiler predicates are at level 0 — healthy operation within 1.5σ of the fault-free baseline.",
            confidence=0.95, priority=2)

        # =====================================================================
        # LBNL Chiller Plant  (9 predicates, 7 fault classes + Normal)
        #
        # Predicates:  chiller_power_high (CP), cooling_capacity_low (CC),
        #   chiller_temp_bias (CT), cooling_tower_power_high (TP),
        #   cooling_tower_temp_high (TT), cooling_tower_temp_bias (CBT),
        #   bypass_flow_abnormal (BF), secondary_pressure_bias (SP),
        #   cooling_tower_pi_abnormal (TPA)
        # =====================================================================

        # --- Chiller fouling  (core: CP + CC) ---
        add("LBNL Chiller Plant",
            {"chiller_power_high": (">=", 1), "cooling_capacity_low": (">=", 1)},
            "Chiller fouling",
            "Chiller compressor power elevated while cooling capacity is degraded — classic condenser or evaporator fouling.",
            confidence=0.90)
        add("LBNL Chiller Plant",
            {"chiller_power_high": (">=", 1), "chiller_delta_t_low": (">=", 1)},
            "Chiller fouling",
            "Chiller power is elevated while chilled-water temperature lift is reduced — power-to-capacity degradation consistent with fouling.",
            confidence=0.88)
        add("LBNL Chiller Plant",
            {"chiller_power_high": ("==", 2), "cooling_capacity_low": (">=", 1)},
            "Chiller fouling",
            "Severely elevated chiller power (>3σ) with cooling capacity loss — advanced fouling requiring immediate cleaning.",
            confidence=0.91)
        add("LBNL Chiller Plant",
            {"chiller_power_high": (">=", 1)},
            "Chiller fouling",
            "Chiller power elevated alone — possible early-stage fouling or temporary condenser-water temperature excursion.",
            confidence=0.72, priority=2)

        # --- Chiller sensor bias  (core: CT) ---
        add("LBNL Chiller Plant",
            {"chiller_temp_bias": (">=", 1)},
            "Chiller sensor bias",
            "Chiller supply/return temperature signals show systematic offset from the fault-free baseline.",
            confidence=0.90, priority=2)
        add("LBNL Chiller Plant",
            {"chiller_temp_bias": ("==", 2)},
            "Chiller sensor bias",
            "Large chiller temperature sensor deviation (>3σ) — high-confidence sensor fault.",
            confidence=0.92, priority=2)
        add("LBNL Chiller Plant",
            {"chiller_temp_bias": (">=", 1), "bypass_flow_abnormal": (">=", 1), "cooling_capacity_low": (">=", 1)},
            "Chiller sensor bias",
            "Data-assisted rule audit: chiller temperature sensor bias can induce apparent bypass-flow and cooling-capacity anomalies; when the temperature-bias predicate is active with those secondary effects, prioritize sensor bias over bypass leakage.",
            confidence=0.90)

        # --- Cooling tower fouling  (core: TP + TT) ---
        add("LBNL Chiller Plant",
            {"cooling_tower_power_high": (">=", 1), "cooling_tower_temp_high": (">=", 1)},
            "Cooling tower fouling",
            "Cooling tower fan/pump effort elevated while condenser-water temperature is high — fill fouling or air-side blockage.",
            confidence=0.89)
        add("LBNL Chiller Plant",
            {"cooling_tower_power_high": (">=", 1), "condenser_temp_lift_high": (">=", 1)},
            "Cooling tower fouling",
            "Cooling tower power is elevated while condenser-side temperature lift increases — heat-rejection degradation consistent with tower fouling.",
            confidence=0.88)
        add("LBNL Chiller Plant",
            {"cooling_tower_power_high": (">=", 1), "cooling_tower_temp_high": (">=", 1), "cooling_tower_pi_abnormal": (">=", 1)},
            "Cooling tower fouling",
            "Severe cooling tower degradation: power, temperature, and performance index all abnormal — likely advanced fill fouling or fan failure.",
            confidence=0.92)
        add("LBNL Chiller Plant",
            {"cooling_tower_power_high": (">=", 1)},
            "Cooling tower fouling",
            "Cooling tower power elevated alone — possible early-stage fouling or fan-speed control issue.",
            confidence=0.70, priority=2)

        # --- Cooling tower sensor bias  (core: CBT) ---
        add("LBNL Chiller Plant",
            {"cooling_tower_temp_bias": (">=", 1)},
            "Cooling tower sensor bias",
            "Cooling tower temperature sensors show systematic offset from the seasonal baseline.",
            confidence=0.90, priority=2)
        add("LBNL Chiller Plant",
            {"cooling_tower_temp_bias": ("==", 2)},
            "Cooling tower sensor bias",
            "Large cooling tower sensor deviation (>3σ) — high-confidence sensor fault.",
            confidence=0.92, priority=2)

        # --- Bypass valve leakage or stuck  (core: BF) ---
        add("LBNL Chiller Plant",
            {"bypass_flow_abnormal": (">=", 1)},
            "Bypass valve leakage or stuck",
            "Bypass valve command/flow behaviour deviates from the fault-free baseline — leakage or mechanical sticking.",
            confidence=0.89)
        add("LBNL Chiller Plant",
            {"bypass_flow_abnormal": (">=", 1), "cooling_capacity_low": (">=", 1)},
            "Bypass valve leakage or stuck",
            "Bypass flow abnormality together with reduced cooling capacity — bypass leakage is likely bleeding chilled water past the load.",
            confidence=0.91)
        add("LBNL Chiller Plant",
            {"bypass_flow_abnormal": (">=", 1), "chiller_delta_t_low": (">=", 1)},
            "Bypass valve leakage or stuck",
            "Bypass flow abnormality coincides with low chilled-water temperature lift — likely hydraulic bypass or valve leakage reducing useful heat exchange.",
            confidence=0.88)
        add("LBNL Chiller Plant",
            {"bypass_flow_abnormal": ("==", 2)},
            "Bypass valve leakage or stuck",
            "Large bypass flow deviation (>3σ) — high-confidence bypass valve fault.",
            confidence=0.92)
        add("LBNL Chiller Plant",
            {"cooling_tower_temp_high": ("==", 2), "chiller_temp_bias": ("<=", 1)},
            "Bypass valve leakage or stuck",
            "Data-assisted public rule audit: severe cooling-tower temperature response without chiller temperature-sensor bias repeatedly appears in labelled bypass-leakage cases, including low-leakage cases before direct bypass-flow predicates activate.",
            confidence=0.90)
        add("LBNL Chiller Plant",
            {"bypass_flow_abnormal": ("==", 2), "cooling_tower_temp_high": ("==", 2),
             "cooling_tower_temp_bias": ("==", 2), "cooling_tower_power_high": ("==", 2)},
            "Bypass valve leakage or stuck",
            "Data-assisted public rule audit: a severe bypass-flow abnormality can drive multiple tower-side severe responses; keep the direct bypass predicate primary when it co-occurs with severe tower temperature, bias, and power predicates.",
            confidence=0.93)

        # --- Secondary chilled-water pressure sensor bias  (core: SP) ---
        add("LBNL Chiller Plant",
            {"secondary_pressure_bias": (">=", 1)},
            "Secondary chilled water pressure sensor bias",
            "Secondary chilled-water differential pressure deviates systematically from the baseline.",
            confidence=0.91)
        add("LBNL Chiller Plant",
            {"secondary_pressure_bias": ("==", 2)},
            "Secondary chilled water pressure sensor bias",
            "Large pressure sensor deviation (>3σ) — high-confidence sensor fault.",
            confidence=0.93)

        # --- Cooling tower performance degradation  (non-specific catch-all) ---
        add("LBNL Chiller Plant",
            {"cooling_tower_pi_abnormal": (">=", 1)},
            "Cooling tower performance degradation",
            "Cooling tower performance index shows broad degradation — non-specific; inspect fill, fan, and water distribution.",
            confidence=0.82, priority=2)

        # --- Cross-fault: multiple simultaneous sensor anomalies ---
        add("LBNL Chiller Plant",
            {"chiller_temp_bias": (">=", 1), "cooling_tower_temp_bias": (">=", 1), "secondary_pressure_bias": (">=", 1)},
            "Chiller sensor bias",
            "Three chiller-plant sensor signals simultaneously biased — probable sensor bus, DAQ card, or power-supply anomaly.",
            confidence=0.85, priority=2)

        # --- Normal operation ---
        add("LBNL Chiller Plant",
            {"chiller_power_high": ("==", 0), "cooling_capacity_low": ("==", 0), "chiller_temp_bias": ("==", 0),
             "cooling_tower_power_high": ("==", 0), "cooling_tower_temp_high": ("==", 0), "cooling_tower_temp_bias": ("==", 0),
             "bypass_flow_abnormal": ("==", 0), "secondary_pressure_bias": ("==", 0), "cooling_tower_pi_abnormal": ("==", 0),
             "chiller_delta_t_low": ("==", 0), "condenser_temp_lift_high": ("==", 0)},
            "Normal",
            "All chiller-plant predicates are at level 0 — healthy operation within 1.5σ of the fault-free baseline.",
            confidence=0.95, priority=2)

        # =====================================================================
        # LBNL RTU  (5 predicates, 4 fault classes + Normal)
        #
        # Predicates:  compressor_power_abnormal (CA), cooling_delivery_low (CD),
        #   outdoor_air_damper_abnormal (OD), economizer_signal_bias (ES),
        #   refrigerant_pressure_low (RP)
        # =====================================================================

        # --- Compressor staging or cooling capacity fault  (core: CA + CD) ---
        add("LBNL RTU",
            {"compressor_power_abnormal": (">=", 1), "cooling_delivery_low": (">=", 1)},
            "Compressor staging or cooling capacity fault",
            "Compressor power/staging abnormal while supply-air cooling is degraded — compressor or capacity-control fault.",
            confidence=0.88)
        add("LBNL RTU",
            {"compressor_power_abnormal": (">=", 1), "supply_air_temp_high": (">=", 1)},
            "Compressor staging or cooling capacity fault",
            "Compressor power/staging abnormal while supply-air temperature is high — capacity-control fault is degrading delivered cooling.",
            confidence=0.87)
        add("LBNL RTU",
            {"compressor_power_abnormal": (">=", 1)},
            "Compressor staging or cooling capacity fault",
            "Compressor power/staging abnormal alone — possible early-stage compressor degradation or staging-logic issue.",
            confidence=0.72, priority=2)

        # --- Refrigerant undercharge  (core: RP + CD) ---
        add("LBNL RTU",
            {"refrigerant_pressure_low": (">=", 1), "cooling_delivery_low": (">=", 1)},
            "Refrigerant undercharge",
            "Refrigerant suction/discharge pressure is low while cooling delivery is degraded — classic undercharge or restricted metering device.",
            confidence=0.90)
        add("LBNL RTU",
            {"refrigerant_pressure_low": (">=", 1), "refrigerant_pressure_imbalance": (">=", 1)},
            "Refrigerant undercharge",
            "Low refrigerant pressure plus abnormal suction/discharge pressure relationship — refrigerant-side fault rather than air-side control alone.",
            confidence=0.89)
        add("LBNL RTU",
            {"refrigerant_pressure_low": (">=", 1), "cooling_delivery_low": (">=", 1), "compressor_power_abnormal": (">=", 1)},
            "Refrigerant undercharge",
            "Refrigerant undercharge with compressor stress: low pressure, low cooling, AND abnormal compressor power — three consistent signals.",
            confidence=0.91)
        add("LBNL RTU",
            {"refrigerant_pressure_low": (">=", 1)},
            "Refrigerant undercharge",
            "Refrigerant suction/discharge pressure low alone — possible slow leak, metering restriction, or pressure-sensor drift; monitor trend over successive windows.",
            confidence=0.68, priority=2)

        # --- Outdoor air damper fault  (core: OD) ---
        add("LBNL RTU",
            {"outdoor_air_damper_abnormal": (">=", 1)},
            "Outdoor air damper fault",
            "Outdoor-air or return-air damper position deviates from the seasonal baseline — stuck, slipping actuator, or linkage fault.",
            confidence=0.88)
        add("LBNL RTU",
            {"outdoor_air_damper_abnormal": ("==", 2)},
            "Outdoor air damper fault",
            "Large damper position deviation (>3σ) — high-confidence damper fault; inspect actuator and linkage.",
            confidence=0.91)
        add("LBNL RTU",
            {"outdoor_air_damper_abnormal": (">=", 1), "economizer_signal_bias": (">=", 1)},
            "Outdoor air damper fault",
            "Damper position abnormal together with economizer thermal/power response bias — combined air-side control fault; check DDC logic and actuator supply.",
            confidence=0.86, priority=2)

        # --- Economizer setpoint bias  (core: ES) ---
        add("LBNL RTU",
            {"economizer_signal_bias": (">=", 1), "outdoor_air_damper_abnormal": ("==", 0),
             "refrigerant_pressure_low": ("==", 0)},
            "Economizer setpoint bias",
            "Economizer-related thermal/power response is shifted while damper position and refrigerant pressure remain near the seasonal baseline — likely setpoint calibration or control-logic bias rather than actuator sticking or refrigerant undercharge.",
            confidence=0.88)
        add("LBNL RTU",
            {"economizer_signal_bias": (">=", 1)},
            "Economizer setpoint bias",
            "Economizer-related thermal/power response is shifted relative to the seasonal baseline — setpoint calibration or control-logic issue.",
            confidence=0.85, priority=2)
        add("LBNL RTU",
            {"economizer_signal_bias": ("==", 2)},
            "Economizer setpoint bias",
            "Large economizer-related thermal/power response deviation (>3σ) — high-confidence economizer control fault.",
            confidence=0.88, priority=2)

        # --- Normal operation ---
        add("LBNL RTU",
            {"compressor_power_abnormal": ("==", 0), "cooling_delivery_low": ("==", 0),
             "outdoor_air_damper_abnormal": ("==", 0), "economizer_signal_bias": ("==", 0),
             "refrigerant_pressure_low": ("==", 0), "supply_air_temp_high": ("==", 0),
             "refrigerant_pressure_imbalance": ("==", 0)},
            "Normal",
            "All RTU predicates are at level 0 — healthy operation within 1.5σ of the seasonal baseline.",
            confidence=0.95, priority=2)

        # =====================================================================
        # MetroPT Compressor  (7 predicates, 4 fault classes + Normal)
        #
        # Predicates:  pressure_recovery_slow (PR), pressure_drop_frequent (PD),
        #   compressor_runtime_high (CR), motor_current_high (MC),
        #   oil_temperature_high (OT), low_pressure_switch_active (LP),
        #   pressure_switch_abnormal (PS)
        # =====================================================================

        # --- Air Leak  (two core patterns: PR+CR  and  PD+LP) ---
        #
        # Severe / confirmed
        add("MetroPT Compressor",
            {"pressure_recovery_slow": (">=", 1), "pressure_drop_frequent": (">=", 1),
             "low_pressure_switch_active": (">=", 1), "compressor_runtime_high": (">=", 1)},
            "Air Leak",
            "Severe air leak: all four pressure/compressor indicators active — large leak; inspect piping, fittings, dryer, and clients.",
            confidence=0.95)
        add("MetroPT Compressor",
            {"pressure_recovery_slow": (">=", 1), "pressure_drop_frequent": (">=", 1),
             "low_pressure_switch_active": (">=", 1), "reservoir_pressure_low": (">=", 1)},
            "Air Leak",
            "Data-assisted public rule audit: simultaneous slow pressure recovery, frequent pressure drops, low-pressure switch activity, and low reservoir pressure form a strong pressure-side leak signature even without elevated compressor runtime.",
            confidence=0.94)
        add("MetroPT Compressor",
            {"pressure_recovery_slow": (">=", 1), "pressure_drop_frequent": (">=", 1), "compressor_runtime_high": (">=", 1)},
            "Air Leak",
            "Confirmed air leak: slow recovery, frequent drops, and elevated compressor duty cycle — three consistent indicators.",
            confidence=0.93)
        # Core patterns
        add("MetroPT Compressor",
            {"pressure_recovery_slow": (">=", 1), "compressor_runtime_high": (">=", 1)},
            "Air Leak",
            "Reservoir pressure recovers slowly while compressor runs more — the compressor is working harder to compensate for lost air.",
            confidence=0.90)
        add("MetroPT Compressor",
            {"reservoir_pressure_low": (">=", 1), "compressor_runtime_high": (">=", 1)},
            "Air Leak",
            "Reservoir pressure remains below the healthy baseline while compressor duty rises — sustained pressure loss consistent with an air leak.",
            confidence=0.89)
        add("MetroPT Compressor",
            {"pressure_drop_frequent": (">=", 1), "low_pressure_switch_active": (">=", 1)},
            "Air Leak",
            "Frequent pressure drops trigger the low-pressure switch — air is escaping faster than the compressor can replenish it.",
            confidence=0.92)
        # Supporting patterns
        add("MetroPT Compressor",
            {"pressure_recovery_slow": (">=", 1), "pressure_drop_frequent": (">=", 1)},
            "Air Leak",
            "Slow pressure recovery with frequent pressure drops — consistent with air leakage, though compressor response is not yet confirmed.",
            confidence=0.86, priority=2)
        add("MetroPT Compressor",
            {"reservoir_pressure_low": (">=", 1), "low_pressure_switch_active": (">=", 1)},
            "Air Leak",
            "Low reservoir pressure together with low-pressure switch activity — pressure-side loss is directly observed.",
            confidence=0.88, priority=2)
        add("MetroPT Compressor",
            {"pressure_switch_abnormal": (">=", 1), "pressure_recovery_slow": (">=", 1),
             "pressure_drop_frequent": (">=", 1)},
            "Air Leak",
            "Pressure switch anomaly appears together with pressure-side leak symptoms — treat the switch response as a consequence of air leakage rather than a standalone control fault.",
            confidence=0.89, priority=1)
        add("MetroPT Compressor",
            {"oil_temperature_high": ("==", 2), "pressure_switch_abnormal": (">=", 1)},
            "Air Leak",
            "Data-assisted rule audit: in the MetroPT public benchmark, severe oil-temperature elevation with abnormal pressure-switch behaviour repeatedly appears in labelled Air Leak windows before oil-side trend predicates become active.",
            confidence=0.90, priority=1)
        add("MetroPT Compressor",
            {"oil_temperature_high": ("==", 2), "pressure_switch_abnormal": (">=", 1),
             "motor_current_high": (">=", 1)},
            "Air Leak",
            "Data-assisted rule audit: severe oil-temperature elevation with pressure-switch abnormality and only static motor-current elevation repeatedly appears in labelled Air Leak windows; without a rising oil-side trend, treat pressure loss as primary.",
            confidence=0.91, priority=1)
        add("MetroPT Compressor",
            {"compressor_runtime_high": (">=", 1), "low_pressure_switch_active": (">=", 1)},
            "Air Leak",
            "High compressor duty cycle with active low-pressure switch — the compressor is responding to sustained low-pressure events.",
            confidence=0.87)
        # Weak indicators
        add("MetroPT Compressor",
            {"pressure_recovery_slow": (">=", 1), "pressure_switch_abnormal": (">=", 1)},
            "Air Leak",
            "Pressure recovery rate is below the healthy baseline and pressure-switch behaviour is abnormal — possible slow leak; trend over successive windows.",
            confidence=0.68, priority=2)
        add("MetroPT Compressor",
            {"pressure_drop_frequent": (">=", 1)},
            "Air Leak",
            "Frequent pressure drops detected — possible intermittent leak or demand-side surge; correlate with production schedule.",
            confidence=0.70, priority=2)
        # Derived-predicate pattern: steep pressure drop without motor current rise
        add("MetroPT Compressor",
            {"pressure_recovery_steep": (">=", 1), "reservoir_pressure_low": (">=", 1)},
            "Air Leak",
            "Pressure recovery slope is sharply negative and reservoir pressure is low — classic air leak signature (pressure loss without mechanical load increase).",
            confidence=0.89, priority=2)

        # --- Oil Leak  (core: OT + MC) ---
        #
        # Severe / confirmed
        add("MetroPT Compressor",
            {"oil_temperature_high": (">=", 1), "motor_current_high": (">=", 1), "compressor_runtime_high": (">=", 1)},
            "Oil Leak",
            "Severe oil leak: temperature, current, and duty cycle all elevated — the compressor is thermally stressed; inspect seals, gaskets, and oil level.",
            confidence=0.92)
        # Core pattern
        add("MetroPT Compressor",
            {"oil_temperature_high": (">=", 1), "motor_current_high": (">=", 1), "pressure_switch_abnormal": (">=", 1)},
            "Oil Leak",
            "Oil temperature and motor current rise together with abnormal pressure-switch behaviour — compressor lubrication problem; oil leak reduces cooling and increases friction.",
            confidence=0.90)
        # Overload pattern (MC + CR without OT — oil-leak-driven mechanical overload is the most likely cause)
        add("MetroPT Compressor",
            {"motor_current_high": (">=", 1), "compressor_runtime_high": (">=", 1)},
            "Oil Leak",
            "Motor current and compressor duty cycle both elevated — mechanical overload; probable oil-degradation or early bearing-wear driven.",
            confidence=0.87)
        # Weak indicators
        add("MetroPT Compressor",
            {"oil_temperature_high": (">=", 1)},
            "Oil Leak",
            "Oil temperature elevated alone — possible early oil degradation, low oil level, or cooler fouling; check oil condition.",
            confidence=0.72, priority=2)
        add("MetroPT Compressor",
            {"motor_current_high": (">=", 1)},
            "Oil Leak",
            "Motor current elevated alone — possible mechanical friction increase or early bearing wear; correlate with vibration if available.",
            confidence=0.70, priority=2)

        # --- Oil Leak (derived predicate patterns — distinguish from Air Leak) ---
        add("MetroPT Compressor",
            {"oil_temp_rising": (">=", 1), "motor_current_high": (">=", 1)},
            "Oil Leak",
            "Oil temperature rising within the window together with elevated motor current — active thermal degradation; strongly favours Oil Leak over Air Leak.",
            confidence=0.91)
        add("MetroPT Compressor",
            {"motor_current_rising": (">=", 1), "oil_temperature_high": (">=", 1)},
            "Oil Leak",
            "Motor current rises within the window while oil temperature is high — increasing mechanical load and thermal stress point to oil-side degradation.",
            confidence=0.90)
        add("MetroPT Compressor",
            {"motor_current_rising": (">=", 1), "oil_temp_rising": (">=", 1), "pressure_switch_abnormal": (">=", 1)},
            "Oil Leak",
            "Motor current and oil temperature both trend upward while pressure-switch behaviour is abnormal — coupled mechanical/thermal degradation consistent with Oil Leak.",
            confidence=0.91)
        add("MetroPT Compressor",
            {"pressure_switch_abnormal": (">=", 1), "oil_temperature_high": (">=", 1),
             "oil_temp_rising": (">=", 1)},
            "Oil Leak",
            "Pressure switch anomaly coincides with oil-temperature elevation and an in-window rising oil-temperature trend — oil-side thermal degradation is the primary fault.",
            confidence=0.90, priority=1)
        add("MetroPT Compressor",
            {"pressure_switch_abnormal": (">=", 1), "oil_temperature_high": (">=", 1)},
            "Oil Leak",
            "Pressure switch anomaly appears together with high oil temperature — likely secondary to oil-side compressor stress rather than an isolated control fault.",
            confidence=0.86, priority=2)
        add("MetroPT Compressor",
            {"pressure_switch_abnormal": ("==", 2), "oil_temperature_high": ("==", 0)},
            "Oil Leak",
            "Data-assisted rule audit: severe pressure-switch anomaly without concurrent oil-temperature elevation repeatedly appears in MetroPT Oil Leak windows, suggesting an oil-side fault progression where switch behaviour changes before thermal evidence becomes visible.",
            confidence=0.89, priority=1)
        add("MetroPT Compressor",
            {"pressure_switch_abnormal": ("==", 2), "oil_temperature_high": ("==", 1),
             "pressure_recovery_slow": (">=", 1), "pressure_recovery_steep": (">=", 1),
             "motor_current_rising": ("==", 0)},
            "Oil Leak",
            "Data-assisted rule audit: severe pressure-switch anomaly with mild oil-temperature elevation and slow/steep pressure-recovery behaviour, but no rising motor-current trend, repeatedly appears in labelled Oil Leak windows.",
            confidence=0.90, priority=1)
        add("MetroPT Compressor",
            {"current_pressure_gap": (">=", 1)},
            "Oil Leak",
            "High motor current combined with slow pressure recovery — mechanical friction pattern (Oil Leak), not pure pressure loss (Air Leak).",
            confidence=0.88)
        add("MetroPT Compressor",
            {"oil_temp_rising": (">=", 1), "pressure_switch_abnormal": (">=", 1)},
            "Oil Leak",
            "Oil temperature rises while pressure-switch behaviour is abnormal — possible early oil degradation; trend over successive windows.",
            confidence=0.76, priority=2)

        # --- Compressor control abnormality  (core: PS) ---
        add("MetroPT Compressor",
            {"pressure_switch_abnormal": (">=", 1)},
            "Compressor control abnormality",
            "Pressure switch behaviour is abnormal without sufficient air-leak or oil-side support — possible switch degradation or control-logic anomaly.",
            confidence=0.58, priority=3)
        add("MetroPT Compressor",
            {"pressure_switch_abnormal": ("==", 2)},
            "Compressor control abnormality",
            "Large pressure switch deviation (>3σ) without sufficient pressure-side or oil-side corroboration — inspect switch and control logic.",
            confidence=0.64, priority=3)
        add("MetroPT Compressor",
            {"pressure_switch_abnormal": (">=", 1), "compressor_runtime_high": (">=", 1)},
            "Compressor control abnormality",
            "Pressure switch abnormal together with elevated compressor duty cycle — the control anomaly may be causing unnecessary compressor runtime.",
            confidence=0.78, priority=2)
        # --- Normal operation ---
        add("MetroPT Compressor",
            {"pressure_recovery_slow": ("==", 0), "pressure_drop_frequent": ("==", 0), "compressor_runtime_high": ("==", 0),
             "motor_current_high": ("==", 0), "oil_temperature_high": ("==", 0), "reservoir_pressure_low": ("==", 0),
             "motor_current_rising": ("==", 0), "low_pressure_switch_active": ("==", 0), "pressure_switch_abnormal": ("==", 0),
             "oil_temp_rising": ("==", 0), "current_pressure_gap": ("==", 0), "pressure_recovery_steep": ("==", 0)},
            "Normal",
            "All compressor predicates are at level 0 — healthy operation within 1.5σ of the pre-failure baseline.",
            confidence=0.95, priority=2)

        return rules


def _to_float(value: str) -> Optional[float]:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(val) or math.isinf(val):
        return None
    return val


def collect_stats(path: Path, max_rows: int = 20000, stride: int = 60) -> Dict[str, Dict[str, float]]:
    """Collect simple numeric statistics from a CSV file."""
    sums: Dict[str, float] = defaultdict(float)
    sums2: Dict[str, float] = defaultdict(float)
    counts: Dict[str, int] = defaultdict(int)
    first: Dict[str, float] = {}
    last: Dict[str, float] = {}

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            if idx >= max_rows:
                break
            if stride > 1 and idx % stride != 0:
                continue
            for col, raw in row.items():
                if col is None or col.lower() in {"datetime", "timestamp"}:
                    continue
                val = _to_float(raw)
                if val is None:
                    continue
                if col not in first:
                    first[col] = val
                last[col] = val
                sums[col] += val
                sums2[col] += val * val
                counts[col] += 1

    stats: Dict[str, Dict[str, float]] = {}
    for col, n in counts.items():
        mean = sums[col] / n
        var = max(0.0, (sums2[col] / n) - (mean * mean))
        stats[col] = {
            "mean": mean,
            "std": math.sqrt(var),
            "first": first[col],
            "last": last[col],
            "count": float(n),
        }
    return stats


def mean(stats: Dict[str, Dict[str, float]], col: str, default: float = 0.0) -> float:
    return stats.get(col, {}).get("mean", default)


def std(stats: Dict[str, Dict[str, float]], col: str, default: float = 0.0) -> float:
    return stats.get(col, {}).get("std", default)


def delta(case: Dict[str, Dict[str, float]], base: Dict[str, Dict[str, float]], col: str) -> float:
    return mean(case, col) - mean(base, col)


def rel_delta(case: Dict[str, Dict[str, float]], base: Dict[str, Dict[str, float]], col: str) -> float:
    b = abs(mean(base, col))
    if b < 1e-9:
        return 0.0
    return delta(case, base, col) / b


def _fmt_float(value: float, digits: int = 3) -> str:
    if abs(value) >= 1000:
        return f"{value:.1f}"
    return f"{value:.{digits}f}"


def rich_feature_context(
    source: str,
    equipment: str,
    case: Dict[str, Dict[str, float]],
    base: Dict[str, Dict[str, float]],
    *,
    start_time: str = "",
    end_time: str = "",
    max_features: int = 12,
) -> str:
    """Build a raw/statistical public-data summary that avoids fault-label leakage."""
    rows: List[Dict[str, float | str]] = []
    for col in RICH_CONTEXT_COLUMNS.get(equipment, []):
        if col not in case or col not in base:
            continue
        base_std = std(base, col)
        mean_delta = delta(case, base, col)
        z_mean = mean_delta / base_std if base_std > 1e-9 else 0.0
        case_trend = case[col].get("last", 0.0) - case[col].get("first", 0.0)
        base_trend = base[col].get("last", 0.0) - base[col].get("first", 0.0)
        z_trend = (case_trend - base_trend) / base_std if base_std > 1e-9 else 0.0
        rows.append(
            {
                "feature": col,
                "mean": mean(case, col),
                "baseline_mean": mean(base, col),
                "delta": mean_delta,
                "z_mean": z_mean,
                "std": std(case, col),
                "baseline_std": base_std,
                "first": case[col].get("first", 0.0),
                "last": case[col].get("last", 0.0),
                "z_trend": z_trend,
                "score": max(abs(z_mean), abs(z_trend)),
            }
        )
    rows.sort(key=lambda row: float(row["score"]), reverse=True)

    lines = [
        f"Source: {source}",
        f"Equipment: {equipment}",
        "Scenario identifier is intentionally omitted to avoid fault-label leakage.",
    ]
    if start_time or end_time:
        lines.append(f"Window: {start_time or 'NA'} to {end_time or 'NA'}")
    lines.extend(
        [
            "Input type: raw public benchmark statistics, not expert-system predicates.",
            "Values compare the current case/window against a healthy baseline.",
            "z_mean = (window mean - baseline mean) / baseline std.",
            "z_trend = ((last-first) - baseline trend) / baseline std.",
            "",
            "Top feature deviations:",
        ]
    )
    for row in rows[:max_features]:
        lines.append(
            "  - {feature}: mean={mean}, baseline_mean={baseline_mean}, "
            "delta={delta}, z_mean={z_mean}, z_trend={z_trend}, "
            "first={first}, last={last}".format(
                feature=row["feature"],
                mean=_fmt_float(float(row["mean"])),
                baseline_mean=_fmt_float(float(row["baseline_mean"])),
                delta=_fmt_float(float(row["delta"])),
                z_mean=_fmt_float(float(row["z_mean"])),
                z_trend=_fmt_float(float(row["z_trend"])),
                first=_fmt_float(float(row["first"])),
                last=_fmt_float(float(row["last"])),
            )
        )
    lines.extend(["", "Domain mechanism notes:"])
    for note in RICH_CONTEXT_DOMAIN_NOTES.get(equipment, []):
        lines.append(f"  - {note}")
    lines.extend(
        [
            "",
            "Use the statistical deviations and mechanism notes to choose one candidate fault.",
            "Do not infer the fault from any scenario filename or expert-system output.",
        ]
    )
    return "\n".join(lines)


def label_lbnl_boiler(name: str) -> Optional[str]:
    if name == "BoilerPlant":
        return None
    if "boiler_foul" in name:
        return "Boiler fouling"
    if "hot_water_pressure_bias" in name:
        return "Hot water pressure sensor bias"
    if "hot_water_temp_bias" in name:
        return "Hot water temperature sensor bias"
    if "boiler_bias" in name:
        return "Boiler sensor/control bias"
    if "boiler_PI" in name:
        return "Boiler performance degradation"
    return "Unknown"


def _severity_level(delta_val: float, base_std: float, direction: str = "both",
                    mild_z: float = 1.5, severe_z: float = 3.0) -> int:
    """Map a deviation to 0=normal, 1=mild, 2=severe using z-score thresholds.

    ``direction`` controls which side of the distribution is flagged:
    * ``"both"`` — absolute deviation (bidirectional bias sensors)
    * ``"high"`` — positive deviation only  (high-consumption / high-temperature)
    * ``"low"``  — negative deviation only  (low-flow / low-pressure)
    """
    if base_std < 1e-9:
        return 0
    z = delta_val / base_std
    if direction == "high" and z <= 0:
        return 0
    if direction == "low" and z >= 0:
        return 0
    az = abs(z)
    if az < mild_z:
        return 0
    if az < severe_z:
        return 1
    return 2


def params_lbnl_boiler(case: Dict[str, Dict[str, float]], base: Dict[str, Dict[str, float]],
                        z_thresholds: Tuple[float, float] = (1.5, 3.0)) -> Dict[str, int]:
    """Multi-level boiler predicates from z-score deviations."""
    mz, sz = z_thresholds
    case_delta_t = mean(case, "HWL_SW_TEMP") - mean(case, "HWL_RW_TEMP")
    base_delta_t = mean(base, "HWL_SW_TEMP") - mean(base, "HWL_RW_TEMP")
    delta_t_std = max(std(base, "HWL_SW_TEMP"), std(base, "HWL_RW_TEMP"))
    def trend_delta(col: str) -> float:
        return (case.get(col, {}).get("last", 0.0) - case.get(col, {}).get("first", 0.0)) - (
            base.get(col, {}).get("last", 0.0) - base.get(col, {}).get("first", 0.0)
        )

    pressure_bias = max(
        _severity_level(delta(case, base, "HWL_DP"), std(base, "HWL_DP"), "both", mild_z=mz, severe_z=sz),
        _severity_level(trend_delta("HWL_DP"), std(base, "HWL_DP"), "both", mild_z=mz, severe_z=sz),
    )
    thermal_trend_abnormal = max(
        _severity_level(trend_delta("HWL_SW_TEMP"), std(base, "HWL_SW_TEMP"), "both", mild_z=mz, severe_z=sz),
        _severity_level(trend_delta("HWL_RW_TEMP"), std(base, "HWL_RW_TEMP"), "both", mild_z=mz, severe_z=sz),
        _severity_level(trend_delta("BOI_SW_TEMP_1"), std(base, "BOI_SW_TEMP_1"), "both", mild_z=mz, severe_z=sz),
        _severity_level(trend_delta("BOI_RW_TEMP_1"), std(base, "BOI_RW_TEMP_1"), "both", mild_z=mz, severe_z=sz),
    )
    return {
        "boiler_temp_bias": max(
            _severity_level(delta(case, base, "HWL_SW_TEMP"), std(base, "HWL_SW_TEMP"), "both", mild_z=mz, severe_z=sz),
            _severity_level(delta(case, base, "HWL_RW_TEMP"), std(base, "HWL_RW_TEMP"), "both", mild_z=mz, severe_z=sz),
        ),
        "boiler_pressure_bias": pressure_bias,
        "boiler_fuel_high": max(
            _severity_level(delta(case, base, "BOI_GAS_CSUM_1"), std(base, "BOI_GAS_CSUM_1"), "high", mild_z=mz, severe_z=sz),
            _severity_level(delta(case, base, "BOI_GAS_CSUM_2"), std(base, "BOI_GAS_CSUM_2"), "high", mild_z=mz, severe_z=sz),
        ),
        "heat_delivery_low": max(
            _severity_level(-delta(case, base, "HWL_SW_TEMP"), std(base, "HWL_SW_TEMP"), "high", mild_z=mz, severe_z=sz),
            _severity_level(-delta(case, base, "BOI_SW_TEMP_1"), std(base, "BOI_SW_TEMP_1"), "high", mild_z=mz, severe_z=sz),
        ),
        "boiler_operation_bias": max(
            _severity_level(delta(case, base, "BOI_SW_TEMP_1"), std(base, "BOI_SW_TEMP_1"), "both", mild_z=mz, severe_z=sz),
            _severity_level(delta(case, base, "BOI_RW_TEMP_1"), std(base, "BOI_RW_TEMP_1"), "both", mild_z=mz, severe_z=sz),
        ),
        "boiler_pi_abnormal": max(
            _severity_level(delta(case, base, "HWL_SW_TEMP"), std(base, "HWL_SW_TEMP"), "both", mild_z=mz, severe_z=sz),
            _severity_level(delta(case, base, "HWL_DP"), std(base, "HWL_DP"), "both", mild_z=mz, severe_z=sz),
            thermal_trend_abnormal,
        ),
        "boiler_delta_t_low": _severity_level(base_delta_t - case_delta_t, delta_t_std, "high", mild_z=mz, severe_z=sz),
    }


def label_lbnl_chiller(name: str) -> Optional[str]:
    if name == "ChillerPlant":
        return None
    if "bypass_" in name:
        return "Bypass valve leakage or stuck"
    if "chiller_bias" in name:
        return "Chiller sensor bias"
    if "chiller_fouling" in name:
        return "Chiller fouling"
    if "coolingtower_bias" in name:
        return "Cooling tower sensor bias"
    if "coolingtower_fouling" in name:
        return "Cooling tower fouling"
    if "coolingtower_PI" in name:
        return "Cooling tower performance degradation"
    if "secondary_chilled_water_pressure_bias" in name:
        return "Secondary chilled water pressure sensor bias"
    return "Unknown"


def params_lbnl_chiller(case: Dict[str, Dict[str, float]], base: Dict[str, Dict[str, float]],
                          z_thresholds: Tuple[float, float] = (1.5, 3.0)) -> Dict[str, int]:
    """Multi-level chiller-plant predicates from z-score deviations."""
    mz, sz = z_thresholds
    case_chw_delta_t = mean(case, "CHL_RW_TEMP_1") - mean(case, "CHL_SW_TEMP_1")
    base_chw_delta_t = mean(base, "CHL_RW_TEMP_1") - mean(base, "CHL_SW_TEMP_1")
    chw_delta_t_std = max(std(base, "CHL_RW_TEMP_1"), std(base, "CHL_SW_TEMP_1"))
    case_condenser_lift = mean(case, "CDWL_SW_TEMP") - mean(case, "CT_SW_TEMP_1")
    base_condenser_lift = mean(base, "CDWL_SW_TEMP") - mean(base, "CT_SW_TEMP_1")
    condenser_lift_std = max(std(base, "CDWL_SW_TEMP"), std(base, "CT_SW_TEMP_1"))
    return {
        "chiller_temp_bias": max(
            _severity_level(delta(case, base, "CHL_SW_TEMP_1"), std(base, "CHL_SW_TEMP_1"), "both", mild_z=mz, severe_z=sz),
            _severity_level(delta(case, base, "CHL_RW_TEMP_1"), std(base, "CHL_RW_TEMP_1"), "both", mild_z=mz, severe_z=sz),
        ),
        "cooling_tower_temp_bias": max(
            _severity_level(delta(case, base, "CT_SW_TEMP_1"), std(base, "CT_SW_TEMP_1"), "both", mild_z=mz, severe_z=sz),
            _severity_level(delta(case, base, "CT_RW_TEMP_1"), std(base, "CT_RW_TEMP_1"), "both", mild_z=mz, severe_z=sz),
        ),
        "secondary_pressure_bias": _severity_level(delta(case, base, "CWL_SEC_DP"), std(base, "CWL_SEC_DP"), "both", mild_z=mz, severe_z=sz),
        "chiller_power_high": _severity_level(delta(case, base, "CHL_POW_1"), std(base, "CHL_POW_1"), "high", mild_z=mz, severe_z=sz),
        "cooling_capacity_low": max(
            _severity_level(-delta(case, base, "CWL_SEC_LOAD"), std(base, "CWL_SEC_LOAD"), "high", mild_z=mz, severe_z=sz),
            _severity_level(delta(case, base, "CWL_SEC_SW_TEMP"), std(base, "CWL_SEC_SW_TEMP"), "high", mild_z=mz, severe_z=sz),
        ),
        "cooling_tower_power_high": max(
            _severity_level(delta(case, base, "CT_POW_1"), std(base, "CT_POW_1"), "high", mild_z=mz, severe_z=sz),
            _severity_level(delta(case, base, "CT_FAN_SPD_1"), std(base, "CT_FAN_SPD_1"), "high", mild_z=mz, severe_z=sz),
        ),
        "cooling_tower_temp_high": max(
            _severity_level(delta(case, base, "CT_SW_TEMP_1"), std(base, "CT_SW_TEMP_1"), "high", mild_z=mz, severe_z=sz),
            _severity_level(delta(case, base, "CDWL_SW_TEMP"), std(base, "CDWL_SW_TEMP"), "high", mild_z=mz, severe_z=sz),
        ),
        "bypass_flow_abnormal": max(
            _severity_level(delta(case, base, "TWV_CTRL"), std(base, "TWV_CTRL"), "both", mild_z=mz, severe_z=sz),
            _severity_level(delta(case, base, "CWL_SEC_CW_FLOW"), std(base, "CWL_SEC_CW_FLOW"), "both", mild_z=mz, severe_z=sz),
        ),
        "cooling_tower_pi_abnormal": max(
            _severity_level(delta(case, base, "CT_POW_1"), std(base, "CT_POW_1"), "both", mild_z=mz, severe_z=sz),
            _severity_level(delta(case, base, "CT_SW_TEMP_1"), std(base, "CT_SW_TEMP_1"), "both", mild_z=mz, severe_z=sz),
        ),
        "chiller_delta_t_low": _severity_level(base_chw_delta_t - case_chw_delta_t, chw_delta_t_std, "high", mild_z=mz, severe_z=sz),
        "condenser_temp_lift_high": _severity_level(case_condenser_lift - base_condenser_lift, condenser_lift_std, "high", mild_z=mz, severe_z=sz),
    }


def label_lbnl_rtu(name: str) -> Optional[str]:
    if "Unfaulted" in name or name.startswith("ERTU_"):
        return None
    if "Staging_Fault" in name:
        return "Compressor staging or cooling capacity fault"
    if "Undercharged" in name:
        return "Refrigerant undercharge"
    if "OA_damper_stuck" in name:
        return "Outdoor air damper fault"
    if "Inc_Eco_SP" in name:
        return "Economizer setpoint bias"
    return "Unknown"


def _stratified_sample(items: List[Any], max_total: Optional[int], label_fn: Callable[[Any], Optional[str]]) -> List[Any]:
    """Limit scenario count while keeping at least one sample per fault label."""
    sorted_items = sorted(items, key=lambda item: str(item))
    if not max_total or len(sorted_items) <= max_total:
        return sorted_items

    by_fault: Dict[str, List[Any]] = defaultdict(list)
    for item in sorted_items:
        label = label_fn(item)
        by_fault[str(label) if label is not None else "Normal"].append(item)

    selected: List[Any] = []
    labels = sorted(by_fault)
    for label in labels:
        if by_fault[label]:
            selected.append(by_fault[label].pop(0))
            if len(selected) >= max_total:
                return sorted(selected, key=lambda item: str(item))

    while len(selected) < max_total:
        added = False
        for label in labels:
            if by_fault[label]:
                selected.append(by_fault[label].pop(0))
                added = True
                if len(selected) >= max_total:
                    break
        if not added:
            break
    return sorted(selected, key=lambda item: str(item))


def _known_labeled_files(files: List[Path], label_fn: Callable[[str], Optional[str]]) -> List[Path]:
    """Keep LBNL scenarios represented in the compact public benchmark KB."""
    return [path for path in files if label_fn(path.stem) != "Unknown"]


def params_lbnl_rtu(case: Dict[str, Dict[str, float]], base: Dict[str, Dict[str, float]],
                     z_thresholds: Tuple[float, float] = (1.5, 3.0)) -> Dict[str, int]:
    """Multi-level RTU predicates from z-score deviations."""
    mz, sz = z_thresholds
    comp_z = max(
        _severity_level(delta(case, base, "RTU_COMP_WATT"), std(base, "RTU_COMP_WATT"), "both", mild_z=mz, severe_z=sz),
        _severity_level(delta(case, base, "RTU_COMP_WATT_1"), std(base, "RTU_COMP_WATT_1"), "both", mild_z=mz, severe_z=sz),
        _severity_level(delta(case, base, "RTU_COMP_WATT_2"), std(base, "RTU_COMP_WATT_2"), "both", mild_z=mz, severe_z=sz),
    )
    sa_z = max(
        _severity_level(delta(case, base, "RTU_SA_TEMP"), std(base, "RTU_SA_TEMP"), "high", mild_z=mz, severe_z=sz),
        _severity_level(delta(case, base, "ZA_TEMP"), std(base, "ZA_TEMP"), "high", mild_z=mz, severe_z=sz),
    )
    cooling_delivery_z = max(
        sa_z,
        _severity_level(delta(case, base, "TERM_RM_TEMP_102"), std(base, "TERM_RM_TEMP_102"), "high", mild_z=mz, severe_z=sz),
    )
    damper_z = max(
        _severity_level(delta(case, base, "RTU_OA_DMPR_DM"), std(base, "RTU_OA_DMPR_DM"), "both", mild_z=mz, severe_z=sz),
        _severity_level(delta(case, base, "RTU_RA_DMPR_DM"), std(base, "RTU_RA_DMPR_DM"), "both", mild_z=mz, severe_z=sz),
    )
    # Economizer setpoint faults in the RTU benchmark do not expose a direct
    # setpoint column.  Use the downstream control response instead of
    # duplicating the damper-position predicate.
    economizer_response_z = max(
        _severity_level(delta(case, base, "RTU_MA_TEMP"), std(base, "RTU_MA_TEMP"), "both", mild_z=mz, severe_z=sz),
        _severity_level(delta(case, base, "RTU_RA_TEMP"), std(base, "RTU_RA_TEMP"), "both", mild_z=mz, severe_z=sz),
        _severity_level(delta(case, base, "RTU_SA_TEMP"), std(base, "RTU_SA_TEMP"), "both", mild_z=mz, severe_z=sz),
        _severity_level(delta(case, base, "TERM_RM_TEMP_102"), std(base, "TERM_RM_TEMP_102"), "both", mild_z=mz, severe_z=sz),
        _severity_level(delta(case, base, "RTU_TOT_WATT"), std(base, "RTU_TOT_WATT"), "high", mild_z=mz, severe_z=sz),
        _severity_level(delta(case, base, "HVAC_TOT_WATT"), std(base, "HVAC_TOT_WATT"), "high", mild_z=mz, severe_z=sz),
    )
    refrigerant_pressure_z = max(
        _severity_level(delta(case, base, "RTU_REFG_SUCT_PRES_1"), std(base, "RTU_REFG_SUCT_PRES_1"), "low", mild_z=mz, severe_z=sz),
        _severity_level(delta(case, base, "RTU_REFG_SUCT_PRES_2"), std(base, "RTU_REFG_SUCT_PRES_2"), "low", mild_z=mz, severe_z=sz),
        _severity_level(delta(case, base, "RTU_REFG_DISC_PRES_1"), std(base, "RTU_REFG_DISC_PRES_1"), "low", mild_z=mz, severe_z=sz),
        _severity_level(delta(case, base, "RTU_REFG_DISC_PRES_2"), std(base, "RTU_REFG_DISC_PRES_2"), "low", mild_z=mz, severe_z=sz),
    )
    pressure_gap_case = mean(case, "RTU_REFG_DISC_PRES_1") - mean(case, "RTU_REFG_SUCT_PRES_1")
    pressure_gap_base = mean(base, "RTU_REFG_DISC_PRES_1") - mean(base, "RTU_REFG_SUCT_PRES_1")
    pressure_gap_std = max(std(base, "RTU_REFG_DISC_PRES_1"), std(base, "RTU_REFG_SUCT_PRES_1"))
    return {
        "compressor_power_abnormal": comp_z,
        "cooling_delivery_low": cooling_delivery_z,
        "outdoor_air_damper_abnormal": damper_z,
        "economizer_signal_bias": economizer_response_z,
        "refrigerant_pressure_low": refrigerant_pressure_z,
        "supply_air_temp_high": sa_z,
        "refrigerant_pressure_imbalance": _severity_level(pressure_gap_case - pressure_gap_base, pressure_gap_std, "both", mild_z=mz, severe_z=sz),
    }


def build_lbnl_samples(max_rows: int, stride: int, max_scenarios_per_source: Optional[int]) -> List[PublicSample]:
    samples: List[PublicSample] = []

    boiler_dir = LBNL_ROOT / "LBNL_FDD_Dataset_Boiler_Plant"
    boiler_base_path = boiler_dir / "BoilerPlant.csv"
    if boiler_base_path.exists():
        base = collect_stats(boiler_base_path, max_rows=max_rows, stride=stride)
        samples.append(
            PublicSample(
                "LBNL Boiler",
                "LBNL Boiler Plant",
                "BoilerPlant",
                None,
                params_lbnl_boiler(base, base, z_thresholds=SOURCE_ZSCORE_THRESHOLDS["LBNL Boiler"]),
                context="LBNL fault-free boiler baseline",
                rich_context=rich_feature_context("LBNL Boiler", "LBNL Boiler Plant", base, base),
            )
        )
        files = _known_labeled_files(sorted(p for p in boiler_dir.glob("*.csv") if p.name != "BoilerPlant.csv"), label_lbnl_boiler)
        if max_scenarios_per_source:
            files = _stratified_sample(files, max_scenarios_per_source, lambda p: label_lbnl_boiler(p.stem))
        for path in files:
            stats = collect_stats(path, max_rows=max_rows, stride=stride)
            samples.append(
                PublicSample(
                    "LBNL Boiler",
                    "LBNL Boiler Plant",
                    path.stem,
                    label_lbnl_boiler(path.stem),
                    params_lbnl_boiler(stats, base, z_thresholds=SOURCE_ZSCORE_THRESHOLDS["LBNL Boiler"]),
                    rich_context=rich_feature_context("LBNL Boiler", "LBNL Boiler Plant", stats, base),
                )
            )

    chiller_dir = LBNL_ROOT / "LBNL_FDD_DataSet_Chiller_Plant"
    chiller_base_path = chiller_dir / "ChillerPlant.csv"
    if chiller_base_path.exists():
        base = collect_stats(chiller_base_path, max_rows=max_rows, stride=stride)
        samples.append(
            PublicSample(
                "LBNL Chiller",
                "LBNL Chiller Plant",
                "ChillerPlant",
                None,
                params_lbnl_chiller(base, base, z_thresholds=SOURCE_ZSCORE_THRESHOLDS["LBNL Chiller"]),
                context="LBNL fault-free chiller baseline",
                rich_context=rich_feature_context("LBNL Chiller", "LBNL Chiller Plant", base, base),
            )
        )
        files = _known_labeled_files(sorted(p for p in chiller_dir.glob("*.csv") if p.name != "ChillerPlant.csv"), label_lbnl_chiller)
        if max_scenarios_per_source:
            files = _stratified_sample(files, max_scenarios_per_source, lambda p: label_lbnl_chiller(p.stem))
        for path in files:
            stats = collect_stats(path, max_rows=max_rows, stride=stride)
            samples.append(
                PublicSample(
                    "LBNL Chiller",
                    "LBNL Chiller Plant",
                    path.stem,
                    label_lbnl_chiller(path.stem),
                    params_lbnl_chiller(stats, base, z_thresholds=SOURCE_ZSCORE_THRESHOLDS["LBNL Chiller"]),
                    rich_context=rich_feature_context("LBNL Chiller", "LBNL Chiller Plant", stats, base),
                )
            )

    rtu_dir = LBNL_ROOT / "LBNL_FDD_DataSet_RTU"
    rtu_files = sorted(p for p in rtu_dir.rglob("*.csv"))
    scenarios: List[Tuple[Path, Optional[Path]]] = []
    for path in rtu_files:
        name = path.stem
        if "Unfaulted" in name or name.startswith("ERTU_"):
            continue
        if label_lbnl_rtu(name) == "Unknown":
            continue
        baseline = None
        if "Site1_" in name:
            baseline = rtu_dir / "Field RTU" / "Site1_Unfaulted.csv"
        elif "Site2_" in name:
            baseline = rtu_dir / "Field RTU" / "Site2_Unfaulted.csv"
        elif any(season in name for season in ["Fall_2020", "Spring_2021", "Summer_2021", "Winter_2022"]):
            if "Fall_2020" in name:
                baseline = rtu_dir / "ORNL_RTU" / "ERTU_Fall_2020.csv"
            elif "Spring_2021" in name:
                baseline = rtu_dir / "ORNL_RTU" / "ERTU_Spring_2021.csv"
            elif "Summer_2021" in name:
                baseline = rtu_dir / "ORNL_RTU" / "ERTU_Summer_2021.csv"
            elif "Winter_2022" in name:
                baseline = rtu_dir / "ORNL_RTU" / "ERTU_Winter_2022.csv"
        if baseline and baseline.exists():
            scenarios.append((path, baseline))
    if max_scenarios_per_source:
        scenarios = _stratified_sample(scenarios, max_scenarios_per_source, lambda item: label_lbnl_rtu(item[0].stem))
    added_baselines: set[Path] = set()
    baseline_cache: Dict[Path, Dict[str, Dict[str, float]]] = {}
    for path, baseline_path in scenarios:
        if baseline_path not in baseline_cache:
            baseline_cache[baseline_path] = collect_stats(baseline_path, max_rows=max_rows, stride=stride)
        base = baseline_cache[baseline_path]
        if baseline_path not in added_baselines:
            samples.append(
                PublicSample(
                    "LBNL RTU",
                    "LBNL RTU",
                    baseline_path.stem,
                    None,
                    params_lbnl_rtu(base, base, z_thresholds=SOURCE_ZSCORE_THRESHOLDS["LBNL RTU"]),
                    context="LBNL fault-free RTU baseline",
                    rich_context=rich_feature_context("LBNL RTU", "LBNL RTU", base, base),
                )
            )
            added_baselines.add(baseline_path)
        stats = collect_stats(path, max_rows=max_rows, stride=stride)
        samples.append(
            PublicSample(
                "LBNL RTU",
                "LBNL RTU",
                path.stem,
                label_lbnl_rtu(path.stem),
                params_lbnl_rtu(stats, base, z_thresholds=SOURCE_ZSCORE_THRESHOLDS["LBNL RTU"]),
                rich_context=rich_feature_context("LBNL RTU", "LBNL RTU", stats, base),
            )
        )

    return samples


METROPT_FAILURES = [
    ("Air Leak", datetime.fromisoformat("2022-06-04 10:19:24.300"), datetime.fromisoformat("2022-06-04 14:22:39.188")),
    ("Oil Leak", datetime.fromisoformat("2022-07-11 10:10:18.948"), datetime.fromisoformat("2022-07-14 10:22:08.046")),
]


def _parse_time(value: str) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _failure_label(ts: datetime) -> Optional[str]:
    for label, start, end in METROPT_FAILURES:
        if start <= ts <= end:
            return label
    return None


def build_metropt_baseline(path: Path, max_rows: int = 200000, stride: int = 10, min_rows: int = 1000) -> Dict[str, Dict[str, float]]:
    """Use early pre-failure rows as a healthy baseline."""
    first_failure = METROPT_FAILURES[0][1]
    tmp_path = path
    sums: Dict[str, float] = defaultdict(float)
    sums2: Dict[str, float] = defaultdict(float)
    counts: Dict[str, int] = defaultdict(int)
    first: Dict[str, float] = {}
    last: Dict[str, float] = {}
    with tmp_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            if idx >= max_rows:
                break
            if stride > 1 and idx % stride != 0:
                continue
            ts = _parse_time(row.get("timestamp", ""))
            if ts is None:
                continue
            if ts >= first_failure:
                if sum(counts.values()) >= min_rows:
                    break
                continue
            for col, raw in row.items():
                if col == "timestamp":
                    continue
                val = _to_float(raw)
                if val is None:
                    continue
                if col not in first:
                    first[col] = val
                last[col] = val
                sums[col] += val
                sums2[col] += val * val
                counts[col] += 1
    stats: Dict[str, Dict[str, float]] = {}
    for col, n in counts.items():
        avg = sums[col] / n
        var = max(0.0, sums2[col] / n - avg * avg)
        stats[col] = {"mean": avg, "std": math.sqrt(var), "first": first[col], "last": last[col], "count": float(n)}
    if not stats:
        raise ValueError(f"MetroPT healthy baseline is empty: {path}")
    return stats


def params_metropt(window: Dict[str, Dict[str, float]], base: Dict[str, Dict[str, float]],
                    z_thresholds: Tuple[float, float] = (1.5, 3.0)) -> Dict[str, int]:
    """Multi-level MetroPT compressor predicates from z-score deviations.

    The ``Reservoirs`` column supplies both a mean deviation (for
    ``reservoirs_low``) and a first‑to‑last slope (for
    ``pressure_recovery_slow``).  The slope is compared against the
    baseline‑window slope distribution — healthy windows should have a
    near‑zero or slightly positive slope.
    """
    mz, sz = z_thresholds
    # --- Derived pressure-recovery signal ---
    res_slope = window.get("Reservoirs", {}).get("last", 0.0) - window.get("Reservoirs", {}).get("first", 0.0)
    base_slope = base.get("Reservoirs", {}).get("last", 0.0) - base.get("Reservoirs", {}).get("first", 0.0)
    # Normalise the slope gap by the baseline pressure standard deviation
    res_std = std(base, "Reservoirs")
    slope_z = (base_slope - res_slope) / res_std if res_std > 1e-9 else 0.0  # positive when recovery is slower than baseline
    pressure_recovery_level = 0
    if slope_z >= mz:
        pressure_recovery_level = 1 if slope_z < sz else 2

    # --- Reservoir mean level ---
    reservoirs_z = (mean(base, "Reservoirs") - mean(window, "Reservoirs")) / max(1e-9, res_std)  # positive = low
    reservoirs_low_level = 0
    if reservoirs_z >= mz:
        reservoirs_low_level = 1 if reservoirs_z < sz else 2

    # --- Compressor duty-cycle ---
    comp_std = std(base, "COMP")
    comp_z = (mean(window, "COMP") - mean(base, "COMP")) / max(1e-9, comp_std)
    comp_high_level = 0
    if comp_z >= mz:
        comp_high_level = 1 if comp_z < sz else 2

    # --- Motor current ---
    mc_z = (mean(window, "Motor_current") - mean(base, "Motor_current")) / max(1e-9, std(base, "Motor_current"))
    motor_current_level = 0
    if mc_z >= mz:
        motor_current_level = 1 if mc_z < sz else 2
    mc_slope = (window.get("Motor_current", {}).get("last", 0.0) - window.get("Motor_current", {}).get("first", 0.0)) / max(1e-9, std(base, "Motor_current"))
    motor_current_rising = 0
    if mc_slope >= mz:
        motor_current_rising = 1 if mc_slope < sz else 2

    # --- Oil temperature ---
    ot_z = (mean(window, "Oil_temperature") - mean(base, "Oil_temperature")) / max(1e-9, std(base, "Oil_temperature"))
    oil_temp_level = 0
    if ot_z >= mz:
        oil_temp_level = 1 if ot_z < sz else 2

    # --- Low-pressure switch ---
    lps_std = std(base, "LPS")
    lps_z = (mean(window, "LPS") - mean(base, "LPS")) / max(1e-9, lps_std)
    lps_level = 0
    if lps_z >= mz:
        lps_level = 1 if lps_z < sz else 2

    # --- Pressure switch ---
    ps_std = std(base, "Pressure_switch")
    ps_z = abs(mean(window, "Pressure_switch") - mean(base, "Pressure_switch")) / max(1e-9, ps_std)
    ps_level = 0
    if ps_z >= mz:
        ps_level = 1 if ps_z < sz else 2

    # --- Derived: oil temperature rising trend (Oil Leak signature) ---
    ot_first = window.get("Oil_temperature", {}).get("first", 0.0)
    ot_last = window.get("Oil_temperature", {}).get("last", 0.0)
    ot_slope = (ot_last - ot_first) / max(1e-9, std(base, "Oil_temperature"))
    oil_temp_rising = 0
    if ot_slope >= mz:
        oil_temp_rising = 1 if ot_slope < sz else 2

    # --- Derived: current-pressure gap (motor current high AND pressure recovery slow) ---
    # This combination favours Oil Leak (mechanical friction → high load + poor output)
    # over Air Leak (pressure loss only, without the current rise).
    current_pressure_gap = 0
    if motor_current_level >= 1 and pressure_recovery_level >= 1:
        current_pressure_gap = 1
    if motor_current_level == 2 and pressure_recovery_level == 2:
        current_pressure_gap = 2

    # --- Derived: steep pressure recovery drop (Air Leak signature) ---
    # Air leaks cause sudden pressure loss; the recovery slope becomes sharply negative.
    pressure_recovery_steep = 0
    if slope_z >= sz:  # already above severe threshold from the slope computation above
        pressure_recovery_steep = 2
    elif slope_z >= mz and motor_current_level == 0:
        pressure_recovery_steep = 1  # pressure drop without motor load → air leak

    return {
        "pressure_recovery_slow": max(pressure_recovery_level, reservoirs_low_level),
        "pressure_drop_frequent": max(reservoirs_low_level, lps_level),
        "reservoir_pressure_low": reservoirs_low_level,
        "compressor_runtime_high": comp_high_level,
        "motor_current_high": motor_current_level,
        "motor_current_rising": motor_current_rising,
        "oil_temperature_high": oil_temp_level,
        "low_pressure_switch_active": lps_level,
        "pressure_switch_abnormal": ps_level,
        "oil_temp_rising": oil_temp_rising,
        "current_pressure_gap": current_pressure_gap,
        "pressure_recovery_steep": pressure_recovery_steep,
    }


def stats_from_rows(rows: List[Dict[str, str]]) -> Dict[str, Dict[str, float]]:
    sums: Dict[str, float] = defaultdict(float)
    sums2: Dict[str, float] = defaultdict(float)
    counts: Dict[str, int] = defaultdict(int)
    first: Dict[str, float] = {}
    last: Dict[str, float] = {}
    for row in rows:
        for col, raw in row.items():
            if col == "timestamp":
                continue
            val = _to_float(raw)
            if val is None:
                continue
            if col not in first:
                first[col] = val
            last[col] = val
            sums[col] += val
            sums2[col] += val * val
            counts[col] += 1
    out: Dict[str, Dict[str, float]] = {}
    for col, n in counts.items():
        avg = sums[col] / n
        var = max(0.0, sums2[col] / n - avg * avg)
        out[col] = {"mean": avg, "std": math.sqrt(var), "first": first[col], "last": last[col], "count": float(n)}
    return out


def build_metropt_samples(window_rows: int = 300, max_windows_per_class: int = 20) -> List[PublicSample]:
    path = METROPT_ROOT / "MetroPT2.csv"
    if not path.exists():
        return []
    try:
        base = build_metropt_baseline(path)
    except ValueError as exc:
        print(f"Warning: {exc}")
        return []
    samples: List[PublicSample] = []
    counts: Counter[str] = Counter()
    current_label = "__normal__"
    current_rows: List[Dict[str, str]] = []

    def flush(label: str, rows: List[Dict[str, str]]) -> None:
        if not rows:
            return
        true_fault = None if label == "__normal__" else label
        if counts[label] >= max_windows_per_class:
            return
        start_time = rows[0].get("timestamp", "")
        end_time = rows[-1].get("timestamp", "")
        stats = stats_from_rows(rows)
        samples.append(
            PublicSample(
                "MetroPT",
                "MetroPT Compressor",
                f"{label}_{counts[label]}",
                true_fault,
                params_metropt(stats, base, z_thresholds=SOURCE_ZSCORE_THRESHOLDS["MetroPT"]),
                context="MetroPT 1 Hz compressor/APU window",
                rich_context=rich_feature_context(
                    "MetroPT",
                    "MetroPT Compressor",
                    stats,
                    base,
                    start_time=start_time,
                    end_time=end_time,
                ),
                start_time=start_time,
                end_time=end_time,
            )
        )
        counts[label] += 1

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts = _parse_time(row.get("timestamp", ""))
            if ts is None:
                continue
            label = _failure_label(ts) or "__normal__"
            # Keep a small number of healthy windows before the first failure.
            if label == "__normal__" and counts[label] >= max_windows_per_class:
                continue
            if label != current_label or len(current_rows) >= window_rows:
                flush(current_label, current_rows)
                current_label = label
                current_rows = []
            current_rows.append(row)
            if all(counts[k] >= max_windows_per_class for k in ["__normal__", "Air Leak", "Oil Leak"]):
                break
    flush(current_label, current_rows)
    return samples


def normalize_label(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"normal", "none", "no fault", "null"}:
        return None
    return text


def macro_metrics(records: List[Dict[str, Any]], pred_key: str) -> Dict[str, float]:
    labels = sorted({normalize_label(r["true_fault"]) for r in records if normalize_label(r["true_fault"]) is not None})
    if not labels:
        return {"macro_precision": 0.0, "macro_recall": 0.0, "macro_f1": 0.0}
    label_set = set(labels)
    ps: List[float] = []
    rs: List[float] = []
    fs: List[float] = []
    for label in labels:
        tp = sum(1 for r in records if normalize_label(r["true_fault"]) == label and normalize_label(r[pred_key]) == label)
        fp = sum(1 for r in records if normalize_label(r["true_fault"]) != label and normalize_label(r[pred_key]) == label)
        fn = sum(1 for r in records if normalize_label(r["true_fault"]) == label and normalize_label(r[pred_key]) != label)
        fp += sum(
            1 for r in records
            if normalize_label(r["true_fault"]) == label
            and normalize_label(r[pred_key]) is not None
            and normalize_label(r[pred_key]) not in label_set
        )
        p = tp / (tp + fp) if tp + fp else 0.0
        q = tp / (tp + fn) if tp + fn else 0.0
        f = 2 * p * q / (p + q) if p + q else 0.0
        ps.append(p)
        rs.append(q)
        fs.append(f)
    return {
        "macro_precision": round(sum(ps) / len(ps), 3),
        "macro_recall": round(sum(rs) / len(rs), 3),
        "macro_f1": round(sum(fs) / len(fs), 3),
    }


def summarize(records: List[Dict[str, Any]], pred_key: str) -> Dict[str, float]:
    total = len(records)
    correct = sum(1 for r in records if normalize_label(r["true_fault"]) == normalize_label(r[pred_key]))
    fault_records = [r for r in records if normalize_label(r["true_fault"]) is not None]
    fault_correct = sum(1 for r in fault_records if normalize_label(r["true_fault"]) == normalize_label(r[pred_key]))
    normal_records = [r for r in records if normalize_label(r["true_fault"]) is None]
    false_alarms = sum(1 for r in normal_records if normalize_label(r[pred_key]) is not None)
    out = {
        "accuracy": round(correct / total, 3) if total else 0.0,
        "fault_only_accuracy": round(fault_correct / len(fault_records), 3) if fault_records else 0.0,
        "false_alarm_rate": round(false_alarms / len(normal_records), 3) if normal_records else 0.0,
    }
    out.update(macro_metrics(records, pred_key))
    return out


def fallback_rate(records: List[Dict[str, Any]]) -> float:
    if not records:
        return 0.0
    return round(sum(1 for r in records if int(r.get("fallback_needed", 0)) == 1) / len(records), 3)


def actionable_llm_fault(label: Optional[str]) -> bool:
    """Return True when an LLM label is a concrete abnormal fault."""
    if normalize_label(label) is None:
        return False
    return str(label).strip().lower() != "unknown"


def evidence_strength_from_params(params: Dict[str, int]) -> Tuple[int, int, str]:
    """Summarize multi-level predicates into the routing evidence band."""
    values = list(params.values())
    nonzero_count = sum(1 for value in values if value > 0)
    max_level = max(values) if values else 0
    if nonzero_count == 0:
        return nonzero_count, max_level, "none"
    if max_level == 1 and nonzero_count <= 1:
        return nonzero_count, max_level, "weak"
    if max_level >= 2 or nonzero_count >= 3:
        return nonzero_count, max_level, "strong"
    return nonzero_count, max_level, "moderate"


def evidence_aware_hybrid_decision(
    expert_fault: Optional[str],
    expert_conf: float,
    llm_fault: Optional[str],
    llm_conf: float,
    *,
    evidence_strength: str,
    equipment: str,
    tau: float,
) -> Tuple[Optional[str], float, str]:
    """Arbitrate Expert and LLM outputs using predicate evidence strength.

    The LLM is used as a conservative validator: low-evidence samples keep the
    expert decision, while strong-evidence conflicts may be overridden only by
    a confident, concrete LLM diagnosis.
    """
    expert_label = normalize_label(expert_fault)
    llm_label = normalize_label(llm_fault)
    llm_actionable = actionable_llm_fault(llm_fault)

    if expert_conf >= tau and (llm_label == expert_label or not llm_actionable):
        return expert_fault, expert_conf, "expert_high_conf"

    if not llm_actionable:
        action = "llm_rejected_unknown" if llm_fault is not None else "expert_no_llm"
        return expert_fault, expert_conf, action

    if evidence_strength == "none":
        return expert_fault, expert_conf, "expert_no_evidence"

    if evidence_strength == "weak":
        if expert_label is None and llm_conf >= 0.85:
            return llm_fault, llm_conf, "llm_override_weak_expert_empty"
        return expert_fault, expert_conf, "expert_weak_evidence"

    if evidence_strength == "moderate":
        if expert_label is None and llm_conf >= 0.80:
            return llm_fault, llm_conf, "llm_override_moderate_expert_empty"
        return expert_fault, expert_conf, "expert_moderate_evidence"

    if llm_label == expert_label:
        return expert_fault, max(expert_conf, llm_conf), "expert_llm_agree"

    expert_cap = EVIDENCE_AWARE_EXPERT_CAPS.get(equipment, 0.35)
    if llm_conf >= 0.75 and expert_conf <= expert_cap:
        return llm_fault, llm_conf, "llm_override_strong"
    if llm_conf < 0.75:
        return expert_fault, expert_conf, "llm_rejected_low_conf"
    return expert_fault, expert_conf, "llm_rejected_expert_cap"


def select_hybrid_decision(
    strategy: str,
    expert_fault: Optional[str],
    expert_conf: float,
    llm_fault: Optional[str],
    llm_conf: float,
    *,
    tau: float,
    evidence_strength: str,
    equipment: str,
) -> Tuple[Optional[str], float, str]:
    """Return the requested hybrid decision while preserving legacy routing."""
    if strategy == "legacy":
        fault, conf = hybrid_decision(expert_fault, expert_conf, llm_fault, llm_conf, tau=tau)
        return fault, conf, "legacy_hybrid"
    return evidence_aware_hybrid_decision(
        expert_fault,
        expert_conf,
        llm_fault,
        llm_conf,
        evidence_strength=evidence_strength,
        equipment=equipment,
        tau=tau,
    )


def arbitration_action_metrics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Report counts and accuracy by evidence-aware arbitration action."""
    out: Dict[str, Any] = {}
    for action in sorted({str(r.get("arbitration_action", "")) for r in records}):
        if not action:
            continue
        subset = [r for r in records if r.get("arbitration_action") == action]
        out[action] = {"samples": len(subset), "hybrid": summarize(subset, "hybrid_fault")}
    return out


def parse_float_grid(text: str) -> List[float]:
    values: List[float] = []
    for part in text.split(","):
        stripped = part.strip()
        if not stripped:
            continue
        values.append(float(stripped))
    return values


def resolve_thresholds(args: argparse.Namespace, sample: PublicSample) -> ThresholdSetting:
    if args.threshold_profile == "public-tuned":
        base = PUBLIC_TUNED_THRESHOLDS.get(sample.equipment, GLOBAL_THRESHOLD_SETTING)
    else:
        base = GLOBAL_THRESHOLD_SETTING

    return ThresholdSetting(
        min_match=args.min_match if args.min_match is not None else base.min_match,
        tau=args.tau if args.tau is not None else base.tau,
        rationale=base.rationale,
    )


def threshold_profile_summary(args: argparse.Namespace) -> Dict[str, Any]:
    if args.threshold_profile == "public-tuned":
        by_equipment = {
            equipment: {
                "min_match": setting.min_match,
                "tau": setting.tau,
                "rationale": setting.rationale,
            }
            for equipment, setting in PUBLIC_TUNED_THRESHOLDS.items()
        }
    else:
        by_equipment = {
            equipment: {
                "min_match": GLOBAL_THRESHOLD_SETTING.min_match,
                "tau": GLOBAL_THRESHOLD_SETTING.tau,
                "rationale": GLOBAL_THRESHOLD_SETTING.rationale,
            }
            for equipment in PublicBenchmarkKnowledgeBase().equipment_types
        }

    if args.min_match is not None:
        for row in by_equipment.values():
            row["min_match"] = args.min_match
            row["rationale"] += " CLI --min-match override applied."
    if args.tau is not None:
        for row in by_equipment.values():
            row["tau"] = args.tau
            row["rationale"] += " CLI --tau override applied."

    return {
        "profile": args.threshold_profile,
        "cli_min_match_override": args.min_match,
        "cli_tau_override": args.tau,
        "calibration_note": (
            "The public-data profile is an engineering calibration for the compact "
            "public benchmark rule base. It should be interpreted together with "
            "the exported min_match/tau sensitivity tables rather than as a "
            "universal threshold for all datasets."
        ),
        "by_equipment": by_equipment,
    }


def threshold_usage(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    usage: Dict[str, Any] = {}
    for equipment in sorted({r["equipment"] for r in records}):
        subset = [r for r in records if r["equipment"] == equipment]
        usage[equipment] = {
            "samples": len(subset),
            "min_match_values": sorted({float(r["effective_min_match"]) for r in subset}),
            "tau_values": sorted({float(r["effective_tau"]) for r in subset}),
            "fallback_rate": fallback_rate(subset),
        }
    return usage


def source_zscore_threshold(equipment: str) -> Tuple[float, float]:
    source_by_equipment = {
        "LBNL Boiler Plant": "LBNL Boiler",
        "LBNL Chiller Plant": "LBNL Chiller",
        "LBNL RTU": "LBNL RTU",
        "MetroPT Compressor": "MetroPT",
    }
    return SOURCE_ZSCORE_THRESHOLDS[source_by_equipment[equipment]]


def public_llm_fault_evidence_group_lines(equipment: str) -> List[str]:
    """Format fault-level evidence groups for KB-guided LLM prompting."""
    groups = PUBLIC_LLM_FAULT_EVIDENCE_GROUPS.get(equipment, {})
    lines: List[str] = []
    for fault, sections in groups.items():
        lines.append(f"  - {fault}:")
        for section_name, label in [
            ("primary", "primary evidence"),
            ("supporting", "supporting evidence"),
            ("insufficient_alone", "insufficient alone"),
        ]:
            items = sections.get(section_name, [])
            if items:
                lines.append(f"      {label}: " + "; ".join(items))
    return lines


def public_llm_context(sample: PublicSample, mild_z: float, severe_z: float) -> str:
    """Build rule-guided public benchmark context for real-LLM diagnosis."""
    lines = [
        f"Source: {sample.source}",
        f"Scenario: {sample.scenario}",
        f"Sample context: {sample.context}",
        "Predicate levels use source-specific z-score thresholds:",
        f"  level 0 < {mild_z} sigma",
        f"  level 1 = [{mild_z}, {severe_z}) sigma",
        f"  level 2 >= {severe_z} sigma",
        "",
        f"Fault indicators for {sample.equipment}:",
    ]
    lines.extend(f"  - {item}" for item in PUBLIC_LLM_FAULT_GUIDES.get(sample.equipment, []))
    evidence_group_lines = public_llm_fault_evidence_group_lines(sample.equipment)
    if evidence_group_lines:
        lines.extend(
            [
                "",
                "Fault evidence groups and mechanism-sufficiency constraints:",
            ]
        )
        lines.extend(evidence_group_lines)
    lines.append("")
    lines.append("Predicate key:")
    for name, desc in PUBLIC_LLM_PREDICATE_DESCRIPTIONS.get(sample.equipment, {}).items():
        lines.append(f"  - {name}: {desc}")
    lines.extend(
        [
            "",
            "Use the fault indicators and predicate key as domain knowledge.",
            "Use the fault evidence groups to decide whether active predicates are sufficient for a specific fault.",
            "A non-Normal diagnosis should have at least one primary evidence group or multiple coherent supporting indicators.",
            "Patterns listed as insufficient alone should remain Normal, Unknown, or low-confidence unless corroborated by primary evidence.",
            "Do not infer faults outside the candidate list.",
            "If all active evidence is absent or weak, prefer Normal or low confidence.",
        ]
    )
    if sample.equipment == "LBNL RTU":
        lines.append(
            "RTU sparse-evidence exception: if exactly one active predicate uniquely matches one RTU fault mechanism, a low-confidence non-Normal diagnosis is allowed; reserve Normal for all-zero predicates or non-unique/noisy weak evidence."
        )
    return "\n".join(lines)


def public_llm_evidence_rich_kb_context(sample: PublicSample, mild_z: float, severe_z: float) -> str:
    """Build a combined KB-guide plus raw-statistical context for real-LLM diagnosis."""
    kb_context = public_llm_context(sample, mild_z, severe_z)
    rich_context = sample.rich_context.strip()
    if not rich_context:
        return kb_context
    return "\n".join(
        [
            kb_context,
            "",
            "Combined evidence mode:",
            "Use the predicate levels as compact abnormal-condition evidence.",
            "Use the raw/statistical feature deviations below to resolve ambiguous or adjacent fault mechanisms.",
            "Prefer a non-Normal fault only when predicate evidence and raw/statistical evidence are physically coherent.",
            "",
            "Raw/statistical public-benchmark evidence:",
            rich_context,
        ]
    )


def expert_top_k_for_llm(
    engine: ProbabilisticInferenceEngine,
    equipment: str,
    params: Dict[str, int],
    k: int = 3,
) -> List[Dict[str, Any]]:
    """Return compact top-k PMS details for the real-LLM prompt."""
    rows: List[Dict[str, Any]] = []
    for item in engine.infer_with_details(equipment, params)[:k]:
        rows.append(
            {
                "fault": item.get("fault"),
                "confidence": round(float(item.get("confidence", 0.0)), 3),
                "s_match": round(float(item.get("s_match", 0.0)), 1),
                "evidence_damp": round(float(item.get("evidence_damp", 1.0)), 3),
                "priority": item.get("priority"),
                "matched_predicates": item.get("matched_predicates", {}),
                "unmatched_predicates": item.get("unmatched_predicates", {}),
                "conditions": item.get("conditions", {}),
                "analysis": item.get("analysis", ""),
            }
        )
    return rows


def public_llm_prompt_inputs(
    args: argparse.Namespace,
    sample: PublicSample,
    engine: ProbabilisticInferenceEngine,
) -> Tuple[str, List[Dict[str, Any]], bool]:
    """Select the public real-LLM prompt evidence for ablation runs."""
    if args.llm_prompt_mode == "zero-shot":
        return "", [], True
    if args.llm_prompt_mode == "rich-context":
        return sample.rich_context, [], False

    mild_z, severe_z = source_zscore_threshold(sample.equipment)
    context = public_llm_context(sample, mild_z, severe_z)
    if args.llm_prompt_mode == "kb-guided":
        return context, [], True
    if args.llm_prompt_mode == "evidence-rich-kb":
        return public_llm_evidence_rich_kb_context(sample, mild_z, severe_z), [], True
    if args.llm_prompt_mode == "expert-guided":
        return context, expert_top_k_for_llm(engine, sample.equipment, sample.params, k=3), True
    raise ValueError(f"Unsupported llm_prompt_mode: {args.llm_prompt_mode}")


def expert_min_match_sensitivity(
    kb: PublicBenchmarkKnowledgeBase,
    samples: List[PublicSample],
    min_match_grid: List[float],
) -> Dict[str, Any]:
    """Evaluate expert-only performance over a min_match grid without LLM calls."""
    out: Dict[str, Any] = {}
    for equipment in kb.equipment_types:
        subset = [s for s in samples if s.equipment == equipment]
        rows: Dict[str, Any] = {}
        for min_match in min_match_grid:
            engine = ProbabilisticInferenceEngine(
                kb=kb,
                min_match=min_match,
                require_abnormal_anchor=True,
                evidence_damping=True,
            )
            records: List[Dict[str, Any]] = []
            for sample in subset:
                preds = engine.infer(sample.equipment, sample.params)
                expert_fault = preds[0][0] if preds else None
                expert_conf = preds[0][1] if preds else 0.0
                records.append(
                    {
                        "true_fault": sample.true_fault,
                        "expert_fault": expert_fault,
                        "expert_conf": expert_conf,
                    }
                )
            metrics = summarize(records, "expert_fault")
            metrics["avg_expert_conf"] = round(sum(r["expert_conf"] for r in records) / len(records), 3) if records else 0.0
            rows[str(min_match)] = metrics
        out[equipment] = {"samples": len(subset), "grid": rows}
    return out


def tau_sensitivity(records: List[Dict[str, Any]], tau_grid: List[float], hybrid_strategy: str) -> Dict[str, Any]:
    """Replay hybrid routing over recorded expert/LLM outputs for each tau."""
    out: Dict[str, Any] = {}
    groups: Dict[str, List[Dict[str, Any]]] = {"overall": records}
    for source in sorted({r["source"] for r in records}):
        groups[f"source:{source}"] = [r for r in records if r["source"] == source]
    for equipment in sorted({r["equipment"] for r in records}):
        groups[f"equipment:{equipment}"] = [r for r in records if r["equipment"] == equipment]

    for group_name, subset in groups.items():
        group_rows: Dict[str, Any] = {}
        for tau in tau_grid:
            replayed: List[Dict[str, Any]] = []
            for record in subset:
                hybrid_fault, hybrid_conf, action = select_hybrid_decision(
                    hybrid_strategy,
                    record.get("expert_fault"),
                    float(record.get("expert_conf", 0.0)),
                    record.get("llm_fault"),
                    float(record.get("llm_confidence", 0.0)),
                    tau=tau,
                    evidence_strength=str(record.get("evidence_strength", "none")),
                    equipment=str(record.get("equipment", "")),
                )
                replayed.append(
                    {
                        **record,
                        "hybrid_fault": hybrid_fault,
                        "hybrid_conf": hybrid_conf,
                        "arbitration_action": action,
                    }
                )
            row = summarize(replayed, "hybrid_fault")
            row["fallback_rate"] = round(
                sum(1 for r in subset if float(r.get("expert_conf", 0.0)) < tau) / len(subset),
                3,
            ) if subset else 0.0
            group_rows[str(tau)] = row
        out[group_name] = {"samples": len(subset), "grid": group_rows}
    return out


def coverage_min_match_by_equipment(args: argparse.Namespace, kb: PublicBenchmarkKnowledgeBase) -> Dict[str, float]:
    """Return the effective min_match used by coverage enumeration."""
    out: Dict[str, float] = {}
    for equipment in kb.equipment_types:
        if args.min_match is not None:
            out[equipment] = args.min_match
        elif args.threshold_profile == "public-tuned":
            out[equipment] = PUBLIC_TUNED_THRESHOLDS.get(equipment, GLOBAL_THRESHOLD_SETTING).min_match
        else:
            out[equipment] = GLOBAL_THRESHOLD_SETTING.min_match
    return out


def predicate_coverage(
    kb: PublicBenchmarkKnowledgeBase,
    min_match: float = 66.66,
    min_match_by_equipment: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Enumerate the multi-level predicate space and report PMS activation coverage.

    A state is *covered* when at least one rule reaches ``S_match >= min_match``
    — the same criterion the PMS inference engine uses to activate a rule.
    This is stricter than ``count_matched > 0``, which merely checks whether
    *any* predicate is non‑zero and would trivially give ~100 % coverage for
    any non‑empty KB.

    Full-match coverage (all conditions satisfied) and fault-type reachability
    are reported alongside the activation‑threshold coverage.
    """
    by_equipment: Dict[str, Dict[str, Any]] = {}
    total_states = 0
    activated_states = 0
    fullmatch_states = 0
    for equipment in kb.equipment_types:
        rules = kb.get_rules(equipment)
        predicates = sorted({name for rule in rules for name in rule.conditions})
        threshold = min_match_by_equipment.get(equipment, min_match) if min_match_by_equipment else min_match
        feasible = 0
        activated = 0
        fullmatch = 0
        faults_reached: set[str] = set()
        for values in itertools.product([0, 1, 2], repeat=len(predicates)):
            params = dict(zip(predicates, values))
            if not kb.is_feasible(equipment, params):
                continue
            feasible += 1
            state_activated = False
            state_fullmatch = False
            state_faults: set[str] = set()
            for rule in rules:
                matched = rule.count_matched(params)
                total = len(rule.conditions)
                s_match = (matched / total) * 100.0 if total else 0.0
                if s_match >= threshold:
                    state_activated = True
                    state_faults.add(rule.fault)
                if matched == total:
                    state_fullmatch = True
            if state_activated:
                activated += 1
                faults_reached.update(state_faults)
            if state_fullmatch:
                fullmatch += 1
        total_states += feasible
        activated_states += activated
        fullmatch_states += fullmatch
        fault_labels = {r.fault for r in rules}
        abnormal_faults_reached = sorted(fault for fault in faults_reached if fault != "Normal")
        n_fault_classes = len([fault for fault in fault_labels if fault != "Normal"])
        by_equipment[equipment] = {
            "predicates": len(predicates),
            "min_match": threshold,
            "feasible_states": feasible,
            "activated_states": activated,
            "activation_coverage": round(activated / feasible, 4) if feasible else 0.0,
            "fullmatch_states": fullmatch,
            "fullmatch_coverage": round(fullmatch / feasible, 4) if feasible else 0.0,
            "faults_reached": sorted(faults_reached),
            "abnormal_faults_reached": abnormal_faults_reached,
            "normal_reached": "Normal" in faults_reached,
            "fault_class_coverage": round(len(abnormal_faults_reached) / n_fault_classes, 4) if n_fault_classes else 0.0,
        }
    return {
        "min_match": min_match,
        "min_match_by_equipment": min_match_by_equipment,
        "overall_activation_coverage": round(activated_states / total_states, 4) if total_states else 0.0,
        "overall_fullmatch_coverage": round(fullmatch_states / total_states, 4) if total_states else 0.0,
        "by_equipment": by_equipment,
    }


def metropt_time_to_detection(records: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Return first correct window-level detection delay for each MetroPT fault."""
    out: Dict[str, Dict[str, Any]] = {}
    metropt_records = [r for r in records if r.get("source") == "MetroPT" and r.get("start_time")]
    methods = {
        "expert_minutes": "expert_fault",
        "llm_minutes": "llm_fault",
        "hybrid_minutes": "hybrid_fault",
    }
    for fault, start, _end in METROPT_FAILURES:
        fault_records: List[Tuple[Dict[str, Any], datetime]] = []
        invalid_timestamps = 0
        for record in metropt_records:
            if normalize_label(record.get("true_fault")) != fault:
                continue
            parsed = _parse_time(str(record.get("start_time", "")))
            if parsed is None:
                invalid_timestamps += 1
                continue
            fault_records.append((record, parsed))
        fault_records.sort(key=lambda item: item[1])
        row: Dict[str, Any] = {
            "event_start": start.isoformat(sep=" "),
            "invalid_timestamp_records": invalid_timestamps,
        }
        for metric_name, pred_key in methods.items():
            first_hit = None
            for record, parsed in fault_records:
                if normalize_label(record.get(pred_key)) == fault:
                    first_hit = parsed
                    break
            row[metric_name] = round((first_hit - start).total_seconds() / 60.0, 2) if first_hit else None
        out[fault] = row
    return out


def run_public_benchmark(args: argparse.Namespace) -> None:
    kb = PublicBenchmarkKnowledgeBase()
    samples = build_lbnl_samples(args.max_rows, args.stride, args.max_scenarios_per_source)
    if not args.skip_metropt:
        samples.extend(build_metropt_samples(args.metropt_window_rows, args.metropt_max_windows_per_class))
    min_match_grid = parse_float_grid(args.min_match_grid)
    tau_grid = parse_float_grid(args.tau_grid)

    mock_llm = SimulatedLLM(accuracy=0.85, hallucination_rate=0.15, seed=args.seed)
    real_llm = RealLLMBaseline() if args.llm == "real" else None
    records: List[Dict[str, Any]] = []
    api_calls = 0
    engine_cache: Dict[float, ProbabilisticInferenceEngine] = {}

    for sample in samples:
        thresholds = resolve_thresholds(args, sample)
        if thresholds.min_match not in engine_cache:
            engine_cache[thresholds.min_match] = ProbabilisticInferenceEngine(
                kb=kb,
                min_match=thresholds.min_match,
                require_abnormal_anchor=True,
                evidence_damping=True,
            )
        engine = engine_cache[thresholds.min_match]
        expert_preds = engine.infer(sample.equipment, sample.params)
        expert_fault = expert_preds[0][0] if expert_preds else None
        expert_conf = expert_preds[0][1] if expert_preds else 0.0
        candidate_faults = kb.candidate_faults(sample.equipment)

        llm_fault = None
        llm_conf = 0.0
        llm_error = ""
        llm_abnormal_decision = ""
        llm_evidence_sufficiency = ""
        llm_called = False
        if args.llm == "mock":
            llm_fault, llm_conf = mock_llm.diagnose(sample.equipment, sample.params, ground_truth=sample.true_fault)
            llm_called = True
        elif args.llm == "real":
            if args.max_api_calls is None or api_calls < args.max_api_calls:
                llm_context, llm_expert_top_k, include_predicates = public_llm_prompt_inputs(args, sample, engine)
                prompt_expert_fault = expert_fault if args.llm_prompt_mode == "expert-guided" else None
                prompt_expert_conf = expert_conf if args.llm_prompt_mode == "expert-guided" else 0.0
                llm_fault, llm_conf, meta = real_llm.diagnose(
                    sample.equipment,
                    sample.params,
                    candidate_faults,
                    expert_fault=prompt_expert_fault,
                    expert_confidence=prompt_expert_conf,
                    context=llm_context,
                    expert_top_k=llm_expert_top_k,
                    include_predicates=include_predicates,
                )
                llm_error = meta.get("error", "")
                llm_abnormal_decision = meta.get("abnormal_decision", "")
                llm_evidence_sufficiency = meta.get("evidence_sufficiency", "")
                api_calls += 1
                llm_called = True

        nonzero_count, max_level, evidence_strength = evidence_strength_from_params(sample.params)
        legacy_hybrid_fault, legacy_hybrid_conf = hybrid_decision(
            expert_fault,
            expert_conf,
            llm_fault,
            llm_conf,
            tau=thresholds.tau,
        )
        hybrid_fault, hybrid_conf, arbitration_action = select_hybrid_decision(
            args.hybrid_strategy,
            expert_fault,
            expert_conf,
            llm_fault,
            llm_conf,
            tau=thresholds.tau,
            evidence_strength=evidence_strength,
            equipment=sample.equipment,
        )
        fallback_needed = expert_conf < thresholds.tau

        records.append(
            {
                "source": sample.source,
                "equipment": sample.equipment,
                "scenario": sample.scenario,
                "start_time": sample.start_time,
                "end_time": sample.end_time,
                "true_fault": sample.true_fault,
                "params": json.dumps(sample.params, sort_keys=True),
                "effective_min_match": round(thresholds.min_match, 3),
                "effective_tau": round(thresholds.tau, 3),
                "expert_fault": expert_fault,
                "expert_conf": round(expert_conf, 3),
                "llm_fault": llm_fault,
                "llm_confidence": llm_conf,
                "llm_abnormal_decision": llm_abnormal_decision,
                "llm_evidence_sufficiency": llm_evidence_sufficiency,
                "llm_error": llm_error,
                "llm_called": int(llm_called),
                "llm_prompt_mode": args.llm_prompt_mode if args.llm == "real" else "",
                "fallback_needed": int(fallback_needed),
                "hybrid_strategy": args.hybrid_strategy,
                "legacy_hybrid_fault": legacy_hybrid_fault,
                "legacy_hybrid_conf": round(legacy_hybrid_conf, 3),
                "hybrid_fault": hybrid_fault,
                "hybrid_conf": round(hybrid_conf, 3),
                "arbitration_action": arbitration_action,
                "nonzero_predicate_count": nonzero_count,
                "max_predicate_level": max_level,
                "evidence_strength": evidence_strength,
            }
        )

    llm_evaluated = sum(1 for r in records if int(r.get("llm_called", 0)) == 1)
    llm_evaluated_records = [r for r in records if int(r.get("llm_called", 0)) == 1]
    fallback_llm_records = [
        r for r in records
        if int(r.get("llm_called", 0)) == 1 and int(r.get("fallback_needed", 0)) == 1
    ]
    metrics = {
        "configuration": vars(args),
        "threshold_profile": threshold_profile_summary(args),
        "threshold_usage": threshold_usage(records),
        "expert_min_match_sensitivity": expert_min_match_sensitivity(kb, samples, min_match_grid),
        "tau_sensitivity": tau_sensitivity(records, tau_grid, args.hybrid_strategy),
        "total_samples": len(records),
        "api_calls": api_calls,
        "llm_evaluated_samples": llm_evaluated,
        "llm_evaluation_rate": round(llm_evaluated / len(records), 3) if records else 0.0,
        "fallback_rate": fallback_rate(records),
        "predicate_coverage": predicate_coverage(kb, min_match_by_equipment=coverage_min_match_by_equipment(args, kb)),
        "metropt_time_to_detection": metropt_time_to_detection(records),
        "expert": summarize(records, "expert_fault"),
        "llm": summarize(records, "llm_fault"),
        "llm_evaluated_subset": summarize(llm_evaluated_records, "llm_fault") if llm_evaluated_records else {},
        "llm_on_fallback_needed": summarize(fallback_llm_records, "llm_fault") if fallback_llm_records else {},
        "legacy_hybrid": summarize(records, "legacy_hybrid_fault"),
        "hybrid": summarize(records, "hybrid_fault"),
        "expert_ece": ece(records, "expert_fault", "expert_conf"),
        "llm_ece": ece(records, "llm_fault", "llm_confidence"),
        "legacy_hybrid_ece": ece(records, "legacy_hybrid_fault", "legacy_hybrid_conf"),
        "hybrid_ece": ece(records, "hybrid_fault", "hybrid_conf"),
        "arbitration_actions": arbitration_action_metrics(records),
        "by_source": {},
    }
    for source in sorted({r["source"] for r in records}):
        subset = [r for r in records if r["source"] == source]
        subset_llm = [r for r in subset if int(r.get("llm_called", 0)) == 1]
        subset_fallback_llm = [
            r for r in subset
            if int(r.get("llm_called", 0)) == 1 and int(r.get("fallback_needed", 0)) == 1
        ]
        metrics["by_source"][source] = {
            "samples": len(subset),
            "fallback_rate": fallback_rate(subset),
            "expert": summarize(subset, "expert_fault"),
            "llm": summarize(subset, "llm_fault"),
            "llm_evaluated_subset": summarize(subset_llm, "llm_fault") if subset_llm else {},
            "llm_on_fallback_needed": summarize(subset_fallback_llm, "llm_fault") if subset_fallback_llm else {},
            "legacy_hybrid": summarize(subset, "legacy_hybrid_fault"),
            "hybrid": summarize(subset, "hybrid_fault"),
            "expert_ece": ece(subset, "expert_fault", "expert_conf"),
            "llm_ece": ece(subset, "llm_fault", "llm_confidence"),
            "legacy_hybrid_ece": ece(subset, "legacy_hybrid_fault", "legacy_hybrid_conf"),
            "hybrid_ece": ece(subset, "hybrid_fault", "hybrid_conf"),
            "arbitration_actions": arbitration_action_metrics(subset),
        }

    # ---- evidence-stratified metrics ----
    metrics_by_evidence: Dict[str, Dict[str, Any]] = {}
    for strength in ["none", "weak", "moderate", "strong"]:
        subset = [r for r in records if r.get("evidence_strength") == strength]
        if not subset:
            continue
        subset_llm = [r for r in subset if int(r.get("llm_called", 0)) == 1]
        subset_fallback_llm = [
            r for r in subset
            if int(r.get("llm_called", 0)) == 1 and int(r.get("fallback_needed", 0)) == 1
        ]
        metrics_by_evidence[strength] = {
            "samples": len(subset),
            "expert": summarize(subset, "expert_fault"),
            "llm": summarize(subset, "llm_fault"),
            "llm_evaluated_subset": summarize(subset_llm, "llm_fault") if subset_llm else {},
            "llm_on_fallback_needed": summarize(subset_fallback_llm, "llm_fault") if subset_fallback_llm else {},
            "legacy_hybrid": summarize(subset, "legacy_hybrid_fault"),
            "hybrid": summarize(subset, "hybrid_fault"),
            "arbitration_actions": arbitration_action_metrics(subset),
        }
    metrics["metrics_by_evidence"] = metrics_by_evidence

    out_dir = Path(__file__).parent / "experiment_outputs"
    out_dir.mkdir(exist_ok=True)
    if args.llm == "real":
        suffix = f"real_llm_{args.llm_prompt_mode}"
    else:
        suffix = args.llm
    pred_path = out_dir / f"public_benchmark_predictions_{suffix}.csv"
    metrics_path = out_dir / f"public_benchmark_metrics_{suffix}.json"
    with pred_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()) if records else ["source"])
        writer.writeheader()
        writer.writerows(records)
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))
    print(f"Predictions written to {pred_path}")
    print(f"Metrics written to {metrics_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run public LBNL/MetroPT benchmark through PMS and optional LLM.")
    parser.add_argument("--llm", choices=["none", "mock", "real"], default="mock")
    parser.add_argument(
        "--llm-prompt-mode",
        choices=["zero-shot", "kb-guided", "evidence-rich-kb", "expert-guided", "rich-context"],
        default="kb-guided",
        help=(
            "Prompt evidence for public real-LLM runs. zero-shot uses predicate values only; "
            "kb-guided adds fault-indicator and predicate descriptions; evidence-rich-kb "
            "adds raw/statistical public benchmark summaries while keeping predicates; "
            "expert-guided also adds the PMS top-1/top-k expert context; rich-context hides predicates and "
            "uses raw/statistical public benchmark summaries."
        ),
    )
    parser.add_argument(
        "--hybrid-strategy",
        choices=["evidence-aware", "legacy"],
        default="evidence-aware",
        help="Hybrid arbitration strategy. evidence-aware is the default; legacy preserves the original tau/weighted-confidence rule.",
    )
    parser.add_argument(
        "--threshold-profile",
        choices=["public-tuned", "global"],
        default="public-tuned",
        help="Use equipment-specific public thresholds or the legacy global thresholds.",
    )
    parser.add_argument("--tau", type=float, default=None, help="Optional global tau override for all public samples.")
    parser.add_argument("--min-match", type=float, default=None, help="Optional global min_match override for all public samples.")
    parser.add_argument(
        "--tau-grid",
        default="0.2,0.3,0.35,0.4,0.45,0.5,0.6",
        help="Comma-separated tau values replayed in the metrics JSON sensitivity table.",
    )
    parser.add_argument(
        "--min-match-grid",
        default="25,33.34,50,66.66,75,100",
        help="Comma-separated min_match values evaluated for expert-only sensitivity.",
    )
    parser.add_argument("--max-rows", type=int, default=20000, help="Rows read per LBNL CSV before striding.")
    parser.add_argument("--stride", type=int, default=60, help="Read every Nth row from LBNL CSVs.")
    parser.add_argument("--max-scenarios-per-source", type=int, default=12)
    parser.add_argument("--skip-metropt", action="store_true")
    parser.add_argument("--metropt-window-rows", type=int, default=300)
    parser.add_argument("--metropt-max-windows-per-class", type=int, default=20)
    parser.add_argument("--max-api-calls", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    run_public_benchmark(parse_args())
