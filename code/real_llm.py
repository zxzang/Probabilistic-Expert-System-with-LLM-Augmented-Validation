"""Real LLM client for diagnostic experiments.

The default provider is DeepSeek through an OpenAI-compatible chat-completions
endpoint. API credentials are read from environment variables, optionally loaded
from the project-level `.env` file via `env_utils.load_dotenv`.
"""

from __future__ import annotations

import json
import os
import ssl
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Tuple

from env_utils import load_dotenv

try:
    import certifi
except ImportError:  # pragma: no cover - fallback for minimal environments.
    certifi = None


@dataclass
class LLMResult:
    fault: Optional[str]
    confidence: float
    raw_text: str = ""
    rationale: str = ""
    abnormal_decision: str = ""
    evidence_sufficiency: str = ""
    error: str = ""


class DeepSeekDiagnosticLLM:
    """DeepSeek-backed diagnostic LLM with a small, stable API."""

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: Optional[float] = None,
        max_retries: int = 2,
    ) -> None:
        load_dotenv()
        self.model = model or os.getenv("DEEPSEEK_MODEL", "deepseek-v4-pro")
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        self.base_url = base_url or os.getenv(
            "DEEPSEEK_BASE_URL",
            "https://api.deepseek.com/v1/chat/completions",
        )
        self.timeout = float(timeout or os.getenv("DEEPSEEK_TIMEOUT", "60"))
        self.max_retries = max(0, max_retries)
        self.ssl_context = self._build_ssl_context()
        if not self.api_key:
            raise RuntimeError(
                "DEEPSEEK_API_KEY is not set. Copy .env.example to .env or set "
                "the variable in your shell."
            )

    @staticmethod
    def _build_ssl_context() -> ssl.SSLContext:
        if certifi is not None:
            return ssl.create_default_context(cafile=certifi.where())
        return ssl.create_default_context()

    @staticmethod
    def _clean_json(raw_text: str) -> Dict[str, Any]:
        text = raw_text.strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.lower().startswith("json"):
                text = text[4:].strip()
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end >= start:
            text = text[start : end + 1]
        return json.loads(text)

    @staticmethod
    def _normalize_fault(value: Any) -> str:
        if value is None:
            return "Unknown"
        fault = str(value).strip()
        if not fault:
            return "Unknown"
        if fault.lower() in {"none", "normal", "no fault", "null"}:
            return "Normal"
        return fault

    def _post(self, payload: Dict[str, Any]) -> str:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.base_url,
            data=data,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout, context=self.ssl_context) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
                return body["choices"][0]["message"]["content"]
            except (urllib.error.URLError, urllib.error.HTTPError, KeyError, json.JSONDecodeError) as exc:
                last_error = exc
                if attempt < self.max_retries:
                    time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"DeepSeek API request failed: {last_error}")

    def diagnose(
        self,
        equipment: str,
        params: Dict[str, Any],
        candidate_faults: Iterable[str],
        expert_fault: Optional[str] = None,
        expert_confidence: float = 0.0,
        context: str = "",
        expert_top_k: Optional[Iterable[Dict[str, Any]]] = None,
        include_predicates: bool = True,
        abnormal_summary: Optional[Iterable[Dict[str, Any]]] = None,
    ) -> LLMResult:
        """Ask the real LLM to diagnose one abnormal-parameter vector.

        The prompt explicitly explains the multi-level predicate encoding
        (0 = normal, 1 = mild, 2 = severe) and provides conservative
        diagnostic heuristics so the LLM does not over-interpret isolated
        level‑1 predicates as faults.
        """
        candidates = [c for c in dict.fromkeys(candidate_faults) if c]
        if "Normal" not in candidates:
            candidates.append("Normal")
        if "Unknown" not in candidates:
            candidates.append("Unknown")

        # ---- detect input format and build appropriate prompt ----
        # Predicate-encoded inputs have keys like  boiler_fuel_high,
        # pressure_recovery_slow, … and values in {0, 1, 2}.
        # Continuous inputs have keys like  pressure, temperature, … and
        # arbitrary float values.
        _predicate_suffixes = (
            "_high", "_low", "_bias", "_abnormal", "_active",
            "_frequent", "_slow", "_stuck",
            "_rising", "_gap", "_steep",
        )
        param_values = [v for v in params.values() if isinstance(v, (int, float))]
        is_predicate_encoded = (
            param_values
            and all(isinstance(v, int) and v in {0, 1, 2} for v in param_values)
            and sum(1 for k in params if any(k.endswith(s) for s in _predicate_suffixes)) >= 2
        )

        if is_predicate_encoded:
            # Multi-level predicate inputs (public benchmark)
            level2_preds = sorted(k for k, v in params.items() if isinstance(v, (int, float)) and v >= 2)
            level1_preds = sorted(k for k, v in params.items() if isinstance(v, (int, float)) and v == 1)
            level0_preds = sorted(k for k, v in params.items() if isinstance(v, (int, float)) and v == 0)

            if include_predicates:
                system_msg = (
                    "You are an industrial fault-diagnosis assistant.  "
                    "You receive abnormal-parameter PREDICATES encoded as severity levels:\n"
                    "  0 = normal      (below the source-specific mild threshold)\n"
                    "  1 = mild        (between the source-specific mild and severe thresholds)\n"
                    "  2 = severe      (at or above the source-specific severe threshold)\n\n"
                    "The exact z-score thresholds, when available, are supplied in the context.  "
                    "Do not assume that every dataset uses 1.5/3.0 sigma thresholds.\n\n"
                    "DIAGNOSTIC RULES you MUST follow:\n"
                    "1. First perform an abnormal-sufficiency self-check before "
                    "choosing any fault label. Set evidence_sufficiency to one "
                    "of none, weak, moderate, or strong, and set "
                    "abnormal_decision to normal, abnormal, or uncertain.\n"
                    "2. A single level-1 predicate is usually measurement noise during "
                    "normal operation - prefer 'Normal' with low-medium confidence (0.3-0.5) "
                    "unless supplied expert context strongly disagrees.\n"
                    "3. Two or more level-1 predicates, or any level-2 predicate, is a "
                    "genuine signal - diagnose the most specific matching fault.\n"
                    "4. When no predicate exceeds level 0, set evidence_sufficiency='none', "
                    "abnormal_decision='normal', and answer 'Normal' with high "
                    "confidence (>= 0.85).\n"
                    "5. When evidence_sufficiency is none or weak, default to "
                    "'Normal' unless multiple physically coherent deviations in "
                    "the context support the same candidate fault. Do not diagnose "
                    "a specific fault from a single isolated mild predicate.\n"
                    "6. Calibrate your confidence:\n"
                    "   - 0.3-0.5 : weak evidence  (single level-1, ambiguous)\n"
                    "   - 0.5-0.7 : moderate evidence  (two level-1, or one level-2)\n"
                    "   - 0.7-0.9 : strong evidence  (three+ level-1, or two+ level-2)\n"
                    "   - 0.9-1.0 : very strong evidence  (multiple level-2 predicates "
                    "matching a single fault signature)\n"
                    "7. Only output faults from the supplied candidate list.  "
                    "8. If the context contains a fault-indicator guide or predicate "
                    "key, use it as domain knowledge for mapping active predicates "
                    "to candidate faults. Prefer the guide over generic word "
                    "association from predicate names.\n"
                    "9. If the context contains fault evidence groups or "
                    "mechanism-sufficiency constraints, use them to decide whether "
                    "the active predicates are sufficient for a specific non-Normal "
                    "fault. Do not promote patterns marked insufficient-alone to a "
                    "specific fault unless corroborating primary evidence is present.\n"
                    "10. Use expert_system_top_k, when supplied, as supporting evidence, but do not "
                    "copy it blindly; compare its matched predicates against the "
                    "fault-indicator guide.\n"
                    "11. If the context also contains raw/statistical feature deviations, "
                    "use them to check whether the predicate-based fault mechanism is "
                    "physically coherent; do not diagnose a specific fault from an "
                    "isolated predicate when the raw/statistical evidence is weak or "
                    "inconsistent.\n"
                    "Return JSON only."
                )
            else:
                system_msg = (
                    "You are an industrial fault-diagnosis assistant. "
                    "You receive a structured public-benchmark statistical summary "
                    "computed from raw time-series windows and healthy baselines. "
                    "Do not assume access to the expert system predicates, rule ranking, "
                    "or scenario filename. Diagnose from the feature deviations, trends, "
                    "candidate fault list, and mechanism notes in the context.\n\n"
                    "DIAGNOSTIC RULES you MUST follow:\n"
                    "1. First perform an abnormal-sufficiency self-check before "
                    "choosing any fault label. Set evidence_sufficiency to one "
                    "of none, weak, moderate, or strong, and set "
                    "abnormal_decision to normal, abnormal, or uncertain.\n"
                    "2. Use large absolute z_mean values as mean-shift evidence and large "
                    "absolute z_trend values as dynamic trend evidence.\n"
                    "3. Prefer Normal when deviations are small, scattered, or physically "
                    "inconsistent with the supplied fault mechanisms.\n"
                    "4. If evidence_sufficiency is none or weak, default to "
                    "'Normal' unless multiple physically coherent deviations support "
                    "the same candidate fault. Do not diagnose Air Leak, Oil Leak, "
                    "or other specific faults from isolated pressure, temperature, "
                    "or trend fluctuations.\n"
                    "5. Prefer the most specific fault whose physical mechanism explains "
                    "the largest and most coherent deviations.\n"
                    "6. Only output faults from the supplied candidate list. "
                    "Return JSON only."
                )
            user_msg = {
                "equipment": equipment,
                "candidate_faults": candidates,
                "context": context,
                "required_json_schema": {
                    "abnormal_decision": "normal, abnormal, or uncertain",
                    "evidence_sufficiency": "none, weak, moderate, or strong",
                    "fault": "one of candidate_faults",
                    "confidence": "number between 0 and 1",
                    "rationale": "short reason referencing the evidence that supports your diagnosis",
                },
            }
            if include_predicates:
                user_msg["predicate_values"] = dict(sorted(params.items()))
                user_msg["severity_summary"] = {
                    "level_2_severe": level2_preds,
                    "level_1_mild": level1_preds,
                    "level_0_normal": level0_preds,
                    "count_level_2": len(level2_preds),
                    "count_level_1": len(level1_preds),
                    "count_level_0": len(level0_preds),
                }
            top_k_rows = list(expert_top_k or [])
            if expert_fault is not None or expert_confidence > 0:
                user_msg["expert_system_opinion"] = {
                    "top_fault": expert_fault,
                    "confidence": expert_confidence,
                    "note": "The expert system uses a probabilistic matching engine. "
                            "Its confidence may be inflated when only one rule activates. "
                            "Use it as a reference, NOT as ground truth.",
                }
            if top_k_rows:
                user_msg["expert_system_top_k"] = top_k_rows
        else:
            # Continuous parameter inputs (synthetic subset)
            system_msg = (
                "You are an industrial fault-diagnosis assistant.  "
                "You receive CONTINUOUS sensor readings together with "
                "FAULT-DETECTION THRESHOLDS in the context.  "
                "Each threshold is a boundary that, when crossed, indicates "
                "an abnormal condition:\n"
                "  - ``>`` means values ABOVE this threshold are abnormally high.\n"
                "  - ``<`` means values BELOW this threshold are abnormally low.\n\n"
                "DIAGNOSTIC RULES you MUST follow:\n"
                "1. Compare each sensor reading against EVERY threshold for that "
                "parameter.  A reading that violates a '>' threshold is too HIGH; "
                "a reading that violates a '<' threshold is too LOW.\n"
                "2. If one parameter crosses multiple thresholds, count it as "
                "one abnormal parameter and use the stricter crossed threshold "
                "as severity evidence.\n"
                "3. When NO readings violate any threshold, answer 'Normal' with "
                "high confidence (≥ 0.85).\n"
                "4. When 1 reading violates a threshold, the evidence is weak — "
                "answer with low confidence (0.3-0.5).\n"
                "5. When 2-3 readings violate thresholds, diagnose the most "
                "specific matching fault from the candidate list.\n"
                "6. Calibrate your confidence by the number and severity of "
                "threshold violations:\n"
                "   - 0.3-0.5 : 1 parameter slightly out of bounds\n"
                "   - 0.5-0.7 : 2 parameters out of bounds\n"
                "   - 0.7-0.9 : 3+ parameters out of bounds, matching a fault pattern\n"
                "   - 0.9-1.0 : multiple parameters far past their thresholds\n"
                "7. Only output faults from the supplied candidate list.  "
                "Return JSON only."
            )
            user_msg = {
                "equipment": equipment,
                "sensor_readings": params,
                "candidate_faults": candidates,
                "context": context,
                "required_json_schema": {
                    "fault": "one of candidate_faults",
                    "confidence": "number between 0 and 1",
                    "rationale": "short reason referencing which readings violate the supplied fault-detection thresholds and what fault that suggests",
                },
            }
            summary_rows = list(abnormal_summary or [])
            if summary_rows:
                user_msg["abnormal_summary"] = summary_rows
            top_k_rows = list(expert_top_k or [])
            if expert_fault is not None or expert_confidence > 0:
                user_msg["expert_system_opinion"] = {
                    "top_fault": expert_fault,
                    "confidence": round(expert_confidence, 3),
                    "note": "The expert system uses a probabilistic matching engine "
                            "trained on domain knowledge.  Use it as a reference, "
                            "NOT as ground truth — it can make mistakes on noisy data.",
                }
            if top_k_rows:
                user_msg["expert_system_top_k"] = top_k_rows
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": json.dumps(user_msg, ensure_ascii=False)},
            ],
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }

        raw = ""
        try:
            raw = self._post(payload)
            parsed = self._clean_json(raw)
            fault = self._normalize_fault(parsed.get("fault"))
            if fault not in candidates:
                fault = "Unknown"
            confidence = float(parsed.get("confidence", 0.0))
            confidence = max(0.0, min(1.0, confidence))
            return LLMResult(
                fault=fault,
                confidence=round(confidence, 3),
                raw_text=raw,
                rationale=str(parsed.get("rationale", "")),
                abnormal_decision=str(parsed.get("abnormal_decision", "")),
                evidence_sufficiency=str(parsed.get("evidence_sufficiency", "")),
            )
        except Exception as exc:
            return LLMResult(fault="Unknown", confidence=0.0, raw_text=raw, error=str(exc))


class RealLLMBaseline:
    """Facade with the same conceptual role as `LLMBaseline`, but real API-backed."""

    def __init__(self, client: Optional[DeepSeekDiagnosticLLM] = None) -> None:
        self.client = client or DeepSeekDiagnosticLLM()

    def diagnose(
        self,
        equipment: str,
        params: Dict[str, Any],
        candidate_faults: Iterable[str],
        expert_fault: Optional[str] = None,
        expert_confidence: float = 0.0,
        context: str = "",
        expert_top_k: Optional[Iterable[Dict[str, Any]]] = None,
        include_predicates: bool = True,
        abnormal_summary: Optional[Iterable[Dict[str, Any]]] = None,
    ) -> Tuple[Optional[str], float, Dict[str, str]]:
        result = self.client.diagnose(
            equipment=equipment,
            params=params,
            candidate_faults=candidate_faults,
            expert_fault=expert_fault,
            expert_confidence=expert_confidence,
            context=context,
            expert_top_k=expert_top_k,
            include_predicates=include_predicates,
            abnormal_summary=abnormal_summary,
        )
        meta = {
            "rationale": result.rationale,
            "raw_text": result.raw_text,
            "abnormal_decision": result.abnormal_decision,
            "evidence_sufficiency": result.evidence_sufficiency,
            "error": result.error,
        }
        return result.fault, result.confidence, meta
