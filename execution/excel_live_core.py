# execution/excel_live_core.py
# ============================================================
# Excel dependency სრულად ამოღებულია.
# წონები და threshold-ები პირდაპირ ENV-იდან იკითხება.
# CORE_VERSION: 2026-03-21.no-excel.v1
# ============================================================
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

CORE_VERSION = "2026-03-21.no-excel.v1"


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        if v is None:
            return default
        return float(v)
    except Exception:
        return default


def _env_float(name: str, default: float) -> float:
    v = os.getenv(name)
    if v is None or str(v).strip() == "":
        return default
    return _safe_float(v, default)


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return str(v).strip().lower() in ("1", "true", "yes", "y", "on")


@dataclass
class CoreInputs:
    trend_strength: float       # 0..1
    structure_ok: bool          # True/False
    volume_score: float         # 0..1
    risk_state: str             # OK / REDUCE / KILL
    confidence_score: float     # 0..1
    volatility_regime: str      # LOW / NORMAL / EXTREME
    liquidity_regime: str = "EXPANSION"     # EXPANSION / CONTRACTION
    macro_risk_level: str = "LOW_RISK"      # LOW_RISK / HIGH_RISK
    shock_absorber: str = "NORMAL"          # NORMAL / REDUCE_EXPOSURE


class ExcelLiveCore:
    """
    ENV-based Live Core — Excel dependency ამოღებულია.

    წონები (ENV-ით კონფიგურირებადი):
      WEIGHT_TREND        default=0.25
      WEIGHT_STRUCTURE    default=0.20
      WEIGHT_VOLUME       default=0.15
      WEIGHT_RISK         default=0.15
      WEIGHT_CONFIDENCE   default=0.15
      WEIGHT_VOLATILITY   default=0.10

    Threshold-ები (ENV-ით კონფიგურირებადი):
      THRESHOLD_TREND     default=0.60
      THRESHOLD_VOLUME    default=0.50
      THRESHOLD_CONF      default=0.64

    Soft Volume Override:
      ENABLE_SOFT_VOLUME_OVERRIDE  default=true
      SOFT_VOLUME_AI_MIN           default=0.60
      SOFT_VOLUME_RELAX            default=0.20
      SOFT_VOLUME_REQUIRE_VOLBAND  default=false
    """

    def __init__(self, workbook_path: str = ""):
        # workbook_path პარამეტრი შენარჩუნებულია უკუთავსებადობისთვის
        # (signal_generator.py-ი კვლავ გადასცემს — უბრალოდ იგნორდება)

        # --- წონები Excel-ის WEIGHT_THRESHOLD_MATRIX-იდან ---
        self.w_trend    = _env_float("WEIGHT_TREND",      0.25)
        self.w_struct   = _env_float("WEIGHT_STRUCTURE",  0.20)
        self.w_volconf  = _env_float("WEIGHT_VOLUME",     0.15)
        self.w_risk     = _env_float("WEIGHT_RISK",       0.15)
        self.w_conf     = _env_float("WEIGHT_CONFIDENCE", 0.15)
        self.w_vol      = _env_float("WEIGHT_VOLATILITY", 0.10)

        # --- threshold-ები Excel-ის WEIGHT_THRESHOLD_MATRIX-იდან ---
        self.th_trend   = _env_float("THRESHOLD_TREND",  0.60)
        self.th_volume  = _env_float("THRESHOLD_VOLUME", 0.50)
        self.th_conf    = _env_float("THRESHOLD_CONF",   0.64)

        # --- AI execute threshold (ENV-ით კონტროლირებადი) ---
        self.ai_execute_min = _env_float("AI_EXECUTE_MIN_SCORE", 0.55)
        self.enable_soft_volume_override  = _env_bool("ENABLE_SOFT_VOLUME_OVERRIDE", True)
        self.soft_volume_ai_min           = _env_float("SOFT_VOLUME_AI_MIN",  0.60)
        self.soft_volume_relax            = _env_float("SOFT_VOLUME_RELAX",   0.20)
        self.soft_volume_require_volband  = _env_bool("SOFT_VOLUME_REQUIRE_VOLBAND", False)

    def _macro_gate(self, inp: CoreInputs) -> str:
        if inp.liquidity_regime == "CONTRACTION":
            return "BLOCK"
        if inp.macro_risk_level == "HIGH_RISK":
            return "BLOCK"
        if inp.shock_absorber == "REDUCE_EXPOSURE":
            return "BLOCK"
        return "ALLOW"

    def _vol_allowed(self, regime: str) -> bool:
        return regime in ("LOW", "NORMAL")

    def _score(self, inp: CoreInputs) -> float:
        risk_num   = 1.0 if inp.risk_state == "OK" else (0.5 if inp.risk_state == "REDUCE" else 0.0)
        vol_num    = 1.0 if inp.volatility_regime == "NORMAL" else (0.8 if inp.volatility_regime == "LOW" else 0.0)
        struct_num = 1.0 if inp.structure_ok else 0.0

        total = (
            inp.trend_strength  * self.w_trend   +
            struct_num          * self.w_struct   +
            inp.volume_score    * self.w_volconf  +
            risk_num            * self.w_risk     +
            inp.confidence_score * self.w_conf    +
            vol_num             * self.w_vol
        )
        return _clamp(total, 0.0, 1.0)

    def decide(self, inp: CoreInputs) -> Dict[str, Any]:
        ai_score   = self._score(inp)
        macro_gate = self._macro_gate(inp)

        trend_ok   = inp.trend_strength    >= self.th_trend
        conf_ok    = inp.confidence_score  >= self.th_conf
        struct_ok  = bool(inp.structure_ok)
        risk_ok    = inp.risk_state != "KILL"
        volband_ok = self._vol_allowed(inp.volatility_regime)

        # strict volume check
        vol_ok_strict = inp.volume_score >= self.th_volume

        # soft volume override
        soft_vol_th = _clamp(self.th_volume - self.soft_volume_relax, 0.0, 1.0)
        vol_ok_soft = False

        if self.enable_soft_volume_override:
            other_gates_ok = (trend_ok and conf_ok and struct_ok and risk_ok)
            volband_req_ok = (volband_ok if self.soft_volume_require_volband else True)
            if other_gates_ok and volband_req_ok and ai_score >= self.soft_volume_ai_min:
                vol_ok_soft = inp.volume_score >= soft_vol_th

        vol_ok = vol_ok_strict or vol_ok_soft

        active_strategy    = "YES" if (trend_ok and struct_ok and vol_ok and conf_ok and risk_ok and volband_ok) else "NO"
        # AI_EXECUTE_MIN_SCORE — ENV-ით კონტროლირებადი (default=0.55)
        # .env-ში: AI_EXECUTE_MIN_SCORE=0.55
        final_trade_decision = "EXECUTE" if (macro_gate == "ALLOW" and active_strategy == "YES" and ai_score >= self.ai_execute_min) else "STAND_BY"

        return {
            "ai_score":              ai_score,
            "macro_gate":            macro_gate,
            "active_strategy":       active_strategy,
            "final_trade_decision":  final_trade_decision,
            "reasons": {
                "core_version":      CORE_VERSION,

                "trend_strength":    inp.trend_strength,
                "trend_th":          self.th_trend,
                "trend_ok":          trend_ok,

                "structure_ok":      struct_ok,

                "volume_score":      inp.volume_score,
                "volume_th":         self.th_volume,
                "volume_th_soft":    soft_vol_th,
                "volume_ok_strict":  vol_ok_strict,
                "volume_ok_soft":    vol_ok_soft,
                "volume_ok":         vol_ok,

                "confidence_score":  inp.confidence_score,
                "conf_th":           self.th_conf,
                "confidence_ok":     conf_ok,

                "risk_state":        inp.risk_state,
                "risk_ok":           risk_ok,

                "volatility_regime": inp.volatility_regime,
                "volband_ok":        volband_ok,

                "soft_volume_override_enabled": bool(self.enable_soft_volume_override),
                "soft_volume_ai_min":           self.soft_volume_ai_min,
                "soft_volume_relax":            self.soft_volume_relax,
                "soft_volume_require_volband":  bool(self.soft_volume_require_volband),
            },
        }
