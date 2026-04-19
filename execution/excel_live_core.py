# execution/excel_live_core.py
# ============================================================
# DCA-optimized Live Core — ENV-only, no hardcoded thresholds.
# CORE_VERSION: 2026-04-05.dca-clean.v2
# ============================================================
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional


CORE_VERSION = "2026-04-05.dca-clean.v2"


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


@dataclass
class CoreInputs:
    trend_strength:    float   # 0..1
    structure_ok:      bool    # True/False
    volume_score:      float   # 0..1
    risk_state:        str     # OK / REDUCE / KILL
    confidence_score:  float   # 0..1
    volatility_regime: str     # LOW / NORMAL / EXTREME
    liquidity_regime:  str = "EXPANSION"
    macro_risk_level:  str = "LOW_RISK"
    shock_absorber:    str = "NORMAL"


class ExcelLiveCore:
    """
    DCA Live Core — ყველა threshold ENV-იდან.
    active_strategy შემოწმება ამოღებულია — მხოლოდ ai_score გადის.

    ENV პარამეტრები:
      WEIGHT_TREND        default=0.40
      WEIGHT_VOLUME       default=0.20
      WEIGHT_CONFIDENCE   default=0.20
      WEIGHT_RISK         default=0.10
      WEIGHT_VOLATILITY   default=0.10

      AI_EXECUTE_MIN_SCORE  default=0.25
      THRESHOLD_VOLUME      default=0.15
    """

    def __init__(self, workbook_path: str = ""):
        # წონები — DCA-ისთვის trend და confidence პრიორიტეტული
        self.w_trend   = _env_float("WEIGHT_TREND",       0.40)
        self.w_volconf = _env_float("WEIGHT_VOLUME",      0.20)
        self.w_conf    = _env_float("WEIGHT_CONFIDENCE",  0.20)
        self.w_risk    = _env_float("WEIGHT_RISK",        0.10)
        self.w_vol     = _env_float("WEIGHT_VOLATILITY",  0.10)

        # AI execute threshold — ENV-იდან, დაბალი default DCA-ისთვის
        self.ai_execute_min = _env_float("AI_EXECUTE_MIN_SCORE", 0.25)

        # Volume threshold — დაბალი, BTC/BNB ყოველთვის გადის
        self.th_volume = _env_float("THRESHOLD_VOLUME", 0.15)

    def _macro_gate(self, inp: CoreInputs) -> str:
        if inp.macro_risk_level == "HIGH_RISK":
            return "BLOCK"
        if inp.shock_absorber == "REDUCE_EXPOSURE":
            return "BLOCK"
        return "ALLOW"

    def _score(self, inp: CoreInputs) -> float:
        risk_num = 1.0 if inp.risk_state == "OK" else (0.5 if inp.risk_state == "REDUCE" else 0.0)
        vol_num  = 1.0 if inp.volatility_regime in ("LOW", "NORMAL") else 0.0

        total = (
            inp.trend_strength   * self.w_trend   +
            inp.volume_score     * self.w_volconf  +
            inp.confidence_score * self.w_conf     +
            risk_num             * self.w_risk     +
            vol_num              * self.w_vol
        )
        return _clamp(total, 0.0, 1.0)

    def decide(self, inp: CoreInputs) -> Dict[str, Any]:
        ai_score   = self._score(inp)
        macro_gate = self._macro_gate(inp)

        risk_ok    = inp.risk_state != "KILL"
        vol_ok     = inp.volume_score >= self.th_volume
        volband_ok = inp.volatility_regime in ("LOW", "NORMAL")

        # DCA: active_strategy შემოწმება ამოღებულია.
        # მხოლოდ 3 პირობა: macro OK + risk OK + ai_score >= threshold
        # trend/struct/conf — score-შია გათვალისწინებული, ცალკე არ ბლოკავს
        final_trade_decision = (
            "EXECUTE"
            if (
                macro_gate == "ALLOW"
                and risk_ok
                and volband_ok
                and ai_score >= self.ai_execute_min
            )
            else "STAND_BY"
        )

        return {
            "ai_score":             ai_score,
            "macro_gate":           macro_gate,
            "active_strategy":      "DCA_SIMPLIFIED",
            "final_trade_decision": final_trade_decision,
            "reasons": {
                "core_version":     CORE_VERSION,
                "trend_strength":   inp.trend_strength,
                "volume_score":     inp.volume_score,
                "volume_th":        self.th_volume,
                "volume_ok":        vol_ok,
                "confidence_score": inp.confidence_score,
                "risk_state":       inp.risk_state,
                "risk_ok":          risk_ok,
                "volatility_regime":inp.volatility_regime,
                "volband_ok":       volband_ok,
                "ai_execute_min":   self.ai_execute_min,
            },
        }
