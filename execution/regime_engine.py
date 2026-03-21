# execution/regime_engine.py
# ============================================================
# Market Regime Engine — v2 (fully corrected + ENV-aligned)
# ============================================================
# ძველი ვერსია (stub):
#   - detect_regime(trend, vol) → მხოლოდ 3 რეჟიმი
#   - apply(regime) → მინიმალური ლოგიკა
#   - signal_generator.py-სთან სრული incompat.
#
# ახალი ვერსია v2:
#   - 5 რეჟიმი: BULL / UNCERTAIN / SIDEWAYS / BEAR / VOLATILE
#   - ENV threshold-ები (REGIME_BULL_TREND_MIN, REGIME_SIDEWAYS_ATR_MAX)
#   - signal_generator.py-ს _vol_regime() + detect_regime() logic-ს ემთხვევა
#   - apply() → სრული trade params override (TP/SL/SKIP)
#   - ATR-based TP/SL multipliers (ENV-კონფიგურირებადი)
#   - SL Cooldown state tracking (per-symbol)
#   - thread-safe (instance-level state, არ არის global)
# ============================================================
from __future__ import annotations

import os
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple

logger = logging.getLogger("gbm")


def _ef(name: str, default: float) -> float:
    v = os.getenv(name)
    try:
        return float(v) if v is not None else default
    except Exception:
        return default


def _ei(name: str, default: int) -> int:
    v = os.getenv(name)
    try:
        return int(v) if v is not None else default
    except Exception:
        return default


# ──────────────────────────────────────────────
# ENV THRESHOLDS (ყველა .env ფაილს ემთხვევა)
# ──────────────────────────────────────────────
_BULL_TREND_MIN    = _ef("REGIME_BULL_TREND_MIN",    0.45)
_SIDEWAYS_ATR_MAX  = _ef("REGIME_SIDEWAYS_ATR_MAX",  0.28)
_VOLATILE_ATR_MIN  = 1.50        # signal_generator-ის _vol_regime() ≥ 2.0 = EXTREME
                                  # 1.5 <= atr < 2.0 → VOLATILE (pre-extreme guard)
_BEAR_TREND_MAX    = -0.10

# ATR-based TP/SL multipliers (Strategy B)
_ATR_TP_BULL       = _ef("ATR_MULT_TP_BULL", 3.0)
_ATR_SL_BULL       = _ef("ATR_MULT_SL_BULL", 1.0)
_ATR_TP_UNCERTAIN  = 2.5
_ATR_SL_UNCERTAIN  = 1.0
_MIN_TP            = _ef("MIN_NET_PROFIT_PCT", 0.50)
_MIN_SL            = 0.20
_MAX_TP            = 4.0
_MAX_SL            = 1.5

# Fixed TP/SL fallback (Strategy A / DEMO mode)
_TP_PCT_FIXED      = _ef("TP_PCT", 1.8)
_SL_PCT_FIXED      = _ef("SL_PCT", 0.5)

# SL Cooldown
_SL_COOLDOWN_N     = _ei("SL_COOLDOWN_AFTER_N",      2)
_SL_PAUSE_SECONDS  = _ei("SL_COOLDOWN_PAUSE_SECONDS", 1800)


class MarketRegimeEngine:
    """
    Regime detection + trade param resolution.

    Usage (signal_generator.py / execution_engine.py):

        from execution.regime_engine import MarketRegimeEngine
        _regime_engine = MarketRegimeEngine()

        # per-tick:
        regime = _regime_engine.detect_regime(trend=0.6, atr_pct=0.45)
        params = _regime_engine.apply(regime, atr_pct=0.45, symbol="BTC/USDT")
        if params["SKIP_TRADING"]:
            continue
        tp = params["TP_PCT"]
        sl = params["SL_PCT"]

        # after trade close:
        _regime_engine.notify_outcome("BTC/USDT", "SL")
        _regime_engine.notify_outcome("BTC/USDT", "TP")
    """

    def __init__(self, config=None):
        # config პარამეტრი შენარჩუნებულია backward-compat-ისთვის
        # (ძველი კოდი MarketRegimeEngine(config) გადასცემს)
        self._config = config

        # SL Cooldown per-symbol state
        self._consecutive_sl: Dict[str, int]             = {}
        self._sl_pause_until: Dict[str, Optional[datetime]] = {}

    # ─────────────────────────────────────────────
    # REGIME DETECTION
    # ─────────────────────────────────────────────

    def detect_regime(self, trend: float, atr_pct: float = 0.0, vol: float = None) -> str:
        """
        5-state regime classifier.

        Args:
            trend:   0..1 trend strength (signal_generator._trend_strength() output)
            atr_pct: ATR as % of price (signal_generator._atr_pct() output)
            vol:     alias for atr_pct (backward compat with old API: detect_regime(trend, vol))

        Returns:
            "BULL" | "UNCERTAIN" | "SIDEWAYS" | "BEAR" | "VOLATILE"
        """
        if vol is not None and atr_pct == 0.0:
            # ძველი API: detect_regime(trend, vol) — vol იყო raw value, არა %
            # ახლა atr_pct-ად ვიყენებთ (safe fallback)
            atr_pct = float(vol)

        # signal_generator-ის _vol_regime() logic:
        # atr_pct >= 2.0 → EXTREME (signal_generator)
        # აქ 1.5 → VOLATILE (pre-extreme guard — trade skip)
        if atr_pct >= _VOLATILE_ATR_MIN:
            return "VOLATILE"

        if atr_pct <= _SIDEWAYS_ATR_MAX and trend < _BULL_TREND_MIN:
            return "SIDEWAYS"

        if trend >= _BULL_TREND_MIN:
            return "BULL"

        if trend <= _BEAR_TREND_MAX:
            return "BEAR"

        return "UNCERTAIN"

    # ─────────────────────────────────────────────
    # TRADE PARAM RESOLUTION
    # ─────────────────────────────────────────────

    def apply(
        self,
        regime: str,
        atr_pct: float = 0.0,
        symbol: str = "",
        buy_time: Optional[datetime] = None,
    ) -> Dict:
        """
        Regime-ის მიხედვით trade params.

        Returns dict:
            SKIP_TRADING  : bool    — True → trade-ს გამოტოვება
            SKIP_REASON   : str     — რატომ გამოვტოვეთ
            TP_PCT        : float   — Take Profit %
            SL_PCT        : float   — Stop Loss %
            REGIME        : str     — detected regime
            COOLDOWN_ACTIVE: bool   — True → SL cooldown პაუზა
        """
        now = buy_time or datetime.utcnow()
        sym = symbol or "_global_"

        # SL Cooldown check
        pause_until = self._sl_pause_until.get(sym)
        if pause_until is not None and now < pause_until:
            remaining = int((pause_until - now).total_seconds())
            logger.info(
                f"[REGIME] SL_COOLDOWN_PAUSE | sym={sym} remaining={remaining}s"
            )
            return {
                "SKIP_TRADING":    True,
                "SKIP_REASON":     "SL_COOLDOWN_PAUSE",
                "TP_PCT":          0.0,
                "SL_PCT":          0.0,
                "REGIME":          regime,
                "COOLDOWN_ACTIVE": True,
            }

        # Skip regimes
        if regime in ("BEAR", "VOLATILE", "SIDEWAYS"):
            return {
                "SKIP_TRADING":    True,
                "SKIP_REASON":     f"REGIME_{regime}",
                "TP_PCT":          0.0,
                "SL_PCT":          0.0,
                "REGIME":          regime,
                "COOLDOWN_ACTIVE": False,
            }

        # ATR-based TP/SL
        tp_pct, sl_pct = self._get_tp_sl(regime, atr_pct)

        return {
            "SKIP_TRADING":    False,
            "SKIP_REASON":     "",
            "TP_PCT":          tp_pct,
            "SL_PCT":          sl_pct,
            "REGIME":          regime,
            "COOLDOWN_ACTIVE": False,
            # Backward compat (ძველი კოდი იყენებდა QUOTE_SIZE)
            "QUOTE_SIZE":      1.0,
        }

    # ─────────────────────────────────────────────
    # SL COOLDOWN TRACKING
    # ─────────────────────────────────────────────

    def notify_outcome(self, symbol: str, outcome: str, buy_time: Optional[datetime] = None) -> None:
        """
        trade დახურვის შემდეგ გამოიძახება.
        outcome: 'SL' | 'TP' | 'MANUAL_SELL'
        """
        sym = symbol or "_global_"
        outcome = str(outcome).upper()
        now = buy_time or datetime.utcnow()

        if outcome == "SL":
            self._consecutive_sl[sym] = self._consecutive_sl.get(sym, 0) + 1
            count = self._consecutive_sl[sym]
            logger.info(f"[REGIME] SL_TRACK | sym={sym} consecutive_sl={count} limit={_SL_COOLDOWN_N}")
            if count >= _SL_COOLDOWN_N:
                pause = now + timedelta(seconds=_SL_PAUSE_SECONDS)
                self._sl_pause_until[sym] = pause
                logger.warning(
                    f"[REGIME] SL_COOLDOWN | sym={sym} {count} consecutive SL → "
                    f"PAUSE {_SL_PAUSE_SECONDS // 60}min until {pause.strftime('%H:%M:%S')} UTC"
                )

        elif outcome in ("TP", "MANUAL_SELL"):
            prev = self._consecutive_sl.get(sym, 0)
            if prev > 0:
                logger.info(f"[REGIME] TP_RESET | sym={sym} consecutive_sl {prev}→0")
            self._consecutive_sl[sym]   = 0
            self._sl_pause_until[sym]   = None

    def reset_cooldown(self, symbol: str) -> None:
        """Recovery პირობები დასრულდა — cooldown reset."""
        sym = symbol or "_global_"
        self._consecutive_sl[sym]  = 0
        self._sl_pause_until[sym]  = None
        logger.info(f"[REGIME] COOLDOWN_RESET | sym={sym}")

    def is_paused(self, symbol: str, now: Optional[datetime] = None) -> bool:
        """True თუ SL Cooldown პაუზა აქტიურია."""
        sym = symbol or "_global_"
        pause = self._sl_pause_until.get(sym)
        if pause is None:
            return False
        check_time = now or datetime.utcnow()
        return check_time < pause

    def get_consecutive_sl(self, symbol: str) -> int:
        return self._consecutive_sl.get(symbol or "_global_", 0)

    # ─────────────────────────────────────────────
    # HELPERS
    # ─────────────────────────────────────────────

    @staticmethod
    def _get_tp_sl(regime: str, atr_pct: float) -> Tuple[float, float]:
        """ATR-based TP/SL % — backtest.py get_b_tp_sl()-ს ემთხვევა."""
        mults = {
            "BULL":      (_ATR_TP_BULL,      _ATR_SL_BULL),
            "UNCERTAIN": (_ATR_TP_UNCERTAIN, _ATR_SL_UNCERTAIN),
        }
        tm, sm = mults.get(regime, (_ATR_TP_UNCERTAIN, _ATR_SL_UNCERTAIN))

        if atr_pct > 0:
            tp = max(_MIN_TP, min(_MAX_TP, atr_pct * tm))
            sl = max(_MIN_SL, min(_MAX_SL, atr_pct * sm))
        else:
            # ATR უცნობია — fallback to fixed
            tp = _TP_PCT_FIXED
            sl = _SL_PCT_FIXED

        return round(tp, 3), round(sl, 3)

    def get_tp_sl(self, regime: str, atr_pct: float = 0.0) -> Tuple[float, float]:
        """Public wrapper — execution_engine.py-ისთვის."""
        return self._get_tp_sl(regime, atr_pct)

    def summary(self) -> Dict:
        """Debug: cooldown state."""
        return {
            sym: {
                "consecutive_sl": self._consecutive_sl.get(sym, 0),
                "pause_until":    str(self._sl_pause_until.get(sym, "none")),
            }
            for sym in set(list(self._consecutive_sl) + list(self._sl_pause_until))
        }
