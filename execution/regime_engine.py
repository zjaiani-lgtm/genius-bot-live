# execution/regime_engine.py
# ============================================================
# Market Regime Adaptive Trading System
# ============================================================
# რეჟიმები:
#   BULL     — ძლიერი uptrend, runner TP-ები
#   BEAR     — downtrend, ახალი BUY ბლოკდება
#   SIDEWAYS — flat/range ბაზარი, სწრაფი turnover
#   VOLATILE — extreme vol, defensive / protective SELL
#   UNCERTAIN— საკმარისი სიგნალი არ არის
# ============================================================

from __future__ import annotations
import os
import logging
from typing import Dict, Any

logger = logging.getLogger("gbm")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ENV — regime thresholds (ადვილად tuneable)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _ef(name: str, default: float) -> float:
    v = os.getenv(name)
    if v is None or str(v).strip() == "":
        return default
    try:
        return float(v)
    except Exception:
        return default


# Regime detection thresholds
BULL_TREND_MIN      = _ef("REGIME_BULL_TREND_MIN",     0.45)  # trend >= ამაზე → BULL
BEAR_TREND_MAX      = _ef("REGIME_BEAR_TREND_MAX",    -0.10)  # trend <= ამაზე → BEAR candidate
SIDEWAYS_ATR_MAX    = _ef("REGIME_SIDEWAYS_ATR_MAX",   0.28)  # ATR% <= 0.28% → SIDEWAYS (fee barrier)
VOLATILE_ATR_MIN    = _ef("REGIME_VOLATILE_ATR_MIN",   1.50)  # ATR% >= 1.50% → VOLATILE
BULL_VOLUME_MIN     = _ef("REGIME_BULL_VOLUME_MIN",    0.40)  # volume score min for BULL

# ATR multipliers per regime
# TP = ATR% × mult  |  SL = ATR% × mult  |  R:R = TP_mult / SL_mult
ATR_TP_BULL         = _ef("ATR_MULT_TP_BULL",          3.0)   # BULL: R:R=3.0
ATR_TP_SIDEWAYS     = _ef("ATR_MULT_TP_SIDE",          3.0)   # SIDE: R:R=3.0
ATR_TP_BEAR         = _ef("ATR_MULT_TP_BEAR",          2.0)   # BEAR: defensive
ATR_TP_VOLATILE     = _ef("ATR_MULT_TP_VOLATILE",      1.5)   # VOL:  tight
ATR_TP_UNCERTAIN    = _ef("ATR_MULT_TP_UNCERTAIN",     2.5)   # UNC:  R:R=2.5

ATR_SL_BULL         = _ef("ATR_MULT_SL_BULL",          1.0)   # BULL SL
ATR_SL_SIDEWAYS     = _ef("ATR_MULT_SL_SIDE",          1.0)   # SIDE SL
ATR_SL_BEAR         = _ef("ATR_MULT_SL_BEAR",          0.5)   # BEAR SL (tight)
ATR_SL_VOLATILE     = _ef("ATR_MULT_SL_VOLATILE",      0.4)   # VOL  SL (very tight)
ATR_SL_UNCERTAIN    = _ef("ATR_MULT_SL_UNCERTAIN",     1.0)   # UNC  SL

# Quote size multipliers per regime
QUOTE_MULT_BULL     = _ef("REGIME_QUOTE_MULT_BULL",    1.0)   # full size
QUOTE_MULT_SIDEWAYS = _ef("REGIME_QUOTE_MULT_SIDE",    0.0)   # SKIP — fee barrier
QUOTE_MULT_BEAR     = _ef("REGIME_QUOTE_MULT_BEAR",    0.0)   # SKIP — protect capital
QUOTE_MULT_VOLATILE = _ef("REGIME_QUOTE_MULT_VOLATILE",0.0)   # SKIP — emergency
QUOTE_MULT_UNCERTAIN= _ef("REGIME_QUOTE_MULT_UNCERTAIN",0.8)  # 80%

# Minimum TP/SL floors (% — safety net)
MIN_TP_PCT          = _ef("REGIME_MIN_TP_PCT",         0.50)  # TP minimum 0.50%
MIN_SL_PCT          = _ef("REGIME_MIN_SL_PCT",         0.20)  # SL minimum 0.20%
MAX_TP_PCT          = _ef("REGIME_MAX_TP_PCT",         4.00)  # TP maximum 4.00%
MAX_SL_PCT          = _ef("REGIME_MAX_SL_PCT",         1.50)  # SL maximum 1.50%


class MarketRegimeEngine:
    """
    Market Regime Adaptive Trading System.

    Input:  trend_strength, atr_pct, vol_score, ai_score
    Output: regime + adaptive params (TP_PCT, SL_PCT, QUOTE_MULT, SKIP_TRADING)

    signal_generator.py-ი ამ კლასს იძახებს ყოველ tick-ზე.
    execution_engine.py-ი adaptive params-ს signal-დან იღებს.
    """

    def __init__(self, config=None):
        # config პარამეტრი შენარჩუნებულია უკუთავსებადობისთვის
        self.config = config or {}

    # ────────────────────────────────────────────────────
    # REGIME DETECTION
    # ────────────────────────────────────────────────────

    def detect_regime(
        self,
        trend: float,
        vol: float = 0.0,
        atr_pct: float = 0.0,
        ai_score: float = 0.0,
    ) -> str:
        """
        ბაზრის რეჟიმის განსაზღვრა.

        Args:
            trend:    trend_strength 0..1 (signal_generator._trend_strength())
            vol:      volume_score 0..1
            atr_pct:  ATR % (signal_generator._atr_pct())
            ai_score: ExcelLiveCore score 0..1

        Returns:
            'BULL' | 'BEAR' | 'SIDEWAYS' | 'VOLATILE' | 'UNCERTAIN'
        """

        # VOLATILE — უმაღლესი პრიორიტეტი
        if atr_pct >= VOLATILE_ATR_MIN:
            return "VOLATILE"

        # SIDEWAYS — flat ბაზარი (ATR ძალიან დაბალი)
        if atr_pct <= SIDEWAYS_ATR_MAX and trend < BULL_TREND_MIN:
            return "SIDEWAYS"

        # BULL — ძლიერი uptrend + volume confirmation
        if trend >= BULL_TREND_MIN and vol >= BULL_VOLUME_MIN:
            return "BULL"

        # BEAR — downtrend
        if trend <= BEAR_TREND_MAX:
            return "BEAR"

        # UNCERTAIN — საკმარისი სიგნალი არ არის
        return "UNCERTAIN"

    # ────────────────────────────────────────────────────
    # ADAPTIVE PARAMS
    # ────────────────────────────────────────────────────

    def get_adaptive_params(
        self,
        regime: str,
        atr_pct: float,
        base_quote: float = 7.0,
    ) -> Dict[str, Any]:
        """
        რეჟიმის მიხედვით TP/SL/Quote გამოთვლა.

        Args:
            regime:     detect_regime()-ის შედეგი
            atr_pct:    ATR % (ბაზრის volatility)
            base_quote: BOT_QUOTE_PER_TRADE ENV-იდან

        Returns:
            dict: TP_PCT, SL_PCT, QUOTE_MULT, QUOTE_SIZE,
                  SKIP_TRADING, REGIME, ATR_PCT
        """

        # ATR multipliers by regime
        _tp_mults = {
            "BULL":      ATR_TP_BULL,
            "SIDEWAYS":  ATR_TP_SIDEWAYS,
            "BEAR":      ATR_TP_BEAR,
            "VOLATILE":  ATR_TP_VOLATILE,
            "UNCERTAIN": ATR_TP_UNCERTAIN,
        }
        _sl_mults = {
            "BULL":      ATR_SL_BULL,
            "SIDEWAYS":  ATR_SL_SIDEWAYS,
            "BEAR":      ATR_SL_BEAR,
            "VOLATILE":  ATR_SL_VOLATILE,
            "UNCERTAIN": ATR_SL_UNCERTAIN,
        }
        _quote_mults = {
            "BULL":      QUOTE_MULT_BULL,
            "SIDEWAYS":  QUOTE_MULT_SIDEWAYS,
            "BEAR":      QUOTE_MULT_BEAR,
            "VOLATILE":  QUOTE_MULT_VOLATILE,
            "UNCERTAIN": QUOTE_MULT_UNCERTAIN,
        }

        tp_mult    = _tp_mults.get(regime, ATR_TP_UNCERTAIN)
        sl_mult    = _sl_mults.get(regime, ATR_SL_UNCERTAIN)
        quote_mult = _quote_mults.get(regime, QUOTE_MULT_UNCERTAIN)

        # ATR-based TP/SL გამოთვლა
        raw_tp = atr_pct * tp_mult
        raw_sl = atr_pct * sl_mult

        # Floor/Ceiling safety
        tp_pct = max(MIN_TP_PCT, min(MAX_TP_PCT, raw_tp))
        sl_pct = max(MIN_SL_PCT, min(MAX_SL_PCT, raw_sl))

        # BEAR/VOLATILE — ახალი trade-ები ბლოკდება
        skip_trading = regime in ("BEAR", "VOLATILE")

        # Quote size
        quote_size = round(base_quote * quote_mult, 2)

        return {
            "TP_PCT":        round(tp_pct, 3),
            "SL_PCT":        round(sl_pct, 3),
            "QUOTE_MULT":    quote_mult,
            "QUOTE_SIZE":    quote_size,
            "SKIP_TRADING":  skip_trading,
            "REGIME":        regime,
            "ATR_PCT":       round(atr_pct, 4),
        }

    # ────────────────────────────────────────────────────
    # MAIN ENTRY (signal_generator.py-იდან გამოძახება)
    # ────────────────────────────────────────────────────

    def apply(
        self,
        trend: float = 0.0,
        vol: float = 0.0,
        atr_pct: float = 0.0,
        ai_score: float = 0.0,
        base_quote: float = 7.0,
    ) -> Dict[str, Any]:
        """
        მთავარი ფუნქცია — detect + adaptive params ერთად.

        signal_generator.py-ი ასე იძახებს:
            adaptive = regime_engine.apply(
                trend=trend, vol=vol_score,
                atr_pct=atrp, ai_score=ai_score,
                base_quote=BOT_QUOTE_PER_TRADE
            )
            sig["adaptive"] = adaptive

        execution_engine.py-ი ასე კითხულობს:
            tp_pct = float(adaptive.get("TP_PCT", self.tp_pct))
            sl_pct = float(adaptive.get("SL_PCT", self.sl_pct))
        """
        regime = self.detect_regime(
            trend=trend, vol=vol, atr_pct=atr_pct, ai_score=ai_score
        )
        params = self.get_adaptive_params(
            regime=regime, atr_pct=atr_pct, base_quote=base_quote
        )

        logger.info(
            f"[REGIME] {regime} | "
            f"trend={trend:.3f} vol={vol:.3f} atr%={atr_pct:.3f} ai={ai_score:.3f} | "
            f"TP={params['TP_PCT']:.3f}% SL={params['SL_PCT']:.3f}% "
            f"quote={params['QUOTE_SIZE']:.2f} skip={params['SKIP_TRADING']}"
        )

        return params

    # ────────────────────────────────────────────────────
    # BACKWARD COMPATIBILITY (main.py-ი კვლავ იყენებდა)
    # ────────────────────────────────────────────────────

    def detect_regime_legacy(self, trend: float, vol: float) -> str:
        """ძველი interface — main.py-ისთვის"""
        return self.detect_regime(trend=trend, vol=vol)
