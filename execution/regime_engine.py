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
#
# FIX HISTORY:
#   2026-03-21 — WORKER_LOOP_ERROR: TypeError '<' not supported between
#                instances of 'str' and 'float'
#
#   ROOT CAUSE (3 ადგილი):
#     1. main.py-ი regime_engine.apply(regime) ასე იძახებდა — string
#        პირდაპირ trend= პოზიციურ არგუმენტში ვარდებოდა.
#     2. signal_outbox.json-დან წაკითხული მნიშვნელობები ხშირად str
#        სახით შემოდის (JSON parse-ის შემდეგ float-ად გადაყვანა
#        ხდებოდა გარეთ, არა ამ კლასში).
#     3. apply() / detect_regime() type annotations float იყო,
#        მაგრამ runtime-ზე validation არ ხდებოდა.
#
#   FIX:
#     • _to_float() helper — ყველა შემომავალ მნიშვნელობაზე
#     • apply() — positional arg-ის ნაცვლად keyword-only signature
#     • detect_regime() — ასევე _to_float() ჭურვი
#     • get_adaptive_params() — atr_pct / base_quote safe cast
#     • apply() — legacy string "BULL"/"BEAR" detection + warning
#     • detect_regime_legacy() — შენარჩუნებულია, გამაგრებული
# ============================================================

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Union

logger = logging.getLogger("gbm")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# INTERNAL HELPER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _to_float(value: Any, default: float = 0.0, name: str = "param") -> float:
    """
    ნებისმიერ მნიშვნელობას გარდაქმნის float-ად.

    - None, "", "None", "null"  → default
    - str რიცხვი ("0.742")     → float(value)
    - უკვე float/int            → float(value)
    - გამოუთვლელი              → default + WARNING

    ეს ფუნქცია არის ROOT FIX — JSON / ENV სტრინგები
    აღარ ჩავარდება TypeError-ში.
    """
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        stripped = value.strip().lower()
        if stripped in ("", "none", "null", "nan", "inf", "-inf"):
            return default
        try:
            return float(stripped)
        except ValueError:
            logger.warning(
                "[REGIME] _to_float: '%s' cannot be parsed as float for '%s', "
                "using default=%.4f", value, name, default
            )
            return default
    # fallback: bool, Decimal, numpy scalar, etc.
    try:
        return float(value)
    except (TypeError, ValueError):
        logger.warning(
            "[REGIME] _to_float: type %s cannot be converted for '%s', "
            "using default=%.4f", type(value).__name__, name, default
        )
        return default


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ENV — regime thresholds (ადვილად tuneable)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _ef(name: str, default: float) -> float:
    """ENV float reader — იგივე helper, ახლა _to_float-ს იყენებს."""
    return _to_float(os.getenv(name), default=default, name=name)


# Regime detection thresholds
BULL_TREND_MIN      = _ef("REGIME_BULL_TREND_MIN",     0.45)
BEAR_TREND_MAX      = _ef("REGIME_BEAR_TREND_MAX",    -0.10)
SIDEWAYS_ATR_MAX    = _ef("REGIME_SIDEWAYS_ATR_MAX",   0.28)
VOLATILE_ATR_MIN    = _ef("REGIME_VOLATILE_ATR_MIN",   1.50)
BULL_VOLUME_MIN     = _ef("REGIME_BULL_VOLUME_MIN",    0.40)

# ATR multipliers per regime (TP / SL)
ATR_TP_BULL         = _ef("ATR_MULT_TP_BULL",          3.0)
ATR_TP_SIDEWAYS     = _ef("ATR_MULT_TP_SIDE",          3.0)
ATR_TP_BEAR         = _ef("ATR_MULT_TP_BEAR",          2.0)
ATR_TP_VOLATILE     = _ef("ATR_MULT_TP_VOLATILE",      1.5)
ATR_TP_UNCERTAIN    = _ef("ATR_MULT_TP_UNCERTAIN",     2.5)

ATR_SL_BULL         = _ef("ATR_MULT_SL_BULL",          1.0)
ATR_SL_SIDEWAYS     = _ef("ATR_MULT_SL_SIDE",          1.0)
ATR_SL_BEAR         = _ef("ATR_MULT_SL_BEAR",          0.5)
ATR_SL_VOLATILE     = _ef("ATR_MULT_SL_VOLATILE",      0.4)
ATR_SL_UNCERTAIN    = _ef("ATR_MULT_SL_UNCERTAIN",     1.0)

# Quote size multipliers per regime
QUOTE_MULT_BULL      = _ef("REGIME_QUOTE_MULT_BULL",     1.0)
QUOTE_MULT_SIDEWAYS  = _ef("REGIME_QUOTE_MULT_SIDE",     0.0)
QUOTE_MULT_BEAR      = _ef("REGIME_QUOTE_MULT_BEAR",     0.0)
QUOTE_MULT_VOLATILE  = _ef("REGIME_QUOTE_MULT_VOLATILE", 0.0)
QUOTE_MULT_UNCERTAIN = _ef("REGIME_QUOTE_MULT_UNCERTAIN",0.8)

# Floor / Ceiling safety
MIN_TP_PCT  = _ef("REGIME_MIN_TP_PCT",  0.50)
MIN_SL_PCT  = _ef("REGIME_MIN_SL_PCT",  0.20)
MAX_TP_PCT  = _ef("REGIME_MAX_TP_PCT",  4.00)
MAX_SL_PCT  = _ef("REGIME_MAX_SL_PCT",  1.50)

# ვალიდური რეჟიმების სია
_VALID_REGIMES = frozenset({"BULL", "BEAR", "SIDEWAYS", "VOLATILE", "UNCERTAIN"})


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ENGINE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class MarketRegimeEngine:
    """
    Market Regime Adaptive Trading System.

    Input:  trend_strength, atr_pct, vol_score, ai_score
    Output: regime + adaptive params (TP_PCT, SL_PCT, QUOTE_MULT, SKIP_TRADING)

    signal_generator.py-ი ამ კლასს იძახებს ყოველ tick-ზე.
    execution_engine.py-ი adaptive params-ს signal-დან იღებს.
    """

    def __init__(self, config=None):
        self.config = config or {}

    # ────────────────────────────────────────────────────
    # REGIME DETECTION
    # ────────────────────────────────────────────────────

    def detect_regime(
        self,
        trend: Union[float, str, None] = 0.0,
        vol:   Union[float, str, None] = 0.0,
        atr_pct: Union[float, str, None] = 0.0,
        ai_score: Union[float, str, None] = 0.0,
    ) -> str:
        """
        ბაზრის რეჟიმის განსაზღვრა.

        ყველა შემომავალი მნიშვნელობა safe cast ხდება float-ად
        _to_float()-ის საშუალებით — JSON string-ები, None, env
        variables პრობლემა არ გამოიწვევს.

        Args:
            trend:    trend_strength 0..1
            vol:      volume_score 0..1
            atr_pct:  ATR % (volatility)
            ai_score: ExcelLiveCore score 0..1

        Returns:
            'BULL' | 'BEAR' | 'SIDEWAYS' | 'VOLATILE' | 'UNCERTAIN'
        """
        # ══ ROOT FIX: safe cast ══════════════════════════
        f_trend    = _to_float(trend,    default=0.0, name="trend")
        f_vol      = _to_float(vol,      default=0.0, name="vol")
        f_atr_pct  = _to_float(atr_pct,  default=0.0, name="atr_pct")
        # ai_score გამოყენებული არ არის detection-ში, მაგრამ
        # კასტი ხდება consistency-სთვის
        _          = _to_float(ai_score, default=0.0, name="ai_score")

        # Priority 1: VOLATILE — extreme volatility
        if f_atr_pct >= VOLATILE_ATR_MIN:
            return "VOLATILE"

        # Priority 2: SIDEWAYS — flat market (fee barrier)
        if f_atr_pct <= SIDEWAYS_ATR_MAX and f_trend < BULL_TREND_MIN:
            return "SIDEWAYS"

        # Priority 3: BULL — strong uptrend + volume
        if f_trend >= BULL_TREND_MIN and f_vol >= BULL_VOLUME_MIN:
            return "BULL"

        # Priority 4: BEAR — downtrend
        if f_trend <= BEAR_TREND_MAX:
            return "BEAR"

        # Fallback: not enough signal
        return "UNCERTAIN"

    # ────────────────────────────────────────────────────
    # ADAPTIVE PARAMS
    # ────────────────────────────────────────────────────

    def get_adaptive_params(
        self,
        regime: str,
        atr_pct: Union[float, str, None] = 0.0,
        base_quote: Union[float, str, None] = 7.0,
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
        # ══ ROOT FIX: safe cast ══════════════════════════
        f_atr_pct   = _to_float(atr_pct,   default=0.0, name="atr_pct")
        f_base_quote = _to_float(base_quote, default=7.0, name="base_quote")

        # ══ Regime validation ════════════════════════════
        if regime not in _VALID_REGIMES:
            logger.warning(
                "[REGIME] get_adaptive_params: unknown regime '%s', "
                "falling back to UNCERTAIN", regime
            )
            regime = "UNCERTAIN"

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

        tp_mult    = _tp_mults[regime]
        sl_mult    = _sl_mults[regime]
        quote_mult = _quote_mults[regime]

        raw_tp = f_atr_pct * tp_mult
        raw_sl = f_atr_pct * sl_mult

        tp_pct = max(MIN_TP_PCT, min(MAX_TP_PCT, raw_tp))
        sl_pct = max(MIN_SL_PCT, min(MAX_SL_PCT, raw_sl))

        skip_trading = regime in ("BEAR", "VOLATILE")
        quote_size   = round(f_base_quote * quote_mult, 2)

        return {
            "TP_PCT":       round(tp_pct, 3),
            "SL_PCT":       round(sl_pct, 3),
            "QUOTE_MULT":   quote_mult,
            "QUOTE_SIZE":   quote_size,
            "SKIP_TRADING": skip_trading,
            "REGIME":       regime,
            "ATR_PCT":      round(f_atr_pct, 4),
        }

    # ────────────────────────────────────────────────────
    # MAIN ENTRY
    # ────────────────────────────────────────────────────

    def apply(
        self,
        trend:      Union[float, str, None] = 0.0,
        vol:        Union[float, str, None] = 0.0,
        atr_pct:    Union[float, str, None] = 0.0,
        ai_score:   Union[float, str, None] = 0.0,
        base_quote: Union[float, str, None] = 7.0,
    ) -> Dict[str, Any]:
        """
        მთავარი ფუნქცია — detect + adaptive params ერთად.

        CRITICAL FIX NOTE:
            თუ main.py-ი ასე იძახებდა:
                adaptive = regime_engine.apply(regime)   # regime="BULL" string
            ახლა ეს აღარ ჩავარდება — trend="BULL" → _to_float() → 0.0 (default)
            + WARNING ლოგში, რომ შეუმჩნეველი არ დარჩეს.

            სწორი გამოძახება:
                adaptive = regime_engine.apply(
                    trend=trend_val,
                    vol=vol_score,
                    atr_pct=atrp,
                    ai_score=ai_score,
                    base_quote=BOT_QUOTE_PER_TRADE,
                )

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
        # ══ Legacy string sentinel detection ═════════════
        # main.py-ი შეიძლება კვლავ გადასცემდეს regime string-ს
        # პირველ positional arg-ში. ვამოწმებთ და ვაფრთხილებთ.
        if isinstance(trend, str) and trend.upper() in _VALID_REGIMES:
            logger.warning(
                "[REGIME] apply() received a regime string ('%s') as 'trend'. "
                "This is a caller bug — main.py must pass numeric trend_strength, "
                "not a regime label. Falling back: trend=0.0, regime forced.",
                trend,
            )
            # Graceful degradation: use string as pre-computed regime
            forced_regime = trend.upper()
            f_atr   = _to_float(atr_pct,    default=0.0, name="atr_pct")
            f_base  = _to_float(base_quote,  default=7.0, name="base_quote")
            params  = self.get_adaptive_params(
                regime=forced_regime, atr_pct=f_atr, base_quote=f_base
            )
            logger.info(
                "[REGIME] %s (forced) | atr%%=%.3f | "
                "TP=%.3f%% SL=%.3f%% quote=%.2f skip=%s",
                forced_regime, f_atr,
                params["TP_PCT"], params["SL_PCT"],
                params["QUOTE_SIZE"], params["SKIP_TRADING"],
            )
            return params

        # ══ Normal flow ═══════════════════════════════════
        regime = self.detect_regime(
            trend=trend, vol=vol, atr_pct=atr_pct, ai_score=ai_score
        )
        params = self.get_adaptive_params(
            regime=regime, atr_pct=atr_pct, base_quote=base_quote
        )

        f_trend    = _to_float(trend,    name="trend")
        f_vol      = _to_float(vol,      name="vol")
        f_atr_pct  = _to_float(atr_pct,  name="atr_pct")
        f_ai_score = _to_float(ai_score, name="ai_score")

        logger.info(
            "[REGIME] %s | trend=%.3f vol=%.3f atr%%=%.3f ai=%.3f | "
            "TP=%.3f%% SL=%.3f%% quote=%.2f skip=%s",
            regime, f_trend, f_vol, f_atr_pct, f_ai_score,
            params["TP_PCT"], params["SL_PCT"],
            params["QUOTE_SIZE"], params["SKIP_TRADING"],
        )

        return params

    # ────────────────────────────────────────────────────
    # BACKWARD COMPATIBILITY
    # ────────────────────────────────────────────────────

    def detect_regime_legacy(
        self,
        trend: Union[float, str, None],
        vol:   Union[float, str, None],
    ) -> str:
        """ძველი interface — main.py-ისთვის. გამაგრებული."""
        return self.detect_regime(trend=trend, vol=vol)
