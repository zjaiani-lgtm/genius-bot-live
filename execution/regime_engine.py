# execution/regime_engine.py
# ============================================================
# Market Regime Engine — v3.1 (ENV-aligned, Production-Ready)
# ============================================================
# v3.1 fixes vs v3:
#
#   FIX-RE-1: _SL_COOLDOWN_N default 2 → 3  (ENV=SL_COOLDOWN_AFTER_N=3)
#   FIX-RE-2: _MIN_TP default 0.50 → 0.25   (ENV=MIN_NET_PROFIT_PCT=0.25)
#   FIX-RE-3: _TP_PCT_FIXED default 1.0 → 1.5 (ENV=TP_PCT=1.5)
#   FIX-RE-4: _SL_PCT_FIXED default 0.70 → 0.80 (ENV=SL_PCT=0.80)
#   FIX-RE-5: _MIN_SL hardcoded 0.25 → ENV MIN_SL_PCT=0.40
#   FIX-RE-6: _MTF_TP_BONUS default 0.20 → 0.25 (ENV=MTF_TP_BONUS=0.25)
#   FIX-RE-7: _MTF_TP_PENALTY default 0.15 → 0.20 (ENV=MTF_TP_PENALTY=0.20)
#   FIX-RE-8: _MTF_BLOCK_DIVERGE default True → False (ENV=MTF_BLOCK_ON_BEAR_DIVERGE=false)
#   FIX-RE-9: _SIDEWAYS_ATR_MAX default 0.18 → 0.20 (ENV=REGIME_SIDEWAYS_ATR_MAX=0.20)
#   FIX-RE-10: _SIZE_BULL_PCT comment $15 → $10 (ENV=MAX_QUOTE_PER_TRADE=10)
#   FIX-RE-11: BOT_QUOTE_PER_TRADE fallback in _get_quote_size 15.0 → 10.0
#   FIX-RE-12: DYNAMIC_SIZE_MIN/MAX fallback 5.0/15.0 → 8.0/10.0
#
# ეტაპი 1 — TP/SL/Size ავტომატური ადაpტაცია:
#     - BULL:      TP=ATR×4.0  SL=ATR×2.0  Size=100% ($10)
#     - UNCERTAIN: TP=ATR×2.0  SL=ATR×1.2  Size=50%  ($5→floor $8 after clamp)
#     - SIDEWAYS:  SKIP
#     - BEAR:      SKIP
#     - VOLATILE:  SKIP
#
# ეტაპი 2 — Confidence threshold ადაpტაცია:
#     - BULL:      BUY_CONFIDENCE_MIN(0.38) × 0.85 = 0.323
#     - UNCERTAIN: BUY_CONFIDENCE_MIN(0.38) × 1.20 = 0.456
#
# ეტაპი 3 — MTF Confirmation:
#     - STRONG:  15m+1h BULL       → TP +25%
#     - WEAK:    15m BULL+1h UNCERT → TP -20%
#     - DIVERGE: MTF_BLOCK_ON_BEAR_DIVERGE=false → არ ბლოკავს (UNCERTAIN+1h BEAR გარდა)
# ============================================================
from __future__ import annotations

import os
import logging
from collections import deque
from datetime import datetime, timedelta
from typing import Deque, Dict, Optional, Tuple

logger = logging.getLogger("gbm")


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

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


def _eb(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "y", "on")


# ─────────────────────────────────────────────
# ENV THRESHOLDS
# ─────────────────────────────────────────────

# Regime detection
_BULL_TREND_MIN   = _ef("REGIME_BULL_TREND_MIN",   0.30)
_SIDEWAYS_ATR_MAX = _ef("REGIME_SIDEWAYS_ATR_MAX", 0.20)   # FIX-RE-9: 0.18 → 0.20 (ENV)
_VOLATILE_ATR_MIN = _ef("REGIME_VOLATILE_ATR_MIN", 1.50)
_BEAR_TREND_MAX   = -0.10

# ──────────────────────────────────────────────────────────
# ეტაპი 1: TP/SL Multipliers — per regime
# ──────────────────────────────────────────────────────────
#   BULL:      ATR×4.0 TP, ATR×2.0 SL  (ENV=ATR_MULT_TP_BULL=4.0 / ATR_MULT_SL_BULL=2.0)
#   UNCERTAIN: ATR×2.0 TP, ATR×1.2 SL
_ATR_TP_BULL        = _ef("ATR_MULT_TP_BULL",      4.0)   # FIX: ENV=4.0 (was default 3.0)
_ATR_SL_BULL        = _ef("ATR_MULT_SL_BULL",      2.0)   # FIX: ENV=2.0 (was default 1.5)

_ATR_TP_UNCERTAIN   = _ef("ATR_MULT_TP_UNCERTAIN", 2.0)
_ATR_SL_UNCERTAIN   = _ef("ATR_MULT_SL_UNCERTAIN", 1.2)

# TP/SL bounds
# FIX-RE-2: _MIN_TP reads ENV=MIN_NET_PROFIT_PCT=0.25 (was hardcoded 0.50)
_MIN_TP = _ef("MIN_NET_PROFIT_PCT", 0.25)
# FIX-RE-5: _MIN_SL reads ENV=MIN_SL_PCT=0.40 (was hardcoded 0.25)
_MIN_SL = _ef("MIN_SL_PCT", 0.40)
_MAX_TP = 4.0
_MAX_SL = 1.8

# Fixed fallback (ATR=0 სიტუაცია)
# FIX-RE-3: ENV=TP_PCT=1.5 (was default 1.0)
_TP_PCT_FIXED = _ef("TP_PCT", 1.5)
# FIX-RE-4: ENV=SL_PCT=0.80 (was default 0.70)
_SL_PCT_FIXED = _ef("SL_PCT", 0.80)

# ──────────────────────────────────────────────────────────
# ეტაპი 1: Position Size — per regime (% of max quote)
# ──────────────────────────────────────────────────────────
# ENV: QUOTE_SIZE_BULL=10.0, QUOTE_SIZE_UNCERTAIN=7.0
# FIX-RE-10: comments updated — max $10 (ENV=MAX_QUOTE_PER_TRADE=10)
_SIZE_BULL_PCT      = _ef("REGIME_SIZE_BULL_PCT",      1.00)  # 100% → $10
_SIZE_UNCERTAIN_PCT = _ef("REGIME_SIZE_UNCERTAIN_PCT", 0.50)  # 50%  → $5 (floor→$8 after clamp)

# QUOTE_SIZE_BULL / QUOTE_SIZE_UNCERTAIN — პირდაპირი USDT ზომა per regime
# თუ დაყენებულია (>0), REGIME_SIZE_*_PCT გამოთვლას override-ავს
# QUOTE_SIZE_BULL=10.0 → BULL-ზე ყოველთვის $10
# QUOTE_SIZE_UNCERTAIN=7.0 → UNCERTAIN-ზე ყოველთვის $7
# 0 = გამორთული, პროცენტული გამოთვლა გამოიყენება
_QUOTE_SIZE_BULL      = _ef("QUOTE_SIZE_BULL",      0.0)   # ENV=10.0 (0=disabled)
_QUOTE_SIZE_UNCERTAIN = _ef("QUOTE_SIZE_UNCERTAIN", 0.0)   # ENV=7.0  (0=disabled)

# ──────────────────────────────────────────────────────────
# ეტაპი 2: Confidence threshold ადაpტაცია
# ──────────────────────────────────────────────────────────
# BULL:      0.38 × 0.85 = 0.323  → ნაკლები სიმკაცრე
# UNCERTAIN: 0.38 × 1.20 = 0.456  → მეტი სიმკაცრე
_CONF_BULL_MULT      = _ef("REGIME_CONF_BULL_MULT",      0.85)
_CONF_UNCERTAIN_MULT = _ef("REGIME_CONF_UNCERTAIN_MULT", 1.20)

# ──────────────────────────────────────────────────────────
# ეტაპი 3: MTF Confirmation
# ──────────────────────────────────────────────────────────
_MTF_ENABLED       = _eb("USE_MTF_FILTER", True)
# FIX-RE-6: ENV=MTF_TP_BONUS=0.25 (was default 0.20)
_MTF_TP_BONUS      = _ef("MTF_TP_BONUS",  0.25)
# FIX-RE-7: ENV=MTF_TP_PENALTY=0.20 (was default 0.15)
_MTF_TP_PENALTY    = _ef("MTF_TP_PENALTY", 0.20)
# FIX-RE-8: ENV=MTF_BLOCK_ON_BEAR_DIVERGE=false (was default True)
# false → BULL+1h_BEAR → TP penalty only, not hard SKIP
# UNCERTAIN+1h_BEAR → always SKIP (unconditional — რჩება)
_MTF_BLOCK_DIVERGE = _eb("MTF_BLOCK_ON_BEAR_DIVERGE", False)

# SL Cooldown — გათიშულია (DCA mode, SL=999%)
# _SL_COOLDOWN_N = 999 → notify_outcome("SL") პაუზას ვეღარ ააქტიურებს
_SL_COOLDOWN_N    = 999   # DCA: disabled
_SL_PAUSE_SECONDS = _ei("SL_COOLDOWN_PAUSE_SECONDS", 1800)

# Regime history depth
_HISTORY_DEPTH = _ei("REGIME_HISTORY_DEPTH", 10)

# ─────────────────────────────────────────────
# REGIME SCORE WEIGHTS (stability check)
# ─────────────────────────────────────────────
_REGIME_STABILITY_MIN = _ef("REGIME_STABILITY_MIN", 0.60)


class MarketRegimeEngine:
    """
    Adaptive Regime-Aware Trading Engine — v3.1

    სამი ეტაპი ერთ კლასში:
      1. TP/SL/Size ავტომატური ადაpტაცია per-regime
      2. Confidence threshold ადაpტაცია per-regime
      3. MTF (15m + 1h) confirmation + TP bonus/penalty

    ENV-aligned defaults (v3.1):
      ATR_MULT_TP_BULL=4.0, ATR_MULT_SL_BULL=2.0
      MTF_TP_BONUS=0.25, MTF_TP_PENALTY=0.20
      MTF_BLOCK_ON_BEAR_DIVERGE=false
      SL_COOLDOWN_AFTER_N=3
      MIN_SL_PCT=0.40, MIN_NET_PROFIT_PCT=0.25

    Usage:
        engine = MarketRegimeEngine()

        regime_15m = engine.detect_regime(trend=0.6, atr_pct=0.45)
        regime_1h  = engine.detect_regime(trend=0.5, atr_pct=0.40)

        params = engine.apply(
            regime        = regime_15m,
            atr_pct       = 0.45,
            symbol        = "BTC/USDT",
            htf_regime    = regime_1h,
            base_conf_min = 0.38,
            base_quote    = 10.0,
        )

        if params["SKIP_TRADING"]:
            continue

        tp       = params["TP_PCT"]        # ეტაპი 1 + MTF bonus
        sl       = params["SL_PCT"]        # ეტაპი 1
        quote    = params["QUOTE_SIZE"]    # ეტაპი 1
        conf_min = params["CONF_MIN"]      # ეტაპი 2
        mtf_ok   = params["MTF_CONFIRMED"] # ეტაპი 3
    """

    def __init__(self, config=None):
        self._config = config  # backward compat

        # SL Cooldown per-symbol (in-memory — DB-based cooldown signal_generator-ში)
        self._consecutive_sl: Dict[str, int]                 = {}
        self._sl_pause_until: Dict[str, Optional[datetime]]  = {}

        # ეტაპი 3: Regime history per-symbol (15m)
        self._regime_history: Dict[str, Deque[str]] = {}

    # ─────────────────────────────────────────────
    # ეტაპი 1+2+3: MAIN ENTRY POINT
    # ─────────────────────────────────────────────

    def apply(
        self,
        regime: str,
        atr_pct: float = 0.0,
        symbol: str = "",
        buy_time: Optional[datetime] = None,
        # ეტაპი 2
        base_conf_min: float = 0.0,
        # ეტაპი 3
        htf_regime: Optional[str] = None,
        base_quote: float = 0.0,
    ) -> Dict:
        """
        სრული adaptive trade params.

        Returns dict with keys:
            SKIP_TRADING   : bool
            SKIP_REASON    : str
            TP_PCT         : float   — ეტაპი 1 + MTF bonus/penalty
            SL_PCT         : float   — ეტაპი 1
            REGIME         : str
            QUOTE_SIZE     : float   — ეტაპი 1
            CONF_MIN       : float   — ეტაპი 2 (adapted threshold)
            MTF_CONFIRMED  : bool    — ეტაპი 3
            MTF_ALIGNMENT  : str     — "STRONG"/"WEAK"/"DIVERGE"/"NEUTRAL"/"N/A"
            REGIME_STABLE  : bool
            COOLDOWN_ACTIVE: bool
        """
        now = buy_time or datetime.utcnow()
        sym = symbol or "_global_"

        # Regime history განახლება
        self._update_history(sym, regime)

        # ── SL Cooldown (in-memory — backward compat with regime engine) ──────
        pause_until = self._sl_pause_until.get(sym)
        if pause_until is not None and now < pause_until:
            remaining = int((pause_until - now).total_seconds())
            logger.info(f"[REGIME] SL_COOLDOWN_ENGINE | sym={sym} remaining={remaining}s")
            return self._skip("SL_COOLDOWN_PAUSE", regime, cooldown=True)

        # ── Skip regimes ──────────────────────────────────────────────────────
        if regime in ("BEAR", "VOLATILE", "SIDEWAYS"):
            logger.info(f"[REGIME] SKIP | sym={sym} regime={regime}")
            return self._skip(f"REGIME_{regime}", regime)

        # ── ეტაპი 3: MTF Confirmation ────────────────────────────────────────
        mtf_result = self._mtf_check(regime, htf_regime, sym)
        if mtf_result["SKIP"]:
            logger.info(
                f"[REGIME] MTF_BLOCK | sym={sym} "
                f"15m={regime} 1h={htf_regime} reason={mtf_result['REASON']}"
            )
            return self._skip(mtf_result["REASON"], regime)

        # ── ეტაპი 1: TP/SL/Size ─────────────────────────────────────────────
        tp_pct, sl_pct = self._get_tp_sl(regime, atr_pct)

        # MTF TP bonus/penalty
        tp_pct = self._apply_mtf_tp(tp_pct, mtf_result["ALIGNMENT"])

        # Position size
        # FIX-RE-11: fallback 15.0 → 10.0 (ENV=BOT_QUOTE_PER_TRADE=10)
        max_q = base_quote if base_quote > 0 else _ef("BOT_QUOTE_PER_TRADE", 10.0)
        quote_size = self._get_quote_size(regime, max_q)

        # ── ეტაპი 2: Confidence threshold ────────────────────────────────────
        conf_min = self._adapt_conf_min(regime, base_conf_min)

        # Regime stability
        stable = self._is_stable(sym)

        logger.info(
            f"[REGIME] OK | sym={sym} regime={regime} "
            f"TP={tp_pct:.3f}% SL={sl_pct:.3f}% "
            f"size=${quote_size:.1f} conf_min={conf_min:.3f} "
            f"mtf={mtf_result['ALIGNMENT']} stable={stable} "
            f"atr%={atr_pct:.3f}"
        )

        return {
            "SKIP_TRADING":    False,
            "SKIP_REASON":     "",
            "TP_PCT":          tp_pct,
            "SL_PCT":          sl_pct,
            "REGIME":          regime,
            "QUOTE_SIZE":      quote_size,
            "CONF_MIN":        conf_min,
            "MTF_CONFIRMED":   mtf_result["CONFIRMED"],
            "MTF_ALIGNMENT":   mtf_result["ALIGNMENT"],
            "REGIME_STABLE":   stable,
            "COOLDOWN_ACTIVE": False,
        }

    # ─────────────────────────────────────────────
    # REGIME DETECTION
    # ─────────────────────────────────────────────

    def detect_regime(
        self,
        trend: float,
        atr_pct: float = 0.0,
        vol: float = None,
    ) -> str:
        """
        5-state classifier: BULL / UNCERTAIN / SIDEWAYS / BEAR / VOLATILE

        Args:
            trend:   0..1 trend strength score
            atr_pct: ATR as % of price
            vol:     backward-compat alias for atr_pct

        Detection order (important — first match wins):
          1. VOLATILE:   atr_pct >= 1.50
          2. BEAR:       trend <= -0.10
          3. SIDEWAYS:   trend < 0.30 AND atr_pct <= 0.30
          4. UNCERTAIN:  trend >= 0.30 AND atr_pct > 0.40  (BULL ტრენდი, მაგრამ მაღალი ვოლატ.)
          5. BULL:       trend >= 0.30 AND atr_pct <= 0.40
          6. UNCERTAIN:  otherwise
        """
        if vol is not None and atr_pct == 0.0:
            atr_pct = float(vol)

        if atr_pct >= _VOLATILE_ATR_MIN:
            return "VOLATILE"

        if trend <= _BEAR_TREND_MAX:
            return "BEAR"

        if trend < _BULL_TREND_MIN and atr_pct <= _SIDEWAYS_ATR_MAX * 1.5:
            return "SIDEWAYS"

        # FIX: UNCERTAIN — BULL ტრენდი მაგრამ ზედმეტად მაღალი ვოლატილობა
        # detect_regime(0.3, 0.5) → UNCERTAIN (was BULL — bug)
        _atr_bull_max = _SIDEWAYS_ATR_MAX * 2.0   # 0.20 × 2.0 = 0.40
        if trend >= _BULL_TREND_MIN and atr_pct <= _atr_bull_max:
            return "BULL"

        return "UNCERTAIN"

    # ─────────────────────────────────────────────
    # ეტაპი 1: TP / SL / SIZE
    # ─────────────────────────────────────────────

    @staticmethod
    def _get_tp_sl(regime: str, atr_pct: float) -> Tuple[float, float]:
        """
        Per-regime ATR multipliers (ENV-aligned):
          BULL:      TP × 4.0,  SL × 2.0  (ATR_MULT_TP_BULL=4.0, ATR_MULT_SL_BULL=2.0)
          UNCERTAIN: TP × 2.0,  SL × 1.2
          fallback:  TP=1.5%,   SL=0.80%  (TP_PCT=1.5, SL_PCT=0.80)
        """
        mults = {
            "BULL":      (_ATR_TP_BULL,      _ATR_SL_BULL),
            "UNCERTAIN": (_ATR_TP_UNCERTAIN, _ATR_SL_UNCERTAIN),
        }
        tm, sm = mults.get(regime, (_ATR_TP_UNCERTAIN, _ATR_SL_UNCERTAIN))

        if atr_pct > 0:
            tp = max(_MIN_TP, min(_MAX_TP, atr_pct * tm))
            sl = max(_MIN_SL, min(_MAX_SL, atr_pct * sm))
        else:
            # ATR=0: fixed fallback from ENV
            tp = _TP_PCT_FIXED
            sl = _SL_PCT_FIXED

        return round(tp, 3), round(sl, 3)

    @staticmethod
    def _get_quote_size(regime: str, max_quote: float) -> float:
        """
        Per-regime position sizing:
          პირველ რიგში QUOTE_SIZE_BULL / QUOTE_SIZE_UNCERTAIN გამოიყენება (პირდაპირი USDT).
          თუ 0-ია — REGIME_SIZE_*_PCT × max_quote გამოიყენება (პროცენტული).
          ყოველთვის DYNAMIC_SIZE_MIN..DYNAMIC_SIZE_MAX-ში ჩაიჭრება.

          BULL:      QUOTE_SIZE_BULL=10.0    → $10
          UNCERTAIN: QUOTE_SIZE_UNCERTAIN=7.0 → $7
        """
        # პირდაპირი USDT ზომა (QUOTE_SIZE_BULL/UNCERTAIN)
        direct = {
            "BULL":      _QUOTE_SIZE_BULL,
            "UNCERTAIN": _QUOTE_SIZE_UNCERTAIN,
        }
        direct_val = direct.get(regime, 0.0)

        if direct_val > 0:
            # პირდაპირი მნიშვნელობა — ENV QUOTE_SIZE_* გამოიყენება
            size = round(direct_val, 2)
        else:
            # პროცენტული fallback — REGIME_SIZE_*_PCT × max_quote
            pcts = {
                "BULL":      _SIZE_BULL_PCT,
                "UNCERTAIN": _SIZE_UNCERTAIN_PCT,
            }
            pct = pcts.get(regime, _SIZE_UNCERTAIN_PCT)
            size = round(max_quote * pct, 2)

        # ყოველთვის DYNAMIC_SIZE_MIN..DYNAMIC_SIZE_MAX-ში
        size = max(
            _ef("DYNAMIC_SIZE_MIN", 8.0),
            min(_ef("DYNAMIC_SIZE_MAX", 10.0), size)
        )
        return size

    # ─────────────────────────────────────────────
    # ეტაპი 2: CONFIDENCE THRESHOLD ADAPTATION
    # ─────────────────────────────────────────────

    @staticmethod
    def _adapt_conf_min(regime: str, base: float) -> float:
        """
        BULL:      base × 0.85  (ნაკლები სიმკაცრე — ტრენდი ჩვენთვის მუშაობს)
        UNCERTAIN: base × 1.20  (მეტი სიმკაცრე — ეჭვიანი ბაზარი)

        ENV: BUY_CONFIDENCE_MIN=0.38
          BULL:      0.38 × 0.85 = 0.323
          UNCERTAIN: 0.38 × 1.20 = 0.456

        bounds: 0.30 .. 0.75
        """
        if base <= 0:
            base = _ef("BUY_CONFIDENCE_MIN", 0.38)   # FIX: 0.44 → 0.38 (ENV)

        mults = {
            "BULL":      _CONF_BULL_MULT,
            "UNCERTAIN": _CONF_UNCERTAIN_MULT,
        }
        mult = mults.get(regime, 1.0)
        adapted = round(base * mult, 3)

        # bounds: 0.30 .. 0.75
        return max(0.30, min(0.75, adapted))

    # ─────────────────────────────────────────────
    # ეტაპი 3: MTF CONFIRMATION
    # ─────────────────────────────────────────────

    @staticmethod
    def _mtf_check(regime_15m: str, regime_1h: Optional[str], symbol: str) -> Dict:
        """
        MTF alignment logic (ENV-aligned v3.1):

          STRONG:  15m=BULL  + 1h=BULL       → TP bonus  +25% (MTF_TP_BONUS=0.25)
          WEAK:    15m=BULL  + 1h=UNCERTAIN  → TP penalty -20% (MTF_TP_PENALTY=0.20)
          DIVERGE: 15m=BULL  + 1h=BEAR       →
                     MTF_BLOCK_ON_BEAR_DIVERGE=false → TP penalty only (no hard SKIP)
          DIVERGE: 15m=UNCERT + 1h=BEAR/VOLATILE → SKIP always (unconditional)
          N/A:     1h data არ არის           → ნეიტრალური
        """
        if not _MTF_ENABLED or regime_1h is None:
            return {"SKIP": False, "REASON": "", "ALIGNMENT": "N/A", "CONFIRMED": False}

        htf = regime_1h.upper()
        ltf = regime_15m.upper()

        # UNCERTAIN + 1h BEAR/VOLATILE → ყოველთვის SKIP (unconditional)
        if ltf == "UNCERTAIN" and htf in ("BEAR", "VOLATILE"):
            return {
                "SKIP":      True,
                "REASON":    f"MTF_UNCERTAIN_HTF_{htf}",
                "ALIGNMENT": "DIVERGE",
                "CONFIRMED": False,
            }

        # BULL + 1h BEAR/VOLATILE
        if ltf == "BULL" and htf in ("BEAR", "VOLATILE"):
            if _MTF_BLOCK_DIVERGE:
                # MTF_BLOCK_ON_BEAR_DIVERGE=true → hard SKIP
                return {
                    "SKIP":      True,
                    "REASON":    f"MTF_BULL_HTF_{htf}_BLOCKED",
                    "ALIGNMENT": "DIVERGE",
                    "CONFIRMED": False,
                }
            else:
                # ENV=false → TP penalty only, no SKIP (trade allowed with reduced TP)
                return {
                    "SKIP":      False,
                    "REASON":    "",
                    "ALIGNMENT": "DIVERGE",
                    "CONFIRMED": False,
                }

        # STRONG: ორივე BULL → TP bonus
        if ltf == "BULL" and htf == "BULL":
            return {"SKIP": False, "REASON": "", "ALIGNMENT": "STRONG", "CONFIRMED": True}

        # WEAK: 15m BULL, 1h UNCERTAIN/SIDEWAYS → TP penalty
        if ltf == "BULL" and htf in ("UNCERTAIN", "SIDEWAYS"):
            return {"SKIP": False, "REASON": "", "ALIGNMENT": "WEAK", "CONFIRMED": False}

        # default: NEUTRAL
        return {"SKIP": False, "REASON": "", "ALIGNMENT": "NEUTRAL", "CONFIRMED": False}

    @staticmethod
    def _apply_mtf_tp(tp_pct: float, alignment: str) -> float:
        """
        MTF alignment-ის მიხედვით TP კორექცია:
          STRONG:          tp × 1.25  (ENV=MTF_TP_BONUS=0.25)
          WEAK / DIVERGE:  tp × 0.80  (ENV=MTF_TP_PENALTY=0.20)
          NEUTRAL / N/A:   unchanged
        Clamped to [_MIN_TP, _MAX_TP].
        """
        if alignment == "STRONG":
            tp_pct = tp_pct * (1.0 + _MTF_TP_BONUS)
        elif alignment in ("WEAK", "DIVERGE"):
            tp_pct = tp_pct * (1.0 - _MTF_TP_PENALTY)
        return round(max(_MIN_TP, min(_MAX_TP, tp_pct)), 3)

    # ─────────────────────────────────────────────
    # REGIME HISTORY & STABILITY
    # ─────────────────────────────────────────────

    def _update_history(self, symbol: str, regime: str) -> None:
        if symbol not in self._regime_history:
            self._regime_history[symbol] = deque(maxlen=_HISTORY_DEPTH)
        self._regime_history[symbol].append(regime)

    def _is_stable(self, symbol: str) -> bool:
        """
        True თუ ბოლო N tick-ის >= REGIME_STABILITY_MIN(60%) ერთი regime-ია.
        """
        hist = self._regime_history.get(symbol)
        if not hist or len(hist) < 3:
            return True  # საკმარისი ისტორია არ არის → ნეიტრალური
        dominant = max(set(hist), key=list(hist).count)
        ratio = list(hist).count(dominant) / len(hist)
        return ratio >= _REGIME_STABILITY_MIN

    def get_regime_history(self, symbol: str) -> list:
        hist = self._regime_history.get(symbol or "_global_")
        return list(hist) if hist else []

    # ─────────────────────────────────────────────
    # SL COOLDOWN (in-memory — regime engine level)
    # NOTE: production-ზე DB-based cooldown signal_generator.py-ში გამოიყენება
    # ─────────────────────────────────────────────

    def notify_outcome(
        self,
        symbol: str,
        outcome: str,
        buy_time: Optional[datetime] = None,
    ) -> None:
        """
        trade დახურვის შემდეგ გამოიძახება.
        outcome: 'SL' | 'TP' | 'MANUAL_SELL'
        """
        sym = symbol or "_global_"
        outcome = str(outcome).upper()
        now = buy_time or datetime.utcnow()

        if outcome == "SL":
            # DCA mode: SL=999% — SL არასოდეს ვარდება → no-op
            pass

        elif outcome in ("TP", "MANUAL_SELL"):
            prev = self._consecutive_sl.get(sym, 0)
            if prev > 0:
                logger.info(f"[REGIME] TP_RESET | sym={sym} consecutive {prev}→0")
            self._consecutive_sl[sym] = 0
            self._sl_pause_until[sym] = None

    def reset_cooldown(self, symbol: str) -> None:
        sym = symbol or "_global_"
        self._consecutive_sl[sym] = 0
        self._sl_pause_until[sym] = None
        logger.info(f"[REGIME] COOLDOWN_RESET | sym={sym}")

    def is_paused(self, symbol: str, now: Optional[datetime] = None) -> bool:
        sym = symbol or "_global_"
        pause = self._sl_pause_until.get(sym)
        if pause is None:
            return False
        return (now or datetime.utcnow()) < pause

    def get_consecutive_sl(self, symbol: str) -> int:
        return self._consecutive_sl.get(symbol or "_global_", 0)

    # ─────────────────────────────────────────────
    # PUBLIC WRAPPERS (backward compat)
    # ─────────────────────────────────────────────

    def get_tp_sl(self, regime: str, atr_pct: float = 0.0) -> Tuple[float, float]:
        """execution_engine.py-სთვის — v2 API შენარჩუნება."""
        return self._get_tp_sl(regime, atr_pct)

    def get_conf_min(self, regime: str, base_conf_min: float = 0.0) -> float:
        """signal_generator.py-სთვის — ადაpტირებული threshold."""
        return self._adapt_conf_min(regime, base_conf_min)

    # ─────────────────────────────────────────────
    # DEBUG / SUMMARY
    # ─────────────────────────────────────────────

    def summary(self) -> Dict:
        """სრული debug state."""
        symbols = set(
            list(self._consecutive_sl)
            + list(self._sl_pause_until)
            + list(self._regime_history)
        )
        return {
            sym: {
                "consecutive_sl": self._consecutive_sl.get(sym, 0),
                "pause_until":    str(self._sl_pause_until.get(sym, "none")),
                "regime_history": list(self._regime_history.get(sym, [])),
                "regime_stable":  self._is_stable(sym),
            }
            for sym in symbols
        }

    # ─────────────────────────────────────────────
    # INTERNAL HELPERS
    # ─────────────────────────────────────────────

    @staticmethod
    def _skip(reason: str, regime: str, cooldown: bool = False) -> Dict:
        return {
            "SKIP_TRADING":    True,
            "SKIP_REASON":     reason,
            "TP_PCT":          0.0,
            "SL_PCT":          0.0,
            "REGIME":          regime,
            "QUOTE_SIZE":      0.0,
            "CONF_MIN":        0.0,
            "MTF_CONFIRMED":   False,
            "MTF_ALIGNMENT":   "N/A",
            "REGIME_STABLE":   False,
            "COOLDOWN_ACTIVE": cooldown,
        }
