# execution/regime_engine.py
# ============================================================
# Market Regime Engine — v3 (Adaptive, Regime-Aware)
# ============================================================
# v2-სთან შედარებით ახალი:
#
#   ეტაპი 1 — TP/SL/Size ავტომატური ადაპტაცია:
#     - BULL:      TP=ATR×3.0  SL=ATR×1.5  Size=100%
#     - UNCERTAIN: TP=ATR×2.0  SL=ATR×1.2  Size=50%
#     - SIDEWAYS:  SKIP (არ ვიყიდით)
#     - BEAR:      SKIP
#     - VOLATILE:  SKIP
#
#   ეტაპი 2 — Confidence threshold ადაპტაცია:
#     - BULL:      BUY_CONFIDENCE_MIN × 0.85  (ნაკლები სიმკაცრე)
#     - UNCERTAIN: BUY_CONFIDENCE_MIN × 1.20  (მეტი სიმკაცრე)
#     - ბოტი ავტომატურად არეგულირებს — .env ცვლილება არ სჭირდება
#
#   ეტაპი 3 — MTF Confirmation score:
#     - 15m + 1h რეჟიმების შედარება
#     - MTF_BONUS / MTF_PENALTY TP-ზე
#     - UNCERTAIN + MTF_BEAR → hard skip
#
#   დამატებითი:
#     - Regime history (ბოლო 10 tick) → trend stability score
#     - Per-symbol regime tracking
#     - სრული debug summary()
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
_BULL_TREND_MIN   = _ef("REGIME_BULL_TREND_MIN",    0.30)
_SIDEWAYS_ATR_MAX = _ef("REGIME_SIDEWAYS_ATR_MAX",  0.18)
_VOLATILE_ATR_MIN = _ef("REGIME_VOLATILE_ATR_MIN",  1.50)
_BEAR_TREND_MAX   = -0.10

# ──────────────────────────────────────────────────────────
# ეტაპი 1: TP/SL Multipliers — per regime
# ──────────────────────────────────────────────────────────
#   BULL: სრული TP, ოდნავ ფართო SL (ნოიზის ათვალისწინებით)
#   UNCERTAIN: შემცირებული TP, კიდევ უფრო ფართო SL (ეჭვიანი ბაზარი)
_ATR_TP_BULL        = _ef("ATR_MULT_TP_BULL",       3.0)
_ATR_SL_BULL        = _ef("ATR_MULT_SL_BULL",       1.5)   # v2: 1.0 → v3: 1.5

_ATR_TP_UNCERTAIN   = _ef("ATR_MULT_TP_UNCERTAIN",  2.0)   # v2: hardcoded 2.5 → v3: ENV
_ATR_SL_UNCERTAIN   = _ef("ATR_MULT_SL_UNCERTAIN",  1.2)   # v2: hardcoded 1.0 → v3: ENV

# TP/SL საზღვრები
_MIN_TP = _ef("MIN_NET_PROFIT_PCT", 0.50)
_MIN_SL = 0.25   # v2: 0.20 → v3: 0.25 (ნოიზის ათვალისწინება)
_MAX_TP = 4.0
_MAX_SL = 1.8    # v2: 1.5 → v3: 1.8

# Fixed fallback (DEMO / ATR=0)
_TP_PCT_FIXED = _ef("TP_PCT", 1.0)
_SL_PCT_FIXED = _ef("SL_PCT", 0.70)

# ──────────────────────────────────────────────────────────
# ეტაპი 1: Position Size — per regime (% of max quote)
# ──────────────────────────────────────────────────────────
_SIZE_BULL_PCT      = _ef("REGIME_SIZE_BULL_PCT",      1.00)  # 100% → $15
_SIZE_UNCERTAIN_PCT = _ef("REGIME_SIZE_UNCERTAIN_PCT", 0.50)  # 50%  → $7.5

# ──────────────────────────────────────────────────────────
# ეტაპი 2: Confidence threshold ადაპტაცია
# ──────────────────────────────────────────────────────────
_CONF_BULL_MULT      = _ef("REGIME_CONF_BULL_MULT",      0.85)  # BULL → threshold × 0.85
_CONF_UNCERTAIN_MULT = _ef("REGIME_CONF_UNCERTAIN_MULT", 1.20)  # UNCERTAIN → threshold × 1.20

# ──────────────────────────────────────────────────────────
# ეტაპი 3: MTF Confirmation
# ──────────────────────────────────────────────────────────
_MTF_ENABLED         = _eb("USE_MTF_FILTER", True)
_MTF_TP_BONUS        = _ef("MTF_TP_BONUS",   0.20)   # 1h BULL + 15m BULL → TP +20%
_MTF_TP_PENALTY      = _ef("MTF_TP_PENALTY", 0.15)   # 1h/15m diverge → TP -15%
_MTF_BLOCK_DIVERGE   = _eb("MTF_BLOCK_ON_BEAR_DIVERGE", True)  # UNCERTAIN+1h_BEAR → SKIP

# SL Cooldown
_SL_COOLDOWN_N    = _ei("SL_COOLDOWN_AFTER_N",      2)
_SL_PAUSE_SECONDS = _ei("SL_COOLDOWN_PAUSE_SECONDS", 1800)

# Regime history depth
_HISTORY_DEPTH = _ei("REGIME_HISTORY_DEPTH", 10)


# ─────────────────────────────────────────────
# REGIME SCORE WEIGHTS (stability check)
# ─────────────────────────────────────────────
_REGIME_STABILITY_MIN = _ef("REGIME_STABILITY_MIN", 0.60)
# ბოლო 10 tick-იდან მინიმუმ 60% ერთი regime → "stable"


class MarketRegimeEngine:
    """
    Adaptive Regime-Aware Trading Engine — v3

    სამი ეტაპი ერთ კლასში:
      1. TP/SL/Size ავტომატური ადაპტაცია per-regime
      2. Confidence threshold ადაปტაცია per-regime
      3. MTF (15m + 1h) confirmation + TP bonus/penalty

    Usage:
        engine = MarketRegimeEngine()

        # per 15m tick:
        regime_15m = engine.detect_regime(trend=0.6, atr_pct=0.45)
        regime_1h  = engine.detect_regime(trend=0.5, atr_pct=0.40)  # MTF

        params = engine.apply(
            regime       = regime_15m,
            atr_pct      = 0.45,
            symbol       = "BTC/USDT",
            htf_regime   = regime_1h,      # ეტაპი 3
            base_conf_min= 0.44,           # ეტაპი 2 — .env BUY_CONFIDENCE_MIN
            base_quote   = 15.0,           # ეტაპი 1 — max position size
        )

        if params["SKIP_TRADING"]:
            continue

        tp            = params["TP_PCT"]
        sl            = params["SL_PCT"]
        quote         = params["QUOTE_SIZE"]      # ეტაპი 1
        conf_min      = params["CONF_MIN"]        # ეტაპი 2 — ადაpტირებული threshold
        mtf_confirmed = params["MTF_CONFIRMED"]   # ეტაპი 3

        # trade დახურვის შემდეგ:
        engine.notify_outcome("BTC/USDT", "SL")
        engine.notify_outcome("BTC/USDT", "TP")
    """

    def __init__(self, config=None):
        self._config = config  # backward compat

        # SL Cooldown per-symbol
        self._consecutive_sl: Dict[str, int]              = {}
        self._sl_pause_until: Dict[str, Optional[datetime]] = {}

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

        Returns:
            SKIP_TRADING   : bool
            SKIP_REASON    : str
            TP_PCT         : float   — ეტაპი 1 (+ MTF bonus/penalty)
            SL_PCT         : float   — ეტაპი 1
            REGIME         : str
            QUOTE_SIZE     : float   — ეტაპი 1
            CONF_MIN       : float   — ეტაპი 2 (ადაpტირებული)
            MTF_CONFIRMED  : bool    — ეტაპი 3
            MTF_ALIGNMENT  : str     — "STRONG"/"WEAK"/"DIVERGE"/"N/A"
            REGIME_STABLE  : bool    — history-based stability
            COOLDOWN_ACTIVE: bool
        """
        now = buy_time or datetime.utcnow()
        sym = symbol or "_global_"

        # regime history განახლება
        self._update_history(sym, regime)

        # ── SL Cooldown ──────────────────────────────
        pause_until = self._sl_pause_until.get(sym)
        if pause_until is not None and now < pause_until:
            remaining = int((pause_until - now).total_seconds())
            logger.info(f"[REGIME] SL_COOLDOWN | sym={sym} remaining={remaining}s")
            return self._skip("SL_COOLDOWN_PAUSE", regime, cooldown=True)

        # ── Skip regimes ─────────────────────────────
        if regime in ("BEAR", "VOLATILE", "SIDEWAYS"):
            logger.info(f"[REGIME] SKIP | sym={sym} regime={regime}")
            return self._skip(f"REGIME_{regime}", regime)

        # ── ეტაპი 3: MTF Confirmation ─────────────────
        mtf_result = self._mtf_check(regime, htf_regime, sym)
        if mtf_result["SKIP"]:
            logger.info(
                f"[REGIME] MTF_BLOCK | sym={sym} "
                f"15m={regime} 1h={htf_regime} reason={mtf_result['REASON']}"
            )
            return self._skip(mtf_result["REASON"], regime)

        # ── ეტაპი 1: TP/SL/Size ──────────────────────
        tp_pct, sl_pct = self._get_tp_sl(regime, atr_pct)

        # MTF TP bonus/penalty
        tp_pct = self._apply_mtf_tp(tp_pct, mtf_result["ALIGNMENT"])

        # Position size
        max_q = base_quote if base_quote > 0 else _ef("BOT_QUOTE_PER_TRADE", 15.0)
        quote_size = self._get_quote_size(regime, max_q)

        # ── ეტაპი 2: Confidence threshold ────────────
        conf_min = self._adapt_conf_min(regime, base_conf_min)

        # Regime stability
        stable = self._is_stable(sym)

        logger.info(
            f"[REGIME] OK | sym={sym} regime={regime} "
            f"TP={tp_pct:.3f}% SL={sl_pct:.3f}% "
            f"size=${quote_size:.1f} conf_min={conf_min:.3f} "
            f"mtf={mtf_result['ALIGNMENT']} stable={stable}"
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
            trend:   0..1 trend strength
            atr_pct: ATR as % of price
            vol:     backward-compat alias for atr_pct
        """
        if vol is not None and atr_pct == 0.0:
            atr_pct = float(vol)

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
    # ეტაპი 1: TP / SL / SIZE
    # ─────────────────────────────────────────────

    @staticmethod
    def _get_tp_sl(regime: str, atr_pct: float) -> Tuple[float, float]:
        """
        Per-regime ATR multipliers:
          BULL:      TP × 3.0,  SL × 1.5
          UNCERTAIN: TP × 2.0,  SL × 1.2
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
            tp = _TP_PCT_FIXED
            sl = _SL_PCT_FIXED

        return round(tp, 3), round(sl, 3)

    @staticmethod
    def _get_quote_size(regime: str, max_quote: float) -> float:
        """
        Per-regime position sizing:
          BULL:      100% of max_quote
          UNCERTAIN: 50%  of max_quote
        """
        pcts = {
            "BULL":      _SIZE_BULL_PCT,
            "UNCERTAIN": _SIZE_UNCERTAIN_PCT,
        }
        pct = pcts.get(regime, _SIZE_UNCERTAIN_PCT)
        size = round(max_quote * pct, 2)
        # hard floor / ceiling
        size = max(_ef("DYNAMIC_SIZE_MIN", 5.0), min(_ef("DYNAMIC_SIZE_MAX", 15.0), size))
        return size

    # ─────────────────────────────────────────────
    # ეტაპი 2: CONFIDENCE THRESHOLD ADAPTATION
    # ─────────────────────────────────────────────

    @staticmethod
    def _adapt_conf_min(regime: str, base: float) -> float:
        """
        BULL:      base × 0.85  (ნაკლები სიმკაცრე — ტრენდი ჩვენთვის მუშაობს)
        UNCERTAIN: base × 1.20  (მეტი სიმკაცრე — ეჭვიანი ბაზარი)

        თუ base=0 (caller არ გადასცემს) → BASE-ს ENV-იდან კითხულობს
        """
        if base <= 0:
            base = _ef("BUY_CONFIDENCE_MIN", 0.44)

        mults = {
            "BULL":      _CONF_BULL_MULT,
            "UNCERTAIN": _CONF_UNCERTAIN_MULT,
        }
        mult = mults.get(regime, 1.0)
        adapted = round(base * mult, 3)

        # საზღვრები: 0.30 .. 0.75
        adapted = max(0.30, min(0.75, adapted))
        return adapted

    # ─────────────────────────────────────────────
    # ეტაპი 3: MTF CONFIRMATION
    # ─────────────────────────────────────────────

    @staticmethod
    def _mtf_check(regime_15m: str, regime_1h: Optional[str], symbol: str) -> Dict:
        """
        MTF alignment logic:

          STRONG:  15m=BULL  + 1h=BULL      → TP bonus +20%
          WEAK:    15m=BULL  + 1h=UNCERTAIN → TP penalty -15%
          DIVERGE: 15m=BULL  + 1h=BEAR      → SKIP (if MTF_BLOCK enabled)
                   15m=UNCERTAIN + 1h=BEAR  → SKIP always
          N/A:     1h data არ არის          → ნეიტრალური
        """
        if not _MTF_ENABLED or regime_1h is None:
            return {"SKIP": False, "REASON": "", "ALIGNMENT": "N/A", "CONFIRMED": False}

        htf = regime_1h.upper()
        ltf = regime_15m.upper()

        # UNCERTAIN + 1h BEAR → ყოველთვის SKIP
        if ltf == "UNCERTAIN" and htf in ("BEAR", "VOLATILE"):
            return {
                "SKIP":      True,
                "REASON":    f"MTF_UNCERTAIN_HTF_{htf}",
                "ALIGNMENT": "DIVERGE",
                "CONFIRMED": False,
            }

        # BULL + 1h BEAR → SKIP თუ MTF_BLOCK ჩართულია
        if ltf == "BULL" and htf in ("BEAR", "VOLATILE"):
            if _MTF_BLOCK_DIVERGE:
                return {
                    "SKIP":      True,
                    "REASON":    f"MTF_BULL_HTF_{htf}_BLOCKED",
                    "ALIGNMENT": "DIVERGE",
                    "CONFIRMED": False,
                }

        # STRONG: ორივე BULL
        if ltf == "BULL" and htf == "BULL":
            return {"SKIP": False, "REASON": "", "ALIGNMENT": "STRONG", "CONFIRMED": True}

        # WEAK: 15m BULL, 1h UNCERTAIN/SIDEWAYS
        if ltf == "BULL" and htf in ("UNCERTAIN", "SIDEWAYS"):
            return {"SKIP": False, "REASON": "", "ALIGNMENT": "WEAK", "CONFIRMED": False}

        # default: ნეიტრალური
        return {"SKIP": False, "REASON": "", "ALIGNMENT": "NEUTRAL", "CONFIRMED": False}

    @staticmethod
    def _apply_mtf_tp(tp_pct: float, alignment: str) -> float:
        """MTF alignment-ის მიხედვით TP კორექცია."""
        if alignment == "STRONG":
            tp_pct = tp_pct * (1.0 + _MTF_TP_BONUS)
        elif alignment in ("WEAK", "DIVERGE"):
            tp_pct = tp_pct * (1.0 - _MTF_TP_PENALTY)
        tp_pct = max(_MIN_TP, min(_MAX_TP, tp_pct))
        return round(tp_pct, 3)

    # ─────────────────────────────────────────────
    # REGIME HISTORY & STABILITY
    # ─────────────────────────────────────────────

    def _update_history(self, symbol: str, regime: str) -> None:
        if symbol not in self._regime_history:
            self._regime_history[symbol] = deque(maxlen=_HISTORY_DEPTH)
        self._regime_history[symbol].append(regime)

    def _is_stable(self, symbol: str) -> bool:
        """
        True თუ ბოლო N tick-ის მინიმუმ REGIME_STABILITY_MIN% ერთი regime-ია.
        მაგ: ბოლო 10-დან 7 BULL → stable=True
        """
        hist = self._regime_history.get(symbol)
        if not hist or len(hist) < 3:
            return True  # საკმარისი ისტორია არ არის → ნეიტრალურად ითვლება
        dominant = max(set(hist), key=list(hist).count)
        ratio = list(hist).count(dominant) / len(hist)
        return ratio >= _REGIME_STABILITY_MIN

    def get_regime_history(self, symbol: str) -> list:
        hist = self._regime_history.get(symbol or "_global_")
        return list(hist) if hist else []

    # ─────────────────────────────────────────────
    # SL COOLDOWN
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
            self._consecutive_sl[sym] = self._consecutive_sl.get(sym, 0) + 1
            count = self._consecutive_sl[sym]
            logger.info(
                f"[REGIME] SL_TRACK | sym={sym} "
                f"consecutive={count} limit={_SL_COOLDOWN_N}"
            )
            if count >= _SL_COOLDOWN_N:
                pause = now + timedelta(seconds=_SL_PAUSE_SECONDS)
                self._sl_pause_until[sym] = pause
                logger.warning(
                    f"[REGIME] SL_COOLDOWN | sym={sym} {count} consecutive SL → "
                    f"PAUSE {_SL_PAUSE_SECONDS // 60}min "
                    f"until {pause.strftime('%H:%M:%S')} UTC"
                )

        elif outcome in ("TP", "MANUAL_SELL"):
            prev = self._consecutive_sl.get(sym, 0)
            if prev > 0:
                logger.info(f"[REGIME] TP_RESET | sym={sym} consecutive {prev}→0")
            self._consecutive_sl[sym]  = 0
            self._sl_pause_until[sym]  = None

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
                "consecutive_sl":  self._consecutive_sl.get(sym, 0),
                "pause_until":     str(self._sl_pause_until.get(sym, "none")),
                "regime_history":  list(self._regime_history.get(sym, [])),
                "regime_stable":   self._is_stable(sym),
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
