# execution/signal_generator.py
import os
import time
import uuid
import logging
from datetime import datetime
from typing import Optional, Dict, Any, Tuple, List

import ccxt

from execution.signal_client import append_signal
from execution.db.repository import (
    has_active_oco_for_symbol,
    has_open_trade_for_symbol,
    get_sl_cooldown_state,
    increment_consecutive_sl,
    reset_consecutive_sl,
    is_sl_pause_active,
    # FIX I-8 FULL: per-symbol SL cooldown (ახალი ცხრილი)
    get_sl_cooldown_state_per_symbol,
    increment_consecutive_sl_per_symbol,
    reset_consecutive_sl_per_symbol,
    is_sl_pause_active_per_symbol,
    get_all_symbol_cooldown_states,
)
from execution.excel_live_core import ExcelLiveCore, CoreInputs
from execution.regime_engine import MarketRegimeEngine

logger = logging.getLogger("gbm")

# -----------------------------
# ENV
# -----------------------------
TIMEFRAME = os.getenv("BOT_TIMEFRAME", "15m").strip()
CANDLE_LIMIT = int(os.getenv("BOT_CANDLE_LIMIT", "80"))
COOLDOWN_SECONDS = int(os.getenv("BOT_SIGNAL_COOLDOWN_SECONDS", "180"))

ALLOW_LIVE_SIGNALS = os.getenv("ALLOW_LIVE_SIGNALS", "false").strip().lower() == "true"

BOT_QUOTE_PER_TRADE = float(os.getenv("BOT_QUOTE_PER_TRADE", "15"))
# MAX_QUOTE_PER_TRADE — exchange_client._guard() hard ceiling
# dynamic sizing ამ მნიშვნელობას ვერ გადააჭარბებს → LIVE_BLOCKED აღარ იქნება
MAX_QUOTE_PER_TRADE = float(os.getenv("MAX_QUOTE_PER_TRADE", "15"))

# Fee-aware edge gate
MIN_MOVE_PCT = float(os.getenv("MIN_MOVE_PCT", "0.60"))
ESTIMATED_ROUNDTRIP_FEE_PCT = float(os.getenv("ESTIMATED_ROUNDTRIP_FEE_PCT", "0.20"))
ESTIMATED_SLIPPAGE_PCT = float(os.getenv("ESTIMATED_SLIPPAGE_PCT", "0.15"))
TP_PCT = float(os.getenv("TP_PCT", "1.3"))
MIN_NET_PROFIT_PCT = float(os.getenv("MIN_NET_PROFIT_PCT", "0.60"))

# ATR sanity
ATR_TO_TP_SANITY_FACTOR = float(os.getenv("ATR_TO_TP_SANITY_FACTOR", "0.20"))

# Optional MA filters
USE_MA_FILTERS = os.getenv("USE_MA_FILTERS", "true").strip().lower() == "true"
MA_GAP_PCT = float(os.getenv("MA_GAP_PCT", "0.15"))

# Extra confidence guard (after Excel decision)
BUY_CONFIDENCE_MIN = float(os.getenv("BUY_CONFIDENCE_MIN", "0.64"))

BLOCK_SIGNALS_WHEN_ACTIVE_OCO = os.getenv("BLOCK_SIGNALS_WHEN_ACTIVE_OCO", "true").strip().lower() == "true"

GEN_DEBUG = os.getenv("GEN_DEBUG", "true").strip().lower() == "true"
GEN_LOG_EVERY_TICK = os.getenv("GEN_LOG_EVERY_TICK", "true").strip().lower() == "true"

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DEAD PARAMS ACTIVATED — ადრე ENV-ში იყო, კოდი არ კითხულობდა
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# 1. Volume filter — 24h volume < MIN_VOLUME_24H → BUY skip
MIN_VOLUME_24H = float(os.getenv("MIN_VOLUME_24H", "0"))

# 2. Signal expiration — signal ts_utc-დან SIGNAL_EXPIRATION_SECONDS გასული → skip
SIGNAL_EXPIRATION_SECONDS = int(os.getenv("SIGNAL_EXPIRATION_SECONDS", "0"))

# 3. AI confidence boost — ai_score * AI_CONFIDENCE_BOOST (>1.0 ამაღლებს score-ს)
AI_CONFIDENCE_BOOST = float(os.getenv("AI_CONFIDENCE_BOOST", "1.0"))

# 4. Trade frequency limits
MAX_TRADES_PER_DAY  = int(os.getenv("MAX_TRADES_PER_DAY",  "0"))
MAX_TRADES_PER_HOUR = int(os.getenv("MAX_TRADES_PER_HOUR", "0"))

# 5. AI_FILTER_LOW_CONFIDENCE — ai_score < threshold → hard reject before any other check
# true = strict mode: ყველა low-confidence signal drop-ი ყველა filter-ის წინ
AI_FILTER_LOW_CONFIDENCE = os.getenv("AI_FILTER_LOW_CONFIDENCE", "false").strip().lower() == "true"
AI_FILTER_MIN_SCORE      = float(os.getenv("BUY_CONFIDENCE_MIN", "0.38"))  # reuses BUY_CONFIDENCE_MIN

# 6. GEN_TEST_SIGNAL — force-emit one test signal for integration testing (true = one shot)
GEN_TEST_SIGNAL = os.getenv("GEN_TEST_SIGNAL", "false").strip().lower() == "true"

# 7. BUY_LIQUIDITY_MIN_SCORE — volume_score minimum for BUY (0=off)
# volume_score < this → skip (stricter than soft-volume-override)
BUY_LIQUIDITY_MIN_SCORE = float(os.getenv("BUY_LIQUIDITY_MIN_SCORE", "0"))

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# #1 RSI + MACD filter
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
USE_RSI_FILTER        = os.getenv("USE_RSI_FILTER", "true").strip().lower() == "true"
RSI_PERIOD            = int(os.getenv("RSI_PERIOD", "14"))
RSI_MIN               = float(os.getenv("RSI_MIN", "35"))
RSI_MAX               = float(os.getenv("RSI_MAX", "70"))
RSI_SELL_MIN          = float(os.getenv("RSI_SELL_MIN", "60"))

USE_MACD_FILTER       = os.getenv("USE_MACD_FILTER", "true").strip().lower() == "true"
MACD_FAST             = int(os.getenv("MACD_FAST", "12"))
MACD_SLOW             = int(os.getenv("MACD_SLOW", "26"))
MACD_SIGNAL_PERIOD    = int(os.getenv("MACD_SIGNAL_PERIOD", "9"))

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# #2 Multi-timeframe confirmation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
USE_MTF_FILTER        = os.getenv("USE_MTF_FILTER", "true").strip().lower() == "true"
MTF_TIMEFRAME         = os.getenv("MTF_TIMEFRAME", "1h").strip()
MTF_CANDLE_LIMIT      = int(os.getenv("MTF_CANDLE_LIMIT", "50"))

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# #3 Trailing Stop
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TRAILING_STOP_ENABLED   = os.getenv("TRAILING_STOP_ENABLED", "false").strip().lower() == "true"
TRAILING_STOP_DISTANCE  = float(os.getenv("TRAILING_STOP_DISTANCE", "0.25"))

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# #4 Dynamic position sizing
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
USE_DYNAMIC_SIZING    = os.getenv("USE_DYNAMIC_SIZING", "true").strip().lower() == "true"
DYNAMIC_SIZE_MIN      = float(os.getenv("DYNAMIC_SIZE_MIN", "5.0"))
DYNAMIC_SIZE_MAX      = float(os.getenv("DYNAMIC_SIZE_MAX", "15.0"))
DYNAMIC_SIZE_AI_LOW   = float(os.getenv("DYNAMIC_SIZE_AI_LOW",  "0.55"))
DYNAMIC_SIZE_AI_HIGH  = float(os.getenv("DYNAMIC_SIZE_AI_HIGH", "0.80"))

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# #5 Partial Take Profit
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
USE_PARTIAL_TP        = os.getenv("USE_PARTIAL_TP", "false").strip().lower() == "true"
PARTIAL_TP1_PCT       = float(os.getenv("PARTIAL_TP1_PCT", "1.5"))
PARTIAL_TP1_SIZE      = float(os.getenv("PARTIAL_TP1_SIZE", "0.5"))

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# #7 Breakeven Stop
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
USE_BREAKEVEN_STOP    = os.getenv("USE_BREAKEVEN_STOP", "true").strip().lower() == "true"
BREAKEVEN_TRIGGER_PCT = float(os.getenv("BREAKEVEN_TRIGGER_PCT", "0.5"))

# Soft structure override (USED ONLY WHEN USE_MA_FILTERS=false)
STRUCT_SOFT_OVERRIDE = os.getenv("STRUCT_SOFT_OVERRIDE", "true").strip().lower() == "true"
STRUCT_SOFT_MIN_TREND = float(os.getenv("STRUCT_SOFT_MIN_TREND", "0.58"))
STRUCT_SOFT_MIN_MA_GAP = float(os.getenv("STRUCT_SOFT_MIN_MA_GAP", "0.35"))
STRUCT_SOFT_REQUIRE_LAST_UP = int(os.getenv("STRUCT_SOFT_REQUIRE_LAST_UP", "2"))

# Excel model path
EXCEL_MODEL_PATH = os.getenv("EXCEL_MODEL_PATH", "/var/data/DYZEN_CAPITAL_OS_AI_LIVE_CORE_READY.xlsx").strip()
if EXCEL_MODEL_PATH.lower().startswith("excel_model_path="):
    EXCEL_MODEL_PATH = EXCEL_MODEL_PATH.split("=", 1)[1].strip()

_last_emit_ts: float = 0.0

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FIX I-1: per-symbol RSI SELL one-shot flag
# RSI >= RSI_SELL_MIN → SELL emit ხდება ერთხელ per open_trade.
# flag ნულდება RSI-ის დაცემისას ან trade close-ზე.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
_rsi_sell_fired: dict = {}  # {symbol: bool}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SL COOLDOWN — 2 SL-ის შემდეგ 30 წუთი პაუზა
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SL_COOLDOWN_COUNT   = int(os.getenv("SL_COOLDOWN_AFTER_N", "2"))
SL_COOLDOWN_PAUSE   = int(os.getenv("SL_COOLDOWN_PAUSE_SECONDS", "1800"))
RECOVERY_CANDLES    = int(os.getenv("RECOVERY_GREEN_CANDLES", "3"))
# FIX: 0.25% → 0.10% default. 15m flat ბაზარზე სანთლები 0.15-0.35%-ია.
# 0.25% ძალიან მაღალია → recovery 30+ წუთი არ გადის.
# ENV-ში: RECOVERY_CANDLE_PCT=0.15 (ან 0.10 flat ბაზრისთვის)
RECOVERY_CANDLE_PCT = float(os.getenv("RECOVERY_CANDLE_PCT", "0.10"))

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SL Cooldown state — DB-based (restart-safe)
# consecutive_sl და sl_pause_until ახლა DB-შია
# memory globals ამოღებულია — deploy/restart bypass შეუძლებელია
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


# -----------------------------
# HELPERS
# -----------------------------
def _now_utc_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _parse_symbols() -> List[str]:
    raw = os.getenv("BOT_SYMBOLS", "").strip()
    if not raw:
        raw = os.getenv("SYMBOL_WHITELIST", "").strip()
    if not raw:
        raw = os.getenv("BOT_SYMBOL", "BTC/USDT").strip()

    syms: List[str] = []
    for s in raw.split(","):
        s = s.strip()
        if not s:
            continue
        syms.append(s.upper())
    return syms


SYMBOLS = _parse_symbols()


def _has_active_oco(symbol: str) -> bool:
    try:
        return has_active_oco_for_symbol(symbol)
    except Exception as e:
        logger.warning(f"[GEN] ACTIVE_OCO_CHECK_FAIL | symbol={symbol} err={e} -> assume active_oco=True")
        return True


def _has_open_trade(symbol: str) -> bool:
    try:
        return has_open_trade_for_symbol(symbol)
    except Exception as e:
        logger.warning(f"[GEN] OPEN_TRADE_CHECK_FAIL | symbol={symbol} err={e} -> assume open_trade=True")
        return True


_CORE: Optional[ExcelLiveCore] = None


def _core() -> ExcelLiveCore:
    global _CORE
    if _CORE is None:
        # Excel dependency ამოღებულია — ExcelLiveCore ახლა ENV-იდან კითხულობს
        _CORE = ExcelLiveCore()
        logger.info(f"[GEN] CORE_LOADED | version=no-excel ENV-based")
    return _CORE


# ─── Regime Engine singleton ───────────────────────────────
_REGIME_ENGINE: Optional[MarketRegimeEngine] = None


def _regime() -> MarketRegimeEngine:
    global _REGIME_ENGINE
    if _REGIME_ENGINE is None:
        _REGIME_ENGINE = MarketRegimeEngine()
        logger.info("[GEN] REGIME_ENGINE_LOADED")
    return _REGIME_ENGINE


def _pct(a: float, b: float) -> float:
    if b == 0:
        return 0.0
    return (a - b) / b * 100.0


def _sma(vals: List[float], n: int) -> float:
    if not vals:
        return 0.0
    if len(vals) < n:
        return sum(vals) / len(vals)
    w = vals[-n:]
    return sum(w) / n


def _atr_pct(ohlcv: List[List[float]], n: int = 14) -> float:
    if len(ohlcv) < n + 1:
        return 0.0
    trs: List[float] = []
    for i in range(-n, 0):
        high = float(ohlcv[i][2])
        low = float(ohlcv[i][3])
        prev_close = float(ohlcv[i - 1][4])
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        trs.append(tr)
    atr = sum(trs) / n
    last_close = float(ohlcv[-1][4])
    return (atr / last_close) * 100.0 if last_close else 0.0


def _vol_regime(atr_pct: float) -> str:
    if atr_pct >= 2.0:
        return "EXTREME"
    if atr_pct <= 0.30:
        return "LOW"
    return "NORMAL"


def _edge_ok(atr_pct: float) -> Tuple[bool, str]:
    if atr_pct < MIN_MOVE_PCT:
        return False, f"ATR_TOO_LOW atr%={atr_pct:.2f} < MIN_MOVE_PCT={MIN_MOVE_PCT:.2f}"

    assumed_gross_edge = TP_PCT
    assumed_cost = ESTIMATED_ROUNDTRIP_FEE_PCT + ESTIMATED_SLIPPAGE_PCT
    assumed_net = assumed_gross_edge - assumed_cost

    if assumed_net < MIN_NET_PROFIT_PCT:
        return False, (
            "EDGE_TOO_SMALL "
            f"TP_PCT={assumed_gross_edge:.2f} cost={assumed_cost:.2f} net={assumed_net:.2f} "
            f"< MIN_NET_PROFIT_PCT={MIN_NET_PROFIT_PCT:.2f}"
        )

    # FIX: ATR_TO_TP_SANITY_FACTOR — dynamic TP-სთან ადაპტირებული შემოწმება
    # ძველი: TP=3.0 × factor=0.10 = min_atr=0.30 → BNB atr=0.23 → BLOCKED_BY_EDGE
    # ლოგი: "ATR_BELOW_TP atr%=0.23 < TP_PCT*ATR_TO_TP_SANITY_FACTOR=0.30 (TP_PCT=3.00 factor=0.10)"
    # პრობლემა: TP=3.0% (regime-based) მაღალია → sanity check ბლოკავს ნორმალურ ბაზარს
    # გამოსწორება: factor=0.10 → 0.06 — 15m flat ბაზარზე atr=0.20-0.35% რეალისტურია
    # min_atr = 3.0 × 0.06 = 0.18% — BNB atr=0.23 გაივლის
    min_atr_for_tp = assumed_gross_edge * ATR_TO_TP_SANITY_FACTOR
    if atr_pct < min_atr_for_tp:
        return False, (
            f"ATR_BELOW_TP atr%={atr_pct:.2f} < TP_PCT*ATR_TO_TP_SANITY_FACTOR={min_atr_for_tp:.2f} "
            f"(TP_PCT={assumed_gross_edge:.2f} factor={ATR_TO_TP_SANITY_FACTOR:.2f})"
        )

    return True, "OK"


def _cooldown_ok() -> bool:
    global _last_emit_ts
    return (time.time() - _last_emit_ts) >= COOLDOWN_SECONDS


def _emit(signal: Dict[str, Any], outbox_path: str) -> None:
    global _last_emit_ts
    append_signal(signal, outbox_path)
    _last_emit_ts = time.time()


def _get_outbox_path() -> str:
    return os.getenv("OUTBOX_PATH") or os.getenv("SIGNAL_OUTBOX_PATH") or "/var/data/signal_outbox.json"


def _notify_sl_event(symbol: str = "") -> None:
    """SL hit — DB-ში counter გაიზარდე (restart-safe).
    FIX I-8 FULL: per-symbol isolation — BTC SL → ETH-ს ვეღარ ბლოკავს.
    """
    # per-symbol (ახალი, primary)
    if symbol:
        new_count_sym = increment_consecutive_sl_per_symbol(
            symbol, pause_seconds=SL_COOLDOWN_PAUSE
        )
        logger.info(
            f"[SL_TRACK_SYM] {symbol} | consecutive_sl={new_count_sym} "
            f"limit={SL_COOLDOWN_COUNT} (DB-saved)"
        )

    # global (backward compat — signal_generator-ის _sl_pause_active() კვლავ global-ს კითხულობს)
    new_count = increment_consecutive_sl(pause_seconds=SL_COOLDOWN_PAUSE)
    logger.info(f"[SL_TRACK] consecutive_sl={new_count} limit={SL_COOLDOWN_COUNT} (DB-saved)")
    if new_count >= SL_COOLDOWN_COUNT:
        logger.warning(
            f"[SL_COOLDOWN] {new_count} consecutive SL → PAUSE {SL_COOLDOWN_PAUSE//60} min "
            f"(saved to DB — restart-safe)"
        )


def _notify_tp_event(symbol: str = "") -> None:
    """TP hit — DB-ში counter reset (restart-safe).
    FIX I-8 FULL: per-symbol reset — მხოლოდ ამ symbol-ის counter ნულდება.
    """
    # per-symbol reset (ახალი, primary)
    if symbol:
        reset_consecutive_sl_per_symbol(symbol)

    # global reset (backward compat)
    state = get_sl_cooldown_state()
    if state["consecutive_sl"] > 0:
        logger.info(f"[SL_TRACK] TP hit → reset consecutive_sl {state['consecutive_sl']}→0 (DB)")
    reset_consecutive_sl()


def _sl_pause_active() -> bool:
    """DB-დან წაიკითხავს — restart-ზეც სწორია. (global — backward compat)"""
    return is_sl_pause_active()


def _sl_pause_active_for_symbol(symbol: str) -> bool:
    """FIX I-8 FULL: symbol-specific pause check — global-ის ნაცვლად.
    True თუ ამ კონკრეტული სიმბოლოს SL პაუზა აქტიურია.
    """
    if not symbol:
        return is_sl_pause_active()  # fallback global
    return is_sl_pause_active_per_symbol(symbol)


def _trades_today_count() -> int:
    """დღეს (UTC) დახურული + ღია trade-ების რაოდენობა DB-დან.
    FIX C-5: ღია (open) trades-ც ითვლება, რომ MAX_TRADES_PER_DAY bypass
    შეუძლებელი იყოს — ადრე მხოლოდ closed trades ითვლებოდა.
    """
    try:
        from execution.db.repository import get_closed_trades, get_trade_stats
        from datetime import datetime, timezone
        today = datetime.now(timezone.utc).date().isoformat()

        # დახურული trade-ები დღეს
        trades = get_closed_trades()
        closed_today = sum(
            1 for t in trades
            if str(t.get("closed_at", "") or "")[:10] == today
        )

        # ღია trade-ები (ნებისმიერ დღეს გახსნილი, ჯერ კიდევ ღია)
        stats = get_trade_stats()
        open_trades = int(stats.get("open_trades", 0))

        return closed_today + open_trades
    except Exception:
        return 0


def _trades_last_hour_count() -> int:
    """ბოლო 60 წუთში დახურული trade-ების რაოდენობა DB-დან."""
    try:
        from execution.db.repository import get_closed_trades
        trades = get_closed_trades()
        from datetime import datetime, timezone, timedelta
        cutoff = datetime.now(timezone.utc) - timedelta(hours=1)
        count = 0
        for t in trades:
            closed_at = t.get("closed_at")
            if not closed_at:
                continue
            try:
                dt = datetime.fromisoformat(str(closed_at).replace("Z", "+00:00"))
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                if dt >= cutoff:
                    count += 1
            except Exception:
                pass
        return count
    except Exception:
        return 0


def _recovery_ok(ohlcv: List[List[float]]) -> Tuple[bool, str]:
    """
    Recovery პირობები პაუზის შემდეგ:
      1. ბოლო RECOVERY_CANDLES (3) სანთლიდან მინ. 2 მწვანეა (close > open)
         FIX: ძველი ლოგიკა ALL 3 green — ძალიან მკაცრი 15m flat ბაზრისთვის
         ახალი: >= ceil(RECOVERY_CANDLES * 2/3) — უმრავლესობა green
      2. ბოლო სანთელი >= RECOVERY_CANDLE_PCT ზომისაა
    """
    if len(ohlcv) < RECOVERY_CANDLES + 1:
        return False, f"not_enough_candles need={RECOVERY_CANDLES+1}"

    candles = ohlcv[-(RECOVERY_CANDLES):]

    green_count = 0
    for c in candles:
        o = float(c[1])  # open
        cl = float(c[4]) # close
        if cl > o:
            green_count += 1

    # FIX: majority green (2 out of 3) instead of all 3
    import math
    min_green = math.ceil(RECOVERY_CANDLES * 2 / 3)
    green_ok = green_count >= min_green

    # ბოლო სანთლის ზომა
    last_c = ohlcv[-1]
    last_open  = float(last_c[1])
    last_close = float(last_c[4])
    if last_open <= 0:
        return False, "last_candle_open_zero"
    last_candle_pct = abs((last_close - last_open) / last_open) * 100.0
    size_ok = last_candle_pct >= RECOVERY_CANDLE_PCT

    ok = green_ok and size_ok
    reason = (
        f"green={green_count}/{RECOVERY_CANDLES} (need>={min_green}) "
        f"last_candle_pct={last_candle_pct:.3f}% >= {RECOVERY_CANDLE_PCT}%={int(size_ok)}"
    )
    return ok, reason


def _tf_seconds(tf: str) -> int:
    tf = (tf or "").strip().lower()
    try:
        if tf.endswith("m"):
            return max(1, int(tf[:-1])) * 60
        if tf.endswith("h"):
            return max(1, int(tf[:-1])) * 3600
        if tf.endswith("d"):
            return max(1, int(tf[:-1])) * 86400
    except Exception:
        pass
    return 900


def _drop_unclosed_candle(ohlcv: List[List[float]], timeframe: str) -> Tuple[List[List[float]], bool]:
    if not ohlcv:
        return ohlcv, False
    last_ts_ms = int(ohlcv[-1][0])
    now_ms = int(time.time() * 1000)
    tf_ms = _tf_seconds(timeframe) * 1000
    if now_ms - last_ts_ms < tf_ms:
        return ohlcv[:-1], True
    return ohlcv, False


# -----------------------------
# EXCHANGE BUILDER
# -----------------------------
def _build_exchange() -> ccxt.Exchange:
    ex_name = os.getenv("EXCHANGE", "binance").strip().lower()
    market_type = os.getenv("MARKET_TYPE", "spot").strip().lower()

    if ex_name == "bybit":
        api_key = os.getenv("BYBIT_API_KEY", "").strip()
        api_secret = os.getenv("BYBIT_API_SECRET", "").strip()
        return ccxt.bybit({
            "enableRateLimit": True,
            "apiKey": api_key,
            "secret": api_secret,
            "options": {"defaultType": market_type},
        })

    api_key = os.getenv("BINANCE_API_KEY", "").strip()
    api_secret = os.getenv("BINANCE_API_SECRET", "").strip()
    return ccxt.binance({
        "enableRateLimit": True,
        "apiKey": api_key,
        "secret": api_secret,
        "options": {"defaultType": market_type},
    })


EXCHANGE = _build_exchange()


# -----------------------------
# FEATURE CALCS
# -----------------------------
def _momentum(closes: List[float], n: int) -> float:
    if len(closes) < n + 1:
        return 0.0
    base = closes[-1 - n]
    if base == 0:
        return 0.0
    return (closes[-1] / base) - 1.0


def _slope_sma(closes: List[float]) -> float:
    if len(closes) < 10:
        return 0.0
    s5 = _sma(closes, 5)
    s10 = _sma(closes, 10)
    if s10 == 0:
        return 0.0
    return (s5 / s10) - 1.0


def _ups_count(closes: List[float], n: int) -> int:
    if len(closes) < n + 1:
        return 0
    ups = 0
    for i in range(-n, 0):
        if closes[i] > closes[i - 1]:
            ups += 1
    return ups


def _trend_strength(closes: List[float], use_ma: bool) -> float:
    if len(closes) < 20:
        return 0.0

    last = closes[-1]
    prev = closes[-2]
    mom1 = _momentum(closes, 1)
    slope = _slope_sma(closes)
    ups3 = _ups_count(closes, 3)

    base = 0.0
    base += 0.35 * (1.0 if last > prev else 0.0)
    base += 0.25 * max(0.0, min(1.0, (mom1 / 0.003)))
    base += 0.20 * max(0.0, min(1.0, (slope / 0.003)))
    base += 0.20 * (ups3 / 3.0)

    if use_ma:
        ma20 = _sma(closes, 20)
        gap_pct = _pct(last, ma20)
        base += 0.15 * max(0.0, min(1.0, gap_pct / 0.6))

    return max(0.0, min(1.0, base))


def _structure_ok(closes: List[float], use_ma: bool, trend_strength: float) -> Tuple[bool, str]:
    if len(closes) < 20:
        return False, "len<20"

    last = closes[-1]
    prev = closes[-2]
    s5 = _sma(closes, 5)
    s10 = _sma(closes, 10)
    ups3 = _ups_count(closes, 3)
    mom10 = _momentum(closes, 10)

    c_last_prev = last > prev
    c_sma = s5 > s10
    c_ups = ups3 >= 2
    c_mom10 = mom10 > -0.002

    if use_ma:
        ma20 = _sma(closes, 20)
        c_ma = last > ma20
        ok = c_last_prev and c_sma and c_ups and c_ma and c_mom10
        reason = (
            f"last>prev={int(c_last_prev)} sma5>sma10={int(c_sma)} ups3>=2={int(c_ups)} "
            f"last>ma20={int(c_ma)} mom10_ok={int(c_mom10)}"
        )
        return ok, reason

    # strict no-MA mode
    strict_ok = c_last_prev and c_sma and c_ups and c_mom10
    if strict_ok:
        reason = (
            f"strict last>prev={int(c_last_prev)} sma5>sma10={int(c_sma)} "
            f"ups3>=2={int(c_ups)} mom10_ok={int(c_mom10)}"
        )
        return True, reason

    # soft no-MA override
    if STRUCT_SOFT_OVERRIDE:
        sma_gap_pct = _pct(s5, s10)  # can be slightly negative
        c_soft_trend = trend_strength >= STRUCT_SOFT_MIN_TREND
        c_soft_last = c_last_prev if STRUCT_SOFT_REQUIRE_LAST_UP > 0 else True
        c_soft_ups = ups3 >= STRUCT_SOFT_REQUIRE_LAST_UP
        c_soft_sma_gap = sma_gap_pct >= (-1.0 * STRUCT_SOFT_MIN_MA_GAP)
        c_soft_mom10 = mom10 > -0.004

        soft_ok = c_soft_trend and c_soft_last and c_soft_ups and c_soft_sma_gap and c_soft_mom10
        reason = (
            f"soft strict=0 last>prev={int(c_last_prev)} sma5>sma10={int(c_sma)} "
            f"ups3={ups3} mom10={mom10:.6f} trend={trend_strength:.3f} sma_gap%={sma_gap_pct:.3f} "
            f"soft_trend_ok={int(c_soft_trend)} soft_last_ok={int(c_soft_last)} "
            f"soft_ups_ok={int(c_soft_ups)} soft_sma_gap_ok={int(c_soft_sma_gap)} "
            f"soft_mom10_ok={int(c_soft_mom10)}"
        )
        return soft_ok, reason

    reason = (
        f"strict last>prev={int(c_last_prev)} sma5>sma10={int(c_sma)} "
        f"ups3>=2={int(c_ups)} mom10_ok={int(c_mom10)}"
    )
    return False, reason


def _volume_score(vols: List[float]) -> Tuple[float, float]:
    if len(vols) < 20:
        return 0.0, 0.0
    v_last = vols[-1]
    v_avg = sum(vols[-20:]) / 20.0
    if v_avg <= 0:
        return 0.0, 0.0
    v_ratio = v_last / v_avg
    score = max(0.0, min(1.0, v_ratio))
    return score, v_ratio


def _confidence_score(closes: List[float], ohlcv: List[List[float]], use_ma: bool) -> float:
    if len(closes) < 20 or len(ohlcv) < 20:
        return 0.0

    last = closes[-1]
    prev = closes[-2]
    atrp = _atr_pct(ohlcv, 14)
    slope = _slope_sma(closes)

    cond_last_prev = 1.0 if last > prev else 0.0
    cond_atr = 1.0 if atrp < 2.0 else 0.0
    cond_slope = max(0.0, min(1.0, slope / 0.003))

    if use_ma:
        ma20 = _sma(closes, 20)
        cond_ma = 1.0 if last > ma20 else 0.0
        raw = (0.35 * cond_ma) + (0.35 * cond_last_prev) + (0.20 * cond_slope) + (0.10 * cond_atr)
    else:
        raw = (0.45 * cond_last_prev) + (0.35 * cond_slope) + (0.20 * cond_atr)

    # AI_CONFIDENCE_BOOST — ENV-ით კონფიგურირებადი score multiplier (default=1.0, ENV=1.15)
    # _clamp 1.0-ზე: boost ამაღლებს score-ს, მაგრამ 1.0-ს ვერ გადააჭარბებს
    boosted = min(1.0, raw * AI_CONFIDENCE_BOOST)
    if GEN_DEBUG and AI_CONFIDENCE_BOOST != 1.0:
        logger.debug(
            f"[CONF_BOOST] raw={raw:.3f} × {AI_CONFIDENCE_BOOST} = {boosted:.3f}"
        )
    return boosted


def _risk_state(vol_regime: str, ai_score: float) -> str:
    if vol_regime == "EXTREME":
        return "KILL"
    if ai_score < 0.35:
        return "REDUCE"
    return "OK"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# #1 RSI
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _rsi(closes: List[float], n: int = 14) -> float:
    """Wilder RSI — სტანდარტული implementation."""
    if len(closes) < n + 1:
        return 50.0  # neutral — საკმარისი data არ არის
    gains, losses = [], []
    for i in range(-n, 0):
        d = closes[i] - closes[i - 1]
        gains.append(max(d, 0.0))
        losses.append(max(-d, 0.0))
    avg_gain = sum(gains) / n
    avg_loss = sum(losses) / n
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# #1 MACD
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _ema(closes: List[float], n: int) -> List[float]:
    """Exponential Moving Average — სრული სერია."""
    if len(closes) < n:
        return [sum(closes) / len(closes)] * len(closes)
    k = 2.0 / (n + 1.0)
    result = [sum(closes[:n]) / n]
    for price in closes[n:]:
        result.append(price * k + result[-1] * (1.0 - k))
    # pad front with first value to match closes length
    pad = len(closes) - len(result)
    return [result[0]] * pad + result


def _macd(closes: List[float], fast: int = 12, slow: int = 26, signal: int = 9
          ) -> Tuple[float, float, float]:
    """
    Returns (macd_line, signal_line, histogram) for the LAST candle.
    macd > signal_line AND histogram > 0 → bullish crossover
    """
    if len(closes) < slow + signal:
        return 0.0, 0.0, 0.0
    ema_fast = _ema(closes, fast)
    ema_slow = _ema(closes, slow)
    # align lengths
    min_len = min(len(ema_fast), len(ema_slow))
    macd_series = [ema_fast[i] - ema_slow[i] for i in range(-min_len, 0)]
    if len(macd_series) < signal:
        return 0.0, 0.0, 0.0
    sig_ema = _ema(macd_series, signal)
    macd_val   = macd_series[-1]
    signal_val = sig_ema[-1]
    hist       = macd_val - signal_val
    return round(macd_val, 8), round(signal_val, 8), round(hist, 8)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# #2 Multi-Timeframe — higher timeframe trend check
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _mtf_trend_ok(symbol: str) -> Tuple[bool, str, str]:
    """
    1h (MTF_TIMEFRAME) trend-ი BULL-ია?
    სწრაფი check: EMA20 > EMA50 AND last > EMA20

    Returns (ok, reason, htf_regime)
      htf_regime — regime_engine-ისთვის: "BULL" | "UNCERTAIN" | "BEAR" | None
      None → data ნაკლებია ან fetch error (caller-ი None-ს გადასცემს apply()-ს)
    """
    try:
        ohlcv_h = EXCHANGE.fetch_ohlcv(symbol, timeframe=MTF_TIMEFRAME, limit=MTF_CANDLE_LIMIT)
        if not ohlcv_h or len(ohlcv_h) < 52:
            return True, "not_enough_data→skip", None  # data-ს ნაკლებობა → არ ვბლოკავთ
        ohlcv_h, _ = _drop_unclosed_candle(ohlcv_h, MTF_TIMEFRAME)
        if len(ohlcv_h) < 52:
            return True, "not_enough_data→skip", None
        closes_h = [float(c[4]) for c in ohlcv_h]
        ema20_h = _ema(closes_h, 20)
        ema50_h = _ema(closes_h, 50)
        last_h  = closes_h[-1]
        ok = (last_h > ema20_h[-1]) and (ema20_h[-1] > ema50_h[-1])

        # ── htf_regime: 1h closes-ზე trend + ATR გამოვთვალოთ regime_engine-სთვის ──
        trend_h = _trend_strength(closes_h, USE_MA_FILTERS)
        atrp_h  = _atr_pct(ohlcv_h, n=14)
        htf_regime = _regime().detect_regime(trend=trend_h, atr_pct=atrp_h)

        reason = (
            f"mtf={MTF_TIMEFRAME} last={last_h:.4f} "
            f"ema20={ema20_h[-1]:.4f} ema50={ema50_h[-1]:.4f} "
            f"htf_regime={htf_regime} ok={ok}"
        )
        return ok, reason, htf_regime
    except Exception as e:
        return True, f"mtf_fetch_err→skip: {e}", None  # fetch error → არ ვბლოკავთ


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# #4 Dynamic position sizing
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _dynamic_quote_size(ai_score: float, base: float) -> float:
    """
    ai_score → quote size:
      ai < AI_LOW  → DYNAMIC_SIZE_MIN
      ai > AI_HIGH → DYNAMIC_SIZE_MAX
      between      → linear interpolation
    """
    if not USE_DYNAMIC_SIZING:
        return base
    lo, hi = DYNAMIC_SIZE_AI_LOW, DYNAMIC_SIZE_AI_HIGH
    if ai_score <= lo:
        return DYNAMIC_SIZE_MIN
    if ai_score >= hi:
        return DYNAMIC_SIZE_MAX
    t = (ai_score - lo) / (hi - lo)
    size = DYNAMIC_SIZE_MIN + t * (DYNAMIC_SIZE_MAX - DYNAMIC_SIZE_MIN)
    return round(size, 2)


def generate_signal() -> Optional[Dict[str, Any]]:
    # BUGFIX: global _rsi_sell_fired ფუნქციის სათავეში — Python requirement
    global _rsi_sell_fired

    outbox_path = _get_outbox_path()

    # core singleton — ერთხელ იქმნება, ყველგან გამოიყენება
    core = _core()

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # GEN_TEST_SIGNAL — integration test one-shot signal
    # GEN_TEST_SIGNAL=true → emit one dummy HOLD signal და return
    # production-ზე false უნდა იყოს
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    if GEN_TEST_SIGNAL:
        test_sym = SYMBOLS[0] if SYMBOLS else "BTC/USDT"
        test_sig = {
            "signal_id":       str(uuid.uuid4()),
            "ts_utc":          _now_utc_iso(),
            "certified_signal": True,
            "final_verdict":   "HOLD",
            "trend":           0.0,
            "atr_pct":         0.0,
            "meta":            {"source": "GEN_TEST_SIGNAL", "symbol": test_sym},
            "execution":       {"symbol": test_sym, "direction": "LONG",
                                "entry": {"type": "MARKET"}, "quote_amount": 0},
        }
        logger.info(f"[GEN_TEST] test signal emitted | symbol={test_sym}")
        append_signal(test_sig, outbox_path)
        return test_sig

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # PROTECTIVE SELL — cooldown-ის გარეშე!
    # ეს ᲧᲝᲕᲔᲚᲗᲕᲘᲡ შემოწმდება — crash/KILL სიტუაციაში
    # cooldown არ ბლოკავს დაცვით SELL-ს
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    for symbol in SYMBOLS:
        active_oco = _has_active_oco(symbol)
        if not active_oco:
            continue

        try:
            ohlcv_quick = EXCHANGE.fetch_ohlcv(symbol, timeframe=TIMEFRAME, limit=30)
        except Exception:
            continue

        if not ohlcv_quick or len(ohlcv_quick) < 20:
            continue

        ohlcv_quick, _ = _drop_unclosed_candle(ohlcv_quick, TIMEFRAME)
        if len(ohlcv_quick) < 20:
            continue

        closes_q = [float(c[4]) for c in ohlcv_quick]
        vols_q   = [float(c[5]) for c in ohlcv_quick]
        atrp_q   = _atr_pct(ohlcv_quick, 14)
        vol_reg_q = _vol_regime(atrp_q)

        if vol_reg_q != "EXTREME":
            continue

        trend_q  = _trend_strength(closes_q, USE_MA_FILTERS)
        struct_q, _ = _structure_ok(closes_q, USE_MA_FILTERS, trend_q)
        vol_sc_q, _ = _volume_score(vols_q)
        conf_q   = _confidence_score(closes_q, ohlcv_quick, USE_MA_FILTERS)

        tmp_q = CoreInputs(
            trend_strength=trend_q, structure_ok=struct_q,
            volume_score=vol_sc_q, risk_state="OK",
            confidence_score=conf_q, volatility_regime=vol_reg_q,
        )
        ai_q = float(core.decide(tmp_q)["ai_score"])
        risk_q = _risk_state(vol_reg_q, ai_q)

        if risk_q == "KILL":
            signal_id = str(uuid.uuid4())
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # FIX #4a: apply() სწორი kwargs-ებით.
            # ძველი კოდი: _regime().apply(trend=, vol=, atr_pct=, ai_score=, base_quote=)
            # regime_engine.apply() signature: apply(regime, atr_pct, symbol, buy_time)
            # trend=/vol=/ai_score=/base_quote= სრულად იგნორდებოდა → atr_pct=0 ყოველთვის
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            regime_sell = _regime().detect_regime(trend=trend_q, atr_pct=atrp_q)
            adaptive_sell = _regime().apply(regime_sell, atr_pct=atrp_q, symbol=symbol)
            sig = {
                "signal_id": signal_id,
                "ts_utc": _now_utc_iso(),
                "certified_signal": True,
                "final_verdict": "SELL",
                # FIX #4b: top-level trend + atr_pct → main.py კითხულობს
                "trend":   round(trend_q, 4),
                "atr_pct": round(atrp_q, 4),
                "meta": {
                    "source": "PROTECTIVE_SELL",
                    "symbol": symbol,
                    "reason": "RISK_KILL_OVERRIDE",
                    "atr_pct": atrp_q,
                    "vol_regime": vol_reg_q,
                    "regime": adaptive_sell.get("REGIME", "VOLATILE"),
                },
                "execution": {
                    "symbol": symbol,
                    "direction": "LONG",
                    "entry": {"type": "MARKET"},
                }
            }
            logger.warning(
                f"[GEN] PROTECTIVE_SELL | symbol={symbol} "
                f"volReg={vol_reg_q} atr%={atrp_q:.2f} ai={ai_q:.3f} — COOLDOWN BYPASSED"
            )
            append_signal(sig, outbox_path)
            return sig

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # COOLDOWN — ჩვეულებრივი signal-ებისთვის
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    if not _cooldown_ok():
        return None

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # TRADE FREQUENCY LIMITS — MAX_TRADES_PER_DAY / HOUR
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    if MAX_TRADES_PER_DAY > 0:
        today_count = _trades_today_count()
        if today_count >= MAX_TRADES_PER_DAY:
            logger.info(
                f"[GEN] BLOCKED_DAILY_LIMIT | today={today_count} >= MAX_TRADES_PER_DAY={MAX_TRADES_PER_DAY}"
            )
            return None

    if MAX_TRADES_PER_HOUR > 0:
        hour_count = _trades_last_hour_count()
        if hour_count >= MAX_TRADES_PER_HOUR:
            logger.info(
                f"[GEN] BLOCKED_HOURLY_LIMIT | last_hour={hour_count} >= MAX_TRADES_PER_HOUR={MAX_TRADES_PER_HOUR}"
            )
            return None

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # SL PAUSE — DB-based, restart-safe
    # consecutive_sl და sl_pause_until DB-შია → deploy bypass შეუძლებელია
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # FIX I-8 FULL: global pause კვლავ ამოწმებს global limit-ს.
    # per-symbol pause → BUY loop-ში symbol-level-ზე ამოწმდება.
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    if _sl_pause_active():
        sl_state = get_sl_cooldown_state()
        pause_ts = sl_state.get("sl_pause_until") or 0.0
        remaining = max(0, int(pause_ts - time.time()))
        logger.info(
            f"[SL_COOLDOWN] GLOBAL PAUSED (DB) | remaining={remaining}s "
            f"({remaining//60}m{remaining%60}s) | "
            f"consecutive_sl={sl_state['consecutive_sl']}"
        )
        return None

    sl_state = get_sl_cooldown_state()
    # FIX: თუ პაუზა დროით გაიარა (is_sl_pause_active()=False) მაგრამ
    # consecutive_sl ჯერ კიდევ >= limit-ია DB-ში → recovery check საჭიროა.
    # თუ recovery_candles=3 და ბაზარი flat-ია (0.15-0.20% სანთლები) →
    # recovery NEVER passes და ბოტი დაბლოკილია indefinitely.
    # FIX: RECOVERY_CANDLE_PCT-ს ENV-ით გასამართავად default 0.10%-ზე
    # (0.25% ძალიან მაღალია 15m flat ბაზრისთვის).
    # ასევე: consecutive_sl DB-ში reset-ი recovery pass-ის შემდეგ სწორდება.
    if sl_state["consecutive_sl"] >= SL_COOLDOWN_COUNT:
        # პაუზა დასრულდა — recovery check
        recovery_passed = False
        for sym in SYMBOLS:
            try:
                ohlcv_r = EXCHANGE.fetch_ohlcv(sym, timeframe=TIMEFRAME, limit=RECOVERY_CANDLES + 5)
            except Exception:
                continue
            if not ohlcv_r or len(ohlcv_r) < RECOVERY_CANDLES + 1:
                continue
            ohlcv_r, _ = _drop_unclosed_candle(ohlcv_r, TIMEFRAME)
            rec_ok, rec_reason = _recovery_ok(ohlcv_r)
            logger.info(
                f"[SL_RECOVERY] symbol={sym} ok={rec_ok} reason={rec_reason} "
                f"consecutive_sl={sl_state['consecutive_sl']} (DB)"
            )
            if rec_ok:
                recovery_passed = True
                break

        if not recovery_passed:
            logger.info(
                f"[SL_RECOVERY] WAITING (DB) | consecutive_sl={sl_state['consecutive_sl']} "
                f"need={RECOVERY_CANDLES} green candles >= {RECOVERY_CANDLE_PCT}%"
            )
            return None
        else:
            logger.warning(
                f"[SL_RECOVERY] PASSED ✅ (DB) | "
                f"consecutive_sl={sl_state['consecutive_sl']}→0 | trading resumed"
            )
            reset_consecutive_sl()

    for symbol in SYMBOLS:
        active_oco = _has_active_oco(symbol)
        open_trade = _has_open_trade(symbol)

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # FIX I-8 FULL: per-symbol SL pause check.
        # BTC-ზე 2 SL → მხოლოდ BTC ბლოკდება, ETH/BNB კვლავ ვაჭრობს.
        # open_trade-ის შემთხვევაში: SELL-ი კვლავ მუშაობს (bypass).
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        if not open_trade and _sl_pause_active_for_symbol(symbol):
            sym_state = get_sl_cooldown_state_per_symbol(symbol)
            pause_ts  = sym_state.get("sl_pause_until") or 0.0
            remaining = max(0, int(pause_ts - time.time()))
            logger.info(
                f"[SL_COOLDOWN_SYM] {symbol} PAUSED | "
                f"remaining={remaining}s ({remaining//60}m{remaining%60}s) | "
                f"consecutive_sl={sym_state['consecutive_sl']}"
            )
            continue

        try:
            ohlcv = EXCHANGE.fetch_ohlcv(symbol, timeframe=TIMEFRAME, limit=CANDLE_LIMIT)
        except Exception as e:
            logger.exception(f"[GEN] FETCH_FAIL | symbol={symbol} tf={TIMEFRAME} err={e}")
            continue

        if not ohlcv or len(ohlcv) < 30:
            if GEN_LOG_EVERY_TICK:
                logger.info(
                    f"[GEN] NO_SIGNAL | symbol={symbol} reason=not_enough_candles got={len(ohlcv) if ohlcv else 0} need>=30"
                )
            continue

        ohlcv, dropped = _drop_unclosed_candle(ohlcv, TIMEFRAME)
        if len(ohlcv) < 30:
            continue

        closes = [float(c[4]) for c in ohlcv]
        vols = [float(c[5]) for c in ohlcv]

        last = closes[-1]
        prev = closes[-2]
        atrp = _atr_pct(ohlcv, 14)

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # MIN_VOLUME_24H filter — 24h volume-ი საკმარისია?
        # vols[-1] = ბოლო კანდელის volume. 24h≈96 candles(15m).
        # v24 = ბოლო 96 კანდელის ჯამი (proximate 24h volume)
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        if MIN_VOLUME_24H > 0:
            candles_per_day = 96  # 15m × 96 = 24h
            v24 = sum(vols[-candles_per_day:]) * last  # USDT-ში
            if v24 < MIN_VOLUME_24H:
                if GEN_DEBUG:
                    logger.info(
                        f"[GEN] BLOCKED_LOW_VOLUME | symbol={symbol} "
                        f"v24_usdt={v24:.0f} < MIN_VOLUME_24H={MIN_VOLUME_24H:.0f}"
                    )
                continue

        vol_reg = _vol_regime(atrp)

        trend = _trend_strength(closes, USE_MA_FILTERS)
        struct_ok, struct_reason = _structure_ok(closes, USE_MA_FILTERS, trend)
        vol_score, v_ratio = _volume_score(vols)
        conf = _confidence_score(closes, ohlcv, USE_MA_FILTERS)

        tmp_inp = CoreInputs(
            trend_strength=trend,
            structure_ok=struct_ok,
            volume_score=vol_score,
            risk_state="OK",
            confidence_score=conf,
            volatility_regime=vol_reg,
        )
        tmp_dec = core.decide(tmp_inp)
        ai_score = float(tmp_dec["ai_score"])
        risk = _risk_state(vol_reg, ai_score)

        inp = CoreInputs(
            trend_strength=trend,
            structure_ok=struct_ok,
            volume_score=vol_score,
            risk_state=risk,
            confidence_score=conf,
            volatility_regime=vol_reg,
        )
        decision = core.decide(inp)

        logger.info(f"[GEN] FINAL_DECISION={decision['final_trade_decision']}")

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # CRITICAL FIX: TREND REVERSAL SELL — decision check-ის წინ!
        # ძველი კოდი: SELL open_trade-ზე decision==EXECUTE-ის შემდეგ იყო
        # → flat ბაზარზე decision=STAND_BY → SELL UNREACHABLE (dead code)
        # ახლა: SELL ყველაზე პირველი შემოწმება — decision-ს არ ელოდება
        # cooldown-საც bypass-ავს (append_signal, არა _emit) — SELL არ ყოვნდება
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        mom1 = _momentum(closes, 1) if len(closes) > 1 else 0.0

        if open_trade:
            # SELL პირობები: trend < -0.15 AND mom1 < -0.01
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # #1 RSI SELL: RSI > RSI_SELL_MIN(60) + trend reversal → ადრე გასვლა
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            rsi_sell = _rsi(closes, RSI_PERIOD) if USE_RSI_FILTER else 50.0

            # FIX I-1: one-shot per open_trade — RSI spike = 1 SELL, არა spam
            rsi_cool_zone = RSI_SELL_MIN - 3
            if rsi_sell < rsi_cool_zone:
                _rsi_sell_fired[symbol] = False
            already_fired    = _rsi_sell_fired.get(symbol, False)
            rsi_sell_trigger = USE_RSI_FILTER and rsi_sell >= RSI_SELL_MIN and not already_fired

            sell_triggered = (trend < -0.15 and mom1 < -0.01) or rsi_sell_trigger

            if sell_triggered:
                signal_id = str(uuid.uuid4())
                sig = {
                    "signal_id": signal_id,
                    "ts_utc": _now_utc_iso(),
                    "certified_signal": True,
                    "final_verdict": "SELL",
                    "trend":   round(trend, 4),
                    "atr_pct": round(atrp, 4),
                    "meta": {
                        "source": "GEN_SIGNAL_SELL",
                        "symbol": symbol,
                        "reason": "RSI_OVERBOUGHT" if rsi_sell_trigger else "TREND_REVERSAL",
                        "trend": trend,
                        "mom1": mom1,
                        "rsi":  round(rsi_sell, 2) if USE_RSI_FILTER else None,
                        "ai_score": float(decision["ai_score"]),
                        "decision_was": decision["final_trade_decision"],
                    },
                    "execution": {
                        "symbol": symbol,
                        "direction": "LONG",
                    }
                }
                logger.info(
                    f"[GEN] TREND_REVERSAL_SELL | symbol={symbol} "
                    f"trend={trend:.3f} mom1={mom1:.4f} "
                    f"rsi={rsi_sell:.1f} rsi_trigger={rsi_sell_trigger} "
                    f"decision_was={decision['final_trade_decision']} — COOLDOWN BYPASSED"
                )
                # append_signal პირდაპირ — cooldown bypass (SELL არ ყოვნდება 60s)
                if rsi_sell_trigger:
                    _rsi_sell_fired[symbol] = True  # I-1: ერთხელ გაისვრა
                append_signal(sig, outbox_path)
                return sig

            # open trade-ია, მაგრამ SELL პირობა არ დასრულდა — BUY-ს ნუ ვცდილობთ
            continue

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # BUY PATH — მხოლოდ open_trade=False შემთხვევაში
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

        # AI_FILTER_LOW_CONFIDENCE — strict pre-filter (ყველა სხვა check-ის წინ)
        if AI_FILTER_LOW_CONFIDENCE:
            raw_ai = float(decision.get("ai_score", 0) or 0)
            if raw_ai < AI_FILTER_MIN_SCORE:
                if GEN_DEBUG:
                    logger.info(
                        f"[GEN] BLOCKED_AI_FILTER | symbol={symbol} "
                        f"ai={raw_ai:.3f} < min={AI_FILTER_MIN_SCORE:.3f}"
                    )
                continue

        # BUY_LIQUIDITY_MIN_SCORE — volume_score hard floor (0=off)
        if BUY_LIQUIDITY_MIN_SCORE > 0 and vol_score < BUY_LIQUIDITY_MIN_SCORE:
            if GEN_DEBUG:
                logger.info(
                    f"[GEN] BLOCKED_LIQUIDITY | symbol={symbol} "
                    f"vol_score={vol_score:.3f} < BUY_LIQUIDITY_MIN_SCORE={BUY_LIQUIDITY_MIN_SCORE:.3f}"
                )
            continue

        # 🚫 BUY BLOCKED — decision check
        if decision["final_trade_decision"] != "EXECUTE":
            logger.info(
                f"[GEN] BLOCKED_BY_CORE | symbol={symbol} "
                f"final={decision['final_trade_decision']} ai={decision['ai_score']:.3f}"
            )

            if GEN_DEBUG:
                logger.info(
                    f"[GEN] BLOCKED_BY_CORE | symbol={symbol} "
                    f"final={decision['final_trade_decision']} "
                    f"ai={decision['ai_score']:.3f} "
                    f"risk={risk} volReg={vol_reg} "
                    f"struct={struct_ok} conf={conf:.3f}"
                )

            continue

        if GEN_DEBUG:
            logger.info(
                f"[GEN] CORE_DECISION | symbol={symbol} ai={decision['ai_score']:.3f} "
                f"macro={decision['macro_gate']} strat={decision['active_strategy']} "
                f"final={decision['final_trade_decision']} risk={risk} "
                f"volReg={vol_reg} atr%={atrp:.2f} last={last:.6f} prev={prev:.6f} "
                f"dropped_last_candle={dropped} outbox={outbox_path}"
            )

            mom1_dbg = _momentum(closes, 1)
            mom10 = _momentum(closes, 10)
            slope = _slope_sma(closes)
            ups3 = _ups_count(closes, 3)
            v5 = sum(vols[-5:]) / 5.0 if len(vols) >= 5 else 0.0
            v20 = sum(vols[-20:]) / 20.0 if len(vols) >= 20 else 0.0
            s5 = _sma(closes, 5)
            s10 = _sma(closes, 10)

            if USE_MA_FILTERS:
                ma20 = _sma(closes, 20)
                ma_gap_abs = abs(_pct(last, ma20))
                logger.info(
                    f"[GEN] DIAG | symbol={symbol} trend={trend:.3f} conf={conf:.3f} struct={struct_ok} "
                    f"vol_score={vol_score:.3f} struct_reason={struct_reason} "
                    f"mom1={mom1_dbg:.6f} mom10={mom10:.6f} slope={slope:.6f} ups3={ups3} "
                    f"sma5={s5:.6f} sma10={s10:.6f} ma_gap%={ma_gap_abs:.3f} "
                    f"v5={v5:.3f} v20={v20:.3f} vRatio={v_ratio:.3f} use_ma={USE_MA_FILTERS}"
                )
            else:
                sma_gap_pct = _pct(s5, s10) if s10 else 0.0
                logger.info(
                    f"[GEN] DIAG | symbol={symbol} trend={trend:.3f} conf={conf:.3f} struct={struct_ok} "
                    f"vol_score={vol_score:.3f} struct_reason={struct_reason} "
                    f"mom1={mom1_dbg:.6f} mom10={mom10:.6f} slope={slope:.6f} ups3={ups3} "
                    f"sma5={s5:.6f} sma10={s10:.6f} sma_gap%={sma_gap_pct:.3f} "
                    f"v5={v5:.3f} v20={v20:.3f} vRatio={v_ratio:.3f} use_ma={USE_MA_FILTERS}"
                )

        # -----------------------------
        # EXTRA LIVE GUARDS
        # -----------------------------
        if USE_MA_FILTERS:
            ma20 = _sma(closes, 20)
            ma_gap_abs = abs(_pct(last, ma20))
            if ma_gap_abs < MA_GAP_PCT:
                if GEN_DEBUG:
                    logger.info(
                        f"[GEN] BLOCKED_BY_MA_GAP | symbol={symbol} gap%={ma_gap_abs:.3f} < MA_GAP_PCT={MA_GAP_PCT:.3f}"
                    )
                continue

        if conf < BUY_CONFIDENCE_MIN:
            if GEN_DEBUG:
                logger.info(
                    f"[GEN] BLOCKED_BY_CONF | symbol={symbol} conf={conf:.3f} < BUY_CONFIDENCE_MIN={BUY_CONFIDENCE_MIN:.3f}"
                )
            continue

        ok_edge, edge_reason = _edge_ok(atrp)
        if not ok_edge:
            if GEN_DEBUG:
                logger.info(f"[GEN] BLOCKED_BY_EDGE | symbol={symbol} reason={edge_reason}")
            continue

        if not ALLOW_LIVE_SIGNALS:
            if GEN_DEBUG:
                logger.info(f"[GEN] BLOCKED_BY_ENV | symbol={symbol} reason=ALLOW_LIVE_SIGNALS=false")
            continue

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # #1 RSI FILTER
        # RSI_MIN(35) <= rsi <= RSI_MAX(70) → BUY zone
        # 35-70: oversold recovery, არა overbought ზონა
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        if USE_RSI_FILTER:
            rsi_val = _rsi(closes, RSI_PERIOD)
            if not (RSI_MIN <= rsi_val <= RSI_MAX):
                if GEN_DEBUG:
                    logger.info(
                        f"[GEN] BLOCKED_BY_RSI | symbol={symbol} rsi={rsi_val:.1f} "
                        f"zone=[{RSI_MIN},{RSI_MAX}]"
                    )
                continue

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # #1 MACD FILTER
        # macd_line > signal_line AND histogram > 0 → bullish momentum
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        if USE_MACD_FILTER:
            macd_line, macd_sig, macd_hist = _macd(closes, MACD_FAST, MACD_SLOW, MACD_SIGNAL_PERIOD)
            if macd_hist <= 0 or macd_line <= macd_sig:
                if GEN_DEBUG:
                    logger.info(
                        f"[GEN] BLOCKED_BY_MACD | symbol={symbol} "
                        f"macd={macd_line:.6f} signal={macd_sig:.6f} hist={macd_hist:.6f}"
                    )
                continue

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # #2 MULTI-TIMEFRAME FILTER
        # 1h EMA20 > EMA50 AND last > EMA20 → higher TF BULL
        # FIX GAP-1: _mtf_trend_ok() ახლა htf_regime-საც აბრუნებს
        # → apply()-ს htf_regime= გადაეცემა → MTF TP bonus/penalty მუშაობს
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        htf_regime: Optional[str] = None
        if USE_MTF_FILTER:
            mtf_ok, mtf_reason, htf_regime = _mtf_trend_ok(symbol)
            if not mtf_ok:
                if GEN_DEBUG:
                    logger.info(
                        f"[GEN] BLOCKED_BY_MTF | symbol={symbol} reason={mtf_reason}"
                    )
                continue

        signal_id = str(uuid.uuid4())

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # MARKET REGIME — ATR-based dynamic TP/SL
        # FIX #4c: apply() სწორი kwargs-ებით.
        # FIX GAP-1: htf_regime= გადაეცემა → MTF bonus/penalty TP-ზე ამუშავდება
        #   STRONG (15m=BULL + 1h=BULL)      → TP × 1.20
        #   WEAK   (15m=BULL + 1h=UNCERTAIN) → TP × 0.85
        #   DIVERGE (1h=BEAR/VOLATILE)        → SKIP_TRADING
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        regime_name = _regime().detect_regime(trend=trend, atr_pct=atrp)
        adaptive = _regime().apply(
            regime_name,
            atr_pct=atrp,
            symbol=symbol,
            htf_regime=htf_regime,          # ← GAP-1 FIX: MTF bonus/penalty ჩართულია
            base_conf_min=BUY_CONFIDENCE_MIN,  # ← ეტაპი 2: adaptive conf threshold
            base_quote=BOT_QUOTE_PER_TRADE,    # ← ეტაპი 1: regime-aware sizing
        )

        # BEAR/VOLATILE/SIDEWAYS → trade ბლოკდება
        if adaptive.get("SKIP_TRADING"):
            if GEN_DEBUG:
                logger.info(
                    f"[GEN] BLOCKED_BY_REGIME | symbol={symbol} "
                    f"regime={adaptive.get('REGIME')} "
                    f"atr%={atrp:.3f} trend={trend:.3f}"
                )
            continue

        # QUOTE_SIZE: dynamic sizing (ai_score-based) ან static BOT_QUOTE_PER_TRADE
        quote_size = adaptive.get("QUOTE_SIZE", 1.0)
        if quote_size <= 0 or quote_size == 1.0:
            quote_size = BOT_QUOTE_PER_TRADE
        if quote_size <= 0:
            continue

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # #4 DYNAMIC POSITION SIZING — ai_score → quote size
        # ai_score=0.55 → DYNAMIC_SIZE_MIN=5 USDT
        # ai_score=0.80 → DYNAMIC_SIZE_MAX=15 USDT
        # between → linear interpolation
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        ai_for_sizing = float(decision["ai_score"])
        quote_size = _dynamic_quote_size(ai_for_sizing, quote_size)
        # exchange_client._guard() hard ceiling — LIVE_BLOCKED-ის თავიდან ასაცილებლად
        quote_size = min(quote_size, MAX_QUOTE_PER_TRADE)
        if quote_size <= 0:
            continue

        # RSI/MACD values for meta (if computed)
        rsi_meta  = round(_rsi(closes, RSI_PERIOD), 2) if USE_RSI_FILTER else None
        macd_meta = None
        if USE_MACD_FILTER:
            ml, ms, mh = _macd(closes, MACD_FAST, MACD_SLOW, MACD_SIGNAL_PERIOD)
            macd_meta = {"macd": round(ml, 6), "signal": round(ms, 6), "hist": round(mh, 6)}

        sig = {
            "signal_id": signal_id,
            "ts_utc": _now_utc_iso(),
            "certified_signal": True,
            "final_verdict": "TRADE",
            "trend":   round(trend, 4),
            "atr_pct": round(atrp, 4),
            "meta": {
                "source":  "DYZEN_EXCEL_LIVE_CORE",
                "symbol":  symbol,
                "decision": decision,
                "regime":  adaptive.get("REGIME"),
                "atr_pct": round(atrp, 4),
                "rsi":     rsi_meta,
                "macd":    macd_meta,
                "ai_score": ai_for_sizing,
                "mtf_tf":  MTF_TIMEFRAME if USE_MTF_FILTER else None,
                "mtf_alignment": adaptive.get("MTF_ALIGNMENT"),   # GAP-1: STRONG/WEAK/DIVERGE/N/A
                "mtf_confirmed": adaptive.get("MTF_CONFIRMED"),   # GAP-1: bool
            },
            "execution": {
                "symbol":       symbol,
                "direction":    "LONG",
                "entry":        {"type": "MARKET"},
                "quote_amount": quote_size,
            },
            "adaptive": {
                "TP_PCT":     adaptive["TP_PCT"],
                "SL_PCT":     adaptive["SL_PCT"],
                "REGIME":     adaptive["REGIME"],
                "ATR_PCT":    round(atrp, 4),
                "QUOTE_SIZE": quote_size,
                # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                # #3 Trailing Stop — execution_engine კითხულობს
                # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                "TRAILING_STOP_ENABLED":   TRAILING_STOP_ENABLED,
                "TRAILING_STOP_DISTANCE":  TRAILING_STOP_DISTANCE,
                # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                # #5 Partial TP — execution_engine კითხულობს
                # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                "USE_PARTIAL_TP":    USE_PARTIAL_TP,
                "PARTIAL_TP1_PCT":   PARTIAL_TP1_PCT,
                "PARTIAL_TP1_SIZE":  PARTIAL_TP1_SIZE,
                # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                # #7 Breakeven Stop — execution_engine კითხულობს
                # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                "USE_BREAKEVEN_STOP":    USE_BREAKEVEN_STOP,
                "BREAKEVEN_TRIGGER_PCT": BREAKEVEN_TRIGGER_PCT,
            },
        }

        _emit(sig, outbox_path)
        return sig

    return None


def run_once(*args, **kwargs) -> Optional[Dict[str, Any]]:
    return generate_signal()


def notify_outcome(outcome: str, symbol: str = "") -> None:
    """
    execution_engine.py-იდან გამოიძახება trade-ის დახურვის შემდეგ.
    outcome: 'SL' ან 'TP' ან 'MANUAL_SELL'
    symbol:  რომელი სიმბოლოს trade დაიხურა (e.g. 'BTC/USDT')

    FIX I-8 FULL: symbol გადაეცემა per-symbol DB isolation-ისთვის.
    BTC-ზე 2 SL → ETH trade-ებს ვეღარ ბლოკავს.
    """
    # BUGFIX: global ფუნქციის სათავეში — Python მოითხოვს ამას
    global _rsi_sell_fired

    outcome_upper = str(outcome).upper()
    sym_tag = f" | symbol={symbol}" if symbol else ""

    if outcome_upper == "SL":
        logger.info(f"[NOTIFY_OUTCOME] SL{sym_tag} → incrementing DB cooldown")
        _notify_sl_event(symbol=symbol)
        # FIX I-1: RSI sell flag reset — ახალი trade-ი = ახალი შანსი
        if symbol:
            _rsi_sell_fired[symbol] = False

    elif outcome_upper in ("TP", "MANUAL_SELL"):
        logger.info(f"[NOTIFY_OUTCOME] {outcome_upper}{sym_tag} → resetting DB cooldown")
        _notify_tp_event(symbol=symbol)
        # FIX I-1: RSI sell flag reset
        if symbol:
            _rsi_sell_fired[symbol] = False
