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
    # FIX GLOBAL-1: global open trade count for MAX_OPEN_TRADES guard
    get_all_open_trades,
)
from execution.excel_live_core import ExcelLiveCore, CoreInputs
from execution.regime_engine import MarketRegimeEngine

logger = logging.getLogger("gbm")

# -----------------------------
# ENV
# -----------------------------
TIMEFRAME = os.getenv("BOT_TIMEFRAME", "15m").strip()
CANDLE_LIMIT = int(os.getenv("BOT_CANDLE_LIMIT", "300"))        # ENV=300
COOLDOWN_SECONDS = int(os.getenv("BOT_SIGNAL_COOLDOWN_SECONDS", "120"))  # ENV=120s

ALLOW_LIVE_SIGNALS = os.getenv("ALLOW_LIVE_SIGNALS", "true").strip().lower() == "true"   # SYNC: false→true (config.py and ENV both true)

BOT_QUOTE_PER_TRADE = float(os.getenv("BOT_QUOTE_PER_TRADE", "10"))   # ENV=10
# MAX_QUOTE_PER_TRADE — exchange_client._guard() hard ceiling
# dynamic sizing ამ მნიშვნელობას ვერ გადააჭარბებს → LIVE_BLOCKED აღარ იქნება
MAX_QUOTE_PER_TRADE = float(os.getenv("MAX_QUOTE_PER_TRADE", "10"))   # ENV=10

# Fee-aware edge gate
MIN_MOVE_PCT = float(os.getenv("MIN_MOVE_PCT", "0.02"))  # DCA: 0.22→0.02 (BTC/BNB always passes)
ESTIMATED_ROUNDTRIP_FEE_PCT = float(os.getenv("ESTIMATED_ROUNDTRIP_FEE_PCT", "0.14"))  # ENV=0.14
ESTIMATED_SLIPPAGE_PCT = 0.0  # DCA: disabled
TP_PCT = float(os.getenv("TP_PCT", "1.5"))                                               # ENV=1.5%
MIN_NET_PROFIT_PCT = float(os.getenv("MIN_NET_PROFIT_PCT", "0.25"))                     # ENV=0.25

# ATR sanity
ATR_TO_TP_SANITY_FACTOR = float(os.getenv("ATR_TO_TP_SANITY_FACTOR", "0.07"))  # SYNC: 0.10→0.07 (LOG: min_atr=0.15% blocked BTC/BNB)

# Optional MA filters
USE_MA_FILTERS = os.getenv("USE_MA_FILTERS", "false").strip().lower() == "true"  # ENV=false
MA_GAP_PCT = float(os.getenv("MA_GAP_PCT", "0.15"))

# Extra confidence guard (after Excel decision)
# FIX WIN-5: BUY_CONFIDENCE_MIN 0.38→0.32
# Flat 15m market confidence_score ≈ 0.30-0.42 range.
# 0.38 ბლოკავდა real signals — ვხედავდით BLOCKED_BY_CONF_STATIC ხშირად.
# 0.32 = meaningful quality gate, still filters noise (score<0.25 = random)
BUY_CONFIDENCE_MIN = float(os.getenv("BUY_CONFIDENCE_MIN", "0.15"))  # DCA: 0.25→0.15 (more signals)

BLOCK_SIGNALS_WHEN_ACTIVE_OCO = os.getenv("BLOCK_SIGNALS_WHEN_ACTIVE_OCO", "true").strip().lower() == "true"

GEN_DEBUG = os.getenv("GEN_DEBUG", "true").strip().lower() == "true"
GEN_LOG_EVERY_TICK = os.getenv("GEN_LOG_EVERY_TICK", "true").strip().lower() == "true"

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DEAD PARAMS ACTIVATED — ადრე ENV-ში იყო, კოდი არ კითხულობდა
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# 1. Volume filter — 24h volume < MIN_VOLUME_24H → BUY skip
MIN_VOLUME_24H = float(os.getenv("MIN_VOLUME_24H", "5000000"))   # DCA: 30M→5M

# 2. Signal expiration — signal ts_utc-დან SIGNAL_EXPIRATION_SECONDS გასული → skip
SIGNAL_EXPIRATION_SECONDS = 0  # DCA: disabled

# 3. AI confidence boost — ai_score * AI_CONFIDENCE_BOOST (>1.0 ამაღლებს score-ს)
AI_CONFIDENCE_BOOST = float(os.getenv("AI_CONFIDENCE_BOOST", "1.05"))  # ENV=1.05

# 4. Trade frequency limits
MAX_TRADES_PER_DAY  = int(os.getenv("MAX_TRADES_PER_DAY",  "10"))  # ENV=10
MAX_TRADES_PER_HOUR = int(os.getenv("MAX_TRADES_PER_HOUR", "3"))   # ENV=3

# FIX GLOBAL-1: MAX_OPEN_TRADES — ადრე config.py-ში განსაზღვრული, მაგრამ
# სიგნალ-გენერატორში ᲐᲠᲐᲡᲝᲓᲔᲡ იმპორტირებული და გამოყენებული.
# ეს ნიშნავდა: MAX_OPEN_TRADES=4 ENV-ში → dead code → ბოტი 3 symbol-ზე
# ერთდროულად შედიოდა შეზღუდვის გარეშე.
MAX_OPEN_TRADES = int(os.getenv("MAX_OPEN_TRADES", "2"))  # ENV=2 (synced)

# 5. AI_FILTER_LOW_CONFIDENCE — ai_score < threshold → hard reject before any other check
# true = strict mode: ყველა low-confidence signal drop-ი ყველა filter-ის წინ
AI_FILTER_LOW_CONFIDENCE = os.getenv("AI_FILTER_LOW_CONFIDENCE", "false").strip().lower() == "true"  # DCA: keep false
AI_FILTER_MIN_SCORE      = float(os.getenv("BUY_CONFIDENCE_MIN", "0.46"))  # ENV=0.46 synced

# 6. GEN_TEST_SIGNAL — force-emit one test signal for integration testing (true = one shot)
GEN_TEST_SIGNAL = os.getenv("GEN_TEST_SIGNAL", "false").strip().lower() == "true"

# 7. BUY_LIQUIDITY_MIN_SCORE — volume_score minimum for BUY (0=off)
# volume_score < this → skip (stricter than soft-volume-override)
# FIX WIN-6: 0.40→0.25. Flat/night market volume_score ≈ 0.20-0.45.
# 0.40 blocked legitimate low-volatility entries. 0.25 = still filters dead volume.
BUY_LIQUIDITY_MIN_SCORE = float(os.getenv("BUY_LIQUIDITY_MIN_SCORE", "0"))  # DCA: off (BTC/BNB always liquid)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# #1 RSI + MACD filter
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
USE_RSI_FILTER        = os.getenv("USE_RSI_FILTER", "false").strip().lower() == "true"  # DCA: off
RSI_PERIOD            = int(os.getenv("RSI_PERIOD", "14"))
RSI_MIN               = float(os.getenv("RSI_MIN", "35"))
RSI_MAX               = float(os.getenv("RSI_MAX", "72"))          # ENV=72
# FIX WIN-1: RSI_SELL_MIN 72→58. 72 = RSI spike-ი ძალიან rare 15m-ზე flat ბაზარში.
# 58 = overbought territory on 15m — exits before momentum exhaustion.
# ადრე: trade-ები 15 SL-ზე დაიხურა, RSI 72-ს არასოდეს მიაღწია.
RSI_SELL_MIN          = float(os.getenv("RSI_SELL_MIN", "72"))     # ENV=72 (synced)

USE_MACD_FILTER       = os.getenv("USE_MACD_FILTER", "false").strip().lower() == "true"  # DCA: off
MACD_FAST             = int(os.getenv("MACD_FAST", "12"))
MACD_SLOW             = int(os.getenv("MACD_SLOW", "26"))
MACD_SIGNAL_PERIOD    = int(os.getenv("MACD_SIGNAL_PERIOD", "9"))

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# #2 Multi-timeframe confirmation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
USE_MTF_FILTER        = os.getenv("USE_MTF_FILTER", "false").strip().lower() == "true"  # DCA: off
MTF_TIMEFRAME         = os.getenv("MTF_TIMEFRAME", "1h").strip()
MTF_CANDLE_LIMIT      = int(os.getenv("MTF_CANDLE_LIMIT", "50"))

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ADX Filter — Average Directional Index trend strength
# USE_ADX_FILTER=true → trade only when ADX >= ADX_MIN_THRESHOLD
# FIX WIN-3: ADX_MIN_THRESHOLD 20→18.
# 15m flat/consolidating market ADX ≈ 15-22.
# 20 ბლოკავდა legit entries flat range breakouts-ზე.
# 18 = filters pure noise (ADX<15), allows range directional moves.
# ADX > 25 = strong trend (still best entries)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
USE_ADX_FILTER        = os.getenv("USE_ADX_FILTER", "false").strip().lower() == "true"  # DCA: off
ADX_MIN_THRESHOLD     = float(os.getenv("ADX_MIN_THRESHOLD", "23.0"))  # ENV=23 (synced)
ADX_PERIOD            = int(os.getenv("ADX_PERIOD", "14"))

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# VWAP Filter — Volume Weighted Average Price entry
# USE_VWAP_FILTER=true → buy only when price <= VWAP × (1 + tolerance)
# ყიდვა VWAP-ზე ქვემოთ ან ახლოს = value zone entry
# FIX WIN-4: VWAP_TOLERANCE 0.006→0.010
# BTC/ETH trend upmoves: ფასი ხშირად 0.6-1.0% above VWAP
# 0.6% ბლოკავდა momentum entries uptrend-ზე
# 1.0% = allows institutional-quality momentum entries
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
USE_VWAP_FILTER       = os.getenv("USE_VWAP_FILTER", "false").strip().lower() == "true"  # DCA: off
VWAP_TOLERANCE        = float(os.getenv("VWAP_TOLERANCE", "0.006"))   # ENV=0.006 (synced)
# VWAP_SESSION_BARS — რამდენი candle გამოიყენოს VWAP გამოთვლაში
# 96 = 24h (15m × 96). 0 = ყველა candle (ძველი ქცევა)
VWAP_SESSION_BARS     = int(os.getenv("VWAP_SESSION_BARS", "96"))      # ENV=96

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TIME-OF-DAY Filter — low liquidity session-ების თავიდან არიდება
# USE_TIME_FILTER=true → trade only during active hours (UTC)
# TRADE_HOUR_START=7, TRADE_HOUR_END=22 → 07:00-22:00 UTC
# (22:00-07:00 UTC = Asia low liquidity + wide spreads)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
USE_TIME_FILTER       = os.getenv("USE_TIME_FILTER", "false").strip().lower() == "true"  # DCA: off
TRADE_HOUR_START_UTC  = int(os.getenv("TRADE_HOUR_START_UTC", "8"))
TRADE_HOUR_END_UTC    = int(os.getenv("TRADE_HOUR_END_UTC", "3"))

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FUNDING RATE FILTER (Bybit Perpetual)
# Crypto-specific institutional signal:
#   funding > +THRESHOLD → longs overheated → avoid BUY
#   funding < -THRESHOLD → shorts overheated → contrarian BUY ok
# USE_FUNDING_FILTER=true → fetch from Bybit Perpetual API
# FUNDING_MAX_LONG_PCT=0.10 → block if funding > 0.10%
# FUNDING_CACHE_SEC=300 → cache 5min (funding updates every 8h)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
USE_FUNDING_FILTER    = os.getenv("USE_FUNDING_FILTER", "false").strip().lower() == "true"  # DCA: off
FUNDING_MAX_LONG_PCT  = float(os.getenv("FUNDING_MAX_LONG_PCT",  "0.10"))  # >0.10% = overbought
FUNDING_MIN_SHORT_PCT = float(os.getenv("FUNDING_MIN_SHORT_PCT", "-0.05")) # <-0.05% = oversold
FUNDING_CACHE_SEC     = int(os.getenv("FUNDING_CACHE_SEC", "300"))         # 5min cache

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MACD SMART MODE
# MACD_SMART_MODE=true → catches early reversals in downtrend
#   Standard:  hist > 0 (0% pass in downtrend)
#   Smart:     hist > 0 OR (hist improving 3 bars AND hist > -ATR×factor)
# MACD_IMPROVING_BARS=3  → need N consecutive improving bars
# MACD_HIST_ATR_FACTOR=0.3 → max negative hist = ATR × 0.3
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MACD_SMART_MODE       = os.getenv("MACD_SMART_MODE", "true").strip().lower() == "true"
# FIX WIN-7: MACD_IMPROVING_BARS 4→3.
# 4 bars on 15m = 60min confirmation — too slow for short moves.
# 3 bars = 45min momentum check — enough signal, faster entry.
MACD_IMPROVING_BARS   = int(os.getenv("MACD_IMPROVING_BARS", "4"))        # ENV=4 (synced)
MACD_HIST_ATR_FACTOR  = float(os.getenv("MACD_HIST_ATR_FACTOR", "0.2"))  # ENV=0.2

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# #3 Trailing Stop
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TRAILING_STOP_ENABLED   = False  # DCA: disabled
# FIX WIN-8: TRAILING_STOP_DISTANCE 0.25→0.35
# BTC/ETH 15m candle noise ≈ 0.20-0.30%. 0.25% trailing = noise trigger.
# 0.35% = beyond typical 15m noise, still locks in profits on real moves.
TRAILING_STOP_DISTANCE  = 0.25  # DCA: disabled (trailing off)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# #4 Dynamic position sizing
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
USE_DYNAMIC_SIZING    = os.getenv("USE_DYNAMIC_SIZING", "true").strip().lower() == "true"
DYNAMIC_SIZE_MIN      = 10.0  # DCA: fixed $10
DYNAMIC_SIZE_MAX      = 10.0  # DCA: fixed $10
DYNAMIC_SIZE_AI_LOW   = float(os.getenv("DYNAMIC_SIZE_AI_LOW",  "0.55"))
DYNAMIC_SIZE_AI_HIGH  = float(os.getenv("DYNAMIC_SIZE_AI_HIGH", "0.80"))

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# #5 Partial Take Profit
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
USE_PARTIAL_TP        = os.getenv("USE_PARTIAL_TP", "true").strip().lower() == "true"    # ENV=true
PARTIAL_TP1_PCT       = float(os.getenv("PARTIAL_TP1_PCT", "1.0"))                        # ENV=1.0%
PARTIAL_TP1_SIZE      = float(os.getenv("PARTIAL_TP1_SIZE", "0.5"))

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# #7 Breakeven Stop
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
USE_BREAKEVEN_STOP    = False  # DCA: disabled
# FIX WIN-9: BREAKEVEN_TRIGGER_PCT 0.40→0.30
# Activates breakeven protection when price is 0.30% above entry (was 0.40%).
# Earlier activation = more trades protected from reverting to SL loss.
# 0.30% is safely above 15m typical noise (0.20%) so minimal false triggers.
BREAKEVEN_TRIGGER_PCT = float(os.getenv("BREAKEVEN_TRIGGER_PCT", "0.48"))  # ENV=0.48 (synced)

# Soft structure override (USED ONLY WHEN USE_MA_FILTERS=false)
STRUCT_SOFT_OVERRIDE = os.getenv("STRUCT_SOFT_OVERRIDE", "false").strip().lower() == "true"  # DCA: off
STRUCT_SOFT_MIN_TREND = float(os.getenv("STRUCT_SOFT_MIN_TREND", "0.25"))        # ENV=0.25
STRUCT_SOFT_MIN_MA_GAP = float(os.getenv("STRUCT_SOFT_MIN_MA_GAP", "0.10"))      # ENV=0.10
STRUCT_SOFT_REQUIRE_LAST_UP = int(os.getenv("STRUCT_SOFT_REQUIRE_LAST_UP", "1")) # ENV=1

_last_emit_ts: float = 0.0

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FIX I-1: per-symbol RSI SELL one-shot flag
# RSI >= RSI_SELL_MIN → SELL emit ხდება ერთხელ per open_trade.
# flag ნულდება RSI-ის დაცემისას ან trade close-ზე.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
_rsi_sell_fired: dict = {}       # {symbol: bool}
_after_hours_sell_ts: dict = {}   # {symbol: float} — last emit timestamp (cooldown)
_protective_sell_ts: dict  = {}   # {symbol: float} — protective sell cooldown

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SL COOLDOWN — 2 SL-ის შემდეგ 30 წუთი პაუზა
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SELL_TREND_THRESHOLD — trend რამდენ დაბლა უნდა ჩავარდეს SELL-ის ტრიგერისთვის
# -0.05 = სუსტი downtrend (default). -0.03 = უფრო მგრძნობიარე, -0.10 = გვიანი გასვლა
SELL_TREND_THRESHOLD = float(os.getenv("SELL_TREND_THRESHOLD", "-0.05"))  # ENV=-0.05

SL_COOLDOWN_COUNT   = int(os.getenv("SL_COOLDOWN_AFTER_N", "99"))      # DCA: გათიშული
SL_COOLDOWN_PAUSE   = int(os.getenv("SL_COOLDOWN_PAUSE_SECONDS", "1800"))
RECOVERY_CANDLES    = int(os.getenv("RECOVERY_GREEN_CANDLES", "3"))
# FIX: 0.25% → 0.10% default. 15m flat ბაზარზე სანთლები 0.15-0.35%-ია.
# 0.25% ძალიან მაღალია → recovery 30+ წუთი არ გადის.
# ENV-ში: RECOVERY_CANDLE_PCT=0.15 (ან 0.10 flat ბაზრისთვის)
RECOVERY_CANDLE_PCT = float(os.getenv("RECOVERY_CANDLE_PCT", "0.05"))  # ENV=0.05 (synced)

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
    """
    Edge quality gate — სამი შემოწმება:
      1. Net profit: TP - fees - slippage >= MIN_NET_PROFIT_PCT
      2. ATR sanity: atr >= TP × ATR_TO_TP_SANITY_FACTOR (dynamic TP-სთან)
      3. MIN_MOVE: atr >= MIN_MOVE_PCT (flat ბაზრის გამოდევნა)
    
    ENV values:
      TP_PCT=1.5, FEE=0.14, SLIP=0.05 → net=1.31 >= MIN_NET=0.25 ✓
      ATR_TO_TP_SANITY_FACTOR=0.07 → min_atr=0.105% (BTC/ETH/BNB ყოველთვის გაივლის)
      MIN_MOVE_PCT=0.12 (LOG: ETH atr=0.17% was blocked at 0.20)
    """
    assumed_gross_edge = TP_PCT
    assumed_cost = ESTIMATED_ROUNDTRIP_FEE_PCT + ESTIMATED_SLIPPAGE_PCT
    assumed_net = assumed_gross_edge - assumed_cost

    # Check 1: Net profit gate
    if assumed_net < MIN_NET_PROFIT_PCT:
        return False, (
            "EDGE_TOO_SMALL "
            f"TP_PCT={assumed_gross_edge:.2f} cost={assumed_cost:.2f} net={assumed_net:.2f} "
            f"< MIN_NET_PROFIT_PCT={MIN_NET_PROFIT_PCT:.2f}"
        )

    # Check 2: ATR sanity vs TP — primary volatility guard
    # ENV=0.07 → min_atr=1.5×0.07=0.105% — BTC/ETH/BNB ყოველთვის გაივლის
    min_atr_for_tp = assumed_gross_edge * ATR_TO_TP_SANITY_FACTOR
    if atr_pct < min_atr_for_tp:
        return False, (
            f"ATR_BELOW_TP atr%={atr_pct:.2f} < TP×factor={min_atr_for_tp:.2f} "
            f"(TP_PCT={assumed_gross_edge:.2f} factor={ATR_TO_TP_SANITY_FACTOR:.2f})"
        )

    # Check 3: MIN_MOVE_PCT — absolute floor (ENV=0.12)
    # SYNC: 0.20→0.12. LOG: ETH atr=0.17% was blocked at 0.20%
    # USE_TIME_FILTER=true (07:00-22:00 UTC) already excludes dead hours
    if atr_pct < MIN_MOVE_PCT:
        return False, f"ATR_TOO_LOW atr%={atr_pct:.2f} < MIN_MOVE_PCT={MIN_MOVE_PCT:.2f}"

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
    api_key    = os.getenv("BINANCE_API_KEY",    "").strip()
    api_secret = os.getenv("BINANCE_API_SECRET", "").strip()
    ex = ccxt.binance({
        "enableRateLimit": True,
        "apiKey":  api_key,
        "secret":  api_secret,
        "options": {
            "defaultType":      "spot",
            "fetchCurrencies":  False,
            "fetchTradingFees": False,
        },
    })
    # load_markets ერთხელ — cache-დება.
    try:
        ex.load_markets()
        logger.info("[GEN] EXCHANGE_MARKETS_CACHED | OK")
    except Exception as e:
        logger.warning(f"[GEN] EXCHANGE_MARKETS_WARN | err={e}")
    return ex


EXCHANGE = _build_exchange()


def _fetch_ohlcv_direct(symbol: str, timeframe: str, limit: int) -> list:
    """
    Binance Klines — პირდაპირი REST call, ccxt load_markets() bypass.
    ccxt.fetch_ohlcv() ყოველ call-ზე load_markets()-ს ამოწმებს → rate limit.
    ეს ფუნქცია პირდაპირ https://api.binance.com/api/v3/klines-ს იძახებს.
    """
    import urllib.request
    import json as _json

    TF_MAP = {
        "1m": "1m", "3m": "3m", "5m": "5m", "15m": "15m",
        "30m": "30m", "1h": "1h", "2h": "2h", "4h": "4h",
        "6h": "6h", "8h": "8h", "12h": "12h", "1d": "1d",
    }
    tf = TF_MAP.get(timeframe, "15m")
    sym = symbol.replace("/", "")  # BTC/USDT → BTCUSDT
    url = f"https://api.binance.com/api/v3/klines?symbol={sym}&interval={tf}&limit={limit}"
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            raw = _json.loads(r.read())
        # Binance კანდელი: [open_time, open, high, low, close, volume, ...]
        return [[c[0], float(c[1]), float(c[2]), float(c[3]), float(c[4]), float(c[5])] for c in raw]
    except Exception as e:
        logger.warning(f"[GEN] FETCH_OHLCV_DIRECT_FAIL | {symbol} err={e} → fallback ccxt")
        return EXCHANGE.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)

# Exchange minimum qty per symbol (spot) — cache to avoid repeated API calls
_market_info_cache: dict = {}  # {symbol: {min_qty, min_notional, qty_step}}


def _get_market_limits(symbol: str) -> dict:
    """
    Exchange spot minimum order constraints.
    Returns: {min_qty: float, min_notional: float, qty_step: float}
    Cached per session — market info changes rarely.
    """
    if symbol in _market_info_cache:
        return _market_info_cache[symbol]
    try:
        mkts = EXCHANGE.load_markets()
        m = mkts.get(symbol, {})
        lim = m.get("limits", {})
        prec = m.get("precision", {})
        result = {
            "min_qty":      float(lim.get("amount", {}).get("min") or 0.0),
            "min_notional": float(lim.get("cost",   {}).get("min") or 10.0),
            "qty_step":     float(prec.get("amount") or 0.0001),
        }
        _market_info_cache[symbol] = result
        return result
    except Exception as _e:
        logger.debug(f"[GEN] MARKET_LIMITS_FAIL | symbol={symbol} err={_e}")
        # safe defaults: ETH=0.0001, BNB=0.001 qty, $10 notional
        default = {"min_qty": 0.0001, "min_notional": 10.0, "qty_step": 0.0001}
        _market_info_cache[symbol] = default
        return default


def _is_sellable_qty(symbol: str, qty: float, price: float) -> tuple:
    """
    True თუ qty Bybit minimum-ს აკმაყოფილებს.
    Returns (ok: bool, reason: str)
    """
    if qty <= 0:
        return False, f"qty={qty} <= 0"
    lim = _get_market_limits(symbol)
    if qty < lim["min_qty"]:
        return False, (
            f"qty={qty:.8f} < min_qty={lim['min_qty']} "
            f"(Bybit {symbol} minimum amount precision)"
        )
    notional = qty * price
    if notional < lim["min_notional"]:
        return False, (
            f"notional={notional:.4f} USDT < min_notional={lim['min_notional']} "
            f"(qty={qty:.8f} × price={price:.4f})"
        )
    return True, "OK"


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


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ADX — Average Directional Index (trend strength 0..100)
# ADX > 25 → ძლიერი trend (trade OK)
# ADX < 20 → sideways (false signals high)
# institutional standard: Renaissance, Two Sigma გამოიყენებს ADX
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _adx(ohlcv: List[List[float]], n: int = 14) -> float:
    """
    Returns ADX value (0..100).
    >25 = strong trend, <20 = sideways/weak.
    """
    if len(ohlcv) < n * 2 + 1:
        return 0.0
    try:
        highs  = [float(c[2]) for c in ohlcv]
        lows   = [float(c[3]) for c in ohlcv]
        closes = [float(c[4]) for c in ohlcv]

        # True Range + Directional Movement
        tr_list, pdm_list, ndm_list = [], [], []
        for i in range(1, len(ohlcv)):
            h, l, pc = highs[i], lows[i], closes[i-1]
            tr = max(h - l, abs(h - pc), abs(l - pc))
            pdm = max(highs[i] - highs[i-1], 0) if (highs[i] - highs[i-1]) > (lows[i-1] - lows[i]) else 0
            ndm = max(lows[i-1] - lows[i], 0) if (lows[i-1] - lows[i]) > (highs[i] - highs[i-1]) else 0
            tr_list.append(tr)
            pdm_list.append(pdm)
            ndm_list.append(ndm)

        # Wilder smoothing
        def _wilder(vals, period):
            result = [sum(vals[:period]) / period]
            for v in vals[period:]:
                result.append((result[-1] * (period - 1) + v) / period)
            return result

        atr_w  = _wilder(tr_list, n)
        pdm_w  = _wilder(pdm_list, n)
        ndm_w  = _wilder(ndm_list, n)

        dx_list = []
        for a, p, nd in zip(atr_w, pdm_w, ndm_w):
            if a == 0:
                dx_list.append(0.0)
                continue
            pdi = (p / a) * 100
            ndi = (nd / a) * 100
            denom = pdi + ndi
            dx_list.append(abs(pdi - ndi) / denom * 100 if denom > 0 else 0.0)

        if len(dx_list) < n:
            return 0.0
        adx = sum(dx_list[-n:]) / n
        return round(adx, 2)
    except Exception:
        return 0.0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# VWAP — Volume Weighted Average Price
# institutional entry rule: buy when price < VWAP (value zone)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _vwap(ohlcv: List[List[float]]) -> float:
    """
    Returns VWAP for given candles.
    Typical price = (high + low + close) / 3
    VWAP = Σ(typical × volume) / Σ(volume)
    VWAP_SESSION_BARS=96 → მხოლოდ ბოლო 96 candle (24h). 0 = ყველა.
    """
    if len(ohlcv) < 2:
        return 0.0
    try:
        # session bars slicing — 96=24h, 0=ყველა candle
        if VWAP_SESSION_BARS > 0 and len(ohlcv) > VWAP_SESSION_BARS:
            candles = ohlcv[-VWAP_SESSION_BARS:]
        else:
            candles = ohlcv
        cum_tp_vol = 0.0
        cum_vol    = 0.0
        for c in candles:
            h, l, cl, v = float(c[2]), float(c[3]), float(c[4]), float(c[5])
            typical = (h + l + cl) / 3.0
            cum_tp_vol += typical * v
            cum_vol    += v
        return cum_tp_vol / cum_vol if cum_vol > 0 else 0.0
    except Exception:
        return 0.0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FUNDING RATE — Bybit Perpetual
# Institutional signal: high funding = crowded longs = reversal risk
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_funding_cache: Dict[str, Tuple[float, float]] = {}  # symbol → (rate, timestamp)


def _get_funding_rate(symbol: str) -> Optional[float]:
    """
    Fetch current funding rate from Bybit Perpetual for spot symbol.
    Returns funding rate as float (e.g. 0.001 = 0.1%) or None on error.

    Caches result for FUNDING_CACHE_SEC seconds (default 5min).
    Funding updates every 8h on Bybit — cache is safe.

    symbol: spot format "BTC/USDT" → linear "BTCUSDT"
    """
    import time as _time
    now = _time.time()

    # Check cache
    if symbol in _funding_cache:
        cached_rate, cached_ts = _funding_cache[symbol]
        if now - cached_ts < FUNDING_CACHE_SEC:
            return cached_rate

    try:
        # Convert spot symbol to Bybit linear format
        fut_symbol = symbol.replace("/", "")   # BTC/USDT → BTCUSDT

        import urllib.request
        import json as _json
        url = (
            f"https://fapi.binance.com/fapi/v1/premiumIndex"
            f"?symbol={fut_symbol}"
        )
        req  = urllib.request.Request(url, headers={"User-Agent": "GeniusBot/1.0"})
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = _json.loads(resp.read().decode())

        rate = float(data.get("lastFundingRate", 0.0)) if data else 0.0
        items = [data] if data else []
        _funding_cache[symbol] = (rate, now)

        if GEN_DEBUG:
            logger.info(
                f"[FUNDING] {symbol} rate={rate*100:.4f}% "
                f"(threshold: max={FUNDING_MAX_LONG_PCT}% min={FUNDING_MIN_SHORT_PCT}%)"
            )
        return rate

    except Exception as e:
        logger.debug(f"[FUNDING] fetch_fail | {symbol} err={e}")
        _funding_cache[symbol] = (0.0, now)  # cache 0 to avoid spam
        return None


def _funding_allows_buy(symbol: str) -> Tuple[bool, str]:
    """
    Returns (ok, reason).
    ok=True  → funding is neutral or favorable for BUY
    ok=False → longs overheated, avoid BUY

    Logic:
      rate > FUNDING_MAX_LONG_PCT  → crowded longs → BLOCK
      rate < FUNDING_MIN_SHORT_PCT → crowded shorts → BUY signal (contrarian)
      otherwise                    → neutral → ALLOW
    """
    if not USE_FUNDING_FILTER:
        return True, "FUNDING_FILTER_DISABLED"

    rate = _get_funding_rate(symbol)
    if rate is None:
        return True, "FUNDING_FETCH_FAILED_ALLOW"  # fail-open: don't block on error

    rate_pct = rate * 100.0  # convert to percentage

    if rate_pct > FUNDING_MAX_LONG_PCT:
        return False, (
            f"FUNDING_OVERHEATED | rate={rate_pct:.4f}% > max={FUNDING_MAX_LONG_PCT}% "
            f"(longs crowded → reversal risk)"
        )
    if rate_pct < FUNDING_MIN_SHORT_PCT:
        # Contrarian: shorts overcrowded = good BUY opportunity
        return True, (
            f"FUNDING_CONTRARIAN_BUY | rate={rate_pct:.4f}% < {FUNDING_MIN_SHORT_PCT}% "
            f"(shorts crowded → squeeze potential)"
        )

    return True, f"FUNDING_OK | rate={rate_pct:.4f}%"


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
        c_soft_last  = c_last_prev if STRUCT_SOFT_REQUIRE_LAST_UP > 0 else True
        c_soft_ups   = ups3 >= STRUCT_SOFT_REQUIRE_LAST_UP
        c_soft_sma_gap = sma_gap_pct >= (-1.0 * STRUCT_SOFT_MIN_MA_GAP)
        # FIX: mom10 hardcoded -0.004 → ENV-დან, default გაფართოვდა -0.05
        _soft_mom10_min = float(os.getenv("STRUCT_SOFT_MIN_MOM10", "-0.02"))  # ENV=-0.02 (was -0.05)
        c_soft_mom10 = mom10 > _soft_mom10_min

        # FIX: soft_ok = მხოლოდ trend + mom10 (ups და sma_gap არჩევითია)
        soft_ok = c_soft_trend and c_soft_mom10
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
    """
    DCA-optimized confidence score — flat ბაზარზე neutral (არ ბლოკავს).

    კომპონენტები (use_ma=False):
      cond_last_prev  (0.30) — ბოლო სანთელი დადებითია
      cond_slope      (0.20) — SMA slope: flat=0.5 neutral, up=1.0, down=0.0
      cond_atr        (0.20) — ATR ნორმალური ზონა (0.05-2.0%)
      cond_rsi        (0.20) — RSI 20-70 ზონა
      cond_ups        (0.10) — ბოლო 3 სანთლიდან 2+ მწვანე
    """
    if len(closes) < 20 or len(ohlcv) < 20:
        return 0.0

    last = closes[-1]
    prev = closes[-2]
    atrp = _atr_pct(ohlcv, 14)
    slope = _slope_sma(closes)

    cond_last_prev = 1.0 if last > prev else 0.0
    cond_atr = 1.0 if (0.05 <= atrp < 2.0) else (0.5 if atrp < 0.05 else 0.0)
    cond_slope = max(0.0, min(1.0, 0.5 + slope / 0.006))
    rsi_val = _rsi(closes, 14)
    cond_rsi = 1.0 if 20.0 <= rsi_val <= 70.0 else 0.3
    ups3 = _ups_count(closes, 3)
    cond_ups = 1.0 if ups3 >= 2 else (0.5 if ups3 == 1 else 0.0)

    if use_ma:
        ma20 = _sma(closes, 20)
        cond_ma = 1.0 if last > ma20 else 0.3
        raw = (0.25*cond_ma + 0.25*cond_last_prev + 0.15*cond_slope + 0.15*cond_atr + 0.15*cond_rsi + 0.05*cond_ups)
    else:
        raw = (0.30*cond_last_prev + 0.20*cond_slope + 0.20*cond_atr + 0.20*cond_rsi + 0.10*cond_ups)

    boosted = min(1.0, raw * AI_CONFIDENCE_BOOST)
    if GEN_DEBUG:
        logger.info(
            f"[CONF] last_prev={cond_last_prev:.1f} slope={cond_slope:.2f} "
            f"atr={cond_atr:.1f} rsi={cond_rsi:.1f}(val={rsi_val:.1f}) "
            f"ups={cond_ups:.1f} raw={raw:.3f} boosted={boosted:.3f}"
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
    min_len = min(len(ema_fast), len(ema_slow))
    macd_series = [ema_fast[i] - ema_slow[i] for i in range(-min_len, 0)]
    if len(macd_series) < signal:
        return 0.0, 0.0, 0.0
    sig_ema = _ema(macd_series, signal)
    macd_val   = macd_series[-1]
    signal_val = sig_ema[-1]
    hist       = macd_val - signal_val
    return round(macd_val, 8), round(signal_val, 8), round(hist, 8)


def _macd_series(closes: List[float], fast: int = 12, slow: int = 26,
                 signal: int = 9, n_bars: int = 5) -> List[float]:
    """
    Returns last n_bars histogram values (oldest→newest).
    Used for MACD Smart Mode: detecting improving momentum.
    Empty list if not enough data.
    """
    if len(closes) < slow + signal + n_bars:
        return []
    ema_fast = _ema(closes, fast)
    ema_slow = _ema(closes, slow)
    min_len  = min(len(ema_fast), len(ema_slow))
    macd_s   = [ema_fast[i] - ema_slow[i] for i in range(-min_len, 0)]
    if len(macd_s) < signal + n_bars:
        return []
    sig_ema  = _ema(macd_s, signal)
    min_s    = min(len(macd_s), len(sig_ema))
    hists    = [macd_s[i] - sig_ema[i] for i in range(-min_s, 0)]
    return hists[-n_bars:]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# #2 Multi-Timeframe — higher timeframe trend check
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _mtf_trend_ok(symbol: str) -> Tuple[bool, str, str]:
    """
    1h (MTF_TIMEFRAME) trend-ი BULL-ია?
    FIX: მკაცრი check — ყველა პირობა უნდა სრულდებოდეს:
      1. last > EMA20  (ფასი EMA20-ზე მაღლა)
      2. EMA20 > EMA50 (მოკლე EMA გრძელ EMA-ზე მაღლა)
      3. trend_h >= 0.25 (1h trend strength მინიმუმი)
      4. ბოლო 3 სანთლიდან 2+ მწვანე (momentum დადებითია)

    Returns (ok, reason, htf_regime)
    """
    try:
        ohlcv_h = _fetch_ohlcv_direct(symbol, MTF_TIMEFRAME, MTF_CANDLE_LIMIT)
        if not ohlcv_h or len(ohlcv_h) < 52:
            return True, "not_enough_data→skip", None
        ohlcv_h, _ = _drop_unclosed_candle(ohlcv_h, MTF_TIMEFRAME)
        if len(ohlcv_h) < 52:
            return True, "not_enough_data→skip", None

        closes_h = [float(c[4]) for c in ohlcv_h]
        ema20_h  = _ema(closes_h, 20)
        ema50_h  = _ema(closes_h, 50)
        last_h   = closes_h[-1]

        # პირობა 1 & 2: EMA სტრუქტურა
        c_ema = (last_h > ema20_h[-1]) and (ema20_h[-1] > ema50_h[-1])

        # პირობა 3: 1h trend strength
        trend_h = _trend_strength(closes_h, USE_MA_FILTERS)
        c_trend = trend_h >= 0.20   # 0.25 → 0.20 (ოდნავ რბილი)

        # პირობა 4: ბოლო 3 სანთლიდან მინიმუმ 2 მწვანე
        last3 = closes_h[-3:]
        opens3 = [float(c[1]) for c in ohlcv_h[-3:]]
        green3 = sum(1 for c, o in zip(last3, opens3) if c > o)
        c_momentum = green3 >= 2

        # FIX: ყველა 3 პირობა
        ok = c_ema and c_trend and c_momentum

        # htf_regime
        atrp_h     = _atr_pct(ohlcv_h, n=14)
        htf_regime = _regime().detect_regime(trend=trend_h, atr_pct=atrp_h)

        reason = (
            f"mtf={MTF_TIMEFRAME} last={last_h:.4f} "
            f"ema20={ema20_h[-1]:.4f} ema50={ema50_h[-1]:.4f} "
            f"trend={trend_h:.3f} green3={green3}/3 "
            f"c_ema={int(c_ema)} c_trend={int(c_trend)} c_mom={int(c_momentum)} "
            f"htf_regime={htf_regime} ok={ok}"
        )
        return ok, reason, htf_regime
    except Exception as e:
        return True, f"mtf_fetch_err→skip: {e}", None


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
            ohlcv_quick = _fetch_ohlcv_direct(symbol, TIMEFRAME, 30)
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
            # cooldown 300s — EXTREME vol სიტუაცია ყოველ 20s-ში spam-ს ბლოკავს
            _ps_last = _protective_sell_ts.get(symbol, 0.0)
            if time.time() - _ps_last < 300:
                if GEN_DEBUG:
                    logger.info(
                        f"[GEN] PROTECTIVE_SELL_COOLDOWN | symbol={symbol} "
                        f"remaining={int(300 - (time.time() - _ps_last))}s"
                    )
                continue
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
            _protective_sell_ts[symbol] = time.time()
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

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # FIX GLOBAL-1: MAX_OPEN_TRADES — global cross-symbol hard limit.
    # ადრე: MAX_OPEN_TRADES=4 ENV-ში განსაზღვრული, მაგრამ ამ ფაილში
    # არასოდეს გამოყენებული → 3 symbol-ზე ერთდროული entry შეუზღუდავი.
    # ახლა: get_all_open_trades() → total count ყველა სიმბოლოზე.
    # BUY loop-ის წინ: total >= MAX_OPEN_TRADES → return None.
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    if MAX_OPEN_TRADES > 0:
        try:
            _total_open = len(get_all_open_trades() or [])
            if _total_open >= MAX_OPEN_TRADES:
                logger.info(
                    f"[GEN] BLOCKED_MAX_OPEN_TRADES | total_open={_total_open} >= MAX_OPEN_TRADES={MAX_OPEN_TRADES}"
                )
                return None
        except Exception as _e:
            logger.warning(f"[GEN] MAX_OPEN_TRADES_CHECK_FAIL | err={_e} → skipped (fail-open)")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # FIX GLOBAL-2 v2: CROSS-SYMBOL CORRELATION FILTER (FULL).
    #
    # პრობლემა v1-ში:
    #   - მხოლოდ BTC↔ETH mutual block (hardcoded strings)
    #   - BNB კორელაციაში არ იყო (BNB/BTC ≈0.75 — significant)
    #   - hardcoded "BTC/USDT" string → fragile (BTCUSDT format breaks)
    #
    # გამოსწორება v2:
    #   - CORRELATED_GROUPS: სიმბოლოების ჯგუფები რომლებშიც
    #     ერთდროულად მხოლოდ 1 პოზიცია არის დაშვებული.
    #   - BTC/ETH/BNB ერთ ჯგუფშია (ყველა BTC-ბეტა coin-ია).
    #   - symbol-ი ნებისმიერ format-ში მუშაობს: "BTC/USDT" ან "BTCUSDT"
    #     → base asset extraction-ით შემოწმება.
    #   - loop-ის წინ snapshot: generate_signal() returns after first BUY
    #     → staleness არ არის (single-threaded, single return per call).
    #   - SELL path bypass: open_trade=True → correlation skip (SELL needs to run).
    #
    # კორელაციის ჯგუფები (ENV-ით override შეიძლება):
    #   CORR_GROUP_1 = BTC,ETH,BNB  (high BTC-beta, ≥0.70 correlation)
    # ENV: CORRELATION_GROUPS="BTC,ETH,BNB|SOL,AVAX" — pipe-separated groups,
    #      comma-separated bases. Default = ყველა symbol ერთ ჯგუფშია.
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _base_asset(sym: str) -> str:
        """Extract base from 'BTC/USDT' → 'BTC', 'BTCUSDT' → 'BTC'."""
        if "/" in sym:
            return sym.split("/")[0].upper()
        # No slash: strip common quote suffixes
        for q in ("USDT", "BUSD", "USDC", "BTC", "ETH", "BNB"):
            if sym.upper().endswith(q) and len(sym) > len(q):
                return sym.upper()[: -len(q)]
        return sym.upper()

    # Parse correlation groups from ENV
    # CORRELATION_GROUPS= ცარიელი → კორელაცია გათიშულია, თითოეული სიმბოლო დამოუკიდებელია
    _raw_corr_groups = os.getenv("CORRELATION_GROUPS", "").strip()
    if _raw_corr_groups:
        _corr_groups: List[set] = [
            {b.strip().upper() for b in grp.split(",") if b.strip()}
            for grp in _raw_corr_groups.split("|")
            if grp.strip()
        ]
    else:
        # DCA: ცარიელი = კორელაცია გათიშული, ყველა symbol დამოუკიდებელია
        _corr_groups = []

    # Snapshot: which bases currently have an open trade (DB query once per call)
    _open_bases: set = set()
    for _s in SYMBOLS:
        try:
            if has_open_trade_for_symbol(_s):
                _open_bases.add(_base_asset(_s))
        except Exception:
            pass

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # SL PAUSE — DB-based, restart-safe, PER-SYMBOL ONLY
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # FIX BUG-2: global _sl_pause_active() check ამოღებულია.
    # პრობლემა: global pause return None-ს ისვრიდა ყველა symbol-ისთვის —
    #   BTC 3 SL → global pause → ETH/BNB-საც ბლოკავდა (per-symbol isolation ტყუილი).
    # გამოსწორება: per-symbol check მხოლოდ BUY loop-ში (line ~1302) — ეს სწორია.
    #   BTC pause → მხოლოდ BTC-ი ბლოკდება, ETH/BNB კვლავ ვაჭრობს.
    # GLOBAL pause-ი ნარჩუნდება მხოლოდ recovery check-ისთვის ქვემოთ.
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    try:
        sl_state = get_sl_cooldown_state()
    except Exception as _sl_err:
        logger.warning(f"[GEN] SL_STATE_FAIL | err={_sl_err} → skipping cooldown check")
        sl_state = {"consecutive_sl": 0, "sl_pause_until": ""}
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
                ohlcv_r = _fetch_ohlcv_direct(sym, TIMEFRAME, RECOVERY_CANDLES + 5)
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
        logger.info(f"[GEN] LOOP_START | symbol={symbol}")
        active_oco = _has_active_oco(symbol)
        open_trade = _has_open_trade(symbol)

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # FIX GLOBAL-2 v2 (per-symbol): CORRELATION GROUP GUARD.
        # თუ ამ symbol-ის ჯგუფში უკვე რომელიმე სიმბოლოს
        # open trade აქვს → BUY დაბლოკილია (double exposure).
        # SELL path-ი bypass-ავს (open_trade=True case ქვემოთ).
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        if not open_trade:
            _sym_base = _base_asset(symbol)
            _blocked_by_corr: Optional[str] = None
            for _grp in _corr_groups:
                if _sym_base in _grp:
                    # ამ ჯგუფის რომელიმე სხვა base open-ია?
                    _conflicting = _grp.intersection(_open_bases) - {_sym_base}
                    if _conflicting:
                        _blocked_by_corr = ",".join(sorted(_conflicting))
                        break
            if _blocked_by_corr:
                if GEN_DEBUG:
                    logger.info(
                        f"[GEN] BLOCKED_CORRELATION | symbol={symbol} "
                        f"corr_group_open={_blocked_by_corr} → same-group exposure blocked"
                    )
                continue

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # FIX I-8 FULL: per-symbol SL pause check.
        # BTC-ზე 2 SL → მხოლოდ BTC ბლოკდება, ETH/BNB კვლავ ვაჭრობს.
        # open_trade-ის შემთხვევაში: SELL-ი კვლავ მუშაობს (bypass).
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        if False and not open_trade and _sl_pause_active_for_symbol(symbol):  # DCA: disabled
            sym_state = get_sl_cooldown_state_per_symbol(symbol)
            pause_ts  = sym_state.get("sl_pause_until") or 0.0
            remaining = max(0, int(pause_ts - time.time()))
            logger.info(
                f"[SL_COOLDOWN_SYM] {symbol} PAUSED | "
                f"remaining={remaining}s ({remaining//60}m{remaining%60}s) | "
                f"consecutive_sl={sym_state['consecutive_sl']}"
            )
            continue

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # TIME-OF-DAY FILTER — low liquidity session-ების გამორიცხვა
        # 00:00-07:00 UTC: Asian session — ვიწრო ბაზარი, false signals
        # 22:00-00:00 UTC: late session — გაფართოებული spread-ები
        # open_trade bypass: თუ trade ღიაა, SELL ყოველთვის მუშაობს
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        from datetime import timezone as _tz
        _utc_hour = datetime.now(_tz.utc).hour
        _in_window = TRADE_HOUR_START_UTC <= _utc_hour < TRADE_HOUR_END_UTC

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # AFTER-HOURS SELL PROTECTION
        # UTC >= TRADE_HOUR_END_UTC (22:00+) + open_trade → session close SELL
        # FIX: cooldown 300s — ერთხელ emit, 5 წუთი პაუზა (LOOP=20s → spam)
        # FIX: min_notional check — partial TP-ის შემდეგ ნარჩენი < minimum
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        if USE_TIME_FILTER and not _in_window and open_trade:
            _ah_last = _after_hours_sell_ts.get(symbol, 0.0)
            _ah_cooldown = 300  # 5 min — loop=20s-ზე max 15 attempt-ი cooldown-ამდე
            if time.time() - _ah_last < _ah_cooldown:
                if GEN_DEBUG:
                    logger.info(
                        f"[GEN] AFTER_HOURS_SELL_COOLDOWN | symbol={symbol} "
                        f"remaining={int(_ah_cooldown - (time.time() - _ah_last))}s"
                    )
            else:
                try:
                    ohlcv_ah = _fetch_ohlcv_direct(symbol, TIMEFRAME, 30)
                    if ohlcv_ah and len(ohlcv_ah) >= 10:
                        ohlcv_ah, _ = _drop_unclosed_candle(ohlcv_ah, TIMEFRAME)
                        closes_ah = [float(c[4]) for c in ohlcv_ah]
                        atrp_ah   = _atr_pct(ohlcv_ah, 14)
                        trend_ah  = _trend_strength(closes_ah, USE_MA_FILTERS)
                        last_price = closes_ah[-1] if closes_ah else 0.0

                        # Min notional guard — execution_engine-ი ყველა open amount-ს ყიდის
                        # minimum: ETH 0.0001, BNB 0.001 qty OR $10 notional
                        # DYNAMIC_SIZE_MIN=8 USDT → $8 / price = qty
                        # partial TP-ის შემდეგ: 50% = $4 → ETH: 4/price qty
                        # safety: $5 minimum notional check
                        _min_notional = 1.0  # USDT — Bybit spot minimum ~$1
                        # quote_in meta-ს execution_engine-ი კითხულობს — არ გვაქვს აქ
                        # signal-ში "close_all": True → execution_engine-ი position-ის
                        # ყველა amount-ს ყიდის (partial TP remainder included)

                        sig_ah = {
                            "signal_id":        str(uuid.uuid4()),
                            "ts_utc":           _now_utc_iso(),
                            "certified_signal": True,
                            "final_verdict":    "SELL",
                            "trend":            round(trend_ah, 4),
                            "atr_pct":          round(atrp_ah, 4),
                            "meta": {
                                "source":    "AFTER_HOURS_SELL",
                                "symbol":    symbol,
                                "reason":    f"SESSION_CLOSE utc_hour={_utc_hour} >= end={TRADE_HOUR_END_UTC}",
                                "close_all": True,   # execution_engine: გამოიყენე all-or-nothing
                                "last_price": round(last_price, 6),
                                "min_notional": _min_notional,
                            },
                            "execution": {"symbol": symbol, "direction": "LONG"},
                        }
                        logger.warning(
                            f"[GEN] AFTER_HOURS_SELL | symbol={symbol} "
                            f"utc_hour={_utc_hour} >= end={TRADE_HOUR_END_UTC} "
                            f"last_price={last_price:.4f} → closing overnight position"
                        )
                        _after_hours_sell_ts[symbol] = time.time()
                        append_signal(sig_ah, outbox_path)
                        return sig_ah
                except Exception as _e:
                    logger.warning(f"[GEN] AFTER_HOURS_SELL_FAIL | symbol={symbol} err={_e}")

        if USE_TIME_FILTER and not open_trade:
            if not _in_window:
                if GEN_DEBUG:
                    logger.info(
                        f"[GEN] BLOCKED_BY_TIME | symbol={symbol} "
                        f"utc_hour={_utc_hour} window=[{TRADE_HOUR_START_UTC},{TRADE_HOUR_END_UTC})"
                    )
                continue

        try:
            ohlcv = _fetch_ohlcv_direct(symbol, TIMEFRAME, CANDLE_LIMIT)
            logger.info(f"[GEN] OHLCV_FETCHED | symbol={symbol} candles={len(ohlcv) if ohlcv else 0}")
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
            # DCA MODE: ღია trade-ზე SELL სიგნალს აღარ ვამოწმებთ signal_generator-ში.
            # გაყიდვა მართავს dca_tp_sl_manager.py — TP hit, SL confirmed, force close.
            # PROTECTIVE_SELL (crash EXTREME) კვლავ მუშაობს protective_sell ბლოკში.
            if GEN_DEBUG:
                logger.info(f"[GEN] OPEN_TRADE | {symbol} → DCA manager handles exit, skipping BUY")
            continue

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # BUY PATH — მხოლოდ open_trade=False შემთხვევაში
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

        # FIX BUG-1: BLOCK_SIGNALS_WHEN_ACTIVE_OCO enforcement
        # active_oco იყო წაკითხული (line ~1294) მაგრამ BUY-ში არ შემოწმდებოდა →
        # OCO-ს არსებობისას ახალი BUY emit-დებოდა → double position
        if BLOCK_SIGNALS_WHEN_ACTIVE_OCO and active_oco:
            if GEN_DEBUG:
                logger.info(
                    f"[GEN] BLOCKED_BY_ACTIVE_OCO | symbol={symbol} "
                    f"BLOCK_SIGNALS_WHEN_ACTIVE_OCO=true → skip BUY"
                )
            continue

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

        # conf < BUY_CONFIDENCE_MIN — static pre-check (adaptive check ქვემოთ, regime apply()-ის შემდეგ)
        # NOTE: ეს მხოლოდ static floor — adaptive (regime-aware) check line ~1763-ზეა
        if conf < BUY_CONFIDENCE_MIN:
            if GEN_DEBUG:
                logger.info(
                    f"[GEN] BLOCKED_BY_CONF_STATIC | symbol={symbol} "
                    f"conf={conf:.3f} < BUY_CONFIDENCE_MIN={BUY_CONFIDENCE_MIN:.3f}"
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
        # #1 MACD FILTER — Smart Mode
        #
        # STANDARD (MACD_SMART_MODE=false):
        #   hist > 0 AND macd > signal → bullish
        #   Problem: 0% pass in downtrend → 0 trades all day
        #
        # SMART MODE (MACD_SMART_MODE=true):
        #   Condition A: hist > 0 AND macd > signal (classic bullish) ✅
        #   Condition B: hist improving N consecutive bars AND
        #                hist > -ATR × factor (recovering, not deep bear)
        #   → catches early reversals = +4-6 trades/day in downtrend
        #   → safe: ATR threshold prevents entries in strong downtrend
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        if USE_MACD_FILTER:
            macd_line, macd_sig, macd_hist = _macd(closes, MACD_FAST, MACD_SLOW, MACD_SIGNAL_PERIOD)

            macd_classic_ok = (macd_hist > 0 and macd_line > macd_sig)
            macd_smart_ok   = False
            macd_mode       = "classic"

            if not macd_classic_ok and MACD_SMART_MODE:
                # Smart: check if histogram is consistently improving
                hist_series = _macd_series(closes, MACD_FAST, MACD_SLOW,
                                           MACD_SIGNAL_PERIOD, MACD_IMPROVING_BARS + 1)
                if len(hist_series) >= MACD_IMPROVING_BARS + 1:
                    # All last N bars improving (each > previous)
                    improving = all(
                        hist_series[i] > hist_series[i - 1]
                        for i in range(-MACD_IMPROVING_BARS, 0)
                    )
                    # Not in deep downtrend: hist > -ATR × factor
                    # FIX MACD-UNIT: macd_hist არის price units (e.g. BTC: dollars)
                    # atrp არის % (e.g. 0.19%). შედარება მოითხოვს ერთ unit-ს.
                    # atrp_price = last_price × atrp / 100 → price units-ში
                    # ძველი: -(atrp × factor) = -(0.19 × 0.2) = -0.038 (% units — მცდარი!)
                    # ახალი: -(last × atrp/100 × factor) = -(66880 × 0.0019 × 0.2) = -25.4 ($ units)
                    _atrp_price = float(last) * atrp / 100.0
                    not_deep_bear = macd_hist > -(_atrp_price * MACD_HIST_ATR_FACTOR)

                    if improving and not_deep_bear:
                        macd_smart_ok = True
                        macd_mode     = f"smart_improving_{MACD_IMPROVING_BARS}bars"

            macd_ok = macd_classic_ok or macd_smart_ok

            if not macd_ok:
                if GEN_DEBUG:
                    logger.info(
                        f"[GEN] BLOCKED_BY_MACD | symbol={symbol} "
                        f"macd={macd_line:.6f} signal={macd_sig:.6f} "
                        f"hist={macd_hist:.6f} smart={MACD_SMART_MODE}"
                    )
                continue

            if GEN_DEBUG and macd_mode != "classic":
                logger.info(
                    f"[GEN] MACD_SMART_PASS | symbol={symbol} "
                    f"mode={macd_mode} hist={macd_hist:.6f} atr={atrp:.4f}"
                )

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # FUNDING RATE FILTER (Crypto-specific institutional signal)
        # High funding = longs overheated = reversal risk → BLOCK BUY
        # Only active when USE_FUNDING_FILTER=true (default: false)
        # Fail-open: fetch error → allow trade (don't miss opportunity)
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        if USE_FUNDING_FILTER:
            funding_ok, funding_reason = _funding_allows_buy(symbol)
            if not funding_ok:
                if GEN_DEBUG:
                    logger.info(
                        f"[GEN] BLOCKED_BY_FUNDING | symbol={symbol} "
                        f"reason={funding_reason}"
                    )
                continue
            if GEN_DEBUG and "CONTRARIAN" in funding_reason:
                logger.info(f"[GEN] FUNDING_BOOST | symbol={symbol} {funding_reason}")

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # ADX FILTER — Average Directional Index trend strength
        # ADX < ADX_MIN_THRESHOLD → sideways/choppy market → false signals
        # ADX >= 25 → strong directional trend → good entry conditions
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        if USE_ADX_FILTER:
            adx_val = _adx(ohlcv, ADX_PERIOD)
            if adx_val < ADX_MIN_THRESHOLD:
                if GEN_DEBUG:
                    logger.info(
                        f"[GEN] BLOCKED_BY_ADX | symbol={symbol} "
                        f"adx={adx_val:.2f} < min={ADX_MIN_THRESHOLD}"
                    )
                continue
            if GEN_DEBUG:
                logger.info(f"[GEN] ADX_OK | symbol={symbol} adx={adx_val:.2f}")

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # VWAP FILTER — Volume Weighted Average Price entry zone
        # buy only when price <= VWAP × (1 + tolerance)
        # price >> VWAP = overextended = bad risk/reward
        # price ≈ VWAP or below = value zone = institutional entry
        # VWAP uses current session candles (intraday context)
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        if USE_VWAP_FILTER:
            vwap_val = _vwap(ohlcv)
            if vwap_val > 0:
                vwap_upper = vwap_val * (1.0 + VWAP_TOLERANCE)
                if last > vwap_upper:
                    if GEN_DEBUG:
                        logger.info(
                            f"[GEN] BLOCKED_BY_VWAP | symbol={symbol} "
                            f"last={last:.4f} vwap={vwap_val:.4f} "
                            f"upper={vwap_upper:.4f} (+{VWAP_TOLERANCE*100:.1f}%)"
                        )
                    continue
                if GEN_DEBUG:
                    pct_from_vwap = (last - vwap_val) / vwap_val * 100
                    logger.info(
                        f"[GEN] VWAP_OK | symbol={symbol} "
                        f"last={last:.4f} vwap={vwap_val:.4f} "
                        f"delta={pct_from_vwap:+.3f}%"
                    )

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

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # ADAPTIVE CONF_MIN — ეტაპი 2 (regime_engine)
        # BULL:      0.38 × 0.85 = 0.323 (ნაკლები სიმკაცრე)
        # UNCERTAIN: 0.38 × 1.20 = 0.456 (მეტი სიმკაცრე)
        # სწორი ადგილი: adaptive განსაზღვრის შემდეგ
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        _eff_conf_min = adaptive.get("CONF_MIN") or BUY_CONFIDENCE_MIN
        if _eff_conf_min <= 0:
            _eff_conf_min = BUY_CONFIDENCE_MIN
        if conf < _eff_conf_min:
            if GEN_DEBUG:
                logger.info(
                    f"[GEN] BLOCKED_BY_CONF_ADAPTIVE | symbol={symbol} "
                    f"conf={conf:.3f} < conf_min={_eff_conf_min:.3f} "
                    f"regime={adaptive.get('REGIME')}"
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
        # ai_score=0.55 → DYNAMIC_SIZE_MIN=8 USDT  (ENV)
        # ai_score=0.80 → DYNAMIC_SIZE_MAX=10 USDT (ENV)
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
