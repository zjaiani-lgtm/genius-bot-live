# execution/config.py
# ============================================================
# სრული კონფიგურაცია — ყველა .env პარამეტრი
# ============================================================
import os
from pathlib import Path


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def _env_bool(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() in ("1", "true", "yes", "y", "on")

def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (ValueError, AttributeError):
        return default

def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (ValueError, AttributeError):
        return default

def _env_str(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


# ─────────────────────────────────────────────
# MODE & SECURITY
# ─────────────────────────────────────────────

# რეჟიმი: DEMO | TESTNET | LIVE
MODE = _env_str("MODE", "DEMO").upper()
if MODE not in ("DEMO", "TESTNET", "LIVE"):
    MODE = "DEMO"

# LIVE/TESTNET დამატებითი დაცვა
LIVE_CONFIRMATION = _env_bool("LIVE_CONFIRMATION", "false")

# Kill switch — Render-ზე default TRUE
KILL_SWITCH = _env_bool("KILL_SWITCH", "true")

# Startup sync
STARTUP_SYNC_ENABLED = _env_bool("STARTUP_SYNC_ENABLED", "true")


# ─────────────────────────────────────────────
# BINANCE API
# ─────────────────────────────────────────────

BINANCE_API_KEY    = _env_str("BINANCE_API_KEY", "")
BINANCE_API_SECRET = _env_str("BINANCE_API_SECRET", "")
BINANCE_LIVE_REST_BASE = _env_str("BINANCE_LIVE_REST_BASE", "https://api.binance.com/api/v3")


# ─────────────────────────────────────────────
# DATABASE
# ─────────────────────────────────────────────

# .env: DB_PATH=/var/data/genius_bot_v2.db
# Render Disk mount: /var/data  ← აუცილებლად ემთხვეოდეს!
DB_PATH = Path(_env_str("DB_PATH", "/var/data/genius_bot_v2.db"))


# ─────────────────────────────────────────────
# TRADING SYMBOLS & TIMEFRAMES
# ─────────────────────────────────────────────

BOT_SYMBOLS   = _env_str("BOT_SYMBOLS",   "BTC/USDT,ETH/USDT,BNB/USDT")
SYMBOL_WHITELIST = _env_str("SYMBOL_WHITELIST", "BTC/USDT,ETH/USDT,BNB/USDT")
BOT_TIMEFRAME = _env_str("BOT_TIMEFRAME", "15m")
MTF_TIMEFRAME = _env_str("MTF_TIMEFRAME", "1h")
BOT_CANDLE_LIMIT = _env_int("BOT_CANDLE_LIMIT", 300)


# ─────────────────────────────────────────────
# POSITION & CAPITAL SIZING
# ─────────────────────────────────────────────

BOT_QUOTE_PER_TRADE  = _env_float("BOT_QUOTE_PER_TRADE",  15.0)
MAX_QUOTE_PER_TRADE  = _env_float("MAX_QUOTE_PER_TRADE",  15.0)
QUOTE_SIZE_BULL      = _env_float("QUOTE_SIZE_BULL",      15.0)
QUOTE_SIZE_UNCERTAIN = _env_float("QUOTE_SIZE_UNCERTAIN",  7.0)
BOT_POSITION_SIZE    = _env_float("BOT_POSITION_SIZE",     0.0)

CAPITAL_USAGE_MAX    = _env_float("CAPITAL_USAGE_MAX",  0.80)
CAPITAL_USAGE_MIN    = _env_float("CAPITAL_USAGE_MIN",  0.30)
MAX_PORTFOLIO_EXPOSURE = _env_float("MAX_PORTFOLIO_EXPOSURE", 0.75)
MAX_SYMBOL_EXPOSURE  = _env_float("MAX_SYMBOL_EXPOSURE", 0.40)

USE_DYNAMIC_SIZING   = _env_bool("USE_DYNAMIC_SIZING", "true")
ALLOW_POSITION_SCALING = _env_bool("ALLOW_POSITION_SCALING", "false")
DYNAMIC_SIZE_MIN     = _env_float("DYNAMIC_SIZE_MIN", 5.0)
DYNAMIC_SIZE_MAX     = _env_float("DYNAMIC_SIZE_MAX", 15.0)
DYNAMIC_SIZE_AI_LOW  = _env_float("DYNAMIC_SIZE_AI_LOW",  0.55)
DYNAMIC_SIZE_AI_HIGH = _env_float("DYNAMIC_SIZE_AI_HIGH", 0.80)

VIRTUAL_START_BALANCE = _env_float("VIRTUAL_START_BALANCE", 100000.0)


# ─────────────────────────────────────────────
# RISK MANAGEMENT
# ─────────────────────────────────────────────

MAX_OPEN_TRADES          = _env_int("MAX_OPEN_TRADES", 5)
MAX_POSITIONS_PER_SYMBOL = _env_int("MAX_POSITIONS_PER_SYMBOL", 1)
MAX_TRADES_PER_DAY       = _env_int("MAX_TRADES_PER_DAY", 25)
MAX_TRADES_PER_HOUR      = _env_int("MAX_TRADES_PER_HOUR", 8)
MAX_CONSECUTIVE_LOSSES   = _env_int("MAX_CONSECUTIVE_LOSSES", 3)
MAX_DAILY_LOSS           = _env_float("MAX_DAILY_LOSS", 2.0)
MAX_ACCOUNT_DRAWDOWN     = _env_float("MAX_ACCOUNT_DRAWDOWN", 7.0)
MAX_RISK_PER_TRADE_PCT   = _env_float("MAX_RISK_PER_TRADE_PCT", 0.0)


# ─────────────────────────────────────────────
# TP / SL / ATR
# ─────────────────────────────────────────────

TP_PCT    = _env_float("TP_PCT",    1.0)
SL_PCT    = _env_float("SL_PCT",    0.70)

ATR_MULT_TP_BULL = _env_float("ATR_MULT_TP_BULL", 3.0)
ATR_MULT_SL_BULL = _env_float("ATR_MULT_SL_BULL", 1.2)   # ტესტი: 1.5
ATR_TO_TP_SANITY_FACTOR = _env_float("ATR_TO_TP_SANITY_FACTOR", 0.15)

USE_PARTIAL_TP   = _env_bool("USE_PARTIAL_TP", "true")
PARTIAL_TP1_PCT  = _env_float("PARTIAL_TP1_PCT", 1.5)
PARTIAL_TP1_SIZE = _env_float("PARTIAL_TP1_SIZE", 0.5)

USE_BREAKEVEN_STOP    = _env_bool("USE_BREAKEVEN_STOP", "true")
BREAKEVEN_TRIGGER_PCT = _env_float("BREAKEVEN_TRIGGER_PCT", 0.3)

TRAILING_STOP_ENABLED  = _env_bool("TRAILING_STOP_ENABLED", "false")
TRAILING_STOP_DISTANCE = _env_float("TRAILING_STOP_DISTANCE", 0.25)

SL_COOLDOWN_AFTER_N      = _env_int("SL_COOLDOWN_AFTER_N", 2)
SL_COOLDOWN_PAUSE_SECONDS = _env_int("SL_COOLDOWN_PAUSE_SECONDS", 1800)
SL_LIMIT_GAP_PCT         = _env_float("SL_LIMIT_GAP_PCT", 0.15)


# ─────────────────────────────────────────────
# SIGNAL FILTERS & AI THRESHOLDS
# ─────────────────────────────────────────────

# AI სიგნალი
AI_CONFIDENCE_BOOST    = _env_float("AI_CONFIDENCE_BOOST",    1.15)
AI_EXECUTE_MIN_SCORE   = _env_float("AI_EXECUTE_MIN_SCORE",   0.55)
AI_SIGNAL_THRESHOLD    = _env_float("AI_SIGNAL_THRESHOLD",    0.45)
AI_FILTER_LOW_CONFIDENCE = _env_bool("AI_FILTER_LOW_CONFIDENCE", "true")

# BUY threshold-ები
BUY_CONFIDENCE_MIN    = _env_float("BUY_CONFIDENCE_MIN",    0.36)  # ტესტი: 0.44
BUY_LIQUIDITY_MIN_SCORE = _env_float("BUY_LIQUIDITY_MIN_SCORE", 0.30)
THRESHOLD_CONF        = _env_float("THRESHOLD_CONF",        0.38)
THRESHOLD_TREND       = _env_float("THRESHOLD_TREND",       0.30)
THRESHOLD_VOLUME      = _env_float("THRESHOLD_VOLUME",      0.50)

# RSI
RSI_MIN      = _env_int("RSI_MIN",      35)   # ტესტი: 32
RSI_MAX      = _env_int("RSI_MAX",      70)
RSI_SELL_MIN = _env_int("RSI_SELL_MIN", 75)

# Volume / Spread
MIN_VOLUME_24H = _env_float("MIN_VOLUME_24H", 30_000_000)
MAX_SPREAD_PCT = _env_float("MAX_SPREAD_PCT",  0.08)
SPREAD_LIMIT_PERCENT = _env_float("SPREAD_LIMIT_PERCENT", 0.12)
MIN_MOVE_PCT   = _env_float("MIN_MOVE_PCT",    0.20)
MIN_NET_PROFIT_PCT = _env_float("MIN_NET_PROFIT_PCT", 0.20)

# Soft volume override
ENABLE_SOFT_VOLUME_OVERRIDE  = _env_bool("ENABLE_SOFT_VOLUME_OVERRIDE", "true")
SOFT_VOLUME_AI_MIN           = _env_float("SOFT_VOLUME_AI_MIN",   0.58)
SOFT_VOLUME_RELAX            = _env_float("SOFT_VOLUME_RELAX",    0.10)
SOFT_VOLUME_REQUIRE_VOLBAND  = _env_bool("SOFT_VOLUME_REQUIRE_VOLBAND", "true")


# ─────────────────────────────────────────────
# FILTERS (MA / MACD / MTF / RSI)
# ─────────────────────────────────────────────

USE_MA_FILTERS  = _env_bool("USE_MA_FILTERS",  "false")
USE_MACD_FILTER = _env_bool("USE_MACD_FILTER", "true")
USE_MTF_FILTER  = _env_bool("USE_MTF_FILTER",  "true")
USE_RSI_FILTER  = _env_bool("USE_RSI_FILTER",  "true")


# ─────────────────────────────────────────────
# REGIME / ADAPTIVE
# ─────────────────────────────────────────────

ADAPTIVE_MODE      = _env_bool("ADAPTIVE_MODE", "true")
MARKET_MODE        = _env_str("MARKET_MODE",  "ADAPTIVE")
STRATEGY_MODE      = _env_str("STRATEGY_MODE", "HYBRID")
TRADE_ACTIVITY     = _env_str("TRADE_ACTIVITY", "HIGH")

REGIME_BULL_TREND_MIN   = _env_float("REGIME_BULL_TREND_MIN",   0.30)
REGIME_SIDEWAYS_ATR_MAX = _env_float("REGIME_SIDEWAYS_ATR_MAX", 0.18)

# Structural soft overrides
STRUCT_SOFT_OVERRIDE      = _env_bool("STRUCT_SOFT_OVERRIDE", "true")
STRUCT_SOFT_MIN_MA_GAP    = _env_float("STRUCT_SOFT_MIN_MA_GAP",  0.10)
STRUCT_SOFT_MIN_TREND     = _env_float("STRUCT_SOFT_MIN_TREND",   0.15)
STRUCT_SOFT_REQUIRE_LAST_UP = _env_int("STRUCT_SOFT_REQUIRE_LAST_UP", 1)

# AI score weights (excel_live_core.py)
WEIGHT_TREND       = _env_float("WEIGHT_TREND",       0.30)
WEIGHT_STRUCTURE   = _env_float("WEIGHT_STRUCTURE",   0.20)
WEIGHT_VOLUME      = _env_float("WEIGHT_VOLUME",      0.13)
WEIGHT_RISK        = _env_float("WEIGHT_RISK",        0.15)
WEIGHT_CONFIDENCE  = _env_float("WEIGHT_CONFIDENCE",  0.15)
WEIGHT_VOLATILITY  = _env_float("WEIGHT_VOLATILITY",  0.07)


# ─────────────────────────────────────────────
# EXECUTION
# ─────────────────────────────────────────────

ENTRY_MODE       = _env_str("ENTRY_MODE", "MARKET")
EXECUTION_STYLE  = _env_str("EXECUTION_STYLE", "FAST")

LIMIT_ENTRY_OFFSET_PCT  = _env_float("LIMIT_ENTRY_OFFSET_PCT",  0.03)
LIMIT_ENTRY_TIMEOUT_SEC = _env_int("LIMIT_ENTRY_TIMEOUT_SEC",   15)

ORDER_RETRY_COUNT  = _env_int("ORDER_RETRY_COUNT",   3)
ORDER_RETRY_DELAY_MS = _env_int("ORDER_RETRY_DELAY_MS", 400)

ESTIMATED_ROUNDTRIP_FEE_PCT = _env_float("ESTIMATED_ROUNDTRIP_FEE_PCT", 0.14)
ESTIMATED_SLIPPAGE_PCT      = _env_float("ESTIMATED_SLIPPAGE_PCT",      0.05)

SELL_BUFFER       = _env_float("SELL_BUFFER",       0.999)
SELL_RETRY_BUFFER = _env_float("SELL_RETRY_BUFFER", 0.998)

BLOCK_SIGNALS_WHEN_ACTIVE_OCO = _env_bool("BLOCK_SIGNALS_WHEN_ACTIVE_OCO", "true")
DEDUPE_ONLY_WHEN_ACTIVE_OCO   = _env_bool("DEDUPE_ONLY_WHEN_ACTIVE_OCO",   "false")

ALLOW_LIVE_SIGNALS = _env_bool("ALLOW_LIVE_SIGNALS", "true")
LIVE_CONFIRMATION  = _env_bool("LIVE_CONFIRMATION",  "true")

LOOP_SLEEP_SECONDS = _env_int("LOOP_SLEEP_SECONDS", 20)
BOT_SIGNAL_COOLDOWN_SECONDS = _env_int("BOT_SIGNAL_COOLDOWN_SECONDS", 120)
SIGNAL_EXPIRATION_SECONDS   = _env_int("SIGNAL_EXPIRATION_SECONDS",   600)

RECOVERY_CANDLE_PCT    = _env_float("RECOVERY_CANDLE_PCT",    0.005)
RECOVERY_GREEN_CANDLES = _env_int("RECOVERY_GREEN_CANDLES",   1)


# ─────────────────────────────────────────────
# EXCEL MODEL
# ─────────────────────────────────────────────

EXCEL_MODEL_PATH = _env_str(
    "EXCEL_MODEL_PATH",
    "/opt/render/project/src/assets/DYZEN_CAPITAL_OS_AI_LIVE_CORE_READY.xlsx"
)
SIGNAL_OUTBOX_PATH = _env_str("SIGNAL_OUTBOX_PATH", "/var/data/signal_outbox.json")


# ─────────────────────────────────────────────
# TELEGRAM
# ─────────────────────────────────────────────

TELEGRAM_NOTIFICATIONS      = _env_bool("TELEGRAM_NOTIFICATIONS", "true")
TELEGRAM_BOT_TOKEN          = _env_str("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID            = _env_str("TELEGRAM_CHAT_ID", "")
TELEGRAM_PARSE_MODE         = _env_str("TELEGRAM_PARSE_MODE", "HTML")
TELEGRAM_TIMEZONE           = _env_str("TELEGRAM_TIMEZONE", "Asia/Tbilisi")
TELEGRAM_REPORT_EVERY_SECONDS = _env_int("TELEGRAM_REPORT_EVERY_SECONDS", 1800)
REPORT_EVERY_SECONDS        = _env_int("REPORT_EVERY_SECONDS", 60)


# ─────────────────────────────────────────────
# DEBUG / TEST
# ─────────────────────────────────────────────

GEN_DEBUG       = _env_bool("GEN_DEBUG",       "true")
GEN_TEST_SIGNAL = _env_bool("GEN_TEST_SIGNAL", "false")
