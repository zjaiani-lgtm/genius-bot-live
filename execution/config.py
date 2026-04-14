# execution/config.py
# ============================================================
# სრული კონფიგურაცია — ყველა .env პარამეტრი
# ============================================================
# SYNC CONTRACT:
#   config.py defaults == signal_generator.py defaults == env_final.txt values
#   ნებისმიერი ცვლილება ამ სამ ფაილში ერთდროულად უნდა მოხდეს.
#   "source of truth" = env_final.txt — ENV ყოველთვის override-ავს defaults-ებს.
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

MODE = _env_str("MODE", "DEMO").upper()
if MODE not in ("DEMO", "TESTNET", "LIVE"):
    MODE = "DEMO"

LIVE_CONFIRMATION    = _env_bool("LIVE_CONFIRMATION",    "true")   # SYNC: was "false"
KILL_SWITCH          = _env_bool("KILL_SWITCH",          "false")  # SYNC: was "true" — dangerous default
STARTUP_SYNC_ENABLED = _env_bool("STARTUP_SYNC_ENABLED", "true")

# ─────────────────────────────────────────────
# BINANCE API
# ─────────────────────────────────────────────

BINANCE_API_KEY        = _env_str("BINANCE_API_KEY",        "")
BINANCE_API_SECRET     = _env_str("BINANCE_API_SECRET",     "")
BINANCE_LIVE_REST_BASE = _env_str("BINANCE_LIVE_REST_BASE", "https://api.binance.com/api/v3")

# ─────────────────────────────────────────────
# DATABASE
# ─────────────────────────────────────────────

DB_PATH = Path(_env_str("DB_PATH", "/var/data/genius_bot_v2.db"))

# ─────────────────────────────────────────────
# TRADING SYMBOLS & TIMEFRAMES
# ─────────────────────────────────────────────

BOT_SYMBOLS      = _env_str("BOT_SYMBOLS",      "BTC/USDT,ETH/USDT,BNB/USDT")
SYMBOL_WHITELIST = _env_str("SYMBOL_WHITELIST", "BTC/USDT,ETH/USDT,BNB/USDT")
BOT_TIMEFRAME    = _env_str("BOT_TIMEFRAME",    "15m")
MTF_TIMEFRAME    = _env_str("MTF_TIMEFRAME",    "1h")
BOT_CANDLE_LIMIT = _env_int("BOT_CANDLE_LIMIT", 300)

# ─────────────────────────────────────────────
# POSITION & CAPITAL SIZING
# ─────────────────────────────────────────────

# SYNC: was 15.0 in config.py — signal_generator.py uses 10 → ENV=10 → aligned to 10
BOT_QUOTE_PER_TRADE  = _env_float("BOT_QUOTE_PER_TRADE",  10.0)   # SYNC: 15→10
MAX_QUOTE_PER_TRADE  = _env_float("MAX_QUOTE_PER_TRADE",  10.0)   # SYNC: 15→10
BOT_POSITION_SIZE    = _env_float("BOT_POSITION_SIZE",     0.0)

CAPITAL_USAGE_MAX      = _env_float("CAPITAL_USAGE_MAX",      0.80)
CAPITAL_USAGE_MIN      = _env_float("CAPITAL_USAGE_MIN",      0.30)

# MAX_ACCOUNT_DRAWDOWN — execution_engine.py __init__ line 180
# % balance drop from session start → KILL all trading (0/999 = disabled)
# FIX CRIT: AttributeError: module 'execution.config' has no attribute 'MAX_ACCOUNT_DRAWDOWN'
MAX_ACCOUNT_DRAWDOWN   = _env_float("MAX_ACCOUNT_DRAWDOWN",   999.0)  # DCA: disabled

USE_DYNAMIC_SIZING     = _env_bool("USE_DYNAMIC_SIZING",     "true")
ALLOW_POSITION_SCALING = _env_bool("ALLOW_POSITION_SCALING", "false")
# SYNC: was 5.0/15.0 in config.py — signal_generator.py uses 8.0/10.0 → align
DYNAMIC_SIZE_AI_LOW  = _env_float("DYNAMIC_SIZE_AI_LOW",  0.55)
DYNAMIC_SIZE_AI_HIGH = _env_float("DYNAMIC_SIZE_AI_HIGH", 0.80)

VIRTUAL_START_BALANCE = _env_float("VIRTUAL_START_BALANCE", 100000.0)

# ─────────────────────────────────────────────
# RISK MANAGEMENT
# ─────────────────────────────────────────────

MAX_OPEN_TRADES          = _env_int("MAX_OPEN_TRADES",          2)    # ENV=2
MAX_POSITIONS_PER_SYMBOL = _env_int("MAX_POSITIONS_PER_SYMBOL", 1)
# SYNC: was 25/8 in config.py — signal_generator.py uses 10/3 → align
MAX_TRADES_PER_DAY       = _env_int("MAX_TRADES_PER_DAY",      10)   # SYNC: 25→10
MAX_TRADES_PER_HOUR      = _env_int("MAX_TRADES_PER_HOUR",      3)   # SYNC: 8→3
MAX_CONSECUTIVE_LOSSES   = _env_int("MAX_CONSECUTIVE_LOSSES",   5)   # SYNC: 3→5
MAX_DAILY_LOSS           = _env_float("MAX_DAILY_LOSS",         3.0)  # SYNC: 2.0→3.0

# ─────────────────────────────────────────────
# TP / SL / ATR
# ─────────────────────────────────────────────

# SYNC: config.py had 1.0 — signal_generator.py uses 1.5 → align to 1.5
TP_PCT = _env_float("TP_PCT", 1.5)   # SYNC: 1.0→1.5
SL_PCT = _env_float("SL_PCT", 999.0)   # ENV=0.80

ATR_MULT_TP_BULL = _env_float("ATR_MULT_TP_BULL", 4.0)   # SYNC: 3.0→4.0
ATR_MULT_SL_BULL = _env_float("ATR_MULT_SL_BULL", 2.0)   # SYNC: 1.2→2.0

# SYNC: signal_generator.py=0.10, config.py=0.15, ENV=0.07 → default=0.07
# LOG: BTC/BNB atr≈0.13-0.14%, old 0.15 factor → min_atr=0.225% BLOCKED everything
ATR_TO_TP_SANITY_FACTOR = _env_float("ATR_TO_TP_SANITY_FACTOR", 0.08)  # ENV=0.08

USE_PARTIAL_TP   = _env_bool("USE_PARTIAL_TP", "true")
# SYNC: config.py had 1.5 — signal_generator.py uses 1.0 → align
PARTIAL_TP1_PCT  = _env_float("PARTIAL_TP1_PCT",  1.0)   # SYNC: 1.5→1.0
PARTIAL_TP1_SIZE = _env_float("PARTIAL_TP1_SIZE", 0.5)

BREAKEVEN_TRIGGER_PCT = _env_float("BREAKEVEN_TRIGGER_PCT", 0.48)  # ENV=0.48

# SYNC: config.py had false/0.25 — signal_generator.py uses true/0.35 → align
TRAILING_STOP_DISTANCE = _env_float("TRAILING_STOP_DISTANCE", 0.25)   # ENV=0.25

# SYNC: config.py had 2 — signal_generator.py uses 3 → align
SL_COOLDOWN_AFTER_N       = _env_int("SL_COOLDOWN_AFTER_N",      3)   # SYNC: 2→3
SL_COOLDOWN_PAUSE_SECONDS = _env_int("SL_COOLDOWN_PAUSE_SECONDS", 1800)
SL_LIMIT_GAP_PCT          = _env_float("SL_LIMIT_GAP_PCT",        0.15)

# SYNC: config.py had 0.005/1 — signal_generator.py uses 0.10/3 → align
RECOVERY_CANDLE_PCT    = _env_float("RECOVERY_CANDLE_PCT",    0.05)  # ENV=0.05
RECOVERY_GREEN_CANDLES = _env_int("RECOVERY_GREEN_CANDLES",   2)     # ENV=2

# ─────────────────────────────────────────────
# SIGNAL FILTERS & AI THRESHOLDS
# ─────────────────────────────────────────────

# SYNC: was 1.15 in config.py — signal_generator.py uses 1.05 → align
AI_CONFIDENCE_BOOST      = _env_float("AI_CONFIDENCE_BOOST",    1.05)  # SYNC: 1.15→1.05
# SYNC: was 0.45 — disabled in ENV (=0) → default 0 = disabled
AI_SIGNAL_THRESHOLD      = _env_float("AI_SIGNAL_THRESHOLD",    0.0)   # SYNC: 0.45→0 (disabled)
# SYNC: was true — blocks too aggressively in live market → false
AI_FILTER_LOW_CONFIDENCE = _env_bool("AI_FILTER_LOW_CONFIDENCE", "false")   # DCA: off

# SYNC: config.py had 0.36/0.30 — signal_generator.py uses 0.32/0.25 → align
BUY_CONFIDENCE_MIN      = _env_float("BUY_CONFIDENCE_MIN",      0.15)  # DCA: 0.46→0.15
BUY_LIQUIDITY_MIN_SCORE = _env_float("BUY_LIQUIDITY_MIN_SCORE", 0.0)   # DCA: off (0=disabled)

# FIX GLOBAL-6: THRESHOLD_CONF / THRESHOLD_TREND / THRESHOLD_VOLUME —
# ეს სამი ცვლადი DEAD CODE-ია: signal_generator.py და execution_engine.py
# BUY_CONFIDENCE_MIN / WEIGHT_* ცვლადებს იყენებს პირდაპირ (os.getenv-ით).
# სამი სახელი, ერთი მნიშვნელობა — კონფუზიის წყარო.
# ყველა ემყარება BUY_CONFIDENCE_MIN-ს. THRESHOLD_* ამოღება safe:
# ბოტი არ კითხულობს ამ ცვლადებს სადამე სიგნალ-გენერაციაში.
THRESHOLD_CONF   = _env_float("THRESHOLD_CONF",   0.32)  # DEAD: alias of BUY_CONFIDENCE_MIN
THRESHOLD_TREND  = _env_float("THRESHOLD_TREND",  0.30)  # DEAD: use REGIME_BULL_TREND_MIN
THRESHOLD_VOLUME = _env_float("THRESHOLD_VOLUME", 0.25)  # DEAD: alias of BUY_LIQUIDITY_MIN_SCORE

# SYNC: config.py had RSI_MAX=70, RSI_SELL_MIN=75
# signal_generator.py uses 72/58 → align
RSI_MIN      = _env_int("RSI_MIN",      35)
RSI_MAX      = _env_int("RSI_MAX",      72)   # SYNC: 70→72
RSI_SELL_MIN = _env_int("RSI_SELL_MIN", 72)   # ENV=72

MIN_VOLUME_24H = _env_float("MIN_VOLUME_24H", 30_000_000)

MAX_SPREAD_PCT     = _env_float("MAX_SPREAD_PCT",      0.08)
MIN_MOVE_PCT       = _env_float("MIN_MOVE_PCT",        0.22)  # ENV=0.22
# SYNC: config.py had 0.20 — signal_generator.py uses 0.25 → align
MIN_NET_PROFIT_PCT = _env_float("MIN_NET_PROFIT_PCT",  0.25)  # SYNC: 0.20→0.25

MIN_SL_PCT = _env_float("MIN_SL_PCT", 0.40)

ENABLE_SOFT_VOLUME_OVERRIDE = _env_bool("ENABLE_SOFT_VOLUME_OVERRIDE", "true")
SOFT_VOLUME_AI_MIN          = _env_float("SOFT_VOLUME_AI_MIN",  0.40)  # SYNC: 0.58→0.40
SOFT_VOLUME_RELAX           = _env_float("SOFT_VOLUME_RELAX",   0.10)
SOFT_VOLUME_REQUIRE_VOLBAND = _env_bool("SOFT_VOLUME_REQUIRE_VOLBAND", "false")  # SYNC: true→false

# ─────────────────────────────────────────────
# FILTERS (MA / MACD / MTF / RSI / ADX / VWAP)
# ─────────────────────────────────────────────

USE_MA_FILTERS     = _env_bool("USE_MA_FILTERS",     "false")
USE_MACD_FILTER    = _env_bool("USE_MACD_FILTER",    "false")   # DCA: off
USE_MTF_FILTER     = _env_bool("USE_MTF_FILTER",     "false")   # DCA: off
USE_RSI_FILTER     = _env_bool("USE_RSI_FILTER",     "false")   # DCA: off
USE_ADX_FILTER     = _env_bool("USE_ADX_FILTER",     "false")   # DCA: off
USE_VWAP_FILTER    = _env_bool("USE_VWAP_FILTER",    "false")   # DCA: off
USE_TIME_FILTER    = _env_bool("USE_TIME_FILTER",    "false")   # DCA: off
USE_FUNDING_FILTER = _env_bool("USE_FUNDING_FILTER", "false")   # DCA: off

ADX_MIN_THRESHOLD = _env_float("ADX_MIN_THRESHOLD", 23.0)  # ENV=23
ADX_PERIOD        = _env_int("ADX_PERIOD", 14)

VWAP_TOLERANCE    = _env_float("VWAP_TOLERANCE", 0.006)   # ENV=0.006
VWAP_SESSION_BARS = _env_int("VWAP_SESSION_BARS", 96)      # ENV=96 (24h window)

MACD_SMART_MODE      = _env_bool("MACD_SMART_MODE",      "true")
MACD_IMPROVING_BARS  = _env_int("MACD_IMPROVING_BARS",   4)   # ENV=4
MACD_HIST_ATR_FACTOR = _env_float("MACD_HIST_ATR_FACTOR", 0.2)

TRADE_HOUR_START_UTC = _env_int("TRADE_HOUR_START_UTC", 7)
TRADE_HOUR_END_UTC   = _env_int("TRADE_HOUR_END_UTC",  22)

FUNDING_MAX_LONG_PCT  = _env_float("FUNDING_MAX_LONG_PCT",  0.10)
FUNDING_MIN_SHORT_PCT = _env_float("FUNDING_MIN_SHORT_PCT", -0.05)

MTF_BLOCK_ON_BEAR_DIVERGE = _env_bool("MTF_BLOCK_ON_BEAR_DIVERGE", "false")
MTF_TP_BONUS    = _env_float("MTF_TP_BONUS",    0.25)
MTF_TP_PENALTY  = _env_float("MTF_TP_PENALTY",  0.20)

# ─────────────────────────────────────────────
# REGIME / ADAPTIVE
# ─────────────────────────────────────────────

ADAPTIVE_MODE  = _env_bool("ADAPTIVE_MODE", "true")
MARKET_MODE    = _env_str("MARKET_MODE",   "ADAPTIVE")
STRATEGY_MODE  = _env_str("STRATEGY_MODE", "HYBRID")
TRADE_ACTIVITY = _env_str("TRADE_ACTIVITY","HIGH")

REGIME_BULL_TREND_MIN    = _env_float("REGIME_BULL_TREND_MIN",    0.30)
REGIME_SIDEWAYS_ATR_MAX  = _env_float("REGIME_SIDEWAYS_ATR_MAX",  0.20)  # SYNC: 0.18→0.20
REGIME_CONF_BULL_MULT    = _env_float("REGIME_CONF_BULL_MULT",    0.85)
REGIME_CONF_UNCERTAIN_MULT = _env_float("REGIME_CONF_UNCERTAIN_MULT", 1.20)
REGIME_STABILITY_MIN     = _env_float("REGIME_STABILITY_MIN",     0.60)

STRUCT_SOFT_OVERRIDE       = _env_bool("STRUCT_SOFT_OVERRIDE",       "true")
STRUCT_SOFT_MIN_MA_GAP     = _env_float("STRUCT_SOFT_MIN_MA_GAP",   0.10)
# SYNC: config.py had 0.15 — signal_generator.py uses 0.25 → align
STRUCT_SOFT_MIN_TREND      = _env_float("STRUCT_SOFT_MIN_TREND",     0.25)  # SYNC: 0.15→0.25
STRUCT_SOFT_MIN_MOM10      = _env_float("STRUCT_SOFT_MIN_MOM10",    -0.02)
STRUCT_SOFT_REQUIRE_LAST_UP = _env_int("STRUCT_SOFT_REQUIRE_LAST_UP", 1)

WEIGHT_TREND      = _env_float("WEIGHT_TREND",      0.30)
WEIGHT_STRUCTURE  = _env_float("WEIGHT_STRUCTURE",  0.20)
WEIGHT_VOLUME     = _env_float("WEIGHT_VOLUME",     0.13)
WEIGHT_RISK       = _env_float("WEIGHT_RISK",       0.15)
WEIGHT_CONFIDENCE = _env_float("WEIGHT_CONFIDENCE", 0.15)
WEIGHT_VOLATILITY = _env_float("WEIGHT_VOLATILITY", 0.07)

# ─────────────────────────────────────────────
# EXECUTION
# ─────────────────────────────────────────────

EXECUTION_STYLE = _env_str("EXECUTION_STYLE", "FAST")

LIMIT_ENTRY_OFFSET_PCT  = _env_float("LIMIT_ENTRY_OFFSET_PCT",  0.03)
LIMIT_ENTRY_TIMEOUT_SEC = _env_int("LIMIT_ENTRY_TIMEOUT_SEC",   15)

ESTIMATED_ROUNDTRIP_FEE_PCT = _env_float("ESTIMATED_ROUNDTRIP_FEE_PCT", 0.14)

SELL_TREND_THRESHOLD = _env_float("SELL_TREND_THRESHOLD", -0.05)  # ENV=-0.05
SELL_BUFFER       = _env_float("SELL_BUFFER",       0.999)
SELL_RETRY_BUFFER = _env_float("SELL_RETRY_BUFFER", 0.998)

BLOCK_SIGNALS_WHEN_ACTIVE_OCO = _env_bool("BLOCK_SIGNALS_WHEN_ACTIVE_OCO", "true")
DEDUPE_ONLY_WHEN_ACTIVE_OCO   = _env_bool("DEDUPE_ONLY_WHEN_ACTIVE_OCO",   "false")

LOOP_SLEEP_SECONDS          = _env_int("LOOP_SLEEP_SECONDS",           20)
BOT_SIGNAL_COOLDOWN_SECONDS = _env_int("BOT_SIGNAL_COOLDOWN_SECONDS",  120)

USE_KELLY_SIZING      = _env_bool("USE_KELLY_SIZING",      "false")  # SYNC: not in old config → add
USE_ADAPTIVE_SIZING   = _env_bool("USE_ADAPTIVE_SIZING",   "true")

PORTFOLIO_ENABLED = _env_bool("PORTFOLIO_ENABLED", "false")

SIGNAL_OUTBOX_PATH = _env_str("SIGNAL_OUTBOX_PATH", "/var/data/signal_outbox.json")

# ─────────────────────────────────────────────
# TELEGRAM
# ─────────────────────────────────────────────

TELEGRAM_NOTIFICATIONS        = _env_bool("TELEGRAM_NOTIFICATIONS",        "true")
TELEGRAM_BOT_TOKEN            = _env_str("TELEGRAM_BOT_TOKEN",             "")
TELEGRAM_CHAT_ID              = _env_str("TELEGRAM_CHAT_ID",               "")
TELEGRAM_PARSE_MODE           = _env_str("TELEGRAM_PARSE_MODE",            "HTML")
TELEGRAM_TIMEZONE             = _env_str("TELEGRAM_TIMEZONE",              "Asia/Tbilisi")
TELEGRAM_REPORT_EVERY_SECONDS = _env_int("TELEGRAM_REPORT_EVERY_SECONDS",  10800)
REPORT_EVERY_SECONDS          = _env_int("REPORT_EVERY_SECONDS",           60)

# ─────────────────────────────────────────────
# DEBUG / TEST
# ─────────────────────────────────────────────

GEN_DEBUG       = _env_bool("GEN_DEBUG",       "true")
GEN_TEST_SIGNAL = _env_bool("GEN_TEST_SIGNAL", "false")

# ─────────────────────────────────────────────
# DCA — Dollar Cost Averaging
# ─────────────────────────────────────────────

DCA_ENABLED              = _env_bool("DCA_ENABLED",              "false")
DCA_MAX_ADD_ONS          = _env_int("DCA_MAX_ADD_ONS",           3)
DCA_MAX_CAPITAL_USDT     = _env_float("DCA_MAX_CAPITAL_USDT",    40.0)
DCA_MAX_TOTAL_USDT       = _env_float("DCA_MAX_TOTAL_USDT",      60.0)
DCA_MAX_DRAWDOWN_PCT     = _env_float("DCA_MAX_DRAWDOWN_PCT",    999.0)
DCA_MIN_NOTIONAL         = _env_float("DCA_MIN_NOTIONAL",        10.0)

# trigger drawdowns per add-on (%) — comma-separated
DCA_ADDON_TRIGGER_PCTS   = _env_str("DCA_ADDON_TRIGGER_PCTS",   "2.0,3.5,5.5")

# add-on sizes (USDT) — comma-separated
DCA_ADDON_SIZES          = _env_str("DCA_ADDON_SIZES",           "10,10,10")

# TP/SL (DCA-ზე SL ბევრად დიდია — averaging სჭირდება სივრცეს)
DCA_TP_PCT               = _env_float("DCA_TP_PCT",              2.0)
DCA_SL_PCT               = _env_float("DCA_SL_PCT",              999.0)

# SL confirmation candles (noise filter)
DCA_SL_CONFIRM_CANDLES   = _env_int("DCA_SL_CONFIRM_CANDLES",    2)

# breakeven trigger (% above avg_entry)
DCA_BREAKEVEN_TRIGGER_PCT = _env_float("DCA_BREAKEVEN_TRIGGER_PCT", 0.5)

# cooldown between add-ons (seconds)
DCA_ADDON_COOLDOWN_SECONDS = _env_int("DCA_ADDON_COOLDOWN_SECONDS", 900)

# minimum recovery score (out of 5)
DCA_MIN_RECOVERY_SCORE   = _env_int("DCA_MIN_RECOVERY_SCORE",    3)
