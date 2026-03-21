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

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SL COOLDOWN — 2 SL-ის შემდეგ 30 წუთი პაუზა
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SL_COOLDOWN_COUNT   = int(os.getenv("SL_COOLDOWN_AFTER_N", "2"))
SL_COOLDOWN_PAUSE   = int(os.getenv("SL_COOLDOWN_PAUSE_SECONDS", "1800"))
RECOVERY_CANDLES    = int(os.getenv("RECOVERY_GREEN_CANDLES", "3"))
RECOVERY_CANDLE_PCT = float(os.getenv("RECOVERY_CANDLE_PCT", "0.25"))

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


def _notify_sl_event() -> None:
    """SL hit — DB-ში counter გაიზარდე (restart-safe)."""
    new_count = increment_consecutive_sl(pause_seconds=SL_COOLDOWN_PAUSE)
    logger.info(f"[SL_TRACK] consecutive_sl={new_count} limit={SL_COOLDOWN_COUNT} (DB-saved)")
    if new_count >= SL_COOLDOWN_COUNT:
        logger.warning(
            f"[SL_COOLDOWN] {new_count} consecutive SL → PAUSE {SL_COOLDOWN_PAUSE//60} min "
            f"(saved to DB — restart-safe)"
        )


def _notify_tp_event() -> None:
    """TP hit — DB-ში counter reset (restart-safe)."""
    state = get_sl_cooldown_state()
    if state["consecutive_sl"] > 0:
        logger.info(f"[SL_TRACK] TP hit → reset consecutive_sl {state['consecutive_sl']}→0 (DB)")
    reset_consecutive_sl()


def _sl_pause_active() -> bool:
    """DB-დან წაიკითხავს — restart-ზეც სწორია."""
    return is_sl_pause_active()


def _recovery_ok(ohlcv: List[List[float]]) -> Tuple[bool, str]:
    """
    Recovery პირობები პაუზის შემდეგ:
      1. ბოლო RECOVERY_CANDLES (3) სანთელი მწვანეა (close > open)
      2. ბოლო სანთელი >= RECOVERY_CANDLE_PCT (0.25%) ზომისაა
    """
    if len(ohlcv) < RECOVERY_CANDLES + 1:
        return False, f"not_enough_candles need={RECOVERY_CANDLES+1}"

    candles = ohlcv[-(RECOVERY_CANDLES):]

    # 3 მწვანე სანთელი
    green_count = 0
    for c in candles:
        o = float(c[1])  # open
        cl = float(c[4]) # close
        if cl > o:
            green_count += 1

    green_ok = green_count >= RECOVERY_CANDLES

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
        f"green={green_count}/{RECOVERY_CANDLES} "
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
        return (0.35 * cond_ma) + (0.35 * cond_last_prev) + (0.20 * cond_slope) + (0.10 * cond_atr)

    return (0.45 * cond_last_prev) + (0.35 * cond_slope) + (0.20 * cond_atr)


def _risk_state(vol_regime: str, ai_score: float) -> str:
    if vol_regime == "EXTREME":
        return "KILL"
    if ai_score < 0.45:
        return "REDUCE"
    return "OK"


def generate_signal() -> Optional[Dict[str, Any]]:
    outbox_path = _get_outbox_path()

    # core singleton — ერთხელ იქმნება, ყველგან გამოიყენება
    core = _core()

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
            # VOLATILE regime params
            adaptive_sell = _regime().apply(
                trend=trend_q, vol=vol_sc_q,
                atr_pct=atrp_q, ai_score=ai_q,
                base_quote=BOT_QUOTE_PER_TRADE,
            )
            sig = {
                "signal_id": signal_id,
                "ts_utc": _now_utc_iso(),
                "certified_signal": True,
                "final_verdict": "SELL",
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
    # SL PAUSE — DB-based, restart-safe
    # consecutive_sl და sl_pause_until DB-შია → deploy bypass შეუძლებელია
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    if _sl_pause_active():
        sl_state = get_sl_cooldown_state()
        pause_ts = sl_state.get("sl_pause_until") or 0.0
        remaining = max(0, int(pause_ts - time.time()))
        logger.info(
            f"[SL_COOLDOWN] PAUSED (DB) | remaining={remaining}s "
            f"({remaining//60}m{remaining%60}s) | "
            f"consecutive_sl={sl_state['consecutive_sl']}"
        )
        return None

    sl_state = get_sl_cooldown_state()
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

        # 🚫 თუ BLOCKED
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

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # TREND REVERSAL SELL — open trade-ზე
        # protective SELL (KILL) უკვე ზემოთ დამუშავდა
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        mom1 = _momentum(closes, 1) if len(closes) > 1 else 0.0

        if open_trade:
            # SELL LOGIC — შერბილებული პირობები:
            # trend < -0.15 (ნაცვლად -0.2) AND mom1 < -0.01 (ნაცვლად -0.02)
            # ეს ნიშნავს: ნელ კლებაზეც გამოვა, არა მხოლოდ crash-ზე
            if trend < -0.15 and mom1 < -0.01:
                signal_id = str(uuid.uuid4())
                sig = {
                    "signal_id": signal_id,
                    "ts_utc": _now_utc_iso(),
                    "certified_signal": True,
                    "final_verdict": "SELL",
                    "meta": {
                        "source": "GEN_SIGNAL_SELL",
                        "symbol": symbol,
                        "reason": "TREND_REVERSAL",
                        "trend": trend,
                        "mom1": mom1,
                        "decision": decision,
                    },
                    "execution": {
                        "symbol": symbol,
                        "direction": "LONG",
                    }
                }
                logger.info(
                    f"[GEN] TREND_REVERSAL_SELL | symbol={symbol} "
                    f"trend={trend:.3f} mom1={mom1:.4f}"
                )
                _emit(sig, outbox_path)
                return sig

            continue

        if decision["final_trade_decision"] != "EXECUTE":
            continue

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

        signal_id = str(uuid.uuid4())

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # MARKET REGIME — ATR-based dynamic TP/SL
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        adaptive = _regime().apply(
            trend=trend,
            vol=vol_score,
            atr_pct=atrp,
            ai_score=float(decision["ai_score"]),
            base_quote=BOT_QUOTE_PER_TRADE,
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

        # QUOTE_SIZE regime-ისგან (BULL=7.0, UNCERTAIN=5.6 და ა.შ.)
        quote_size = adaptive.get("QUOTE_SIZE", BOT_QUOTE_PER_TRADE)
        if quote_size <= 0:
            continue

        sig = {
            "signal_id": signal_id,
            "ts_utc": _now_utc_iso(),
            "certified_signal": True,
            "final_verdict": "TRADE",
            "meta": {
                "source": "DYZEN_EXCEL_LIVE_CORE",
                "symbol": symbol,
                "decision": decision,
                "regime": adaptive.get("REGIME"),
                "atr_pct": round(atrp, 4),
            },
            "execution": {
                "symbol": symbol,
                "direction": "LONG",
                "entry": {"type": "MARKET"},
                "quote_amount": quote_size,
            },
            # execution_engine.py-ი ამ dict-ს კითხულობს:
            # tp_pct = float(adaptive.get("TP_PCT", self.tp_pct))
            # sl_pct = float(adaptive.get("SL_PCT", self.sl_pct))
            "adaptive": {
                "TP_PCT":       adaptive["TP_PCT"],
                "SL_PCT":       adaptive["SL_PCT"],
                "REGIME":       adaptive["REGIME"],
                "ATR_PCT":      adaptive["ATR_PCT"],
                "QUOTE_SIZE":   quote_size,
            },
        }

        _emit(sig, outbox_path)
        return sig

    return None


def run_once(*args, **kwargs) -> Optional[Dict[str, Any]]:
    return generate_signal()


def notify_outcome(outcome: str) -> None:
    """
    main.py-იდან გამოიძახება trade-ის დახურვის შემდეგ.
    outcome: 'SL' ან 'TP' ან 'MANUAL_SELL'
    ამ ფუნქციის გამოძახება execution_engine.py-ში:
      after close_trade(...) → from execution.signal_generator import notify_outcome
                                notify_outcome(outcome)
    """
    if str(outcome).upper() == "SL":
        _notify_sl_event()
    elif str(outcome).upper() in ("TP", "MANUAL_SELL"):
        _notify_tp_event()
