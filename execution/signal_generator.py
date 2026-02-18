# execution/signal_generator.py
import os
import time
import uuid
import logging
from datetime import datetime
from typing import Optional, Dict, Any, Tuple, List

import ccxt

from execution.signal_client import append_signal
from execution.db.repository import has_active_oco_for_symbol
from execution.excel_live_core import ExcelLiveCore, CoreInputs

logger = logging.getLogger("gbm")

# -----------------------------
# ENV HELPERS
# -----------------------------
def _env_str(*keys: str, default: str = "") -> str:
    for k in keys:
        v = os.getenv(k)
        if v is not None and str(v).strip() != "":
            return str(v).strip()
    return default


def _env_int(*keys: str, default: int) -> int:
    s = _env_str(*keys, default=str(default))
    try:
        return int(s)
    except Exception:
        return default


def _env_float(*keys: str, default: float) -> float:
    s = _env_str(*keys, default=str(default))
    try:
        return float(s)
    except Exception:
        return default


def _env_bool(*keys: str, default: bool = False) -> bool:
    s = _env_str(*keys, default=("true" if default else "false")).lower()
    return s in ("1", "true", "yes", "y", "on")


# -----------------------------
# CONFIG (supports both naming styles)
# -----------------------------
TIMEFRAME = _env_str("BOT_TIMEFRAME", "TIMEFRAME", default="15m")
CANDLE_LIMIT = _env_int("BOT_CANDLE_LIMIT", "CANDLE_LIMIT", default=80)
COOLDOWN_SECONDS = _env_int("BOT_SIGNAL_COOLDOWN_SECONDS", "BOT_COOLDOWN_SECONDS", default=180)

# Your env uses AUTO_TRADING=true; legacy uses ALLOW_LIVE_SIGNALS=true
ALLOW_LIVE_SIGNALS = _env_bool("ALLOW_LIVE_SIGNALS", "AUTO_TRADING", default=False)

# USDT per trade (prevents NOTIONAL issues when you size in quote)
BOT_QUOTE_PER_TRADE = _env_float("BOT_QUOTE_PER_TRADE", default=15.0)

BLOCK_SIGNALS_WHEN_ACTIVE_OCO = _env_bool("BLOCK_SIGNALS_WHEN_ACTIVE_OCO", default=True)

GEN_DEBUG = _env_bool("GEN_DEBUG", default=True)
GEN_LOG_EVERY_TICK = _env_bool("GEN_LOG_EVERY_TICK", default=True)
XRAY_DEBUG = _env_bool("XRAY_DEBUG", default=True)  # turn off if you want

# ---- Excel model path (supports EXCEL_MODEL_PATH and your EXCEL_PATH) ----
EXCEL_MODEL_PATH = _env_str(
    "EXCEL_MODEL_PATH",
    "EXCEL_PATH",
    default="/var/data/DYZEN_CAPITAL_OS_AI_LIVE_CORE_READY.xlsx",
).strip()

# sanitize common misconfig like: EXCEL_MODEL_PATH=EXCEL_MODEL_PATH=/opt/render/...xlsx
if EXCEL_MODEL_PATH.lower().startswith("excel_model_path="):
    EXCEL_MODEL_PATH = EXCEL_MODEL_PATH.split("=", 1)[1].strip()

_last_emit_ts: float = 0.0
_last_signature: Optional[Tuple[str, str]] = None  # reserved (if you later want de-dup)

# ---- Exchange (Binance) ----
BINANCE_API_KEY = _env_str("BINANCE_API_KEY", default="")
BINANCE_API_SECRET = _env_str("BINANCE_API_SECRET", default="")

EXCHANGE = ccxt.binance({
    "enableRateLimit": True,
    "apiKey": BINANCE_API_KEY,
    "secret": BINANCE_API_SECRET,
})

# Load Excel core once
_CORE: Optional[ExcelLiveCore] = None


def _now_utc_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _parse_symbols() -> List[str]:
    raw = _env_str("BOT_SYMBOLS", "SYMBOL_WHITELIST", "BOT_SYMBOL", default="BTC/USDT")
    syms = []
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
        # safe default: assume active_oco to avoid uncontrolled trades
        logger.warning(f"[GEN] ACTIVE_OCO_CHECK_FAIL | symbol={symbol} err={e} -> assume active_oco=True")
        return True


def _resolve_excel_path(env_path: str) -> str:
    """
    Resolve excel file path robustly:
    - uses env_path if exists
    - falls back to /var/data and /opt/render assets
    - provides strong debug context if nothing found
    """
    candidates = [
        env_path,
        "/var/data/DYZEN_CAPITAL_OS_AI_LIVE_CORE_READY.xlsx",
        "/opt/render/project/src/assets/DYZEN_CAPITAL_OS_AI_LIVE_CORE_READY.xlsx",
    ]

    for p in candidates:
        if p and os.path.exists(p):
            return p

    # strong debug info
    try:
        assets_list = os.listdir("/opt/render/project/src/assets")
    except Exception:
        assets_list = []

    try:
        var_data_list = os.listdir("/var/data")
    except Exception:
        var_data_list = []

    raise FileNotFoundError(
        f"EXCEL_MODEL_NOT_FOUND | env={env_path} | "
        f"assets={assets_list} | var_data={var_data_list}"
    )


def _core() -> ExcelLiveCore:
    global _CORE
    if _CORE is None:
        resolved = _resolve_excel_path(EXCEL_MODEL_PATH)
        logger.info(
            f"[GEN] EXCEL_PATH | env={EXCEL_MODEL_PATH} resolved={resolved} "
            f"exists_env={os.path.exists(EXCEL_MODEL_PATH)}"
        )
        _CORE = ExcelLiveCore(resolved)
        logger.info(f"[GEN] EXCEL_CORE_LOADED | path={resolved}")
    return _CORE


def _pct(a: float, b: float) -> float:
    if b == 0:
        return 0.0
    return (a - b) / b * 100.0


def _sma(vals: List[float], n: int) -> float:
    if len(vals) < n:
        return sum(vals) / max(1, len(vals))
    w = vals[-n:]
    return sum(w) / n


def _atr_pct(ohlcv: List[List[float]], n: int = 14) -> float:
    if len(ohlcv) < n + 1:
        return 0.0
    trs = []
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
    # tweakable but sane defaults:
    if atr_pct >= 2.0:
        return "EXTREME"
    if atr_pct <= 0.30:
        return "LOW"
    return "NORMAL"


def _trend_strength(last: float, ma20: float) -> float:
    # normalize separation above MA20 into 0..1
    gap_pct = _pct(last, ma20)  # last vs ma20
    # map: 0% -> 0.5, 0.4% -> ~0.7, 1% -> ~0.9
    x = (gap_pct / 1.0)  # 1% scale
    return max(0.0, min(1.0, 0.5 + (x * 0.4)))


def _structure_ok(closes: List[float]) -> bool:
    if len(closes) < 10:
        return False
    last = closes[-1]
    ma20 = _sma(closes, 20)
    prev = closes[-2]
    last5 = closes[-5:]
    last10 = closes[-10:]
    return (last > ma20) and (last > prev) and (sum(last5) / 5.0 > sum(last10) / 10.0)


def _volume_score(vols: List[float]) -> float:
    if len(vols) < 20:
        return 0.0
    v_last = vols[-1]
    v_avg = sum(vols[-20:]) / 20.0
    if v_avg <= 0:
        return 0.0
    ratio = v_last / v_avg  # 1.0 is normal
    # normalize: 0..2 mapped to 0..1
    return max(0.0, min(1.0, ratio / 2.0))


def _confidence_score(closes: List[float], ohlcv: List[List[float]]) -> float:
    # confidence combines trend alignment + momentum + volatility sanity
    last = closes[-1]
    prev = closes[-2]
    ma20 = _sma(closes, 20)
    atrp = _atr_pct(ohlcv, 14)

    cond1 = 1.0 if last > ma20 else 0.0
    cond2 = 1.0 if last > prev else 0.0
    cond3 = 1.0 if atrp < 2.0 else 0.0  # avoid extreme

    return (0.45 * cond1) + (0.35 * cond2) + (0.20 * cond3)


def _risk_state(vol_regime: str, ai_score: float) -> str:
    # minimal protective logic:
    if vol_regime == "EXTREME":
        return "KILL"
    if ai_score < 0.45:
        return "REDUCE"
    return "OK"


def _cooldown_ok() -> bool:
    global _last_emit_ts
    return (time.time() - _last_emit_ts) >= COOLDOWN_SECONDS


def _emit(signal: Dict[str, Any], outbox_path: str) -> None:
    global _last_emit_ts
    append_signal(signal, outbox_path)
    _last_emit_ts = time.time()


def _get_outbox_path() -> str:
    # supports both env names
    return (
        os.getenv("OUTBOX_PATH")
        or os.getenv("SIGNAL_OUTBOX_PATH")
        or "/var/data/signal_outbox.json"
    )


def generate_signal() -> Optional[Dict[str, Any]]:
    """
    Excel Live Core based generator:
    - If no active OCO: emits TRADE only when final_trade_decision == EXECUTE.
    - If active OCO: can emit SELL if risk_state == KILL (protective override).
    """
    outbox_path = _get_outbox_path()

    if not _cooldown_ok():
        return None

    core = _core()

    for symbol in SYMBOLS:
        active_oco = _has_active_oco(symbol)

        try:
            t0 = time.time()
            ohlcv = EXCHANGE.fetch_ohlcv(symbol, timeframe=TIMEFRAME, limit=CANDLE_LIMIT)
            dt_ms = int((time.time() - t0) * 1000)
            if GEN_DEBUG:
                logger.info(f"[GEN] FETCH_OK | symbol={symbol} tf={TIMEFRAME} candles={len(ohlcv) if ohlcv else 0} dt={dt_ms}ms")
        except Exception as e:
            logger.exception(f"[GEN] FETCH_FAIL | symbol={symbol} tf={TIMEFRAME} err={e}")
            continue

        if not ohlcv or len(ohlcv) < 30:
            if GEN_LOG_EVERY_TICK:
                logger.info(f"[GEN] NO_SIGNAL | symbol={symbol} reason=not_enough_candles got={len(ohlcv) if ohlcv else 0} need>=30")
            continue

        closes = [float(c[4]) for c in ohlcv]
        vols = [float(c[5]) for c in ohlcv]
        last = closes[-1]
        ma20 = _sma(closes, 20)
        atrp = _atr_pct(ohlcv, 14)
        vol_reg = _vol_regime(atrp)

        trend = _trend_strength(last, ma20)
        struct_ok = _structure_ok(closes)
        vol_score = _volume_score(vols)
        conf = _confidence_score(closes, ohlcv)

        # First pass ai_score without risk (risk uses ai_score)
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

        if GEN_DEBUG:
            logger.info(
                f"[GEN] CORE_DECISION | symbol={symbol} "
                f"ai={decision['ai_score']:.3f} macro={decision['macro_gate']} strat={decision['active_strategy']} "
                f"final={decision['final_trade_decision']} risk={risk} volReg={vol_reg} atr%={atrp:.2f} "
                f"last={last:.6f} ma20={ma20:.6f} outbox={outbox_path}"
            )

            # ===== STRATEGY X-RAY (SAFE) =====
            if XRAY_DEBUG:
                try:
                    logger.info(
                        "[XRAY] symbol=%s conf=%.3f trend=%.3f volScore=%.3f volReg=%s risk=%s strat=%s activeOCO=%s allowLive=%s",
                        symbol,
                        float(conf),
                        float(trend),
                        float(vol_score),
                        vol_reg,
                        risk,
                        decision.get("active_strategy"),
                        str(active_oco),
                        str(ALLOW_LIVE_SIGNALS),
                    )
                except Exception as e:
                    logger.exception("[XRAY] FAIL | symbol=%s err=%s", symbol, e)

        # Protective SELL if active OCO and risk is KILL
        if active_oco and risk == "KILL":
            signal_id = str(uuid.uuid4())
            sig = {
                "signal_id": signal_id,
                "ts_utc": _now_utc_iso(),
                "certified_signal": True,
                "final_verdict": "SELL",
                "meta": {
                    "source": "DYZEN_EXCEL_LIVE_CORE",
                    "symbol": symbol,
                    "reason": "RISK_KILL_OVERRIDE",
                    "decision": decision,
                },
                "execution": {
                    "symbol": symbol,
                    "direction": "LONG",
                    "entry": {"type": "MARKET"},
                }
            }
            _emit(sig, outbox_path)
            return sig

        # If active OCO → we do not open new TRADE (risk-first)
        if active_oco and BLOCK_SIGNALS_WHEN_ACTIVE_OCO:
            continue

        # TRADE only if final decision says EXECUTE
        if decision["final_trade_decision"] != "EXECUTE":
            continue

        # Safety: don't emit live trades if not allowed
        if not ALLOW_LIVE_SIGNALS:
            if GEN_DEBUG:
                logger.info(f"[GEN] BLOCKED_BY_ENV | symbol={symbol} reason=ALLOW_LIVE_SIGNALS/AUTO_TRADING=false")
            continue

        # build TRADE signal
        signal_id = str(uuid.uuid4())
        sig = {
            "signal_id": signal_id,
            "ts_utc": _now_utc_iso(),
            "certified_signal": True,
            "final_verdict": "TRADE",
            "meta": {
                "source": "DYZEN_EXCEL_LIVE_CORE",
                "symbol": symbol,
                "decision": decision,
            },
            "execution": {
                "symbol": symbol,
                "direction": "LONG",
                "entry": {"type": "MARKET"},
                "quote_amount": BOT_QUOTE_PER_TRADE,  # size in USDT (helps NOTIONAL)
            }
        }

        _emit(sig, outbox_path)
        return sig

    return None


# -----------------------------
# COMPATIBILITY ENTRYPOINTS
# -----------------------------
def run_once(*args, **kwargs) -> Optional[Dict[str, Any]]:
    """
    Backwards-compatible entrypoint expected by bootstrap:
    some versions do: `from execution.signal_generator import run_once`.
    We ignore args/kwargs intentionally.
    """
    return generate_signal()
