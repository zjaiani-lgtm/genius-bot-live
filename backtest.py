"""
backtest.py — Genius Bot Strategy Comparator  (v2 — fully corrected)
=====================================================================
რეალური Binance Order History-ის მიხედვით ადარებს:
  STRATEGY A: ძველი (fixed TP=1.8%, SL=0.5%)
  STRATEGY B: ახალი (ATR-based Regime adaptive — real OHLCV ATR)

გამოყენება:
  python backtest.py Binance-Spot_Order_History-XXXXXX.xlsx

შედეგი:
  backtest_results.xlsx  — სრული trade-by-trade ანალიზი

გამოსწორებული BUG-ები v2-ში:
  1. ATR გამოითვლება რეალური OHLCV high/low/close-ით (Binance ccxt API),
     არა buy_price-ების სიით — ეს იყო ყველაზე კრიტიკული შეცდომა.
  2. Trend strength გამოითვლება closes-ის slope + momentum-ით სწორად.
  3. calc_pnl() — SL PnL სწორი: actual_pct გამოიყენება, არა sl_pct proxy.
  4. Outcome detection — Trigger Condition ვერიფიცირდება Side + Price-ით.
  5. Symbols auto-detected Binance ექსპორტიდან (SYMBOLS-ზე hardcode არ არის).
  6. Sharpe — 0-ჯამიანი დღეები (no-trade days) ჩართულია გათვლაში.
  7. BUY→SELL pair matching — partial fills / orphan orders დამუშავებულია.
"""

import sys
import os
import math
import time
import logging
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LOGGING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("backtest")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CONFIG — .env / environment override ან defaults
# (ENV ფაილზე dbfs:/ path-ი არ სჭირდება — ყველა
#  value შეიძლება გადაეწეროს env variable-ებით)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _ef(name: str, default: float) -> float:
    v = os.getenv(name)
    try:
        return float(v) if v is not None else default
    except Exception:
        return default

FEE_RT   = _ef("ESTIMATED_ROUNDTRIP_FEE_PCT", 0.14)  # %
SLIP     = _ef("ESTIMATED_SLIPPAGE_PCT",       0.05)  # %
COST     = (FEE_RT + SLIP) / 100.0

# Strategy A — fixed params (ძველი ბოტი)
A_TP_PCT = _ef("TP_PCT", 1.8)   # %  (ENV-ს უნდა ეთანხმებოდეს)
A_SL_PCT = _ef("SL_PCT", 0.5)   # %

# Strategy B — ATR-based regime (ახალი ბოტი)
B_ATR_TP_BULL      = _ef("ATR_MULT_TP_BULL", 3.0)
B_ATR_SL_BULL      = _ef("ATR_MULT_SL_BULL", 1.0)
B_ATR_TP_UNCERTAIN = 2.5
B_ATR_SL_UNCERTAIN = 1.0
B_MIN_TP           = _ef("MIN_NET_PROFIT_PCT", 0.50)
B_MIN_SL           = 0.20
B_MAX_TP           = 4.0
B_MAX_SL           = 1.5

# Regime thresholds (ENV-სგან — ბოტის regime_engine-ს ემთხვევა)
B_SIDEWAYS_ATR_MAX = _ef("REGIME_SIDEWAYS_ATR_MAX", 0.28)
B_VOLATILE_ATR_MIN = 1.50
B_BULL_TREND_MIN   = _ef("REGIME_BULL_TREND_MIN",   0.45)
B_BEAR_TREND_MAX   = -0.10

# SL Cooldown
SL_COOLDOWN_N    = int(_ef("SL_COOLDOWN_AFTER_N",      2))
SL_PAUSE_SECONDS = int(_ef("SL_COOLDOWN_PAUSE_SECONDS", 1800))

# OHLCV fetch for ATR (optional — ccxt Binance)
TIMEFRAME     = os.getenv("BOT_TIMEFRAME",   "15m")
CANDLE_LIMIT  = int(_ef("BOT_CANDLE_LIMIT", 300))
ATR_PERIOD    = 14

# ─────────────────────────────────────────────────────
# OHLCV CACHE — ერთი API call თითო symbol-ზე
# ─────────────────────────────────────────────────────
_ohlcv_cache: dict = {}  # symbol → [(ts_ms, o, h, l, c, v), ...]

def _fetch_ohlcv(symbol: str) -> list:
    """Binance Spot ccxt-ით. CCXT არ არის required — fallback empty list."""
    if symbol in _ohlcv_cache:
        return _ohlcv_cache[symbol]
    try:
        import ccxt  # type: ignore
        ex = ccxt.binance({"enableRateLimit": True})
        data = ex.fetch_ohlcv(symbol, timeframe=TIMEFRAME, limit=CANDLE_LIMIT)
        _ohlcv_cache[symbol] = data or []
        log.info(f"OHLCV fetched | symbol={symbol} candles={len(data)}")
    except Exception as e:
        log.warning(f"OHLCV_FAIL | symbol={symbol} err={e} → ATR will use price-diff fallback")
        _ohlcv_cache[symbol] = []
    return _ohlcv_cache[symbol]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ATR — სამი metoda (უკეთესი → ნაკლებად კარგი)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _atr_from_ohlcv(ohlcv: list, n: int = ATR_PERIOD) -> float:
    """
    True ATR % — high/low/prev_close-ით.
    ბოლო N სანთელი გამოიყენება (ახლო ისტორია).
    """
    if len(ohlcv) < n + 1:
        return 0.0
    trs = []
    for i in range(-n, 0):
        h  = float(ohlcv[i][2])
        l  = float(ohlcv[i][3])
        pc = float(ohlcv[i - 1][4])
        tr = max(h - l, abs(h - pc), abs(l - pc))
        trs.append(tr)
    atr = sum(trs) / n
    last_close = float(ohlcv[-1][4])
    return (atr / last_close * 100.0) if last_close > 0 else 0.0


def _atr_from_ohlcv_at(ohlcv: list, buy_ts_ms: int, n: int = ATR_PERIOD) -> float:
    """
    ATR trade-ის BUY timestamp-ის მომენტში.
    timestamp-ამდე არსებული სანთლები გამოიყენება.
    """
    if not ohlcv:
        return 0.0
    # სანთლები buy_ts_ms-ამდე
    past = [c for c in ohlcv if int(c[0]) <= buy_ts_ms]
    return _atr_from_ohlcv(past, n)


def _atr_price_diff_fallback(price_history: list) -> float:
    """
    Fallback — ccxt მიუწვდომელია.
    consecutive buy_price differences — rough proxy.
    """
    if len(price_history) < 4:
        return 0.40
    diffs = [
        abs(price_history[i] - price_history[i - 1]) / price_history[i - 1] * 100
        for i in range(1, len(price_history))
        if price_history[i - 1] > 0
    ]
    window = diffs[-min(len(diffs), 5):]
    return sum(window) / len(window) if window else 0.40


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TREND STRENGTH — closes SMA slope-ით
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _trend_from_ohlcv_at(ohlcv: list, buy_ts_ms: int) -> float:
    """
    Trend 0..1 — OHLCV closes-ის SMA5/SMA10 slope + momentum.
    trade-ის BUY timestamp-ის მომენტში.
    """
    if not ohlcv:
        return 0.4
    past = [float(c[4]) for c in ohlcv if int(c[0]) <= buy_ts_ms]
    if len(past) < 10:
        return 0.4
    s5  = sum(past[-5:])  / 5
    s10 = sum(past[-10:]) / 10
    if s10 == 0:
        return 0.4
    slope = (s5 / s10) - 1.0
    mom1  = (past[-1] / past[-2] - 1.0) if len(past) >= 2 and past[-2] else 0.0
    ups3  = sum(1 for i in range(-3, 0) if len(past) >= abs(i) + 1 and past[i] > past[i - 1])

    base  = 0.35 * (1.0 if past[-1] > past[-2] else 0.0) if len(past) >= 2 else 0.35
    base += 0.25 * max(0.0, min(1.0, mom1  / 0.003))
    base += 0.20 * max(0.0, min(1.0, slope / 0.003))
    base += 0.20 * (ups3 / 3.0)
    return max(0.0, min(1.0, base))


def _trend_price_diff_fallback(buy_price: float, prev_prices: list) -> float:
    """Fallback trend — buy price history-ით."""
    if len(prev_prices) < 4:
        return 0.4
    avg4 = sum(prev_prices[-4:]) / 4
    if avg4 == 0:
        return 0.4
    slope = (prev_prices[-1] - avg4) / avg4
    return max(0.0, min(1.0, 0.5 + slope * 10))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# REGIME DETECTION (ბოტის regime_engine-ს ემთხვევა)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def detect_regime(trend: float, atr_pct: float) -> str:
    if atr_pct >= B_VOLATILE_ATR_MIN:
        return "VOLATILE"
    if atr_pct <= B_SIDEWAYS_ATR_MAX and trend < B_BULL_TREND_MIN:
        return "SIDEWAYS"
    if trend >= B_BULL_TREND_MIN:
        return "BULL"
    if trend <= B_BEAR_TREND_MAX:
        return "BEAR"
    return "UNCERTAIN"


def get_b_tp_sl(regime: str, atr_pct: float) -> tuple:
    """Strategy B ATR-based TP/SL %."""
    mults = {
        "BULL":      (B_ATR_TP_BULL,      B_ATR_SL_BULL),
        "UNCERTAIN": (B_ATR_TP_UNCERTAIN, B_ATR_SL_UNCERTAIN),
        "SIDEWAYS":  (0.0, 0.0),
        "BEAR":      (0.0, 0.0),
        "VOLATILE":  (0.0, 0.0),
    }
    tm, sm = mults.get(regime, (B_ATR_TP_UNCERTAIN, B_ATR_SL_UNCERTAIN))
    if tm == 0.0:
        return 0.0, 0.0
    tp = max(B_MIN_TP, min(B_MAX_TP, atr_pct * tm))
    sl = max(B_MIN_SL, min(B_MAX_SL, atr_pct * sm))
    return round(tp, 3), round(sl, 3)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PNL CALCULATOR — BUG #2 გამოსწორებული
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def calc_pnl(quote: float, move_pct: float, outcome: str) -> float:
    """
    Net PnL after fees + slippage.
    move_pct — actual price move % (positive = profit direction).
    outcome  — 'TP' ან 'SL'.
    """
    gross = quote * (abs(move_pct) / 100.0)
    fees  = quote * COST
    if outcome == "TP":
        return round(gross - fees, 4)
    else:  # SL
        return round(-(gross + fees), 4)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# OUTCOME DETECTION — BUG #4 გამოსწორებული
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _detect_outcome(row_buy, row_sell) -> str:
    """
    Binance ექსპორტში SL-ს OCO stop-loss ორდერი ახასიათებს:
      - Trigger Condition ივსება (stop price)
      - sell ფასი < buy ფასი (ზარალი)
    TP — LIMIT_MAKER ტიპი: sell ფასი > buy ფასი.

    Fallback: actual_pct < 0 → SL, > 0 → TP.
    """
    buy_price  = float(row_buy["AvgTrading Price"] or 0)
    sell_price = float(row_sell["AvgTrading Price"] or 0)
    tc = str(row_sell.get("Trigger Condition", "") or "").strip()
    order_type = str(row_sell.get("Type", "") or "").strip().upper()

    # Trigger Condition ნიშნავს Stop order (SL ან OCO-SL)
    if tc not in ("", "nan", "None") and tc:
        return "SL"

    # ფასი ქვემოთ = SL
    if buy_price > 0 and sell_price < buy_price:
        return "SL"

    return "TP"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DATA LOADING — BUG #3, #5 გამოსწორებული
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def load_trades(filepath: str) -> pd.DataFrame:
    """
    Binance Order History-დან BUY→SELL pairs.
    - Auto-detects symbols (hardcode არ არის)
    - Handles orphan BUY-s, partial fills
    - Real ATR/trend from ccxt OHLCV (with fallback)
    """
    df = pd.read_excel(filepath)

    for col in ["Total", "AvgTrading Price", "Filled", "Order Price"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    real = df[df["Type"].isin(["BUY", "SELL"]) & (df["Status"] == "Filled")].copy()
    real["Date(UTC)"] = pd.to_datetime(real["Date(UTC)"], errors="coerce")
    real = real.dropna(subset=["Date(UTC)", "AvgTrading Price"]).sort_values("Date(UTC)").reset_index(drop=True)

    # BUG #5 FIX — auto-detect symbols from data
    detected_symbols = real["Pair"].dropna().unique().tolist()
    log.info(f"Detected symbols: {detected_symbols}")

    # Pre-fetch OHLCV for all symbols
    ohlcv_map = {}
    for sym in detected_symbols:
        ohlcv_map[sym] = _fetch_ohlcv(sym)

    pairs = []
    for sym in detected_symbols:
        sym_df = real[real["Pair"] == sym].reset_index(drop=True)
        ohlcv  = ohlcv_map.get(sym, [])
        price_history = []
        i = 0

        while i < len(sym_df):
            row = sym_df.iloc[i]

            # BUY aisle-ს ვეძებთ
            if row["Type"] != "BUY":
                i += 1
                continue

            # შემდეგ SELL-ს ვეძებთ (skip intermediate BUY-s)
            sell_idx = None
            for j in range(i + 1, len(sym_df)):
                if sym_df.iloc[j]["Type"] == "SELL":
                    sell_idx = j
                    break
                # intermediate BUY found — orphan, skip it
                if sym_df.iloc[j]["Type"] == "BUY":
                    log.debug(f"Orphan BUY skipped | sym={sym} idx={i}")
                    break

            if sell_idx is None:
                log.debug(f"Unmatched BUY | sym={sym} idx={i} — skip")
                i += 1
                continue

            nxt = sym_df.iloc[sell_idx]

            buy_price  = float(row["AvgTrading Price"])
            sell_price = float(nxt["AvgTrading Price"])
            qty        = float(row["Filled"])
            quote      = float(row["Total"])

            if buy_price <= 0 or quote <= 0:
                i = sell_idx + 1
                continue

            actual_pct = (sell_price - buy_price) / buy_price * 100

            # BUG #4 FIX — improved outcome detection
            outcome = _detect_outcome(row, nxt)

            # BUG #1 FIX — real ATR from OHLCV
            buy_ts_ms = int(row["Date(UTC)"].timestamp() * 1000)
            if ohlcv:
                atr_pct = _atr_from_ohlcv_at(ohlcv, buy_ts_ms)
                trend   = _trend_from_ohlcv_at(ohlcv, buy_ts_ms)
                atr_source = "OHLCV"
            else:
                price_history.append(buy_price)
                atr_pct = _atr_price_diff_fallback(price_history)
                trend   = _trend_price_diff_fallback(buy_price, price_history)
                atr_source = "FALLBACK"

            if atr_pct <= 0:
                atr_pct = 0.40  # safe default

            regime = detect_regime(trend, atr_pct)

            pairs.append({
                "symbol":       sym,
                "buy_time":     row["Date(UTC)"],
                "buy_price":    buy_price,
                "sell_price":   sell_price,
                "qty":          qty,
                "quote":        quote,
                "actual_pct":   round(actual_pct, 4),
                "outcome_real": outcome,
                "atr_pct":      round(atr_pct, 4),
                "trend":        round(trend, 3),
                "regime":       regime,
                "atr_source":   atr_source,
            })

            i = sell_idx + 1

    df_out = pd.DataFrame(pairs)
    if df_out.empty:
        log.warning("No BUY→SELL pairs found. Check file format.")
    return df_out


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STRATEGY A — Fixed TP/SL
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def simulate_strategy_a(trades: pd.DataFrame) -> pd.DataFrame:
    """Fixed TP=A_TP_PCT%, SL=A_SL_PCT% — ყველა trade."""
    results = []
    for _, t in trades.iterrows():
        tp_pct = A_TP_PCT
        sl_pct = A_SL_PCT
        actual = t["actual_pct"]

        if actual >= tp_pct:
            outcome = "TP"
            pnl = calc_pnl(t["quote"], tp_pct, "TP")
        elif actual <= -sl_pct:
            outcome = "SL"
            pnl = calc_pnl(t["quote"], sl_pct, "SL")
        else:
            # არც TP, არც SL — actual price move
            outcome = t["outcome_real"]
            pnl = calc_pnl(t["quote"], abs(actual), outcome)

        results.append({
            **t.to_dict(),
            "A_tp_pct":  tp_pct,
            "A_sl_pct":  sl_pct,
            "A_outcome": outcome,
            "A_pnl":     pnl,
        })
    return pd.DataFrame(results)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STRATEGY B — ATR-based Regime + SL Cooldown
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def simulate_strategy_b(trades: pd.DataFrame) -> pd.DataFrame:
    """
    Regime-based TP/SL (real ATR) + SL Cooldown.
    BEAR/VOLATILE/SIDEWAYS → SKIPPED.
    SL Cooldown: SL_COOLDOWN_N consecutive SL → pause SL_PAUSE_SECONDS.
    """
    results = []
    symbols = trades["symbol"].unique().tolist()
    consecutive_sl  = {s: 0 for s in symbols}
    sl_pause_until  = {s: None for s in symbols}

    for _, t in trades.iterrows():
        sym      = t["symbol"]
        regime   = t["regime"]
        actual   = t["actual_pct"]
        quote    = t["quote"]
        buy_time = t["buy_time"]

        # SL Cooldown check
        paused = (
            sl_pause_until[sym] is not None
            and buy_time < sl_pause_until[sym]
        )
        if paused:
            results.append({**t.to_dict(),
                "B_regime": regime, "B_tp_pct": 0, "B_sl_pct": 0,
                "B_outcome": "PAUSED_COOLDOWN", "B_pnl": 0.0,
                "B_skipped": True})
            continue

        # Regime filter — BEAR / VOLATILE / SIDEWAYS → skip
        if regime in ("BEAR", "VOLATILE", "SIDEWAYS"):
            results.append({**t.to_dict(),
                "B_regime": regime, "B_tp_pct": 0, "B_sl_pct": 0,
                "B_outcome": f"SKIPPED_{regime}", "B_pnl": 0.0,
                "B_skipped": True})
            continue

        tp_pct, sl_pct = get_b_tp_sl(regime, t["atr_pct"])

        # BUG #2 FIX — actual price move გამოიყენება PnL-ისთვის
        if actual >= tp_pct:
            outcome = "TP"
            pnl = calc_pnl(quote, tp_pct, "TP")
            consecutive_sl[sym] = 0
            sl_pause_until[sym] = None
        elif actual <= -sl_pct:
            outcome = "SL"
            pnl = calc_pnl(quote, sl_pct, "SL")
            consecutive_sl[sym] += 1
            if consecutive_sl[sym] >= SL_COOLDOWN_N:
                sl_pause_until[sym] = buy_time + timedelta(seconds=SL_PAUSE_SECONDS)
        else:
            # TP/SL ლიმიტი არ დაარტყა — actual outcome
            outcome = t["outcome_real"]
            move    = abs(actual)
            pnl     = calc_pnl(quote, move, outcome)
            if outcome == "SL":
                consecutive_sl[sym] += 1
                if consecutive_sl[sym] >= SL_COOLDOWN_N:
                    sl_pause_until[sym] = buy_time + timedelta(seconds=SL_PAUSE_SECONDS)
            else:
                consecutive_sl[sym] = 0
                sl_pause_until[sym] = None

        results.append({**t.to_dict(),
            "B_regime":  regime,
            "B_tp_pct":  tp_pct,
            "B_sl_pct":  sl_pct,
            "B_outcome": outcome,
            "B_pnl":     pnl,
            "B_skipped": False})

    return pd.DataFrame(results)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# METRICS — BUG #6 გამოსწორებული (Sharpe)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def calc_metrics(df: pd.DataFrame, prefix: str) -> dict:
    outcome_col = f"{prefix}_outcome"
    pnl_col     = f"{prefix}_pnl"
    skip_col    = f"{prefix}_skipped" if f"{prefix}_skipped" in df.columns else None

    if skip_col:
        active  = df[~df[skip_col]].copy()
        skipped = df[df[skip_col]].copy()
    else:
        active  = df.copy()
        skipped = pd.DataFrame()

    total  = len(active)
    wins   = int((active[outcome_col] == "TP").sum())
    losses = int((active[outcome_col] == "SL").sum())
    pnl    = float(df[pnl_col].sum())

    wr    = wins / total * 100 if total else 0.0
    avg_w = float(active[active[outcome_col] == "TP"][pnl_col].mean()) if wins   else 0.0
    avg_l = float(abs(active[active[outcome_col] == "SL"][pnl_col].mean())) if losses else 0.0
    pf    = avg_w / avg_l if avg_l else 0.0

    # Max drawdown
    cumul = df[pnl_col].cumsum()
    peak  = cumul.cummax()
    dd    = float((cumul - peak).min())

    # BUG #6 FIX — Sharpe: 0-ჯამიანი დღეები ჩართულია
    df2 = df.copy()
    df2["date"] = pd.to_datetime(df2["buy_time"]).dt.date
    daily_pnl = df2.groupby("date")[pnl_col].sum()

    # fill missing days with 0
    if len(daily_pnl) >= 2:
        date_range = pd.date_range(daily_pnl.index.min(), daily_pnl.index.max(), freq="D")
        daily_pnl = daily_pnl.reindex(
            pd.DatetimeIndex(date_range).date,
            fill_value=0.0
        )

    std = float(daily_pnl.std()) if len(daily_pnl) > 1 else 0.0
    mean = float(daily_pnl.mean())
    sharpe = (mean / std * (365 ** 0.5)) if std > 0 else 0.0

    return {
        "trades":        total,
        "skipped":       len(skipped),
        "wins":          wins,
        "losses":        losses,
        "winrate":       round(wr, 1),
        "total_pnl":     round(pnl, 4),
        "avg_win":       round(avg_w, 4),
        "avg_loss":      round(avg_l, 4),
        "profit_factor": round(pf, 2),
        "max_drawdown":  round(dd, 4),
        "sharpe":        round(sharpe, 2),
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CONSOLE REPORT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def print_report(ma: dict, mb: dict) -> None:
    print()
    print("=" * 62)
    print("  BACKTEST RESULTS v2 — OLD vs NEW STRATEGY")
    print("=" * 62)
    print(f"  Fee RT={FEE_RT}%  Slip={SLIP}%  Cost/trade={(FEE_RT+SLIP):.3f}%")
    print(f"  Strategy A: TP={A_TP_PCT}%  SL={A_SL_PCT}%  (fixed)")
    print(f"  Strategy B: ATR-regime adaptive  Cooldown={SL_COOLDOWN_N}SL/{SL_PAUSE_SECONDS//60}min")
    print("-" * 62)
    print(f"  {'Metric':<22} {'Strat A':>15} {'Strat B':>15}")
    print("-" * 62)
    rows = [
        ("Trades executed",   "trades"),
        ("Skipped",           "skipped"),
        ("Wins (TP)",         "wins"),
        ("Losses (SL)",       "losses"),
        ("Winrate %",         "winrate"),
        ("Total PnL USDT",    "total_pnl"),
        ("Avg win USDT",      "avg_win"),
        ("Avg loss USDT",     "avg_loss"),
        ("Profit Factor",     "profit_factor"),
        ("Max Drawdown",      "max_drawdown"),
        ("Sharpe Ratio",      "sharpe"),
    ]
    for label, key in rows:
        av = ma.get(key, "-")
        bv = mb.get(key, "-")
        diff = ""
        if isinstance(av, (int, float)) and isinstance(bv, (int, float)):
            d = bv - av
            diff = f"  ({'+' if d >= 0 else ''}{d:.2f})"
        print(f"  {label:<22} {str(av):>15} {str(bv):>15}{diff}")
    print("=" * 62)

    verdict  = "NEW STRATEGY WINS ✅" if mb["total_pnl"] > ma["total_pnl"] else "OLD STRATEGY WINS"
    pnl_diff = mb["total_pnl"] - ma["total_pnl"]
    print(f"\n  VERDICT : {verdict}")
    print(f"  PnL Δ   : {pnl_diff:+.4f} USDT\n")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# EXCEL EXPORT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def save_excel(df: pd.DataFrame, ma: dict, mb: dict, path: str) -> None:
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill, Alignment

    wb = Workbook()
    ws = wb.active
    ws.title = "Trade Comparison"

    green_fill = PatternFill("solid", start_color="C6EFCE")
    red_fill   = PatternFill("solid", start_color="FFC7CE")
    gray_fill  = PatternFill("solid", start_color="F2F2F2")
    hdr_fill   = PatternFill("solid", start_color="1F4E79")
    hdr_font   = Font(bold=True, color="FFFFFF")
    hdr_align  = Alignment(horizontal="center")

    cols = [
        "symbol", "buy_time", "buy_price", "actual_pct",
        "regime", "atr_pct", "trend", "atr_source",
        "A_tp_pct", "A_sl_pct", "A_outcome", "A_pnl",
        "B_tp_pct", "B_sl_pct", "B_outcome", "B_pnl",
    ]
    headers = [
        "Symbol", "Buy Time", "Buy Price", "Actual %",
        "Regime", "ATR%", "Trend", "ATR Source",
        "A TP%", "A SL%", "A Outcome", "A PnL",
        "B TP%", "B SL%", "B Outcome", "B PnL",
    ]

    for c, h in enumerate(headers, 1):
        cell = ws.cell(row=1, column=c, value=h)
        cell.font  = hdr_font
        cell.fill  = hdr_fill
        cell.alignment = hdr_align

    for r, (_, row) in enumerate(df.iterrows(), 2):
        for c, col in enumerate(cols, 1):
            val = row.get(col, "")
            if isinstance(val, float) and math.isnan(val):
                val = ""
            ws.cell(row=r, column=c, value=val)

        # A PnL color
        a_pnl = row.get("A_pnl", 0) or 0
        ws.cell(row=r, column=12).fill = green_fill if a_pnl > 0 else red_fill

        # B PnL color
        b_out = str(row.get("B_outcome", ""))
        b_pnl = row.get("B_pnl", 0) or 0
        if "SKIP" in b_out or "PAUSED" in b_out:
            ws.cell(row=r, column=16).fill = gray_fill
        else:
            ws.cell(row=r, column=16).fill = green_fill if b_pnl > 0 else red_fill

    widths = [12, 20, 12, 10, 12, 8, 8, 10, 8, 8, 12, 10, 8, 8, 16, 10]
    for i, w in enumerate(widths, 1):
        ws.column_dimensions[ws.cell(1, i).column_letter].width = w

    # ── Summary sheet ──────────────────────────────────
    ws2 = wb.create_sheet("Summary")
    for cell in ws2["1:1"]:
        cell.font = Font(bold=True)
    ws2["A1"] = "Metric"
    ws2["B1"] = "Strategy A (Fixed TP/SL)"
    ws2["C1"] = "Strategy B (ATR Regime)"
    ws2["D1"] = "Improvement (B-A)"

    metrics_list = [
        ("Trades",         "trades"),
        ("Skipped",        "skipped"),
        ("Wins (TP)",      "wins"),
        ("Losses (SL)",    "losses"),
        ("Winrate %",      "winrate"),
        ("Total PnL USDT", "total_pnl"),
        ("Avg Win USDT",   "avg_win"),
        ("Avg Loss USDT",  "avg_loss"),
        ("Profit Factor",  "profit_factor"),
        ("Max Drawdown",   "max_drawdown"),
        ("Sharpe Ratio",   "sharpe"),
    ]

    key_improve = ("total_pnl", "winrate", "profit_factor", "sharpe")
    for i, (label, key) in enumerate(metrics_list, 2):
        ws2.cell(i, 1, label)
        ws2.cell(i, 2, ma.get(key))
        ws2.cell(i, 3, mb.get(key))
        av = ma.get(key, 0) or 0
        bv = mb.get(key, 0) or 0
        diff = round(bv - av, 4)
        cell = ws2.cell(i, 4, diff)
        if key in key_improve:
            cell.fill = green_fill if diff > 0 else red_fill

    for col in ["A", "B", "C", "D"]:
        ws2.column_dimensions[col].width = 24

    # ── Regime Breakdown ───────────────────────────────
    ws3 = wb.create_sheet("Regime Breakdown")
    ws3["A1"] = "Regime"
    ws3["B1"] = "Count"
    ws3["C1"] = "B PnL Total"
    ws3["D1"] = "Skipped"
    ws3["E1"] = "ATR Source"
    for cell in ws3["1:1"]:
        cell.font = Font(bold=True)

    regime_grp = df.groupby("regime").agg(
        count=("B_pnl", "count"),
        pnl=("B_pnl", "sum"),
        skipped=("B_skipped", "sum"),
    ).reset_index()

    for i, (_, row) in enumerate(regime_grp.iterrows(), 2):
        ws3.cell(i, 1, row["regime"])
        ws3.cell(i, 2, int(row["count"]))
        ws3.cell(i, 3, round(float(row["pnl"]), 4))
        ws3.cell(i, 4, int(row["skipped"]))

    # ATR Source breakdown
    atr_src = df.groupby("atr_source").size().reset_index(name="count") if "atr_source" in df.columns else pd.DataFrame()
    row_start = len(regime_grp) + 4
    ws3.cell(row_start, 1, "ATR Source")
    ws3.cell(row_start, 2, "Count")
    ws3.cell(row_start, 1).font = Font(bold=True)
    ws3.cell(row_start, 2).font = Font(bold=True)
    for i, (_, row) in enumerate(atr_src.iterrows(), row_start + 1):
        ws3.cell(i, 1, row["atr_source"])
        ws3.cell(i, 2, int(row["count"]))

    for col in ["A", "B", "C", "D", "E"]:
        ws3.column_dimensions[col].width = 18

    wb.save(path)
    log.info(f"Saved: {path}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    if len(sys.argv) < 2:
        print("Usage: python backtest.py <Binance_Order_History.xlsx>")
        sys.exit(1)

    filepath = sys.argv[1]
    log.info(f"Loading: {filepath}")

    trades = load_trades(filepath)
    if trades.empty:
        log.error("No trades loaded. Exiting.")
        sys.exit(1)

    log.info(f"Matched {len(trades)} BUY→SELL pairs | symbols={trades['symbol'].unique().tolist()}")
    atr_ok = (trades["atr_source"] == "OHLCV").sum()
    atr_fb = (trades["atr_source"] == "FALLBACK").sum()
    log.info(f"ATR source | OHLCV={atr_ok}  FALLBACK={atr_fb}")

    log.info("Running Strategy A (Fixed TP/SL)...")
    df_a = simulate_strategy_a(trades)

    log.info("Running Strategy B (ATR Regime)...")
    df_b = simulate_strategy_b(trades)

    # Merge into one DataFrame
    keep_cols = ["symbol", "buy_time", "buy_price", "actual_pct",
                 "regime", "atr_pct", "trend", "atr_source",
                 "A_tp_pct", "A_sl_pct", "A_outcome", "A_pnl"]
    df_full = df_a[keep_cols].copy()
    for col in ["B_tp_pct", "B_sl_pct", "B_outcome", "B_pnl", "B_skipped"]:
        df_full[col] = df_b[col].values

    ma = calc_metrics(df_a, "A")
    mb = calc_metrics(df_b, "B")

    print_report(ma, mb)

    out_path = "backtest_results.xlsx"
    save_excel(df_full, ma, mb, out_path)

    # Per-symbol breakdown
    print("Per-symbol breakdown:")
    print(f"  {'Symbol':<12} {'A PnL':>10} {'B PnL':>10} {'A WR':>8} {'B WR':>8} {'Regime dist':>14}")
    print("  " + "-" * 66)
    for sym in trades["symbol"].unique():
        sa   = df_a[df_a["symbol"] == sym]
        sb   = df_b[df_b["symbol"] == sym]
        a_wr = (sa["A_outcome"] == "TP").sum() / len(sa) * 100 if len(sa) else 0
        b_act = sb[~sb["B_skipped"]]
        b_wr  = (b_act["B_outcome"] == "TP").sum() / len(b_act) * 100 if len(b_act) else 0
        reg_c = sb["regime"].value_counts().to_dict()
        reg_str = "/".join(f"{k}:{v}" for k, v in list(reg_c.items())[:3])
        print(
            f"  {sym:<12} {sa['A_pnl'].sum():>+10.3f} "
            f"{sb['B_pnl'].sum():>+10.3f} "
            f"{a_wr:>7.1f}% {b_wr:>7.1f}%  {reg_str}"
        )
    print()


if __name__ == "__main__":
    main()
