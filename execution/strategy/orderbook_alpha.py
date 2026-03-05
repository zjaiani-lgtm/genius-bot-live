from __future__ import annotations

# --- V2 ADAPTIVE THRESHOLD (lightweight) ---
def _adaptive_relax(value: float, base: float, relax: float = 0.05) -> bool:
    return value >= (base - relax)


from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from execution.indicators import ema, rsi, atr
from execution.strategy.pressure_analyzer import log_pressure

import logging
log = logging.getLogger("strategy")


@dataclass(frozen=True)
class Signal:
    action: str
    reason: str
    atr_value: float
    features: np.ndarray


def compute_long_signal(
    df15: pd.DataFrame,
    df30: pd.DataFrame,
    df1h: pd.DataFrame,
    ema_fast: int,
    ema_slow: int,
    rsi_period: int,
    rsi_min: float,
    atr_period: int,
) -> Optional[Signal]:

    print("ORDERBOOK_ALPHA_RUNNING")

    need = max(ema_slow, rsi_period, atr_period) + 5
    if len(df15) < need or len(df30) < ema_slow + 5 or len(df1h) < ema_slow + 5:
        return None

    c15 = df15["close"].astype(float)
    h15 = df15["high"].astype(float)
    l15 = df15["low"].astype(float)

    ef15 = ema(c15, ema_fast)
    es15 = ema(c15, ema_slow)
    r15 = rsi(c15, rsi_period)
    a15 = atr(h15, l15, c15, atr_period)

    up15 = float(ef15.iloc[-1]) > float(es15.iloc[-1])
    up30 = float(
        ema(df30["close"].astype(float), ema_fast).iloc[-1]
    ) > float(
        ema(df30["close"].astype(float), ema_slow).iloc[-1]
    )
    up1h = float(
        ema(df1h["close"].astype(float), ema_fast).iloc[-1]
    ) > float(
        ema(df1h["close"].astype(float), ema_slow).iloc[-1]
    )

    rsi_ok = float(r15.iloc[-1]) >= rsi_min
    atr_val = float(a15.iloc[-1])

    if atr_val <= 0:
        return Signal("HOLD", "ATR_ZERO", atr_val, np.zeros(6, dtype=float))

    # distance from slow EMA
    dist = (float(c15.iloc[-1]) - float(es15.iloc[-1])) / max(float(es15.iloc[-1]), 1e-12)
    not_too_extended = dist < 0.05

    # --- pressure analyzer ---
    log_pressure(
        symbol="REGIME=NEUTRAL",
        up15=up15,
        up30=up30,
        up1h=up1h,
        rsi_ok=rsi_ok,
        not_too_extended=not_too_extended,
        atr_ok=(atr_val > 0),
    )

    # --- BUY MATRIX LOG ---
    log.info(
        "[BUY_MATRIX] REGIME=%s | 15m=%s 30m=%s 1h=%s RSI=%s EXT=%s",
        "NEUTRAL",
        up15,
        up30,
        up1h,
        rsi_ok,
        not_too_extended,
    )

    # --- BUY DEBUG ---
    if not (up15 and up30 and up1h and rsi_ok and not_too_extended):
        log.info(
            "BUY_BLOCKED up15=%s up30=%s up1h=%s rsi=%s ext=%s",
            up15,
            up30,
            up1h,
            rsi_ok,
            not_too_extended,
        )

    # --- BUY CONDITION ---
    if up15 and up30 and up1h and rsi_ok and not_too_extended:

        atr_pct = atr_val / max(float(c15.iloc[-1]), 1e-12)

        slope_fast = float(ef15.iloc[-1] - ef15.iloc[-5]) / max(float(c15.iloc[-1]), 1e-12)
        slope_slow = float(es15.iloc[-1] - es15.iloc[-5]) / max(float(c15.iloc[-1]), 1e-12)

        feats = np.array(
            [
                float(dist),
                float(r15.iloc[-1]) / 100.0,
                float(atr_pct),
                float(slope_fast),
                float(slope_slow),
                float(up30 and up1h),
            ],
            dtype=float,
        )

        return Signal("BUY", "TREND_OK", atr_val, feats)

    # --- ANALYZER METADATA ---
    trend_ok = (up15 and up30 and up1h)

    reason_str = (
        f"FILTERS_FAIL:"
        f"trend={trend_ok},"
        f"rsi={rsi_ok},"
        f"ext={not_too_extended},"
        f"dist={dist:.4f},"
        f"rsi_val={float(r15.iloc[-1]):.2f}"
    )

    return Signal(
        "HOLD",
        reason_str,
        atr_val,
        np.array(
            [
                dist,
                float(r15.iloc[-1]) / 100.0,
                0,
                0,
                0,
                float(up30 and up1h),
            ],
            dtype=float,
        ),
    )
