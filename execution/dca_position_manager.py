# execution/dca_position_manager.py
# ============================================================
# DCA Position Manager — Add-on logic, average price calculation,
# falling knife protection, recovery signal scoring.
#
# ENV პარამეტრები:
#   DCA_ENABLED=true
#   DCA_MAX_ADD_ONS=3
#   DCA_MAX_CAPITAL_USDT=40.0
#   DCA_ADDON_TRIGGER_PCTS=2.0,3.5,5.5
#   DCA_ADDON_SIZES=10,10,10
#   DCA_SL_CONFIRM_CANDLES=2
#   DCA_ADDON_COOLDOWN_SECONDS=900
# ============================================================
from __future__ import annotations

import os
import logging
import math
import time
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("gbm")


def _ef(name: str, default: float) -> float:
    try:
        v = os.getenv(name)
        return float(v) if v is not None else default
    except Exception:
        return default


def _ei(name: str, default: int) -> int:
    try:
        v = os.getenv(name)
        return int(v) if v is not None else default
    except Exception:
        return default


def _eb(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "y", "on")


def _parse_list_float(name: str, default: List[float]) -> List[float]:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return [float(x.strip()) for x in raw.split(",") if x.strip()]
    except Exception:
        return default


def _rsi(closes: List[float], period: int = 14) -> float:
    if len(closes) < period + 1:
        return 50.0
    gains, losses = [], []
    for i in range(-period, 0):
        d = closes[i] - closes[i - 1]
        gains.append(max(d, 0.0))
        losses.append(max(-d, 0.0))
    avg_g = sum(gains) / period
    avg_l = sum(losses) / period
    if avg_l == 0:
        return 100.0
    rs = avg_g / avg_l
    return 100.0 - (100.0 / (1.0 + rs))


def _ema(closes: List[float], n: int) -> List[float]:
    if len(closes) < n:
        avg = sum(closes) / len(closes)
        return [avg] * len(closes)
    k = 2.0 / (n + 1.0)
    result = [sum(closes[:n]) / n]
    for price in closes[n:]:
        result.append(price * k + result[-1] * (1.0 - k))
    pad = len(closes) - len(result)
    return [result[0]] * pad + result


def _macd_hist_series(closes: List[float], fast: int = 12, slow: int = 26,
                      signal: int = 9, n: int = 5) -> List[float]:
    if len(closes) < slow + signal + n:
        return []
    ema_f = _ema(closes, fast)
    ema_s = _ema(closes, slow)
    ml = len(min(ema_f, ema_s, key=len))
    macd_s = [ema_f[i] - ema_s[i] for i in range(-ml, 0)]
    if len(macd_s) < signal + n:
        return []
    sig_ema = _ema(macd_s, signal)
    ms = len(min(macd_s, sig_ema, key=len))
    hists = [macd_s[i] - sig_ema[i] for i in range(-ms, 0)]
    return hists[-n:]


def _atr_pct(ohlcv: List[List[float]], n: int = 14) -> float:
    if len(ohlcv) < n + 1:
        return 0.0
    trs = []
    for i in range(-n, 0):
        h = float(ohlcv[i][2])
        l = float(ohlcv[i][3])
        pc = float(ohlcv[i - 1][4])
        trs.append(max(h - l, abs(h - pc), abs(l - pc)))
    atr = sum(trs) / n
    last_close = float(ohlcv[-1][4])
    return (atr / last_close * 100.0) if last_close else 0.0


def _volume_score(vols: List[float], n: int = 20) -> float:
    if len(vols) < n + 1:
        return 0.5
    avg = sum(vols[-n:]) / n
    if avg <= 0:
        return 0.0
    return min(1.0, vols[-1] / avg)


def recalculate_average(
    old_qty: float,
    old_avg: float,
    new_qty: float,
    new_price: float,
) -> Dict[str, float]:
    old_value = old_qty * old_avg
    new_value = new_qty * new_price
    total_qty = old_qty + new_qty
    if total_qty <= 0:
        return {
            "avg_entry_price": old_avg,
            "total_qty": old_qty,
            "added_quote": new_qty * new_price,
        }
    new_avg = (old_value + new_value) / total_qty
    return {
        "avg_entry_price": round(new_avg, 8),
        "total_qty": round(total_qty, 8),
        "added_quote": round(new_qty * new_price, 6),
    }


def score_recovery_signals(ohlcv: List[List[float]]) -> Tuple[int, Dict[str, Any]]:
    if len(ohlcv) < 30:
        return 0, {"error": "not_enough_candles"}

    closes = [float(c[4]) for c in ohlcv]
    vols   = [float(c[5]) for c in ohlcv]
    details: Dict[str, Any] = {}
    score = 0

    rsi = _rsi(closes, 14)
    rsi_ok = 20.0 <= rsi <= 48.0
    details["rsi"] = round(rsi, 2)
    details["rsi_ok"] = rsi_ok
    if rsi_ok:
        score += 1

    hist = _macd_hist_series(closes, n=3)
    macd_ok = len(hist) >= 2 and hist[-1] > hist[-2]
    details["macd_hist_last"] = round(hist[-1], 8) if hist else None
    details["macd_ok"] = macd_ok
    if macd_ok:
        score += 1

    vol_s = _volume_score(vols, 20)
    vol_ok = vol_s >= 0.80
    details["vol_score"] = round(vol_s, 3)
    details["vol_ok"] = vol_ok
    if vol_ok:
        score += 1

    atr = _atr_pct(ohlcv, 14)
    atr_ok = atr < 2.0
    details["atr_pct"] = round(atr, 4)
    details["atr_ok"] = atr_ok
    if atr_ok:
        score += 1

    last = ohlcv[-1]
    o, h, l, c = float(last[1]), float(last[2]), float(last[3]), float(last[4])
    body = abs(c - o)
    lower_wick = min(o, c) - l
    wick_ok = (lower_wick > body * 0.5) if body > 0 else False
    details["lower_wick_pct"] = round(lower_wick / body, 3) if body > 0 else 0
    details["wick_ok"] = wick_ok
    if wick_ok:
        score += 1

    details["score"] = score
    details["min_required"] = 3
    details["pass"] = score >= 3

    return score, details


class DCAPositionManager:
    """
    ADDON CASCADE SYSTEM — Add-on + LIFO Rotation Manager.

    L2 ZONE (ADD-ON active):
      avg-დან -1%,-2.2%,-3.5%,-5%,-6.5% trigger → ADD-ON (pyramid-down)
      sizes: $12,$15,$18,$15,$10 — ყველაზე დიდი bounce probability ზონაში

    L3 ZONE (ROTATION active):
      ADD-ONs exhausted + price კვლავ ეცემა → LIFO rotation
      ყველაზე ძვირი unit sell → proceeds reinvest @ current
      → avg ეცემა, TP ახლოვდება, ზარალი მინიმიზდება
    """

    def __init__(self) -> None:
        self.enabled      = _eb("DCA_ENABLED", False)

        # ── L2 ZONE: ADD-ON პარამეტრები ──────────────────────
        self.max_add_ons  = _ei("DCA_MAX_ADD_ONS", 5)
        self.max_capital  = _ef("DCA_MAX_CAPITAL_USDT", 80.0)
        self.max_drawdown = _ef("DCA_MAX_DRAWDOWN_PCT", 999.0)

        # pyramid-down triggers: bounce prob 95%→82%→71%→58%→44%
        self.trigger_pcts = _parse_list_float(
            "DCA_ADDON_TRIGGER_PCTS", [1.0, 2.2, 3.5, 5.0, 6.5]
        )
        # pyramid-down sizes: ყველაზე დიდი სადაც bounce 71% (level 3)
        self.addon_sizes = _parse_list_float(
            "DCA_ADDON_SIZES", [12.0, 15.0, 18.0, 15.0, 10.0]
        )
        # cooldown 3 წუთი (ადრე 15) — სწრაფ ვარდნაში ADD-ON არ გამოტოვდეს
        self.addon_cooldown = _ei("DCA_ADDON_COOLDOWN_SECONDS", 180)
        # recovery score გათიშულია — drawdown-based trigger საკმარისია
        self.min_recovery_score = _ei("DCA_MIN_RECOVERY_SCORE", 0)

        # ── L3 ZONE: LIFO ROTATION პარამეტრები ───────────────
        self.rotation_enabled     = _eb("DCA_ROTATION_ENABLED", True)
        # drop from last ADD-ON price to trigger rotation
        self.rotation_trigger_pct = _ef("DCA_ROTATION_TRIGGER_PCT", 1.5)
        # cooldown 5 წუთი rotation-ებს შორის (fee leak თავიდანაცილება)
        self.rotation_cooldown    = _ei("DCA_ROTATION_COOLDOWN_SECONDS", 300)

        logger.info(
            f"[DCA] DCAPositionManager init | enabled={self.enabled} "
            f"max_add_ons={self.max_add_ons} max_capital={self.max_capital} "
            f"triggers={self.trigger_pcts} sizes={self.addon_sizes} "
            f"cooldown={self.addon_cooldown}s | "
            f"rotation={self.rotation_enabled} "
            f"rot_trigger={self.rotation_trigger_pct}% "
            f"rot_cooldown={self.rotation_cooldown}s"
        )

    # ─────────────────────────────────────────────────────────
    # L2 ZONE: ADD-ON LOGIC
    # ─────────────────────────────────────────────────────────

    def should_add_on(
        self,
        position: Dict[str, Any],
        current_price: float,
        ohlcv: List[List[float]],
    ) -> Tuple[bool, str]:
        """
        True + "OK" → ADD-ON შეიძლება.
        False + reason → ვერ გაიხსნება.

        6 გეიტი:
          1. max_add_ons limit
          2. trigger list bounds
          3. capital limit
          4. price dropped enough (drawdown >= trigger)
          5. max drawdown — force close territory
          6. cooldown between add-ons
        """
        if not self.enabled:
            return False, "DCA_DISABLED"

        n           = int(position.get("add_on_count", 0))
        max_n       = int(position.get("max_add_ons", self.max_add_ons))
        max_cap     = float(position.get("max_capital", self.max_capital))
        total_spent = float(position.get("total_quote_spent", 0.0))
        avg_entry   = float(position.get("avg_entry_price", 0.0))

        if n >= max_n:
            return False, f"MAX_ADD_ONS_REACHED ({n}/{max_n})"

        if n >= len(self.trigger_pcts):
            return False, f"NO_TRIGGER_DEFINED_FOR_LEVEL_{n}"

        next_size = self.addon_sizes[n] if n < len(self.addon_sizes) else self.addon_sizes[-1]
        if total_spent + next_size > max_cap:
            return False, f"MAX_CAPITAL_REACHED ({total_spent:.1f}+{next_size:.1f}>{max_cap:.1f})"

        if avg_entry <= 0:
            return False, "AVG_ENTRY_ZERO"
        drawdown = (avg_entry - current_price) / avg_entry * 100.0
        trigger  = self.trigger_pcts[n]
        if drawdown < trigger:
            return False, f"DRAWDOWN_{drawdown:.2f}%_<_TRIGGER_{trigger:.1f}%"

        if drawdown > self.max_drawdown:
            return False, f"DRAWDOWN_{drawdown:.2f}%_>_MAX_{self.max_drawdown:.1f}%_FORCE_CLOSE"

        last_ts = float(position.get("last_add_on_ts", 0.0) or 0.0)
        if last_ts > 0 and (time.time() - last_ts) < self.addon_cooldown:
            remaining = int(self.addon_cooldown - (time.time() - last_ts))
            return False, f"ADDON_COOLDOWN_ACTIVE ({remaining}s remaining)"

        logger.info(
            f"[DCA] ADD_ON_OK | level={n+1} drawdown={drawdown:.2f}% "
            f"trigger={trigger:.1f}% size={next_size}"
        )
        return True, "OK"

    def get_addon_size(self, add_on_count: int) -> float:
        """ADD-ON ზომა USDT-ში level-ის მიხედვით."""
        n = int(add_on_count)
        if n < len(self.addon_sizes):
            return float(self.addon_sizes[n])
        return float(self.addon_sizes[-1]) if self.addon_sizes else 10.0

    def get_trigger_pct(self, add_on_count: int) -> float:
        """Trigger drawdown % level-ის მიხედვით."""
        n = int(add_on_count)
        if n < len(self.trigger_pcts):
            return float(self.trigger_pcts[n])
        return float(self.trigger_pcts[-1]) if self.trigger_pcts else 5.0

    # ─────────────────────────────────────────────────────────
    # L3 ZONE: LIFO ROTATION
    # ─────────────────────────────────────────────────────────

    def should_rotate(
        self,
        position: Dict[str, Any],
        current_price: float,
    ) -> Tuple[bool, str]:
        """
        L3 zone: ADD-ONs exhausted + price კვლავ ეცემა → LIFO rotation.

        LIFO მათემატიკა:
          ძვირი unit sell → avg ეცემა (FIFO-ზე avg ამაღლდება!)
          proceeds reinvest @ current → qty არ იცვლება
          → TP ახლოვდება ყოველ rotation-ზე

        გეიტები:
          1. rotation_enabled=false → skip
          2. add_on_count < max_add_ons → L2 zone, ADD-ON ჯერ active
          3. drop from last_addon_price >= rotation_trigger_pct (1.5%)
          4. cooldown: 300s rotation-ებს შორის
        """
        if not self.rotation_enabled:
            return False, "ROTATION_DISABLED"

        if not self.enabled:
            return False, "DCA_DISABLED"

        n     = int(position.get("add_on_count", 0))
        max_n = int(position.get("max_add_ons", self.max_add_ons))

        if n < max_n:
            return False, f"L2_ZONE_ACTIVE ({n}/{max_n} add-ons used)"

        # reference: last ADD-ON price (ან avg თუ არ გვაქვს)
        last_addon_price = float(
            position.get("last_addon_price") or
            position.get("avg_entry_price") or 0.0
        )
        if last_addon_price <= 0:
            return False, "NO_REFERENCE_PRICE"

        drop_from_last = (last_addon_price - current_price) / last_addon_price * 100.0
        if drop_from_last < self.rotation_trigger_pct:
            return False, (
                f"DROP_{drop_from_last:.2f}%_<_ROT_TRIGGER_{self.rotation_trigger_pct:.1f}%"
            )

        last_rot_ts = float(position.get("last_rotation_ts") or 0.0)
        if last_rot_ts > 0:
            elapsed = time.time() - last_rot_ts
            if elapsed < self.rotation_cooldown:
                remaining = int(self.rotation_cooldown - elapsed)
                return False, f"ROTATION_COOLDOWN ({remaining}s remaining)"

        logger.info(
            f"[DCA] ROTATION_OK | L3 zone | "
            f"drop_from_last={drop_from_last:.2f}% "
            f"trigger={self.rotation_trigger_pct:.1f}% → LIFO"
        )
        return True, "ROTATE_LIFO"

    def get_lifo_unit(
        self,
        dca_orders: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """
        LIFO: ყველაზე მაღალ ფასზე ნაყიდი unit.

        რატომ LIFO და არა FIFO:
          LIFO sell: avg ეცემა → TP ახლოვდება ✅
          FIFO sell: avg ამაღლდება → TP შორდება ❌

        მხოლოდ INITIAL და ADD_ON order-ები — ROTATION_REINVEST გამოვრიცხოთ
        (reinvest units-ი უკვე current price-ზეა → LIFO-სთვის არ ვიყენებთ)
        """
        if not dca_orders:
            return None

        eligible = [
            o for o in dca_orders
            if str(o.get("order_type", "")).upper() in (
                "INITIAL", "LAYER2_INITIAL", "CASCADE_LAYER",
            ) or str(o.get("order_type", "")).startswith("ADD_ON_")
        ]

        if not eligible:
            eligible = list(dca_orders)

        return max(eligible, key=lambda o: float(o.get("entry_price", 0.0)))


# module-level singleton
_dca_manager: Optional[DCAPositionManager] = None


def get_dca_manager() -> DCAPositionManager:
    global _dca_manager
    if _dca_manager is None:
        _dca_manager = DCAPositionManager()
    return _dca_manager
