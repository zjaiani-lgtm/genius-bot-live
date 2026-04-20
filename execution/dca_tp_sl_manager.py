# execution/dca_tp_sl_manager.py
# ============================================================
# DCA TP/SL Manager — TP გამოთვლა avg_entry-დან.
# SL=999% (გათიშული), Breakeven=გათიშული.
#
# FORCE CLOSE — 2 პირობა:
#   1. MAX_OPEN_DAYS=7  → პოზიცია > 7 დღე ღიაა → დახურვა
#   2. MAX_DRAWDOWN_PCT=15.0 → avg-დან -15% → დახურვა
#
# ENV პარამეტრები:
#   DCA_TP_PCT=0.55         ← L1-L2 TP პროცენტი
#   DCA_SL_PCT=999.0        ← გათიშული (DCA ფილოსოფია)
#   FORCE_CLOSE_MAX_DAYS=7  ← მაქს დღე ღია (0=გათიშული)
#   FORCE_CLOSE_DRAWDOWN_PCT=15.0 ← მაქს drawdown % (0=გათიშული)
# ============================================================
from __future__ import annotations

import os
import logging
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger("gbm")


def _ef(name: str, default: float) -> float:
    try:
        v = os.getenv(name)
        return float(v) if v is not None else default
    except Exception:
        return default


class DCATpSlManager:
    """
    DCA TP/SL მართვა.
    SL=999% (გათიშული) — ბოტი TP-ს ელოდება.
    Breakeven და ForceClose გათიშულია.
    """

    def __init__(self) -> None:
        self.tp_pct = _ef("DCA_TP_PCT", 0.55)
        self.sl_pct = _ef("DCA_SL_PCT", 999.0)

        # FORCE CLOSE პარამეტრები
        self.force_close_max_days     = _ef("FORCE_CLOSE_MAX_DAYS", 7.0)
        self.force_close_drawdown_pct = _ef("FORCE_CLOSE_DRAWDOWN_PCT", 15.0)

        logger.info(
            f"[DCA] DCATpSlManager init | TP={self.tp_pct}% SL={self.sl_pct}% "
            f"force_close_days={self.force_close_max_days} "
            f"force_close_drawdown={self.force_close_drawdown_pct}%"
        )

    def calculate(self, avg_entry_price: float) -> Dict[str, float]:
        """
        avg_entry-დან TP და SL გამოთვლა.
        გამოიძახება პოზიციის გახსნისას და ყოველ add-on-ის შემდეგ.
        """
        avg = float(avg_entry_price)
        tp  = round(avg * (1.0 + self.tp_pct / 100.0), 6)
        sl  = round(avg * (1.0 - self.sl_pct / 100.0), 6)
        return {
            "tp_price": tp,
            "sl_price": sl,
            "tp_pct":   self.tp_pct,
            "sl_pct":   self.sl_pct,
        }

    def is_sl_confirmed(
        self,
        sl_price: float,
        ohlcv: Any,
    ) -> Tuple[bool, str]:
        """
        DCA სტრატეგია: SL confirmation გათიშულია.
        ბოტი არ ყიდის SL-ზე — ინახავს პოზიციას.
        """
        # DCA: SL disabled — hold until TP
        return False, "SL_DISABLED_DCA_MODE"

    def check_breakeven(
        self,
        avg_entry_price: float,
        current_price: float,
        current_sl_price: float,
    ) -> Tuple[bool, float]:
        """
        DCA სტრატეგია: Breakeven სრულად გათიშულია.
        ბოტი ინახავს პოზიციას სანამ TP-ს არ მიაღწევს.
        """
        # DCA: breakeven disabled — hold until TP
        return False, current_sl_price

    def should_force_close(
        self,
        position: Dict[str, Any],
        current_price: float,
    ) -> Tuple[bool, str]:
        """
        FORCE CLOSE — 2 პირობა:

        1. MAX DAYS: პოზიცია > FORCE_CLOSE_MAX_DAYS დღე ღიაა
           → კაპიტალი ჩაკეტილია → პატარა ზარალით დახურვა ჯობია
           → FORCE_CLOSE_MAX_DAYS=0 → გათიშული

        2. MAX DRAWDOWN: avg-დან -FORCE_CLOSE_DRAWDOWN_PCT%
           → ბაზარიძლიერ ეცემა → SHORT hedge-ს ვეღარ ანაზღაურებს
           → FORCE_CLOSE_DRAWDOWN_PCT=0 → გათიშული
        """
        # ── 1. MAX DAYS check ────────────────────────────────────────
        if self.force_close_max_days > 0:
            try:
                from datetime import datetime, timezone
                opened_raw = position.get("opened_at") or position.get("created_at") or ""
                if opened_raw:
                    opened_str = str(opened_raw).replace("Z", "+00:00")
                    opened_dt  = datetime.fromisoformat(opened_str)
                    if opened_dt.tzinfo is None:
                        opened_dt = opened_dt.replace(tzinfo=timezone.utc)
                    now_dt   = datetime.now(timezone.utc)
                    days_open = (now_dt - opened_dt).total_seconds() / 86400.0

                    if days_open >= self.force_close_max_days:
                        reason = (
                            f"MAX_DAYS_OPEN | "
                            f"open={days_open:.1f}d >= limit={self.force_close_max_days:.0f}d"
                        )
                        logger.warning(f"[FORCE_CLOSE] {position.get('symbol')} | {reason}")
                        return True, reason
            except Exception as _e:
                logger.warning(f"[FORCE_CLOSE] days_check_fail | err={_e}")

        # ── 2. MAX DRAWDOWN check ────────────────────────────────────
        if self.force_close_drawdown_pct > 0:
            try:
                avg_entry = float(position.get("avg_entry_price") or 0.0)
                if avg_entry > 0 and current_price > 0:
                    drawdown_pct = (avg_entry - current_price) / avg_entry * 100.0
                    if drawdown_pct >= self.force_close_drawdown_pct:
                        reason = (
                            f"MAX_DRAWDOWN | "
                            f"drawdown={drawdown_pct:.2f}% >= limit={self.force_close_drawdown_pct:.1f}%"
                        )
                        logger.warning(f"[FORCE_CLOSE] {position.get('symbol')} | {reason}")
                        return True, reason
            except Exception as _e:
                logger.warning(f"[FORCE_CLOSE] drawdown_check_fail | err={_e}")

        return False, "OK"


# module-level singleton
_tp_sl_mgr: Optional[DCATpSlManager] = None


def get_tp_sl_manager() -> DCATpSlManager:
    global _tp_sl_mgr
    if _tp_sl_mgr is None:
        _tp_sl_mgr = DCATpSlManager()
    return _tp_sl_mgr
