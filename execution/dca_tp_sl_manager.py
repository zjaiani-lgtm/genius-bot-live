# execution/dca_tp_sl_manager.py
# ============================================================
# DCA TP/SL Manager — ADDON CASCADE SYSTEM
# SL=999% (გათიშული), Breakeven=გათიშული.
#
# ADAPTIVE TP — ზონის მიხედვით:
#   L2 zone (add_on_count < max_add_ons): TP = DCA_TP_PCT (0.55%)
#   L3 zone (add_on_count >= max_add_ons): TP = CASCADE_TP_L3_PCT (0.35%)
#     L3-ზე bounce მოთხოვნა მცირეა — 0.35% საკმარისი avg-დან
#     fee = 0.2% (2 trades), net = 0.15% — LIFO rotation-ს ფარავს
#
# FORCE CLOSE — 2 safety net:
#   1. FORCE_CLOSE_MAX_DAYS=10   → 10 დღეზე force exit
#   2. FORCE_CLOSE_DRAWDOWN_PCT=22.0 → -22% avg-დან → exit
#      (LIFO rotation-ი ვერ გადარჩინა)
#
# ENV:
#   DCA_TP_PCT=0.55
#   CASCADE_TP_L3_PCT=0.35
#   DCA_SL_PCT=999.0
#   DCA_MAX_ADD_ONS=5
#   FORCE_CLOSE_MAX_DAYS=10
#   FORCE_CLOSE_DRAWDOWN_PCT=22.0
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


def _ei(name: str, default: int) -> int:
    try:
        v = os.getenv(name)
        return int(v) if v is not None else default
    except Exception:
        return default


class DCATpSlManager:
    """
    ADDON CASCADE SYSTEM TP/SL Manager.

    TP ადაპტური — ზონის მიხედვით:
      L2 zone: 0.55% (ADD-ON active, სწრაფი scalp)
      L3 zone: 0.35% (LIFO rotation, პატარა bounce საკმარისი)

    SL: 999% = გათიშული.
    LIFO rotation არის SL-ის ჩანაცვლება.
    """

    def __init__(self) -> None:
        self.tp_pct    = _ef("DCA_TP_PCT",          0.55)  # L2 zone
        self.tp_pct_l3 = _ef("CASCADE_TP_L3_PCT",   0.35)  # L3 zone
        self.sl_pct    = _ef("DCA_SL_PCT",           999.0) # გათიშული

        self.force_close_max_days     = _ef("FORCE_CLOSE_MAX_DAYS",     10.0)
        self.force_close_drawdown_pct = _ef("FORCE_CLOSE_DRAWDOWN_PCT", 22.0)

        # L3 boundary განსაზღვრა
        self.max_add_ons = _ei("DCA_MAX_ADD_ONS", 5)

        logger.info(
            f"[DCA] DCATpSlManager init | "
            f"TP_L2={self.tp_pct}% TP_L3={self.tp_pct_l3}% SL=OFF | "
            f"force_close_days={self.force_close_max_days}d "
            f"force_close_drawdown={self.force_close_drawdown_pct}%"
        )

    def _get_tp_pct(self, position: Optional[Dict[str, Any]] = None) -> float:
        """
        L2 vs L3 TP განსაზღვრა.
        add_on_count >= max_add_ons → L3 zone → 0.35%
        add_on_count <  max_add_ons → L2 zone → 0.55%
        """
        if position is None:
            return self.tp_pct
        n = int(position.get("add_on_count", 0) or 0)
        return self.tp_pct_l3 if n >= self.max_add_ons else self.tp_pct

    def calculate(
        self,
        avg_entry_price: float,
        position: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        """
        avg_entry-დან TP/SL გამოთვლა.
        position გადაეცემა L2/L3 zone განსაზღვრისთვის.
        გამოიძახება: პოზიციის გახსნისას, ADD-ON-ის შემდეგ, LIFO rotation-ის შემდეგ.
        """
        avg    = float(avg_entry_price)
        tp_pct = self._get_tp_pct(position)
        tp     = round(avg * (1.0 + tp_pct / 100.0), 6)
        sl     = round(avg * (1.0 - self.sl_pct / 100.0), 6)
        return {
            "tp_price": tp,
            "sl_price": sl,
            "tp_pct":   tp_pct,
            "sl_pct":   self.sl_pct,
        }

    def calculate_rotation_tp(self, new_avg: float) -> float:
        """
        LIFO rotation-ის შემდეგ ახალი TP.
        L3 zone TP (0.35%) — პატარა bounce = TP hit.
        """
        return round(new_avg * (1.0 + self.tp_pct_l3 / 100.0), 6)

    def is_sl_confirmed(self, sl_price: float, ohlcv: Any) -> Tuple[bool, str]:
        """DCA: SL disabled. LIFO rotation არის SL-ის ჩანაცვლება."""
        return False, "SL_DISABLED_DCA_MODE"

    def check_breakeven(
        self,
        avg_entry_price: float,
        current_price: float,
        current_sl_price: float,
    ) -> Tuple[bool, float]:
        """DCA: Breakeven გათიშულია."""
        return False, current_sl_price

    def should_force_close(
        self,
        position: Dict[str, Any],
        current_price: float,
    ) -> Tuple[bool, str]:
        """
        FORCE CLOSE — safety net LIFO rotation-ის შემდეგ.

        1. MAX DAYS (10): კაპიტალი 10+ დღე ჩაკეტილია → გათავისუფლება
        2. MAX DRAWDOWN (22%): LIFO rotation-მა ვერ გადარჩინა → force exit
           -22% @ $80 invested = -$17.6 realized loss
           vs კლასიკური SL -2% = -$1.48 (ანომალური სცენარი)
        """
        # ── 1. MAX DAYS ──────────────────────────────────────────────
        if self.force_close_max_days > 0:
            try:
                from datetime import datetime, timezone
                opened_raw = position.get("opened_at") or position.get("created_at") or ""
                if opened_raw:
                    opened_str = str(opened_raw).replace("Z", "+00:00")
                    opened_dt  = datetime.fromisoformat(opened_str)
                    if opened_dt.tzinfo is None:
                        opened_dt = opened_dt.replace(tzinfo=timezone.utc)
                    days_open = (datetime.now(timezone.utc) - opened_dt).total_seconds() / 86400.0
                    if days_open >= self.force_close_max_days:
                        reason = (
                            f"MAX_DAYS_OPEN | "
                            f"open={days_open:.1f}d >= limit={self.force_close_max_days:.0f}d"
                        )
                        logger.warning(f"[FORCE_CLOSE] {position.get('symbol')} | {reason}")
                        return True, reason
            except Exception as _e:
                logger.warning(f"[FORCE_CLOSE] days_check_fail | err={_e}")

        # ── 2. MAX DRAWDOWN ───────────────────────────────────────────
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


_tp_sl_mgr: Optional[DCATpSlManager] = None


def get_tp_sl_manager() -> DCATpSlManager:
    global _tp_sl_mgr
    if _tp_sl_mgr is None:
        _tp_sl_mgr = DCATpSlManager()
    return _tp_sl_mgr
