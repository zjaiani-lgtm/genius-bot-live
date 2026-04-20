# execution/dca_tp_sl_manager.py
# ============================================================
# DCA TP/SL Manager — TP გამოთვლა avg_entry-დან.
# SL=999% (გათიშული), Breakeven=გათიშული, ForceClose=გათიშული.
#
# ENV პარამეტრები:
#   DCA_TP_PCT=0.55   ← L1-L2 TP პროცენტი
#   DCA_SL_PCT=999.0  ← გათიშული (DCA ფილოსოფია)
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

        logger.info(
            f"[DCA] DCATpSlManager init | TP={self.tp_pct}% SL={self.sl_pct}%"
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
        DCA სტრატეგია: Force close სრულად გათიშულია.
        ბოტი ინახავს პოზიციას სანამ TP-ს არ მიაღწევს.
        არავითარი იძულებითი გაყიდვა არ არის.
        """
        # DCA: force close disabled — hold until TP
        return False, "OK"


# module-level singleton
_tp_sl_mgr: Optional[DCATpSlManager] = None


def get_tp_sl_manager() -> DCATpSlManager:
    global _tp_sl_mgr
    if _tp_sl_mgr is None:
        _tp_sl_mgr = DCATpSlManager()
    return _tp_sl_mgr
