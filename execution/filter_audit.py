from __future__ import annotations

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict


@dataclass
class FilterStats:
    total_signals: int = 0
    buy_signals: int = 0
    blocked_signals: int = 0

    trend_fail: int = 0
    rsi_fail: int = 0
    extension_fail: int = 0


class FilterAuditEngine:

    def __init__(self):
        self.stats = FilterStats()

    def process_signal(
        self,
        trend_ok: bool,
        rsi_ok: bool,
        extension_ok: bool,
        action: str,
        pnl: float | None = None,
    ) -> None:

        self.stats.total_signals += 1

        if action == "BUY":
            self.stats.buy_signals += 1
        else:
            self.stats.blocked_signals += 1

        if not trend_ok:
            self.stats.trend_fail += 1
        if not rsi_ok:
            self.stats.rsi_fail += 1
        if not extension_ok:
            self.stats.extension_fail += 1

    def summary(self) -> Dict[str, float]:
        total = max(self.stats.total_signals, 1)

        return {
            "total_signals": self.stats.total_signals,
            "buy_signals": self.stats.buy_signals,
            "blocked_signals": self.stats.blocked_signals,
            "trend_fail_%": 100 * self.stats.trend_fail / total,
            "rsi_fail_%": 100 * self.stats.rsi_fail / total,
            "extension_fail_%": 100 * self.stats.extension_fail / total,
        }

    def print_report(self) -> None:
        report = self.summary()

        print("\n===== FILTER AUDIT REPORT =====")
        for k, v in report.items():
            print(f"{k}: {v}")
        print("================================\n")
