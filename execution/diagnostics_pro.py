# diagnostics_pro.py

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

@dataclass
class CheckResult:
    name: str
    ok: bool
    msg: str = ""
    severity: str = "INFO"

@dataclass
class Report:
    results: List[CheckResult] = field(default_factory=list)

    def add(self, name, ok, msg="", severity="INFO"):
        self.results.append(CheckResult(name, ok, msg, severity))

    def print(self):
        print("\n===== PRO DIAGNOSTIC REPORT =====\n")
        for r in self.results:
            icon = "✔" if r.ok else "❌"
            print(f"{icon} [{r.severity}] {r.name} -> {r.msg}")
        print("\n==============================\n")

class Adapter:
    def get_trade(self, signal_id): raise NotImplementedError
    def get_oco_status(self, link_id): raise NotImplementedError
    def get_order(self, order_id): raise NotImplementedError
    def get_position(self, symbol): raise NotImplementedError

def run_pro_diagnostics(adapter: Adapter, signal_id: str, link_id: str):
    rep = Report()

    trade = adapter.get_trade(signal_id)
    if not trade:
        rep.add("TRADE_EXISTS", False, "trade missing", "CRITICAL")
        rep.print()
        return

    tp = adapter.get_order(trade.get("tp_order_id"))
    sl = adapter.get_order(trade.get("sl_order_id"))
    pos = adapter.get_position(trade.get("symbol"))

    rep.add("ORDER_LINK", tp is not None and sl is not None)
    rep.add("POSITION_SYNC", pos.get("qty", 0) == 0 if trade.get("status") in ["CLOSED_TP","CLOSED_SL"] else True)

    rep.print()
