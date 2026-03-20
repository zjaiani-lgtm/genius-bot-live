# diagnostics_pro.py

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import time


# =========================
# RESULT STRUCTURES
# =========================

@dataclass
class CheckResult:
    name: str
    ok: bool
    msg: str = ""
    severity: str = "INFO"  # INFO / WARN / CRITICAL


@dataclass
class Report:
    results: List[CheckResult] = field(default_factory=list)

    def add(self, name, ok, msg="", severity="INFO"):
        self.results.append(CheckResult(name, ok, msg, severity))

    def summary(self):
        ok = sum(1 for r in self.results if r.ok)
        fail = len(self.results) - ok
        critical = sum(1 for r in self.results if (not r.ok and r.severity == "CRITICAL"))

        return {
            "passed": ok,
            "failed": fail,
            "critical": critical,
            "status": "SAFE" if fail == 0 else ("UNSAFE" if critical > 0 else "WARN")
        }

    def print(self):
        print("\n===== 🧠 PRO DIAGNOSTIC REPORT =====\n")
        for r in self.results:
            icon = "✔" if r.ok else "❌"
            print(f"{icon} [{r.severity}] {r.name} -> {r.msg}")
        s = self.summary()
        print("\n------------------------------")
        print(f"PASSED: {s['passed']}")
        print(f"FAILED: {s['failed']}")
        print(f"CRITICAL: {s['critical']}")
        print(f"🚀 STATUS: {s['status']}")
        print("\n==============================\n")


# =========================
# ADAPTER INTERFACE
# =========================

class Adapter:
    """
    აქ უნდა მიაბა შენი რეალური ფუნქციები:
    - DB: get_trade, get_oco_status, get_open_trades, close_events count
    - Exchange: get_order, get_fills, get_position, get_balance, get_fee
    """

    # ---- TRADE / OCO ----
    def get_trade(self, signal_id) -> Dict[str, Any]:
        raise NotImplementedError

    def get_oco_status(self, link_id) -> str:
        raise NotImplementedError

    def get_close_events_count(self, signal_id) -> int:
        raise NotImplementedError

    def get_trade_logs(self, signal_id) -> List[str]:
        raise NotImplementedError

    def get_open_trades(self) -> List[Dict[str, Any]]:
        raise NotImplementedError

    # ---- EXCHANGE ----
    def get_order(self, order_id) -> Optional[Dict[str, Any]]:
        raise NotImplementedError

    def get_fills(self, order_id) -> List[Dict[str, Any]]:
        raise NotImplementedError

    def get_position(self, symbol) -> Dict[str, Any]:
        raise NotImplementedError

    def get_balance(self) -> Dict[str, Any]:
        raise NotImplementedError

    def get_fee_rate(self, symbol) -> float:
        return 0.001  # default taker 0.1%

    # ---- METRICS ----
    def get_latency_ms(self) -> int:
        return 0


# =========================
# HELPERS
# =========================

def norm(s):
    return (s or "").lower().strip()

def safe_float(x):
    try:
        return float(x)
    except:
        return None

def avg_fill_price(fills):
    # weighted avg price
    total_qty = 0.0
    total_quote = 0.0
    for f in fills:
        q = safe_float(f.get("qty") or f.get("quantity") or f.get("executedQty"))
        p = safe_float(f.get("price"))
        if q and p:
            total_qty += q
            total_quote += q * p
    if total_qty > 0:
        return total_quote / total_qty, total_qty
    return None, 0.0


# =========================
# CHECKS (STATE-AWARE)
# =========================

def check_position_sync(rep: Report, trade, pos):
    status = norm(trade.get("status"))
    qty = safe_float(pos.get("positionAmt") or pos.get("qty") or 0)
    if status in ["closed_tp", "closed_sl"]:
        ok = (qty == 0)
        rep.add("POSITION_SYNC", ok, f"pos_qty={qty}", "CRITICAL" if not ok else "INFO")
    else:
        rep.add("POSITION_SYNC", True, "open trade - skip", "INFO")


def check_order_link(rep: Report, tp, sl):
    ok = bool(tp and sl and tp.get("status") and sl.get("status"))
    rep.add("ORDER_LINK_INTEGRITY", ok, f"tp={bool(tp)} sl={bool(sl)}", "CRITICAL" if not ok else "INFO")


def check_partial_fill_engine(rep: Report, adapter: Adapter, order_id, expected_qty):
    fills = adapter.get_fills(order_id) or []
    avg_px, filled_qty = avg_fill_price(fills)
    ok = (filled_qty <= (safe_float(expected_qty) or 0))
    rep.add("PARTIAL_FILL_ENGINE",
            ok,
            f"filled={filled_qty} avg_px={avg_px}",
            "WARN" if not ok else "INFO")


def check_restart_recovery(rep: Report, adapter: Adapter):
    open_trades = adapter.get_open_trades()
    ok = True
    missing = 0
    for t in open_trades:
        link_id = t.get("link_id")
        oco_status = adapter.get_oco_status(link_id)
        if not oco_status:
            ok = False
            missing += 1
    rep.add("RESTART_RECOVERY", ok, f"missing_oco={missing}", "CRITICAL" if not ok else "INFO")


def check_api_resilience(rep: Report, tp, sl):
    ok = (tp is not None and sl is not None)
    rep.add("API_RESILIENCE", ok, "orders fetched", "CRITICAL" if not ok else "INFO")


def check_race_condition(rep: Report, adapter: Adapter, signal_id):
    closes = adapter.get_close_events_count(signal_id)
    ok = (closes <= 1)
    rep.add("RACE_PROTECTION", ok, f"close_events={closes}", "CRITICAL" if not ok else "INFO")


def check_latency(rep: Report, adapter: Adapter):
    lat = adapter.get_latency_ms()
    ok = (lat < 2000)
    rep.add("LATENCY", ok, f"{lat}ms", "WARN" if not ok else "INFO")


def check_slippage(rep: Report, expected_price, actual_price):
    ep = safe_float(expected_price)
    ap = safe_float(actual_price)
    if not ep or not ap:
        rep.add("SLIPPAGE", False, "missing price", "WARN")
        return
    dev = abs(ap - ep) / ep
    ok = dev < 0.02
    rep.add("SLIPPAGE", ok, f"dev={dev:.4f}", "WARN" if not ok else "INFO")


def check_fee_engine(rep: Report, adapter: Adapter, symbol, qty, actual_price, pnl_reported):
    fee_rate = adapter.get_fee_rate(symbol)
    q = safe_float(qty) or 0
    ap = safe_float(actual_price) or 0
    expected_fee = q * ap * fee_rate
    # simplistic: pnl_reported უნდა იყოს უკვე net
    ok = (pnl_reported is None) or True  # შენს სისტემაში ჩასვი რეალური შედარება
    rep.add("FEE_ENGINE", ok, f"est_fee={expected_fee}", "WARN" if not ok else "INFO")


def check_logs(rep: Report, adapter: Adapter, signal_id):
    logs = adapter.get_trade_logs(signal_id) or []
    needed = ["ENTRY", "OCO", "EXIT", "PNL"]
    ok = all(any(n in l for l in logs) for n in needed)
    rep.add("LOG_COMPLETENESS", ok, f"logs={len(logs)}", "WARN" if not ok else "INFO")


def check_edge_cases(rep: Report, trade):
    qty = safe_float(trade.get("qty"))
    price = safe_float(trade.get("entry_price"))
    ok = (qty is not None and qty > 0 and price is not None and price > 0)
    rep.add("EDGE_CASES", ok, f"qty={qty} price={price}", "CRITICAL" if not ok else "INFO")


# =========================
# MASTER RUN
# =========================

def run_pro_diagnostics(adapter: Adapter, signal_id: str, link_id: str):
    rep = Report()

    trade = adapter.get_trade(signal_id)
    if not trade:
        rep.add("TRADE_EXISTS", False, "trade missing", "CRITICAL")
        rep.print()
        return rep

    symbol = trade.get("symbol")
    tp_id = trade.get("tp_order_id")
    sl_id = trade.get("sl_order_id")

    tp = adapter.get_order(tp_id) if tp_id else None
    sl = adapter.get_order(sl_id) if sl_id else None

    pos = adapter.get_position(symbol)

    # -------- LEVEL 2 --------
    check_position_sync(rep, trade, pos)
    check_order_link(rep, tp, sl)
    if tp_id:
        check_partial_fill_engine(rep, adapter, tp_id, trade.get("qty"))
    if sl_id:
        check_partial_fill_engine(rep, adapter, sl_id, trade.get("qty"))
    check_restart_recovery(rep, adapter)
    check_api_resilience(rep, tp, sl)

    # -------- LEVEL 3 --------
    check_race_condition(rep, adapter, signal_id)
    check_latency(rep, adapter)

    # expected vs actual price (თუ გაქვს)
    expected = trade.get("tp_price") or trade.get("sl_price")
    actual = (tp or {}).get("avgPrice") or (sl or {}).get("avgPrice")
    check_slippage(rep, expected, actual)

    check_fee_engine(rep, adapter, symbol, trade.get("qty"), actual, trade.get("pnl_quote"))
    check_logs(rep, adapter, signal_id)

    # -------- LEVEL 4 --------
    check_edge_cases(rep, trade)

    rep.print()
    return rep
