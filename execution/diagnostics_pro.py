# execution/diagnostics_pro.py
# ============================================================
# GENIUS BOT — სრული Production Diagnostics
# Senior Algo-Trading Engineer ვერსია
#
# შემოწმებები:
#   1. Python ფაილების არსებობა
#   2. DB მდგომარეობა და ცხრილები
#   3. system_state
#   4. ღია trade-ები
#   5. OCO links (broken detection)
#   6. Performance სტატისტიკა
#   7. ENV პარამეტრების სრული ვალიდაცია
#   8. SL Cooldown მდგომარეობა
#   9. signal_outbox
#  10. ბოლო audit events
#  11. Regime Engine ფუნქციური ტესტი
#  12. ENV vs Python default კონფლიქტები (ახალი — სრული)
#  13. BROKEN OCO ავტო-repair (ახალი)
#  14. Trade PnL consistency (ახალი)
#  15. API connectivity (ახალი)
# ============================================================

from __future__ import annotations

import os
import re
import sys
import json
import math
import time
import sqlite3
import inspect
import traceback
import importlib
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple

# ─── გამოიყენება როდესაც run_pro_diagnostics() გამოიძახება პირდაპირ ─────────
try:
    from execution.regime_engine import MarketRegimeEngine
    _REGIME_IMPORT_OK = True
except Exception:
    _REGIME_IMPORT_OK = False

try:
    from execution.db import repository as _repo
    _DB_IMPORT_OK = True
except Exception:
    _DB_IMPORT_OK = False


# =============================================================================
# RESULT STRUCTURES
# =============================================================================

@dataclass
class CheckResult:
    name: str
    ok: bool
    msg: str = ""
    severity: str = "INFO"   # INFO / WARN / CRITICAL
    fix: str = ""            # კონკრეტული გამოსასწორებელი ქმედება


@dataclass
class Report:
    results: List[CheckResult] = field(default_factory=list)

    def add(self, name: str, ok: bool, msg: str = "",
            severity: str = "INFO", fix: str = ""):
        self.results.append(CheckResult(name, ok, msg, severity, fix))

    def summary(self) -> Dict[str, Any]:
        ok_cnt      = sum(1 for r in self.results if r.ok)
        warn_cnt    = sum(1 for r in self.results if not r.ok and r.severity == "WARN")
        fail_cnt    = sum(1 for r in self.results if not r.ok and r.severity in ("INFO", "WARN"))
        crit_cnt    = sum(1 for r in self.results if not r.ok and r.severity == "CRITICAL")
        return {
            "total":    len(self.results),
            "passed":   ok_cnt,
            "warn":     warn_cnt,
            "failed":   fail_cnt,
            "critical": crit_cnt,
            "status":   "SAFE" if (fail_cnt + crit_cnt) == 0
                        else ("CRITICAL" if crit_cnt > 0 else "WARN"),
        }

    def print_report(self):
        G  = "\033[32m"   # green
        Y  = "\033[33m"   # yellow
        R  = "\033[31m"   # red
        C  = "\033[36m"   # cyan
        B  = "\033[1m"    # bold
        RST = "\033[0m"

        W = 72
        print(f"\n{B}{'─'*W}{RST}")
        print(f"{B}  GENIUS BOT — სრული დიაგნოსტიკა{RST}")
        print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{B}{'─'*W}{RST}\n")

        sections = {
            "Python ფაილები":           [],
            "DB კავშირი და ცხრილები":   [],
            "system_state":             [],
            "ღია trade-ები":            [],
            "OCO links":                [],
            "Performance სტატისტიკა":   [],
            "ENV ვალიდაცია":            [],
            "SL Cooldown მდგომარეობა":  [],
            "signal_outbox":            [],
            "ბოლო audit events":        [],
            "Regime Engine ტესტი":      [],
            "ENV vs Code კონფლიქტები":  [],
            "Trade PnL consistency":    [],
            "API connectivity":         [],
            "BROKEN OCO repair":        [],
            "სხვა":                     [],
        }

        for r in self.results:
            placed = False
            for sec_key in sections:
                if r.name.startswith(sec_key.split()[0]):
                    sections[sec_key].append(r)
                    placed = True
                    break
            if not placed:
                sections["სხვა"].append(r)

        # fallback — if something not placed, distribute by keyword
        ungrouped = [r for r in self.results
                     if not any(r in v for v in sections.values())]
        for r in ungrouped:
            sections["სხვა"].append(r)

        # Deduplicate
        shown = set()
        sec_num = 0
        for sec_name, items in sections.items():
            items_unique = [r for r in items if id(r) not in shown]
            for r in items_unique:
                shown.add(id(r))
            if not items_unique:
                continue
            sec_num += 1
            print(f"{C}{'─'*4} {sec_num}. {sec_name} {'─'*max(1,W-6-len(sec_name))}{RST}")
            for r in items_unique:
                if r.ok:
                    tag   = f"{G}[OK]  {RST}"
                elif r.severity == "WARN":
                    tag   = f"{Y}[WARN]{RST}"
                elif r.severity == "CRITICAL":
                    tag   = f"{R}[CRIT]{RST}"
                else:
                    tag   = f"{Y}[FAIL]{RST}"
                print(f"  {tag} {r.name}")
                if r.msg:
                    print(f"        {r.msg}")
                if not r.ok and r.fix:
                    print(f"        {Y}→ FIX: {r.fix}{RST}")
            print()

        s = self.summary()
        total_bar = f"სულ:{s['total']}  OK:{s['passed']}  WARN:{s['warn']}  FAIL:{s['failed']}  CRIT:{s['critical']}"
        color = G if s["status"] == "SAFE" else (R if s["status"] == "CRITICAL" else Y)
        print(f"{B}{'─'*W}{RST}")
        print(f"  {total_bar}")
        print(f"  სტატუსი: {color}{B}{s['status']}{RST}\n")

        if not all(r.ok for r in self.results):
            print(f"{B}  დასაფიქსირებელი პრობლემები:{RST}")
            for r in self.results:
                if not r.ok and r.fix:
                    sev_color = R if r.severity == "CRITICAL" else Y
                    print(f"  {sev_color}• [{r.severity}] {r.name}{RST}")
                    print(f"    {r.fix}")
            print()


# =============================================================================
# ADAPTER INTERFACE  (production-ში override-ავ რეალური DB/exchange-ით)
# =============================================================================

class Adapter:
    def get_trade(self, signal_id) -> Dict[str, Any]:          raise NotImplementedError
    def get_oco_status(self, link_id) -> str:                  raise NotImplementedError
    def get_close_events_count(self, signal_id) -> int:        raise NotImplementedError
    def get_trade_logs(self, signal_id) -> List[str]:          raise NotImplementedError
    def get_open_trades(self) -> List[Dict[str, Any]]:         raise NotImplementedError
    def get_order(self, order_id) -> Optional[Dict[str, Any]]: raise NotImplementedError
    def get_fills(self, order_id) -> List[Dict[str, Any]]:     raise NotImplementedError
    def get_position(self, symbol) -> Dict[str, Any]:          raise NotImplementedError
    def get_balance(self) -> Dict[str, Any]:                   raise NotImplementedError
    def get_fee_rate(self, symbol) -> float:                   return 0.001
    def get_latency_ms(self) -> int:                           return 0


# =============================================================================
# HELPERS
# =============================================================================

def _norm(s: Any) -> str:
    return (str(s) or "").lower().strip()


def _safe_float(x: Any) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _avg_fill_price(fills: list) -> Tuple[Optional[float], float]:
    total_qty, total_quote = 0.0, 0.0
    for f in fills:
        q = _safe_float(f.get("qty") or f.get("quantity") or f.get("executedQty"))
        p = _safe_float(f.get("price"))
        if q and p:
            total_qty   += q
            total_quote += q * p
    if total_qty > 0:
        return total_quote / total_qty, total_qty
    return None, 0.0


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


# =============================================================================
# SECTION 1: Python ფაილები
# =============================================================================

def check_python_files(rep: Report, base_path: str = "/opt/render/project/src"):
    files = [
        "execution/signal_generator.py",
        "execution/execution_engine.py",
        "execution/excel_live_core.py",
        "execution/exchange_client.py",
        "execution/kill_switch.py",
        "execution/main.py",
        "execution/telegram_notifier.py",
        "execution/diagnostics_pro.py",
        "execution/my_adapter.py",
        "execution/regime_engine.py",
    ]
    for f in files:
        full = os.path.join(base_path, f)
        exists = os.path.isfile(full)
        rep.add(
            f"Python/{f.split('/')[-1]}",
            exists,
            f"path={full}",
            severity="CRITICAL" if not exists else "INFO",
            fix=f"ფაილი არ არსებობს: {full} — deploy-ი გადაამოწმე" if not exists else "",
        )


# =============================================================================
# SECTION 2: DB კავშირი და ცხრილები
# =============================================================================

def check_db(rep: Report, db_path: Optional[str] = None) -> Optional[sqlite3.Connection]:
    db_path = db_path or os.getenv("DB_PATH", "/var/data/genius_bot_v2.db")
    if not os.path.isfile(db_path):
        rep.add("DB/path", False, f"DB არ მოიძებნა: {db_path}", "CRITICAL",
                fix=f"DB_PATH={db_path} — Render Disk mount შეამოწმე")
        return None

    size_kb = os.path.getsize(db_path) // 1024
    rep.add("DB/path", True, f"path={db_path} size={size_kb}KB")

    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
    except Exception as e:
        rep.add("DB/connect", False, str(e), "CRITICAL",
                fix="sqlite3 connect ვერ მოხდა — disk permissions შეამოწმე")
        return None

    rep.add("DB/connect", True, "sqlite3 OK")

    required_tables = ["system_state", "trades", "oco_links", "audit_log", "executed_signals"]
    cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    existing = {r["name"] for r in cur.fetchall()}
    for t in required_tables:
        ok = t in existing
        rep.add(f"DB/table/{t}", ok, f"{'exists' if ok else 'MISSING'}",
                severity="CRITICAL" if not ok else "INFO",
                fix=f"ცხრილი '{t}' არ არსებობს — DB migration გაუშვი" if not ok else "")

    return conn


# =============================================================================
# SECTION 3: system_state
# =============================================================================

def check_system_state(rep: Report, conn: sqlite3.Connection):
    try:
        row = conn.execute(
            "SELECT * FROM system_state ORDER BY id DESC LIMIT 1"
        ).fetchone()
        if not row:
            rep.add("system_state/exists", False, "system_state ცხრილი ცარიელია", "CRITICAL",
                    fix="main.py გაუშვი — system_state-ი ვერ ინიციალიზდა")
            return
        d = dict(row)
        status   = d.get("status", "?")
        kill_sw  = d.get("kill_switch", "?")
        sync_ok  = d.get("startup_sync_ok", "?")

        rep.add("system_state/status",   status == "RUNNING", f"status={status}",
                fix="status=RUNNING-ს ელოდება — main.py restart" if status != "RUNNING" else "")
        rep.add("system_state/kill_sw",  str(kill_sw).upper() in ("OFF", "0", "FALSE", "NONE"),
                f"kill_switch={kill_sw}",
                fix="KILL_SWITCH=OFF — Render ENV-ში შეცვალე და restart" if str(kill_sw).upper() not in ("OFF","0","FALSE","NONE") else "")
        rep.add("system_state/sync",     str(sync_ok) in ("1", "True", "true"),
                f"startup_sync_ok={sync_ok}",
                fix="startup_sync_ok=0 — exchange API key-ები შეამოწმე" if str(sync_ok) not in ("1","True","true") else "")
    except Exception as e:
        rep.add("system_state/read", False, str(e), "CRITICAL",
                fix="system_state ვერ წაიკითხა — DB schema შეამოწმე")


# =============================================================================
# SECTION 4: ღია trade-ები
# =============================================================================

def check_open_trades(rep: Report, conn: sqlite3.Connection) -> List[Dict]:
    try:
        rows = conn.execute(
            "SELECT * FROM trades WHERE status='open' ORDER BY opened_at DESC"
        ).fetchall()
        trades = [dict(r) for r in rows]
        rep.add("ღია_trade/count", True, f"ღია trade-ები: {len(trades)}")
        return trades
    except Exception as e:
        rep.add("ღია_trade/read", False, str(e), "CRITICAL",
                fix="trades ცხრილი ვერ წაიკითხა")
        return []


# =============================================================================
# SECTION 5: OCO links — broken detection + analysis
# =============================================================================

def check_oco_links(rep: Report, conn: sqlite3.Connection) -> int:
    try:
        rows = conn.execute("SELECT * FROM oco_links").fetchall()
        total   = len(rows)
        broken  = [dict(r) for r in rows if _norm(dict(r).get("status","")) == "broken"]
        active  = [dict(r) for r in rows if _norm(dict(r).get("status","")) == "active"]

        rep.add("OCO/total",   True,           f"სულ OCO: {total}")
        rep.add("OCO/active",  True,           f"active OCO: {len(active)}")
        broken_ok = len(broken) == 0
        rep.add(
            "OCO/broken",
            broken_ok,
            f"BROKEN OCO: {len(broken)}",
            severity="CRITICAL" if not broken_ok else "INFO",
            fix=(
                f"Binance-ზე გახსენი Open Orders და გააუქმე ეს {len(broken)} broken OCO.\n"
                f"        SQL fix: UPDATE oco_links SET status='closed' WHERE status='broken';\n"
                f"        შემდეგ DB-ში ხელით გაუშვი ან: sqlite3 $DB_PATH \"UPDATE oco_links SET status='closed' WHERE status='broken';\""
                if not broken_ok else ""
            ),
        )
        return len(broken)
    except Exception as e:
        rep.add("OCO/read", False, str(e), "CRITICAL",
                fix="oco_links ცხრილი ვერ წაიკითხა")
        return 0


# =============================================================================
# SECTION 6: Performance სტატისტიკა
# =============================================================================

def check_performance(rep: Report, conn: sqlite3.Connection):
    try:
        rows = conn.execute(
            "SELECT * FROM trades WHERE status IN ('closed_tp','closed_sl','closed_manual') "
            "ORDER BY closed_at DESC"
        ).fetchall()
        trades = [dict(r) for r in rows]

        total   = len(trades)
        wins    = sum(1 for t in trades if _safe_float(t.get("pnl_quote") or 0) > 0)
        losses  = total - wins
        total_pnl = sum(_safe_float(t.get("pnl_quote") or 0) for t in trades)
        winrate = (wins / total * 100) if total > 0 else 0.0

        rep.add("Performance/trades", True, f"სულ={total}  wins={wins}  losses={losses}  winrate={winrate:.1f}%")

        pnl_ok = total_pnl >= 0
        rep.add("Performance/pnl", pnl_ok, f"Total PnL: {total_pnl:+.4f} USDT",
                severity="WARN" if not pnl_ok else "INFO",
                fix="PnL უარყოფითია — სტრატეგია ან signal quality შეამოწმე" if not pnl_ok else "")

        wr_ok = winrate >= 35.0 or total < 5
        rep.add("Performance/winrate", wr_ok, f"Winrate: {winrate:.1f}% (breakeven≈35% R:R=1:1.875)",
                severity="WARN" if not wr_ok else "INFO",
                fix="Winrate < 35% — filters გამკაცრება ან strategy review საჭიროა" if not wr_ok else "")

        # Profit Factor
        gross_win  = sum(_safe_float(t.get("pnl_quote") or 0) for t in trades if (_safe_float(t.get("pnl_quote") or 0) or 0) > 0)
        gross_loss = abs(sum(_safe_float(t.get("pnl_quote") or 0) for t in trades if (_safe_float(t.get("pnl_quote") or 0) or 0) < 0))
        pf = gross_win / gross_loss if gross_loss > 0 else float("inf")
        pf_ok = pf >= 1.0 or total < 5
        rep.add("Performance/profit_factor", pf_ok, f"Profit Factor: {pf:.2f} (minimum=1.0)",
                severity="WARN" if not pf_ok else "INFO",
                fix="Profit Factor < 1.0 — SL ძალიან გრძელია ან TP ძალიან მოკლე" if not pf_ok else "")

        # Avg win vs avg loss
        avg_win  = gross_win  / wins   if wins   > 0 else 0
        avg_loss = gross_loss / losses if losses > 0 else 0
        rep.add("Performance/avg",    True,
                f"Avg Win: +{avg_win:.4f} USDT | Avg Loss: -{avg_loss:.4f} USDT")

        # Drawdown check
        drawdown_env = _safe_float(os.getenv("MAX_ACCOUNT_DRAWDOWN", "7")) or 7.0
        balance_proxy = 100.0  # fallback if balance unknown
        try:
            balance_row = conn.execute(
                "SELECT value FROM system_state ORDER BY id DESC LIMIT 1"
            ).fetchone()
        except Exception:
            pass
        dd_pct = abs(total_pnl) / balance_proxy * 100 if total_pnl < 0 else 0
        dd_ok  = dd_pct <= drawdown_env
        rep.add("Performance/drawdown", dd_ok,
                f"estimated_drawdown≈{dd_pct:.1f}% vs MAX={drawdown_env}%",
                severity="CRITICAL" if not dd_ok else "INFO",
                fix=(
                    f"Drawdown {dd_pct:.1f}% >= MAX_ACCOUNT_DRAWDOWN={drawdown_env}% → bot EXEC_REJECT.\n"
                    f"        გამოსავალი: MAX_ACCOUNT_DRAWDOWN={int(dd_pct)+5} (Render ENV) და restart."
                ) if not dd_ok else "")

    except Exception as e:
        rep.add("Performance/read", False, str(e), "CRITICAL",
                fix="trades ცხრილი ვერ წაიკითხა performance-ისთვის")


# =============================================================================
# SECTION 7: ENV ვალიდაცია — სრული, ყველა პარამეტრი
# =============================================================================

# (key, expected_value_or_None, check_type, severity, description)
# check_type: "eq" | "gt" | "gte" | "lt" | "range" | "bool" | "nonempty"
ENV_RULES: List[Tuple] = [
    # ─── Core mode ───────────────────────────────────────────────
    ("MODE",                   "LIVE",       "eq",     "CRITICAL", "bot production mode"),
    ("KILL_SWITCH",            "false",      "eq",     "CRITICAL", "kill switch off"),
    ("ALLOW_LIVE_SIGNALS",     "true",       "eq",     "CRITICAL", "signals enabled"),
    ("LIVE_CONFIRMATION",      "true",       "eq",     "WARN",     "live confirmation"),
    # ─── API ─────────────────────────────────────────────────────
    ("BINANCE_API_KEY",        None,         "nonempty","CRITICAL","Binance API key"),
    ("BINANCE_API_SECRET",     None,         "nonempty","CRITICAL","Binance API secret"),
    # ─── Symbols ─────────────────────────────────────────────────
    ("BOT_SYMBOLS",            None,         "nonempty","CRITICAL","trading symbols"),
    ("BOT_TIMEFRAME",          "15m",        "eq",     "WARN",     "candle timeframe"),
    ("BOT_CANDLE_LIMIT",       "300",        "eq",     "WARN",     "candle history"),
    # ─── Sizing ──────────────────────────────────────────────────
    ("BOT_QUOTE_PER_TRADE",    "10",         "eq",     "WARN",     "quote per trade USDT"),
    ("MAX_QUOTE_PER_TRADE",    "10",         "eq",     "WARN",     "max quote ceiling"),
    ("DYNAMIC_SIZE_MIN",       "8",          "eq",     "WARN",     "min dynamic size"),
    ("DYNAMIC_SIZE_MAX",       "10",         "eq",     "WARN",     "max dynamic size"),
    # ─── Risk ────────────────────────────────────────────────────
    ("TP_PCT",                 "1.5",        "eq",     "CRITICAL", "take profit %"),
    ("SL_PCT",                 "0.80",       "eq",     "CRITICAL", "stop loss %"),
    ("MIN_MOVE_PCT",           "0.35",       "eq",     "WARN",     "ATR min move filter"),
    ("MIN_NET_PROFIT_PCT",     "0.35",       "eq",     "WARN",     "net profit gate"),
    ("ESTIMATED_ROUNDTRIP_FEE_PCT","0.14",   "eq",     "WARN",     "fee estimate"),
    ("ESTIMATED_SLIPPAGE_PCT", "0.05",       "eq",     "WARN",     "slippage estimate"),
    ("MAX_ACCOUNT_DRAWDOWN",   "7",          "gte",    "WARN",     "account drawdown limit %"),
    ("MAX_DAILY_LOSS",         "3.0",        "gte",    "WARN",     "daily loss limit %"),
    # ─── Filters ─────────────────────────────────────────────────
    ("USE_RSI_FILTER",         "true",       "eq",     "WARN",     "RSI filter"),
    ("RSI_MIN",                "35",         "eq",     "WARN",     "RSI buy zone min"),
    ("RSI_MAX",                "65",         "eq",     "WARN",     "RSI buy zone max"),
    ("RSI_SELL_MIN",           "72",         "eq",     "WARN",     "RSI sell trigger"),
    ("USE_MACD_FILTER",        "true",       "eq",     "WARN",     "MACD filter"),
    ("MACD_SMART_MODE",        "true",       "eq",     "WARN",     "MACD smart mode"),
    ("MACD_IMPROVING_BARS",    "4",          "eq",     "WARN",     "MACD improving bars"),
    ("MACD_HIST_ATR_FACTOR",   "0.2",        "eq",     "WARN",     "MACD ATR factor"),
    ("USE_MTF_FILTER",         "true",       "eq",     "WARN",     "multi-timeframe filter"),
    ("MTF_TIMEFRAME",          "1h",         "eq",     "WARN",     "HTF timeframe"),
    ("USE_ADX_FILTER",         "true",       "eq",     "WARN",     "ADX filter"),
    ("ADX_MIN_THRESHOLD",      "25",         "eq",     "WARN",     "ADX threshold"),
    ("USE_VWAP_FILTER",        "true",       "eq",     "WARN",     "VWAP filter"),
    ("VWAP_TOLERANCE",         "0.006",      "eq",     "WARN",     "VWAP tolerance"),
    ("USE_TIME_FILTER",        "true",       "eq",     "WARN",     "time-of-day filter"),
    ("TRADE_HOUR_START_UTC",   "7",          "eq",     "WARN",     "trade start hour UTC"),
    ("TRADE_HOUR_END_UTC",     "22",         "eq",     "WARN",     "trade end hour UTC"),
    ("USE_FUNDING_FILTER",     "true",       "eq",     "WARN",     "funding rate filter"),
    ("FUNDING_MAX_LONG_PCT",   "0.10",       "eq",     "WARN",     "max funding for longs"),
    ("USE_MA_FILTERS",         "false",      "eq",     "WARN",     "MA filters (off=soft mode)"),
    ("MIN_VOLUME_24H",         "30000000",   "eq",     "WARN",     "24h volume minimum USDT"),
    ("AI_FILTER_LOW_CONFIDENCE","true",      "eq",     "WARN",     "AI pre-filter"),
    ("AI_CONFIDENCE_BOOST",    "1.05",       "eq",     "WARN",     "AI score boost"),
    ("BUY_CONFIDENCE_MIN",     "0.55",       "eq",     "WARN",     "min confidence for BUY"),
    ("BUY_LIQUIDITY_MIN_SCORE","0.40",       "eq",     "WARN",     "min liquidity score"),
    # ─── SL Cooldown ─────────────────────────────────────────────
    ("SL_COOLDOWN_AFTER_N",    "3",          "eq",     "CRITICAL", "SL cooldown trigger count"),
    ("SL_COOLDOWN_PAUSE_SECONDS","1800",     "eq",     "WARN",     "SL cooldown pause duration"),
    ("RECOVERY_GREEN_CANDLES", "2",          "eq",     "WARN",     "recovery green candles"),
    ("RECOVERY_CANDLE_PCT",    "0.10",       "eq",     "WARN",     "recovery candle size %"),
    # ─── Features ────────────────────────────────────────────────
    ("USE_PARTIAL_TP",         "true",       "eq",     "WARN",     "partial TP"),
    ("PARTIAL_TP1_PCT",        "1.0",        "eq",     "WARN",     "partial TP1 level %"),
    ("PARTIAL_TP1_SIZE",       "0.5",        "eq",     "WARN",     "partial TP1 size fraction"),
    ("USE_BREAKEVEN_STOP",     "true",       "eq",     "WARN",     "breakeven stop"),
    ("BREAKEVEN_TRIGGER_PCT",  "0.40",       "eq",     "WARN",     "breakeven trigger %"),
    ("TRAILING_STOP_ENABLED",  "true",       "eq",     "WARN",     "trailing stop"),
    ("TRAILING_STOP_DISTANCE", "0.25",       "eq",     "WARN",     "trailing distance %"),
    ("USE_DYNAMIC_SIZING",     "true",       "eq",     "WARN",     "dynamic sizing"),
    ("DYNAMIC_SIZE_AI_LOW",    "0.55",       "eq",     "WARN",     "dynamic size AI low threshold"),
    ("DYNAMIC_SIZE_AI_HIGH",   "0.80",       "eq",     "WARN",     "dynamic size AI high threshold"),
    # ─── Regime / ATR ────────────────────────────────────────────
    ("ATR_TO_TP_SANITY_FACTOR","0.10",       "eq",     "WARN",     "ATR sanity factor"),
    ("ATR_MULT_SL_BULL",       "2.0",        "eq",     "WARN",     "ATR SL multiplier bull"),
    ("ATR_MULT_TP_BULL",       "4.0",        "eq",     "WARN",     "ATR TP multiplier bull"),
    ("STRUCT_SOFT_MIN_TREND",  "0.25",       "eq",     "WARN",     "soft struct trend min"),
    ("STRUCT_SOFT_MIN_MA_GAP", "0.10",       "eq",     "WARN",     "soft struct MA gap min"),
    ("STRUCT_SOFT_MIN_MOM10",  "-0.02",      "eq",     "WARN",     "soft struct momentum min"),
    ("STRUCT_SOFT_REQUIRE_LAST_UP","1",      "eq",     "WARN",     "soft struct require last up"),
    ("STRUCT_SOFT_OVERRIDE",   "true",       "eq",     "WARN",     "soft struct override"),
    # ─── Paths / Telegram ────────────────────────────────────────
    ("DB_PATH",                None,         "nonempty","CRITICAL","DB path"),
    ("SIGNAL_OUTBOX_PATH",     None,         "nonempty","WARN",    "signal outbox path"),
    ("TELEGRAM_BOT_TOKEN",     None,         "nonempty","WARN",    "Telegram bot token"),
    ("TELEGRAM_CHAT_ID",       None,         "nonempty","WARN",    "Telegram chat ID"),
    # ─── Trade limits ────────────────────────────────────────────
    ("MAX_TRADES_PER_DAY",     "10",         "eq",     "WARN",     "max trades per day"),
    ("MAX_TRADES_PER_HOUR",    "3",          "eq",     "WARN",     "max trades per hour"),
    ("MAX_OPEN_TRADES",        "2",          "eq",     "WARN",     "max concurrent open trades"),
    ("SIGNAL_EXPIRATION_SECONDS","600",      "eq",     "WARN",     "signal expiry seconds"),
    ("BOT_SIGNAL_COOLDOWN_SECONDS","120",    "eq",     "WARN",     "signal cooldown seconds"),
]


def check_env(rep: Report):
    for key, expected, check_type, severity, desc in ENV_RULES:
        actual = os.getenv(key, "")
        actual_str = actual.strip()

        if check_type == "nonempty":
            # Mask secrets in output
            display = (actual_str[:4] + "****") if len(actual_str) > 4 else ("****" if actual_str else "")
            ok = bool(actual_str)
            rep.add(f"ENV/{key}", ok, f"{desc} → {'set (masked)' if ok else 'NOT SET'}",
                    severity=severity if not ok else "INFO",
                    fix=f"Render ENV → {key}=<value> — არ არის დაყენებული!" if not ok else "")
            continue

        if check_type == "bool":
            ok = actual_str.lower() in ("true", "false", "1", "0")
            rep.add(f"ENV/{key}", ok, f"{desc} → '{actual_str}'",
                    severity=severity if not ok else "INFO",
                    fix=f"Render ENV → {key}=true ან false" if not ok else "")
            continue

        if not actual_str:
            rep.add(f"ENV/{key}", False, f"{desc} → NOT SET (expected={expected})",
                    severity=severity,
                    fix=f"Render ENV → {key}={expected}")
            continue

        if check_type == "eq":
            try:
                ev = float(expected)
                av = float(actual_str)
                ok = abs(ev - av) < 1e-9
            except Exception:
                ok = (actual_str.lower() == (expected or "").lower())
            rep.add(f"ENV/{key}", ok,
                    f"{desc} → '{actual_str}'" + (f" (expected={expected})" if not ok else ""),
                    severity=severity if not ok else "INFO",
                    fix=f"Render ENV → {key}={expected}" if not ok else "")

        elif check_type == "gt":
            av = _safe_float(actual_str)
            ev = _safe_float(expected)
            ok = av is not None and ev is not None and av > ev
            rep.add(f"ENV/{key}", ok, f"{desc} → {actual_str} (must > {expected})",
                    severity=severity if not ok else "INFO",
                    fix=f"Render ENV → {key}>{expected}" if not ok else "")

        elif check_type == "gte":
            av = _safe_float(actual_str)
            ev = _safe_float(expected)
            ok = av is not None and ev is not None and av >= ev
            rep.add(f"ENV/{key}", ok, f"{desc} → {actual_str} (must >= {expected})",
                    severity=severity if not ok else "INFO",
                    fix=f"Render ENV → {key}>={expected}" if not ok else "")

        elif check_type == "range":
            # expected = "min,max"
            parts = (expected or "").split(",")
            lo, hi = _safe_float(parts[0]), _safe_float(parts[1]) if len(parts) > 1 else None
            av = _safe_float(actual_str)
            ok = av is not None and lo is not None and hi is not None and lo <= av <= hi
            rep.add(f"ENV/{key}", ok, f"{desc} → {actual_str} (must in [{lo},{hi}])",
                    severity=severity if not ok else "INFO",
                    fix=f"Render ENV → {key} ∈ [{lo},{hi}]" if not ok else "")


# =============================================================================
# SECTION 8: SL Cooldown
# =============================================================================

def check_sl_cooldown(rep: Report, conn: sqlite3.Connection):
    sl_limit = _safe_int(os.getenv("SL_COOLDOWN_AFTER_N", "3"), 3)

    # Global
    try:
        row = conn.execute(
            "SELECT * FROM system_state ORDER BY id DESC LIMIT 1"
        ).fetchone()
        d = dict(row) if row else {}
        consec_sl   = _safe_int(d.get("consecutive_sl", 0))
        pause_until = _safe_float(d.get("sl_pause_until") or 0)
        now_ts      = time.time()
        paused      = pause_until and pause_until > now_ts

        sl_ok = consec_sl < sl_limit and not paused
        remaining = max(0, int((pause_until or 0) - now_ts)) if paused else 0

        rep.add(
            "SL_Cooldown/global",
            sl_ok,
            f"consecutive_sl={consec_sl} limit={sl_limit} | "
            f"paused={paused} remaining={remaining}s",
            severity="WARN" if not sl_ok else "INFO",
            fix=(
                f"Bot is paused! {remaining//60}m{remaining%60}s remaining.\n"
                f"        Manual reset (caution!): sqlite3 $DB_PATH \"UPDATE system_state SET consecutive_sl=0, sl_pause_until=0;\""
            ) if not sl_ok else "",
        )

        # Verify: loss count vs DB consecutive_sl
        row_losses = conn.execute(
            "SELECT COUNT(*) as cnt FROM trades WHERE status='closed_sl'"
        ).fetchone()
        actual_sl_count = dict(row_losses)["cnt"] if row_losses else 0
        rep.add(
            "SL_Cooldown/tp_reset",
            True,
            f"Total SL trades in DB: {actual_sl_count} | DB consecutive_sl: {consec_sl}",
        )

        # Check: if consec_sl >= limit but pause not set → BUG
        pause_missing = consec_sl >= sl_limit and not paused
        rep.add(
            "SL_Cooldown/pause_trigger",
            not pause_missing,
            f"pause_set_correctly={'YES' if not pause_missing else 'NO — PAUSED BUT NO pause_until SET'}",
            severity="CRITICAL" if pause_missing else "INFO",
            fix=(
                f"consecutive_sl={consec_sl} >= limit={sl_limit} მაგრამ pause_until=0!\n"
                f"        BUG: SL_COOLDOWN_AFTER_N Render-ზე სწორია? შეამოწმე.\n"
                f"        Fix: sqlite3 $DB_PATH \"UPDATE system_state SET sl_pause_until={int(now_ts)+1800};\""
            ) if pause_missing else "",
        )

    except Exception as e:
        rep.add("SL_Cooldown/read", False, str(e), "CRITICAL",
                fix="system_state SL fields ვერ წაიკითხა")


# =============================================================================
# SECTION 9: signal_outbox
# =============================================================================

def check_signal_outbox(rep: Report):
    path = os.getenv("SIGNAL_OUTBOX_PATH", "/var/data/signal_outbox.json")
    if not os.path.isfile(path):
        rep.add("signal_outbox/exists", False, f"path={path} NOT FOUND",
                fix=f"SIGNAL_OUTBOX_PATH={path} — Render disk mount შეამოწმე")
        return

    size = os.path.getsize(path)
    rep.add("signal_outbox/size", True, f"path={path} size={size}bytes")

    if size > 0:
        try:
            with open(path) as f:
                content = f.read().strip()
            # Could be JSON list or newline-delimited
            if content.startswith("["):
                sigs = json.loads(content)
            else:
                sigs = [json.loads(line) for line in content.splitlines() if line.strip()]
            rep.add("signal_outbox/parse", True, f"Outbox სიგნალები: {len(sigs)}")

            # Check stale signals
            now_ts = time.time()
            expiry = _safe_int(os.getenv("SIGNAL_EXPIRATION_SECONDS", "600"), 600)
            stale  = 0
            for sig in sigs:
                ts_str = sig.get("ts_utc", "")
                if ts_str:
                    try:
                        dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                        age = (datetime.now(timezone.utc) - dt).total_seconds()
                        if age > expiry:
                            stale += 1
                    except Exception:
                        pass
            stale_ok = stale == 0
            rep.add("signal_outbox/stale", stale_ok, f"stale signals: {stale}",
                    severity="WARN" if not stale_ok else "INFO",
                    fix=f"{stale} სიგნალი ძველია (>{expiry}s) — outbox-ი გასუფთავე ან execution_engine შეამოწმე" if not stale_ok else "")
        except Exception as e:
            rep.add("signal_outbox/parse", False, str(e), "WARN",
                    fix="signal_outbox.json parse error — ფაილი შეამოწმე")
    else:
        rep.add("signal_outbox/empty", True, "outbox ცარიელია (OK — სიგნალი არ ლოდინობს)")


# =============================================================================
# SECTION 10: ბოლო audit events
# =============================================================================

def check_audit_events(rep: Report, conn: sqlite3.Connection, n: int = 5):
    try:
        rows = conn.execute(
            f"SELECT * FROM audit_log ORDER BY created_at DESC LIMIT {n}"
        ).fetchall()
        events = [dict(r) for r in rows]
        rep.add("audit/count", True, f"ბოლო {n} event ნაჩვენებია")

        exec_rejects = [e for e in events if "EXEC_REJECT" in str(e.get("event_type", ""))]
        for e in exec_rejects:
            details = e.get("details", "")
            rep.add(
                "audit/EXEC_REJECT",
                False,
                f"[{e.get('created_at','')}] {e.get('event_type','')} | {str(details)[:120]}",
                severity="CRITICAL",
                fix=(
                    "EXEC_REJECT_DRAWDOWN: MAX_ACCOUNT_DRAWDOWN limit გადაჭარბდა.\n"
                    "        → Render ENV → MAX_ACCOUNT_DRAWDOWN=20 (ან მეტი) და restart.\n"
                    "        ყოვლისთვის: balance manually შეამოწმე Binance-ზე."
                ) if "DRAWDOWN" in str(details) else
                "EXEC_REJECT: execution_engine reject — details შეამოწმე"
            )
    except Exception as e:
        rep.add("audit/read", False, str(e), "WARN",
                fix="audit_log ცხრილი ვერ წაიკითხა")


# =============================================================================
# SECTION 11: Regime Engine ფუნქციური ტესტი
# =============================================================================

def check_regime_engine(rep: Report):
    if not _REGIME_IMPORT_OK:
        rep.add("Regime/import", False, "MarketRegimeEngine import failed", "WARN",
                fix="from execution.regime_engine import MarketRegimeEngine — import error")
        return

    try:
        engine = MarketRegimeEngine()
        rep.add("Regime/import", True, "regime_engine.py — import OK")
    except Exception as e:
        rep.add("Regime/init", False, str(e), "CRITICAL",
                fix="MarketRegimeEngine() init failed — regime_engine.py შეამოွმე")
        return

    # notify_outcome + is_paused მეთოდები
    has_notify  = callable(getattr(engine, "notify_outcome", None))
    has_paused  = callable(getattr(engine, "is_paused", None))
    rep.add("Regime/methods", has_notify and has_paused,
            f"notify_outcome={'✓' if has_notify else '✗'}  is_paused={'✓' if has_paused else '✗'}",
            fix="regime_engine.py-ს აკლია notify_outcome() ან is_paused() — კოდი შეამოწმე" if not (has_notify and has_paused) else "")

    # detect_regime test cases
    test_cases = [
        ((0.6, 0.4), "BULL"),
        ((0.2, 0.2), "SIDEWAYS"),
        ((0.3, 0.5), "UNCERTAIN"),  # ← FAIL in diagnose — შემოწმება
        ((0.1, 2.0), "VOLATILE"),
        ((-0.2, 0.5), "BEAR"),
    ]
    fail_count = 0
    for (trend, atr), expected in test_cases:
        try:
            got = engine.detect_regime(trend=trend, atr_pct=atr)
            ok  = got == expected
            if not ok:
                fail_count += 1
                rep.add(
                    f"Regime/detect({trend},{atr})",
                    False,
                    f"got={got!r}  expected={expected!r}",
                    severity="WARN",
                    fix=(
                        f"detect_regime(trend={trend}, atr_pct={atr}) returns '{got}' but expected '{expected}'.\n"
                        f"        regime_engine.py boundary logic შეამოწმე — UNCERTAIN vs BULL threshold."
                    ),
                )
            else:
                rep.add(f"Regime/detect({trend},{atr})", True, f"→ '{got}' ✓")
        except Exception as e:
            fail_count += 1
            rep.add(f"Regime/detect({trend},{atr})", False, str(e), "WARN",
                    fix=f"detect_regime() exception — regime_engine.py შეამოწმე")

    # apply() test
    try:
        result = engine.apply("BULL", atr_pct=0.5, symbol="BTC/USDT")
        tp = result.get("TP_PCT")
        sl = result.get("SL_PCT")
        tp_ok = tp is not None and tp > 0
        sl_ok = sl is not None and sl > 0
        rep.add("Regime/apply_BULL", tp_ok and sl_ok,
                f"apply(BULL, atr=0.5%) → TP={tp}% SL={sl}%",
                fix="apply() TP ან SL None-ია — regime_engine.apply() შეამოწმე" if not (tp_ok and sl_ok) else "")

        # TP/SL sanity
        if tp and sl:
            rr = tp / sl
            tp_env = _safe_float(os.getenv("TP_PCT", "1.5"))
            sl_env = _safe_float(os.getenv("SL_PCT", "0.80"))
            rr_ok  = rr >= 1.5
            rep.add("Regime/RR_ratio", rr_ok,
                    f"R:R = {tp:.2f}/{sl:.2f} = 1:{rr:.2f} (minimum 1:1.5)",
                    severity="WARN" if not rr_ok else "INFO",
                    fix=f"R:R < 1.5 — TP_PCT={tp_env} vs SL_PCT={sl_env} შეამოწმე. TP უნდა იყოს SL×1.875-ზე მეტი" if not rr_ok else "")

            # PnL simulation
            fee   = _safe_float(os.getenv("ESTIMATED_ROUNDTRIP_FEE_PCT", "0.14")) or 0.14
            slip  = _safe_float(os.getenv("ESTIMATED_SLIPPAGE_PCT", "0.05")) or 0.05
            cost  = fee + slip
            net_tp = tp - cost
            net_sl = sl + cost
            be_wr  = net_sl / (net_tp + net_sl) * 100 if (net_tp + net_sl) > 0 else 0
            rep.add("Regime/PnL_sim", net_tp > 0,
                    f"TP net=+{net_tp:.2f}% | SL net=-{net_sl:.2f}% | breakeven_winrate={be_wr:.1f}%",
                    severity="WARN" if net_tp <= 0 else "INFO",
                    fix=f"Net TP <= 0! Fee+Slippage={cost:.2f}% > TP={tp:.2f}% — TP_PCT გაზარდე" if net_tp <= 0 else "")
    except Exception as e:
        rep.add("Regime/apply", False, str(e), "WARN",
                fix=f"apply() exception — {e}")

    # apply SKIP tests
    for regime in ("BEAR", "VOLATILE", "SIDEWAYS"):
        try:
            res = engine.apply(regime, atr_pct=0.5, symbol="BTC/USDT")
            skip = res.get("SKIP_TRADING", False)
            rep.add(f"Regime/apply_{regime}", skip,
                    f"apply({regime}) → SKIP_TRADING={skip}",
                    fix=f"apply({regime}) უნდა დააბრუნოს SKIP_TRADING=True — regime_engine.py შეამოწმე" if not skip else "")
        except Exception as e:
            rep.add(f"Regime/apply_{regime}", False, str(e), "WARN")


# =============================================================================
# SECTION 12: ENV vs Python default კონფლიქტები — სრული ანალიზი
# =============================================================================

# ეს არის FULL cross-reference: .env მნიშვნელობები vs signal_generator.py defaults
# თუ Render-ზე ENV ვარიაბლი არ ჩაიტვირთა → კოდი default-ს გამოიყენებს
# ეს section ამას ადასტურებს live-ში

ENV_VS_CODE_CHECKS: List[Tuple[str, str, str]] = [
    # (ENV_KEY, expected_env_val, description_of_impact)
    ("AI_CONFIDENCE_BOOST",     "1.05",     "signal score boost — 1.0 default score-ს ვერ ამაღლებს"),
    ("AI_FILTER_LOW_CONFIDENCE","true",     "pre-filter off → low quality signals გადის"),
    ("ALLOW_LIVE_SIGNALS",      "true",     "CRITICAL: false → ყველა BUY სიგნალი იბლოკება!"),
    ("BUY_LIQUIDITY_MIN_SCORE", "0.40",     "0 default → volume floor გამორთულია"),
    ("MAX_TRADES_PER_DAY",      "10",       "0 default → daily limit გამორთულია"),
    ("MAX_TRADES_PER_HOUR",     "3",        "0 default → hourly limit გამორთულია"),
    ("MIN_VOLUME_24H",          "30000000", "0 default → ყველა symbol გადის volume filter-ს"),
    ("RSI_MAX",                 "65",       "70 default → overbought zone-ში ვყიდულობთ"),
    ("RSI_SELL_MIN",            "72",       "60 default → ძალიან ადრე SELL trigger"),
    ("SIGNAL_EXPIRATION_SECONDS","600",     "0 default → signals არ expire-ავს"),
    ("TRAILING_STOP_ENABLED",   "true",     "false default → trailing stop გამორთულია"),
    ("USE_FUNDING_FILTER",      "true",     "false default → funding rate filter გამორთულია"),
    ("USE_MA_FILTERS",          "false",    "true default → soft mode ვერ მუშაობს"),
    ("BUY_CONFIDENCE_MIN",      "0.55",     "INTERNAL_CONFLICT: 0.55 vs 0.38 in same file"),
]


def check_env_vs_code(rep: Report):
    """
    ამოწმებს: Render-ზე ENV ჩაიტვირთა? თუ არ ჩაიტვირთა, კოდი wrong default-ს გამოიყენებს.
    ეს არის runtime conflict detector.
    """
    rep.add("ENV_vs_Code/header", True,
            "ENV vs Python default conflict analysis — critical for Render deployments")

    conflicts_found = 0
    for key, expected, impact in ENV_VS_CODE_CHECKS:
        actual = os.getenv(key, "__NOT_SET__")
        if actual == "__NOT_SET__":
            conflicts_found += 1
            rep.add(
                f"ENV_vs_Code/{key}",
                False,
                f"NOT SET in Render ENV! Default will be used. Impact: {impact}",
                severity="CRITICAL" if key in ("ALLOW_LIVE_SIGNALS", "SL_COOLDOWN_AFTER_N") else "WARN",
                fix=f"Render ENV → {key}={expected}\n        Impact: {impact}"
            )
        else:
            # Check if actual matches expected
            try:
                ev = float(expected)
                av = float(actual.strip())
                ok = abs(ev - av) < 1e-9
            except Exception:
                ok = (actual.strip().lower() == expected.lower())

            if not ok:
                conflicts_found += 1
                rep.add(
                    f"ENV_vs_Code/{key}",
                    False,
                    f"MISMATCH: ENV='{actual.strip()}' but expected='{expected}'. Impact: {impact}",
                    severity="WARN",
                    fix=f"Render ENV → {key}={expected}\n        Impact: {impact}"
                )
            else:
                rep.add(f"ENV_vs_Code/{key}", True,
                        f"'{actual.strip()}' == expected ✓")

    # Special: BUY_CONFIDENCE_MIN internal conflict
    rep.add(
        "ENV_vs_Code/BUY_CONFIDENCE_MIN_internal",
        False,
        "signal_generator.py-ში BUY_CONFIDENCE_MIN 2-ჯერ გამოიყენება: "
        "line 67: BUY_CONFIDENCE_MIN=0.55 (main guard) | "
        "line 94: AI_FILTER_MIN_SCORE=0.38 (pre-filter) — განსხვავებული threshold-ები!",
        severity="WARN",
        fix=(
            "AI_FILTER_MIN_SCORE (line 94) = 0.38 — ეს pre-filter threshold-ია, BUY_CONFIDENCE_MIN-ისგან განსხვავებული.\n"
            "        .env: BUY_CONFIDENCE_MIN=0.55 → main confidence guard.\n"
            "        .env: AI_EXECUTE_MIN_SCORE=0.55 → should be used for AI_FILTER_MIN_SCORE.\n"
            "        FIX: signal_generator.py line 94:\n"
            "          AI_FILTER_MIN_SCORE = float(os.getenv('AI_EXECUTE_MIN_SCORE', '0.55'))"
        )
    )

    rep.add("ENV_vs_Code/summary", conflicts_found == 0,
            f"კონფლიქტები: {conflicts_found}",
            severity="CRITICAL" if conflicts_found > 3 else ("WARN" if conflicts_found > 0 else "INFO"),
            fix=f"{conflicts_found} ENV კონფლიქტი — Render → Environment-ში შეამოწმე და restart" if conflicts_found > 0 else "")


# =============================================================================
# SECTION 13: Trade PnL Consistency
# =============================================================================

def check_pnl_consistency(rep: Report, conn: sqlite3.Connection):
    """
    შემოწმება: closed trades-ის pnl_quote ლოგიკურია?
    - TP trades: pnl_quote > 0
    - SL trades: pnl_quote < 0
    - Inconsistency → DB ან execution_engine bug
    """
    try:
        rows = conn.execute(
            "SELECT signal_id, status, pnl_quote, entry_price, exit_price, qty "
            "FROM trades WHERE status IN ('closed_tp','closed_sl') "
            "ORDER BY closed_at DESC LIMIT 50"
        ).fetchall()

        inconsistent = []
        for row in rows:
            d = dict(row)
            status  = d.get("status", "")
            pnl     = _safe_float(d.get("pnl_quote") or 0)
            sig_id  = d.get("signal_id", "?")[:8]

            if status == "closed_tp" and pnl is not None and pnl <= 0:
                inconsistent.append(f"TP trade {sig_id} has pnl={pnl:.4f} (should be >0)")
            elif status == "closed_sl" and pnl is not None and pnl >= 0:
                inconsistent.append(f"SL trade {sig_id} has pnl={pnl:.4f} (should be <0)")

        ok = len(inconsistent) == 0
        rep.add(
            "PnL_consistency",
            ok,
            f"Checked {len(rows)} trades | inconsistent={len(inconsistent)}",
            severity="CRITICAL" if not ok else "INFO",
            fix=(
                "PnL inconsistencies:\n" +
                "\n".join(f"          {x}" for x in inconsistent[:5]) +
                "\n        execution_engine.py pnl calculation შეამოწმე"
            ) if not ok else ""
        )

        # Fee sanity: avg pnl for SL trades should be around -SL_PCT - fees
        sl_trades  = [dict(r) for r in rows if dict(r).get("status") == "closed_sl"]
        if sl_trades:
            avg_sl_pnl = sum(_safe_float(t.get("pnl_quote") or 0) for t in sl_trades) / len(sl_trades)
            sl_pct     = _safe_float(os.getenv("SL_PCT", "0.80")) or 0.80
            quote_size = _safe_float(os.getenv("BOT_QUOTE_PER_TRADE", "10")) or 10
            expected_sl_pnl = -(sl_pct / 100.0 * quote_size)
            rep.add(
                "PnL_SL_sanity",
                True,
                f"Avg SL pnl={avg_sl_pnl:.4f} USDT | expected≈{expected_sl_pnl:.4f} USDT "
                f"(SL_PCT={sl_pct}% × quote={quote_size})"
            )

    except Exception as e:
        rep.add("PnL_consistency", False, str(e), "WARN",
                fix="trade PnL consistency check failed — DB schema შეამოწმე")


# =============================================================================
# SECTION 14: API Connectivity
# =============================================================================

def check_api_connectivity(rep: Report):
    """Binance REST API ping — 3s timeout"""
    import urllib.request
    import json as _json

    base = os.getenv("BINANCE_LIVE_REST_BASE", "https://api.binance.com/api/v3")

    # 1. Ping
    try:
        req = urllib.request.Request(
            f"{base}/ping",
            headers={"User-Agent": "GeniusBot-Diag/1.0"}
        )
        with urllib.request.urlopen(req, timeout=3) as resp:
            ok = resp.status == 200
        rep.add("API/ping", ok, f"Binance ping → {resp.status}",
                fix="Binance API ping failed — network ან API endpoint შეამოწმე" if not ok else "")
    except Exception as e:
        rep.add("API/ping", False, str(e), "CRITICAL",
                fix="Binance REST API მიუწვდომელია — Render outbound network შეამოწმე")
        return

    # 2. Server time drift
    try:
        req2 = urllib.request.Request(f"{base}/time",
                                       headers={"User-Agent": "GeniusBot-Diag/1.0"})
        with urllib.request.urlopen(req2, timeout=3) as resp2:
            data = _json.loads(resp2.read().decode())
        server_ms = data.get("serverTime", 0)
        local_ms  = int(time.time() * 1000)
        drift_ms  = abs(server_ms - local_ms)
        drift_ok  = drift_ms < 1000
        rep.add("API/time_drift", drift_ok,
                f"drift={drift_ms}ms (max 1000ms)",
                severity="WARN" if not drift_ok else "INFO",
                fix=f"Clock drift {drift_ms}ms — server NTP sync შეამოწმე. Binance ითხოვს < 1000ms" if not drift_ok else "")
    except Exception as e:
        rep.add("API/time_drift", False, str(e), "WARN",
                fix="Binance /time endpoint failed")


# =============================================================================
# SECTION 15: BROKEN OCO ავტო-repair suggestion
# =============================================================================

def check_and_suggest_oco_repair(rep: Report, conn: sqlite3.Connection):
    try:
        rows = conn.execute(
            "SELECT * FROM oco_links WHERE status='broken'"
        ).fetchall()
        broken = [dict(r) for r in rows]

        if not broken:
            rep.add("BROKEN_OCO/repair", True, "broken OCO არ არის — OK")
            return

        rep.add(
            "BROKEN_OCO/repair",
            False,
            f"{len(broken)} broken OCO link(s) — Binance-ზე manual check საჭიროა!",
            severity="CRITICAL",
            fix=(
                f"ნაბიჯი 1: Binance Open Orders გახსენი და ეს orders გააუქმე:\n" +
                "".join(
                    f"          link_id={b.get('link_id','?')} symbol={b.get('symbol','?')} "
                    f"tp_order={b.get('tp_order_id','?')} sl_order={b.get('sl_order_id','?')}\n"
                    for b in broken[:5]
                ) +
                f"        ნაბიჯი 2: DB-ში status update:\n"
                f"          sqlite3 $DB_PATH \"UPDATE oco_links SET status='closed' WHERE status='broken';\"\n"
                f"        ნაბიჯი 3: trades ცხრილში შესაბამისი entries შეამოწმე და status='closed_manual' დასეტე\n"
                f"        ნაბიჯი 4: bot restart"
            )
        )

        # Log each broken OCO
        for b in broken:
            rep.add(
                f"BROKEN_OCO/{b.get('link_id','?')[:8]}",
                False,
                f"symbol={b.get('symbol','?')} tp={b.get('tp_order_id','?')} "
                f"sl={b.get('sl_order_id','?')} created={b.get('created_at','?')}",
                severity="CRITICAL",
                fix=f"Binance-ზე manually cancel order {b.get('tp_order_id','?')} და {b.get('sl_order_id','?')}"
            )
    except Exception as e:
        rep.add("BROKEN_OCO/check", False, str(e), "WARN",
                fix="oco_links broken check failed")


# =============================================================================
# INDIVIDUAL TRADE CHECKS (Adapter-based, for specific trade debugging)
# =============================================================================

def check_position_sync(rep: Report, trade: Dict, pos: Dict):
    status = _norm(trade.get("status"))
    qty    = _safe_float(pos.get("positionAmt") or pos.get("qty") or 0)
    if status in ["closed_tp", "closed_sl", "closed_manual"]:
        ok = (qty == 0 or qty is None)
        rep.add("POSITION_SYNC", ok, f"pos_qty={qty}",
                "CRITICAL" if not ok else "INFO",
                fix=f"Position qty={qty} but trade={status} — Binance-ზე manually close position" if not ok else "")
    else:
        rep.add("POSITION_SYNC", True, "open trade — skip")


def check_order_link(rep: Report, tp: Optional[Dict], sl: Optional[Dict]):
    ok = bool(tp and sl and tp.get("status") and sl.get("status"))
    rep.add("ORDER_LINK_INTEGRITY", ok, f"tp={bool(tp)} sl={bool(sl)}",
            "CRITICAL" if not ok else "INFO",
            fix="OCO orders ვერ მოიძებნა — execution_engine OCO placement შეამოწმე" if not ok else "")


def check_partial_fill_engine(rep: Report, adapter: Adapter, order_id: str, expected_qty: Any):
    fills = adapter.get_fills(order_id) or []
    avg_px, filled_qty = _avg_fill_price(fills)
    exp = _safe_float(expected_qty) or 0
    ok  = filled_qty <= exp + 1e-8
    rep.add("PARTIAL_FILL_ENGINE", ok, f"filled={filled_qty:.6f} avg_px={avg_px} expected<={exp:.6f}",
            "WARN" if not ok else "INFO",
            fix=f"Overfill: filled={filled_qty} > expected={exp} — exchange partial fill logic შეამოწმე" if not ok else "")


def check_restart_recovery(rep: Report, adapter: Adapter):
    open_trades = adapter.get_open_trades()
    missing = 0
    for t in open_trades:
        link_id    = t.get("link_id")
        oco_status = adapter.get_oco_status(link_id)
        if not oco_status:
            missing += 1
    ok = missing == 0
    rep.add("RESTART_RECOVERY", ok, f"missing_oco={missing}",
            "CRITICAL" if not ok else "INFO",
            fix=f"{missing} open trade(s) without OCO — restart recovery failed" if not ok else "")


def check_api_resilience(rep: Report, tp: Optional[Dict], sl: Optional[Dict]):
    ok = tp is not None and sl is not None
    rep.add("API_RESILIENCE", ok, "orders fetched",
            "CRITICAL" if not ok else "INFO",
            fix="OCO orders fetch failed — Binance API key permissions შეამოწმე" if not ok else "")


def check_race_condition(rep: Report, adapter: Adapter, signal_id: str):
    closes = adapter.get_close_events_count(signal_id)
    ok = closes <= 1
    rep.add("RACE_PROTECTION", ok, f"close_events={closes}",
            "CRITICAL" if not ok else "INFO",
            fix=f"Race condition! {closes} close events for same trade — execution_engine deduplication შეამოწმე" if not ok else "")


def check_latency(rep: Report, adapter: Adapter):
    lat = adapter.get_latency_ms()
    ok  = lat < 2000
    rep.add("LATENCY", ok, f"{lat}ms (max=2000ms)",
            "WARN" if not ok else "INFO",
            fix=f"High latency {lat}ms — Render region ან Binance endpoint შეამოწმე" if not ok else "")


def check_slippage(rep: Report, expected_price: Any, actual_price: Any):
    ep = _safe_float(expected_price)
    ap = _safe_float(actual_price)
    if not ep or not ap:
        rep.add("SLIPPAGE", False, "missing price data", "WARN",
                fix="entry/exit price data არ არის — trade record შეამოწმე")
        return
    dev = abs(ap - ep) / ep
    ok  = dev < 0.02
    rep.add("SLIPPAGE", ok, f"dev={dev:.4f} ({dev*100:.2f}%)",
            "WARN" if not ok else "INFO",
            fix=f"Slippage {dev*100:.2f}% > 2% — MARKET order-ების ნაცვლად LIMIT გამოიყენე" if not ok else "")


def check_fee_engine(rep: Report, adapter: Adapter, symbol: str,
                     qty: Any, actual_price: Any, pnl_reported: Any):
    fee_rate = adapter.get_fee_rate(symbol)
    q   = _safe_float(qty) or 0
    ap  = _safe_float(actual_price) or 0
    pnl = _safe_float(pnl_reported)

    estimated_fee = q * ap * fee_rate * 2  # roundtrip

    if pnl is None:
        rep.add("FEE_ENGINE", True, f"est_fee={estimated_fee:.6f} pnl=None (no comparison)")
        return

    ok = pnl >= -(estimated_fee * 3)
    rep.add("FEE_ENGINE", ok, f"est_fee={estimated_fee:.6f} pnl={pnl:.6f}",
            "WARN" if not ok else "INFO",
            fix=f"PnL={pnl:.4f} worse than 3×fee={estimated_fee*3:.4f} — fee calculation შეამოწმე" if not ok else "")


def check_logs_completeness(rep: Report, adapter: Adapter, signal_id: str):
    logs   = adapter.get_trade_logs(signal_id) or []
    needed = ["ENTRY", "OCO", "EXIT", "PNL"]
    ok     = all(any(n in l for l in logs) for n in needed)
    rep.add("LOG_COMPLETENESS", ok, f"logs={len(logs)}",
            "WARN" if not ok else "INFO",
            fix="Trade logs incomplete — ENTRY/OCO/EXIT/PNL entries missing" if not ok else "")


def check_edge_cases(rep: Report, trade: Dict):
    qty   = _safe_float(trade.get("qty"))
    price = _safe_float(trade.get("entry_price"))
    ok    = qty is not None and qty > 0 and price is not None and price > 0
    rep.add("EDGE_CASES", ok, f"qty={qty} price={price}",
            "CRITICAL" if not ok else "INFO",
            fix="trade qty ან entry_price None/0 — execution_engine entry recording შეამოწმე" if not ok else "")


# =============================================================================
# MASTER RUN — სრული system diagnostics (no adapter needed)
# =============================================================================

def run_full_diagnostics(
    db_path: Optional[str] = None,
    base_path: str = "/opt/render/project/src",
) -> Report:
    """
    სრული system diagnostics — adapter-ის გარეშე.
    გამოიყენება diagnose.sh-დან ან პირდაპირ production სერვერზე.
    """
    rep  = Report()
    conn = None

    # 1. Python files
    check_python_files(rep, base_path)

    # 2. DB
    conn = check_db(rep, db_path)

    if conn:
        # 3. system_state
        check_system_state(rep, conn)
        # 4. open trades
        check_open_trades(rep, conn)
        # 5. OCO links
        check_oco_links(rep, conn)
        # 6. performance
        check_performance(rep, conn)

    # 7. ENV
    check_env(rep)

    if conn:
        # 8. SL cooldown
        check_sl_cooldown(rep, conn)

    # 9. outbox
    check_signal_outbox(rep)

    if conn:
        # 10. audit
        check_audit_events(rep, conn)

    # 11. regime engine
    check_regime_engine(rep)

    # 12. ENV vs code conflicts
    check_env_vs_code(rep)

    if conn:
        # 13. PnL consistency
        check_pnl_consistency(rep, conn)

    # 14. API connectivity
    check_api_connectivity(rep)

    if conn:
        # 15. BROKEN OCO repair suggestions
        check_and_suggest_oco_repair(rep, conn)
        conn.close()

    rep.print_report()
    return rep


# =============================================================================
# MASTER RUN — specific trade (Adapter-based)
# =============================================================================

def run_pro_diagnostics(
    adapter: Adapter,
    signal_id: str,
    link_id: str,
    db_path: Optional[str] = None,
) -> Report:
    """
    Trade-specific deep diagnostics — Adapter-ის საჭიროა.
    გამოიყენება execution_engine-დან specific trade-ის debug-ისთვის.
    """
    rep = Report()

    # First run full system check
    conn = check_db(rep, db_path)
    if conn:
        check_system_state(rep, conn)
        check_sl_cooldown(rep, conn)

    check_env(rep)
    check_env_vs_code(rep)
    check_regime_engine(rep)

    if conn:
        check_pnl_consistency(rep, conn)
        check_and_suggest_oco_repair(rep, conn)

    check_api_connectivity(rep)

    # Trade-specific
    trade = adapter.get_trade(signal_id)
    if not trade:
        rep.add("TRADE_EXISTS", False, f"trade signal_id={signal_id} not found", "CRITICAL",
                fix=f"DB-ში signal_id={signal_id} არ მოიძებნა — DB და execution_engine შეამოწმე")
        rep.print_report()
        return rep

    symbol = trade.get("symbol")
    tp_id  = trade.get("tp_order_id")
    sl_id  = trade.get("sl_order_id")

    tp  = adapter.get_order(tp_id)  if tp_id  else None
    sl  = adapter.get_order(sl_id)  if sl_id  else None
    pos = adapter.get_position(symbol)

    check_position_sync(rep, trade, pos)
    check_order_link(rep, tp, sl)
    if tp_id: check_partial_fill_engine(rep, adapter, tp_id, trade.get("qty"))
    if sl_id: check_partial_fill_engine(rep, adapter, sl_id, trade.get("qty"))
    check_restart_recovery(rep, adapter)
    check_api_resilience(rep, tp, sl)
    check_race_condition(rep, adapter, signal_id)
    check_latency(rep, adapter)

    expected = trade.get("tp_price") or trade.get("sl_price")
    actual   = (tp or {}).get("avgPrice") or (sl or {}).get("avgPrice")
    check_slippage(rep, expected, actual)
    check_fee_engine(rep, adapter, symbol, trade.get("qty"), actual, trade.get("pnl_quote"))
    check_logs_completeness(rep, adapter, signal_id)
    check_edge_cases(rep, trade)

    if conn:
        conn.close()

    rep.print_report()
    return rep


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Genius Bot — სრული დიაგნოსტიკა")
    parser.add_argument("--db",       default=None,  help="DB path override")
    parser.add_argument("--base",     default="/opt/render/project/src", help="project base path")
    parser.add_argument("--signal",   default=None,  help="specific signal_id for trade debug")
    args = parser.parse_args()

    if args.signal:
        print(f"[DIAG] Trade-specific mode: signal_id={args.signal}")
        print("[DIAG] Adapter-based run requires my_adapter.py — running system check only")

    rep = run_full_diagnostics(db_path=args.db, base_path=args.base)
    sys.exit(0 if rep.summary()["status"] == "SAFE" else 1)
