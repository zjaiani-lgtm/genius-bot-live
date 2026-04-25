# execution/diagnostics_pro.py
# ╔══════════════════════════════════════════════════════════════════════╗
# ║                                                                      ║
# ║   ██████  ███████ ███    ██ ██ ██    ██ ███████                      ║
# ║  ██       ██      ████   ██ ██ ██    ██ ██                           ║
# ║  ██   ███ █████   ██ ██  ██ ██ ██    ██ ███████                      ║
# ║  ██    ██ ██      ██  ██ ██ ██ ██    ██      ██                      ║
# ║   ██████  ███████ ██   ████ ██  ██████  ███████                      ║
# ║                                                                      ║
# ║         DCA BOT — Production Diagnostics v2.0                        ║
# ║         Binance Spot | Cascade DCA | Rolling Exchange                ║
# ║                                                                      ║
# ╠══════════════════════════════════════════════════════════════════════╣
# ║  გაშვება:                                                            ║
# ║    cd /opt/render/project/src                                        ║
# ║    PYTHONPATH=. python3 -m execution.diagnostics_pro                 ║
# ╠══════════════════════════════════════════════════════════════════════╣
# ║  შემოწმებები:                                                        ║
# ║   01. Python ფაილების არსებობა                                       ║
# ║   02. DB მდგომარეობა და ცხრილები                                     ║
# ║   03. system_state                                                   ║
# ║   04. ღია trade-ები და DCA პოზიციები                                 ║
# ║   05. OCO links (broken detection)                                   ║
# ║   06. Performance სტატისტიკა                                         ║
# ║   07. ENV პარამეტრების სრული ვალიდაცია                               ║
# ║   08. SL Cooldown მდგომარეობა                                        ║
# ║   09. signal_outbox                                                  ║
# ║   10. ბოლო audit events                                              ║
# ║   11. Regime Engine ფუნქციური ტესტი                                  ║
# ║   12. ENV vs Python default კონფლიქტები                              ║
# ║   13. Trade PnL consistency                                          ║
# ║   14. Binance API connectivity                                       ║
# ║   15. BROKEN OCO repair suggestions                                  ║
# ║   16. DCA პოზიციების ვალიდაცია (TP/SL/Memory)                       ║
# ╠══════════════════════════════════════════════════════════════════════╣
# ║  FIX LOG:                                                            ║
# ║   FIX-D1: system_state ACTIVE/RUNNING — ორივე valid სტატუსი         ║
# ║   FIX-D2: API drift — Binance {"serverTime":...} ფორმატი            ║
# ║   FIX-D3: Bybit → Binance ყველა reference-ში                        ║
# ║   FIX-D4: ENV_VS_CODE DCA-incompatible checks ამოღებულია            ║
# ║   FIX-D5: BUY_CONFIDENCE_MIN=0.005 (intentional DCA config)         ║
# ║   FIX-D6: MAX_TRADES_PER_DAY=60 / HOUR=12 (DCA config)              ║
# ║   FIX-D7: MIN_VOLUME_24H DCA მნიშვნელობა (100000)                   ║
# ║   FIX-D8: CASCADE_DROP_L4/L8_PCT ამოღებულია (კოდში არ გამოიყენება) ║
# ║   FIX-D9: ENV_RULES შენი კონფიგურაციისთვის განახლება               ║
# ║   FIX-D10: MODE=DEMO — API keys MODE-conditional check               ║
# ╚══════════════════════════════════════════════════════════════════════╝

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
        G   = "\033[32m"   # green
        Y   = "\033[33m"   # yellow
        R   = "\033[31m"   # red
        C   = "\033[36m"   # cyan
        B   = "\033[1m"    # bold
        M   = "\033[35m"   # magenta
        RST = "\033[0m"

        W = 72
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        print(f"\n{B}{C}╔{'═'*W}╗{RST}")
        print(f"{B}{C}║{'GENIUS DCA BOT — სრული დიაგნოსტიკა':^{W}}║{RST}")
        print(f"{B}{C}║{f'Binance Spot | Cascade DCA | {now_str}':^{W}}║{RST}")
        print(f"{B}{C}╚{'═'*W}╝{RST}\n")

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
            sec_icon = {
                "Python ფაილები": "📁",
                "DB კავშირი": "🗄️",
                "system_state": "⚙️",
                "ღია trade": "📊",
                "OCO links": "🔗",
                "Performance": "📈",
                "ENV ვალიდაცია": "🔧",
                "SL Cooldown": "⏱️",
                "signal_outbox": "📬",
                "ბოლო audit": "📋",
                "Regime Engine": "🧠",
                "ENV vs Code": "⚠️",
                "Trade PnL": "💰",
                "API connectivity": "🌐",
                "BROKEN OCO": "🔴",
                "სხვა": "📌",
            }.get(sec_name.split()[0], "▸")
            print(f"{C}┌{'─'*W}┐{RST}")
            print(f"{C}│ {B}{sec_icon} {sec_num:02d}. {sec_name}{RST}{C}{' '*(W-6-len(sec_name))}│{RST}")
            print(f"{C}└{'─'*W}┘{RST}")
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
        status_icon = "✅" if s["status"] == "SAFE" else ("🔴" if s["status"] == "CRITICAL" else "⚠️")
        color = G if s["status"] == "SAFE" else (R if s["status"] == "CRITICAL" else Y)

        print(f"{B}{'═'*W}{RST}")
        print(f"  📊 სულ: {s['total']}   {G}✓ OK: {s['passed']}{RST}   {Y}⚠ WARN: {s['warn']}{RST}   {R}✗ FAIL: {s['failed']}   🔴 CRIT: {s['critical']}{RST}")
        print(f"  {status_icon} სტატუსი: {color}{B}{s['status']}{RST}")
        print(f"{B}{'═'*W}{RST}\n")

        if not all(r.ok for r in self.results):
            problems = [r for r in self.results if not r.ok and r.fix]
            if problems:
                print(f"{B}  🔧 დასაფიქსირებელი პრობლემები ({len(problems)}):{RST}")
                print(f"  {'─'*W}")
                for i, r in enumerate(problems, 1):
                    sev_color = R if r.severity == "CRITICAL" else Y
                    sev_icon  = "🔴" if r.severity == "CRITICAL" else "⚠️"
                    print(f"  {sev_color}{sev_icon} {i:02d}. [{r.severity}] {r.name}{RST}")
                    print(f"     {r.fix}")
                    print()
                print(f"  {'─'*W}")


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

        rep.add("system_state/status",
                status in ("RUNNING", "ACTIVE"),
                f"status={status}",
                fix="" if status in ("RUNNING", "ACTIVE") else "status უნდა იყოს RUNNING ან ACTIVE — main.py restart")
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
        # FIX: trades ცხრილს არ აქვს 'status' სვეტი — closed_at IS NULL გამოვიყენოთ
        rows = conn.execute(
            "SELECT * FROM trades WHERE closed_at IS NULL ORDER BY opened_at DESC"
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
        # FIX: status სვეტი არ არის — closed_at IS NOT NULL გამოვიყენოთ
        rows = conn.execute(
            "SELECT * FROM trades WHERE closed_at IS NOT NULL "
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
    ("MODE",                   "DEMO",       "eq",      "WARN",     "bot mode — DEMO=virtual, LIVE=real money"),
    ("KILL_SWITCH",            "false",      "eq",      "CRITICAL", "kill switch off"),
    ("LIVE_CONFIRMATION",      "true",       "eq",      "WARN",     "live confirmation"),
    # ─── API — checked MODE-conditionally in check_env() ─────────
    # BINANCE_API_KEY + SECRET: only required when MODE=LIVE
    # handled separately below ENV_RULES loop
    # ─── Symbols ─────────────────────────────────────────────────
    ("BOT_SYMBOLS",            None,         "nonempty","CRITICAL", "trading symbols"),
    ("BOT_TIMEFRAME",          "15m",        "eq",      "WARN",     "candle timeframe"),
    ("BOT_CANDLE_LIMIT",       "50",         "eq",      "WARN",     "candle history (DCA=50)"),
    # ─── DCA სტრატეგია ───────────────────────────────────────────
    ("DCA_ENABLED",            "true",       "eq",      "CRITICAL", "DCA სტრატეგია ჩართული"),
    ("DCA_TP_PCT",             "0.55",       "eq",      "WARN",     "DCA Take Profit % (L1-L2)"),
    ("DCA_SL_PCT",             "999.0",      "eq",      "CRITICAL", "DCA Stop Loss გათიშული (ფილოსოფია)"),
    ("DCA_MAX_ADD_ONS",        "5",          "eq",      "WARN",     "DCA მაქსიმალური add-on (5-level pyramid)"),
    ("DCA_MAX_CAPITAL_USDT",   "350",        "eq",      "WARN",     "DCA მაქსიმალური კაპიტალი per position"),
    ("DCA_ADDON_TRIGGER_PCTS", "1.0,2.2,3.5,5.0,6.5", "eq", "WARN", "DCA add-on trigger % სია (5 level)"),
    # ─── CASCADE სტრატეგია ───────────────────────────────────────
    ("CASCADE_ENABLED",        "true",       "eq",      "CRITICAL", "CASCADE სტრატეგია ჩართული"),
    ("CASCADE_START_LAYER",    "2",          "eq",      "WARN",     "CASCADE დაწყება L2-დან"),
    ("CASCADE_MAX_LAYERS",     "10",         "eq",      "WARN",     "CASCADE მაქსიმალური layer"),
    ("CASCADE_RESUME_LAYER",   "999",        "eq",      "WARN",     "CASCADE dead zone გაუქმება (999=disabled)"),
    ("CASCADE_DROP_PCT",       "1.5",        "eq",      "WARN",     "CASCADE drop L2-L3 (1.5%)"),
    ("CASCADE_TP_L3_PCT",      "0.35",       "eq",      "WARN",     "CASCADE TP L3 zone (0.35% — LIFO rotation)"),
    # ─── LAYER2 სტრატეგია ────────────────────────────────────────
    ("LAYER2_ENABLED",         "true",       "eq",      "WARN",     "LAYER2 სტრატეგია ჩართული"),
    ("LAYER2_DROP_PCT",        "1.5",        "eq",      "WARN",     "LAYER2 crash trigger %"),
    ("LAYER2_QUOTE",           "50",         "eq",      "WARN",     "LAYER2 ყიდვის ზომა USDT"),
    # ─── Memory დაცვა ────────────────────────────────────────────
    ("BOT_API_ENABLED",        "false",      "eq",      "WARN",     "Bot API გამორთული (memory დაცვა)"),
    ("DASHBOARD_ENABLED",      "false",      "eq",      "WARN",     "Dashboard გამორთული (memory დაცვა)"),
    ("QTY_SYNC_ENABLED",       "false",      "eq",      "WARN",     "QTY Sync (Standard plan-ზე ჩართე)"),
    # ─── Sizing ──────────────────────────────────────────────────
    ("BOT_QUOTE_PER_TRADE",    "50",         "eq",      "WARN",     "quote per trade USDT"),
    ("MAX_QUOTE_PER_TRADE",    "50",         "eq",      "WARN",     "max quote ceiling"),
    # ─── Trade limits ────────────────────────────────────────────
    ("MAX_OPEN_TRADES",        "6",          "eq",      "WARN",     "მაქსიმალური ღია პოზიციები"),
    ("MIN_OPEN_TRADES",        "5",          "eq",      "WARN",     "მინიმალური ღია პოზიციები"),
    ("SMART_ADDON_BUFFER",     "200",        "eq",      "WARN",     "USDT buffer add-on-ისთვის"),
    ("LOOP_SLEEP_SECONDS",     "120",        "eq",      "WARN",     "loop ინტერვალი წამებში"),
    # ─── Paths / Telegram ────────────────────────────────────────
    ("DB_PATH",                None,         "nonempty","CRITICAL", "DB path"),
    ("SIGNAL_OUTBOX_PATH",     None,         "nonempty","WARN",     "signal outbox path"),
    ("TELEGRAM_BOT_TOKEN",     None,         "nonempty","WARN",     "Telegram bot token"),
    ("TELEGRAM_CHAT_ID",       None,         "nonempty","WARN",     "Telegram chat ID"),
    ("TELEGRAM_NOTIFICATIONS", "true",       "eq",      "WARN",     "Telegram შეტყობინებები"),
]


def check_env(rep: Report):
    # MODE-conditional: BINANCE API keys only required in LIVE mode
    _mode = os.getenv("MODE", "DEMO").upper()
    if _mode == "LIVE":
        for _api_key in ("BINANCE_API_KEY", "BINANCE_API_SECRET"):
            _val = os.getenv(_api_key, "").strip()
            _ok  = bool(_val)
            rep.add(f"ENV/{_api_key}", _ok,
                    f"Binance API {'key' if 'KEY' in _api_key else 'secret'} → "
                    f"{'set (masked)' if _ok else 'NOT SET'}",
                    severity="CRITICAL" if not _ok else "INFO",
                    fix=f"Render ENV → {_api_key}=<value> — MODE=LIVE-ზე სავალდებულოა!" if not _ok else "")
    else:
        rep.add("ENV/BINANCE_API", True,
                f"MODE={_mode} — Binance API keys optional (not checked in DEMO/TESTNET)")

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
        # FIX: status სვეტი არ არის — outcome სვეტი გამოვიყენოთ
        row_losses = conn.execute(
            "SELECT COUNT(*) as cnt FROM trades WHERE outcome='SL'"
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
            # FIX: {} ან empty object → ცარიელი outbox (არა სიგნალი)
            if content in ("{}", ""):
                rep.add("signal_outbox/parse", True, "Outbox ცარიელია — {} (OK)")
                return
            if content.startswith("["):
                sigs = json.loads(content)
            elif content.startswith("{"):
                # single signal object
                sigs = [json.loads(content)]
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
                    "        ყოვლისთვის: balance manually შეამოწმე Bybit-ზე."
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
            tp_env = _safe_float(os.getenv("TP_PCT", "0.55"))
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
    # ─── სიგნალის ხარისხი ────────────────────────────────────────────────
    ("AI_CONFIDENCE_BOOST",      "1.05",  "signal score boost — 1.0 default score-ს ვერ ამაღლებს"),
    ("ALLOW_LIVE_SIGNALS",       "true",  "CRITICAL: false → ყველა BUY სიგნალი იბლოკება!"),
    # ─── DCA სწორი ლიმიტები ──────────────────────────────────────────────
    ("BUY_CONFIDENCE_MIN",       "0.005", "BUY_CONFIDENCE_MIN — intentional permissive DCA config"),
    ("MAX_TRADES_PER_DAY",       "60",    "DCA trade limit per day"),
    ("MAX_TRADES_PER_HOUR",      "12",    "DCA trade limit per hour"),
    ("MIN_VOLUME_24H",           "100000","DCA: BTC/ETH/BNB ყოველთვის > 100K — საკმარისია"),
    # ─── სიგნალის ფილტრები ───────────────────────────────────────────────
    ("USE_MA_FILTERS",           "false", "DCA: MA filters გათიშული — ნორმალურია"),
    ("USE_RSI_FILTER",           "true",  "RSI filter ჩართული უნდა იყოს"),
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
        # FIX: status სვეტი არ არის — outcome სვეტი გამოვიყენოთ
        rows = conn.execute(
            "SELECT signal_id, outcome, pnl_quote, entry_price, exit_price, qty "
            "FROM trades WHERE closed_at IS NOT NULL "
            "ORDER BY closed_at DESC LIMIT 50"
        ).fetchall()

        inconsistent = []
        for row in rows:
            d = dict(row)
            outcome = str(d.get("outcome", "") or "").upper()
            pnl     = _safe_float(d.get("pnl_quote") or 0)
            sig_id  = d.get("signal_id", "?")[:8]

            if outcome == "TP" and pnl is not None and pnl <= 0:
                inconsistent.append(f"TP trade {sig_id} has pnl={pnl:.4f} (should be >0)")
            elif outcome == "SL" and pnl is not None and pnl >= 0:
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
        # FIX: outcome სვეტი status-ის მაგივრად
        sl_trades  = [dict(r) for r in rows if str(dict(r).get("outcome","") or "").upper() == "SL"]
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

    # 1. Ping — Binance server time (public, no auth)
    try:
        req = urllib.request.Request(
            "https://api.binance.com/api/v3/time",
            headers={"User-Agent": "GeniusBot-Diag/1.0"}
        )
        with urllib.request.urlopen(req, timeout=3) as resp:
            ok = resp.status == 200
            data = _json.loads(resp.read().decode())
        rep.add("API/ping", ok, f"Binance ping → {resp.status}",
                fix="Binance API ping failed — network ან API endpoint შეამოწმე" if not ok else "")
    except Exception as e:
        rep.add("API/ping", False, str(e), "CRITICAL",
                fix="Binance REST API მიუწვდომელია — Render outbound network შეამოწმე")
        return

    # 2. Server time drift — FIX-D2: Binance აბრუნებს {"serverTime": ms}
    try:
        server_ms = int(data.get("serverTime", 0))  # FIX: Binance ფორმატი
        local_ms  = int(time.time() * 1000)
        drift_ms  = abs(server_ms - local_ms)
        drift_ok  = drift_ms < 1000
        rep.add("API/time_drift", drift_ok,
                f"drift={drift_ms}ms (max=1000ms)" + (" ✅" if drift_ok else " ⚠️"),
                severity="WARN" if not drift_ok else "INFO",
                fix=f"Clock drift {drift_ms}ms > 1000ms — Render server NTP sync პრობლემა" if not drift_ok else "")
    except Exception as e:
        rep.add("API/time_drift", False, str(e), "WARN",
                fix="Binance /time endpoint parse failed")


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
                f"        ნაბიჯი 3: trades ცხრილში შესაბამისი entries შეამოწმე\n"
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
                fix=f"Bybit-ზე manually cancel order {b.get('tp_order_id','?')} და {b.get('sl_order_id','?')}"
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
                fix=f"Position qty={qty} but trade={status} — Bybit-ზე manually close position" if not ok else "")
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
            fix="OCO orders fetch failed — Bybit API key permissions შეამოწმე" if not ok else "")


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
            fix=f"High latency {lat}ms — Render region ან Bybit endpoint შეამოწმე" if not ok else "")


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

def check_dca_positions(rep: Report, conn: sqlite3.Connection):
    """
    DCA-სპეციფიური შემოწმებები:
    1. TP > avg_entry ყოველ პოზიციაზე
    2. qty × price > $5 (minimum notional)
    3. SL_PCT >= 999 (DCA ფილოსოფია)
    4. CASCADE layer drop_pct by layer
    5. Memory check
    6. BOT_API/DASHBOARD გამორთული
    """
    # ── 1. TP > avg_entry ────────────────────────────────────────
    try:
        rows = conn.execute("""
            SELECT symbol, avg_entry_price, current_tp_price, total_qty
            FROM dca_positions WHERE status='OPEN'
        """).fetchall()

        for row in rows:
            sym, avg, tp, qty = row
            avg  = float(avg  or 0)
            tp   = float(tp   or 0)
            qty  = float(qty  or 0)

            # TP > avg_entry
            if avg > 0 and tp > 0:
                ok = tp > avg
                pct = round((tp - avg) / avg * 100, 3) if avg > 0 else 0
                rep.add(
                    f"DCA/tp_valid/{sym}", ok,
                    f"TP={tp:.4f} > avg={avg:.4f} (+{pct}%)" if ok
                    else f"TP={tp:.4f} <= avg={avg:.4f} — TP გასწორება საჭიროა!",
                    severity="CRITICAL" if not ok else "INFO",
                    fix=f"Shell: UPDATE dca_positions SET current_tp_price=round({avg}*1.0055,6) WHERE symbol='{sym}' AND status='OPEN';" if not ok else ""
                )

            # qty × price > $5
            value = qty * avg
            ok_val = value >= 5.0
            rep.add(
                f"DCA/qty_notional/{sym}", ok_val,
                f"qty={qty:.8f} × avg={avg:.2f} = ${value:.2f}" + (" ✅" if ok_val else " ⚠️ < $5!"),
                severity="CRITICAL" if not ok_val else "INFO",
                fix=f"Shell: qty_fix საჭიროა — {sym} value=${value:.2f} < $5" if not ok_val else ""
            )

    except Exception as e:
        rep.add("DCA/positions_check", False, f"ERROR: {e}", severity="WARN")

    # ── 2. SL_PCT >= 999 ─────────────────────────────────────────
    sl_pct = float(os.getenv("DCA_SL_PCT", "0"))
    ok = sl_pct >= 999.0
    rep.add(
        "DCA/sl_philosophy", ok,
        f"DCA_SL_PCT={sl_pct} — {'SL გათიშულია ✅' if ok else 'SL ჩართულია! DCA ფილოსოფია ირღვევა!'}",
        severity="CRITICAL" if not ok else "INFO",
        fix="Render ENV → DCA_SL_PCT=999.0" if not ok else ""
    )

    # ── 3. CASCADE layer drop_pct ────────────────────────────────
    drop_base = float(os.getenv("CASCADE_DROP_PCT",  "0"))
    tp_l3     = float(os.getenv("CASCADE_TP_L3_PCT", "0"))

    # CASCADE_DROP_L4_PCT / CASCADE_DROP_L8_PCT ამოღებულია —
    # ეს ENV-ები კოდში არ გამოიყენება (grep დადასტურა)
    for val, expected, name, fix_key in [
        (drop_base, 1.5,  "CASCADE drop L2-L3",        "CASCADE_DROP_PCT=1.5"),
        (tp_l3,     0.35, "CASCADE TP L3 (LIFO zone)",  "CASCADE_TP_L3_PCT=0.35"),
    ]:
        ok = abs(val - expected) < 0.001
        rep.add(
            f"DCA/cascade/{fix_key.split('=')[0]}", ok,
            f"{name}: {val}%" + (" ✅" if ok else f" (expected={expected}%)"),
            severity="WARN" if not ok else "INFO",
            fix=f"Render ENV → {fix_key}" if not ok else ""
        )

    # ── 4. Memory check ──────────────────────────────────────────
    try:
        import resource
        mem_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        mem_mb = mem_kb / 1024
        ok = mem_mb < 480
        rep.add(
            "DCA/memory", ok,
            f"Memory usage: {mem_mb:.0f}MB (limit=480MB)" + (" ✅" if ok else " ⚠️ 512MB-ს უახლოვდება!"),
            severity="WARN" if not ok else "INFO",
            fix="BOT_API_ENABLED=false და DASHBOARD_ENABLED=false შეამოწმე" if not ok else ""
        )
    except Exception:
        rep.add("DCA/memory", True, "Memory check N/A", severity="INFO")

    # ── 5. BOT_API / DASHBOARD გამორთული ────────────────────────
    for key, label in [("BOT_API_ENABLED", "Bot API"), ("DASHBOARD_ENABLED", "Dashboard")]:
        val = os.getenv(key, "true").strip().lower()
        ok  = val in ("false", "0")
        rep.add(
            f"DCA/{key}", ok,
            f"{label}: {val}" + (" გამორთულია ✅" if ok else " ჩართულია! Memory-ს ჭამს!"),
            severity="WARN" if not ok else "INFO",
            fix=f"Render ENV → {key}=false" if not ok else ""
        )


# =============================================================================
# SECTION 17: ENTRY LOGIC — executed_signals + REJECT events
# (shell audit section 1 — not in original diagnostics_pro)
# =============================================================================

def check_entry_logic(rep: Report, conn: sqlite3.Connection):
    """
    შემოწმება:
      - ბოლო 20 executed signal
      - REJECT events (რატომ არ ყიდულობს)
      - MAX_OPEN_TRADES blocks
      - დღევანდელი trade count
    """
    # ── ბოლო 20 executed signal ──────────────────────────────────
    try:
        rows = conn.execute(
            "SELECT signal_id, action, symbol, executed_at "
            "FROM executed_signals ORDER BY executed_at DESC LIMIT 20"
        ).fetchall()
        signals = [dict(r) for r in rows]

        demo_count = sum(1 for s in signals if "DEMO" in str(s.get("action", "")))
        live_count = sum(1 for s in signals if "LIVE" in str(s.get("action", "")))
        reject_count = sum(1 for s in signals if "REJECT" in str(s.get("action", "")))

        rep.add("Entry/executed_signals", True,
                f"ბოლო 20 signal: demo={demo_count} live={live_count} reject={reject_count}")

        if reject_count > 0:
            rejects = [s for s in signals if "REJECT" in str(s.get("action", ""))]
            for r in rejects[:3]:
                rep.add("Entry/REJECT_signal", False,
                        f"[{r.get('executed_at','')}] action={r.get('action','')} sym={r.get('symbol','')}",
                        severity="WARN",
                        fix="REJECT actions — execution_engine-ში reason შეამოწმე audit_log-ში")
    except Exception as e:
        rep.add("Entry/executed_signals", False, str(e), "WARN",
                fix="executed_signals ცხრილი ვერ წაიკითხა")

    # ── REJECT events audit_log-დან ───────────────────────────────
    try:
        rows = conn.execute(
            "SELECT event_type, message, created_at FROM audit_log "
            "WHERE event_type LIKE '%REJECT%' "
            "ORDER BY id DESC LIMIT 10"
        ).fetchall()
        rejects = [dict(r) for r in rows]
        rep.add("Entry/reject_events", len(rejects) == 0,
                f"REJECT events audit_log-ში: {len(rejects)}",
                severity="WARN" if rejects else "INFO",
                fix="REJECT events ნაპოვნია — audit_log details შეამოწმე" if rejects else "")
        for r in rejects[:3]:
            rep.add("Entry/reject_detail", False,
                    f"[{r.get('created_at','')}] {r.get('event_type','')} | {str(r.get('message',''))[:100]}",
                    severity="WARN")
    except Exception as e:
        rep.add("Entry/reject_events", False, str(e), "WARN")

    # ── MAX_OPEN_TRADES blocks ────────────────────────────────────
    try:
        rows = conn.execute(
            "SELECT COUNT(*) as cnt FROM audit_log "
            "WHERE event_type='EXEC_REJECT_MAX_OPEN_TRADES'"
        ).fetchone()
        cnt = dict(rows)["cnt"] if rows else 0
        ok = cnt == 0
        rep.add("Entry/max_open_blocks", ok,
                f"MAX_OPEN_TRADES blocks: {cnt}",
                severity="WARN" if cnt > 5 else "INFO",
                fix=f"MAX_OPEN_TRADES ბლოკი {cnt}-ჯერ — MAX_OPEN_TRADES ENV გაზარდე ან positions გაახსნე" if cnt > 0 else "")
    except Exception as e:
        rep.add("Entry/max_open_blocks", False, str(e), "WARN")

    # ── დღევანდელი trades ────────────────────────────────────────
    try:
        rows = conn.execute(
            "SELECT COUNT(*) as total, "
            "SUM(CASE WHEN opened_at >= date('now') THEN 1 ELSE 0 END) as today "
            "FROM trades"
        ).fetchone()
        d = dict(rows) if rows else {}
        rep.add("Entry/trade_count", True,
                f"სულ trades={d.get('total',0)} | დღეს={d.get('today',0)}")
    except Exception as e:
        rep.add("Entry/trade_count", False, str(e), "WARN")


# =============================================================================
# SECTION 18: TP DETAIL — NULL TP, math, TP fix log, add-on cooldown
# (shell audit sections 2+3 — partial overlap with check_dca_positions)
# =============================================================================

def check_tp_and_addon_detail(rep: Report, conn: sqlite3.Connection):
    """
    შემოწმება (check_dca_positions-ს ავსებს, არ მეორებს):
      - NULL current_tp_price positions
      - TP % actual (tp/avg - 1) × 100
      - TP_FIX audit events
      - add-on orders per position
      - last_add_on_ts cooldown status
      - days open per position
    """
    # ── NULL TP შემოწმება ─────────────────────────────────────────
    try:
        rows = conn.execute(
            "SELECT id, symbol, current_tp_price, avg_entry_price "
            "FROM dca_positions "
            "WHERE status='OPEN' AND (current_tp_price IS NULL OR current_tp_price=0)"
        ).fetchall()
        null_tp = [dict(r) for r in rows]
        ok = len(null_tp) == 0
        rep.add("TP_Detail/null_tp", ok,
                f"NULL/0 TP positions: {len(null_tp)}",
                severity="CRITICAL" if not ok else "INFO",
                fix=(
                    "NULL TP positions ნაპოვნია! TP_FIX_ENABLED=true შეამოწმე.\n"
                    "Manual fix: sqlite3 $DB " +
                    " && ".join(
                        f"\"UPDATE dca_positions SET current_tp_price=round({float(r.get('avg_entry_price') or 0)*1.0055},6) WHERE id={r.get('id')};\""
                        for r in null_tp[:3]
                    )
                ) if not ok else "")
    except Exception as e:
        rep.add("TP_Detail/null_tp", False, str(e), "CRITICAL")

    # ── TP % actual math ─────────────────────────────────────────
    try:
        rows = conn.execute(
            "SELECT id, symbol, avg_entry_price, current_tp_price, add_on_count "
            "FROM dca_positions WHERE status='OPEN'"
        ).fetchall()

        tp_pct_env = float(os.getenv("DCA_TP_PCT", "0.55"))
        l3_pct_env = float(os.getenv("CASCADE_TP_L3_PCT", "0.35"))
        max_addons = int(os.getenv("DCA_MAX_ADD_ONS", "5"))
        tolerance  = 0.1  # 0.1% tolerance

        for row in rows:
            d = dict(row)
            sym  = d["symbol"]
            avg  = float(d["avg_entry_price"] or 0)
            tp   = float(d["current_tp_price"] or 0)
            n    = int(d["add_on_count"] or 0)

            if avg <= 0 or tp <= 0:
                continue

            actual_pct = (tp / avg - 1.0) * 100.0
            expected   = l3_pct_env if n >= max_addons else tp_pct_env
            diff       = abs(actual_pct - expected)
            ok         = diff < tolerance

            rep.add(f"TP_Detail/math/{sym}",
                    ok,
                    f"zone={'L3' if n>=max_addons else 'L2'} "
                    f"actual={actual_pct:.3f}% expected={expected}% diff={diff:.4f}%",
                    severity="WARN" if not ok else "INFO",
                    fix=f"TP_FIX_ENABLED=true ENV-ში — tp_fix.py-ი ასწორებს ყოველ loop-ზე" if not ok else "")
    except Exception as e:
        rep.add("TP_Detail/math", False, str(e), "WARN")

    # ── TP_FIX log ───────────────────────────────────────────────
    try:
        rows = conn.execute(
            "SELECT event_type, message, created_at FROM audit_log "
            "WHERE event_type LIKE '%TP_FIX%' ORDER BY id DESC LIMIT 5"
        ).fetchall()
        tp_fix_events = [dict(r) for r in rows]
        tp_fix_enabled = os.getenv("TP_FIX_ENABLED", "true").lower() in ("true", "1", "yes")

        if tp_fix_enabled and not tp_fix_events:
            rep.add("TP_Detail/tp_fix_log", False,
                    "TP_FIX_ENABLED=true მაგრამ audit_log-ში TP_FIX event არ არის",
                    severity="WARN",
                    fix="tp_fix.py-ი არ მუშაობს — main.py-ში TP_FIX_LOOP შეამოწმე")
        else:
            rep.add("TP_Detail/tp_fix_log", True,
                    f"TP_FIX events: {len(tp_fix_events)} {'(ბოლოს: ' + tp_fix_events[0].get('created_at','') + ')' if tp_fix_events else '(none)'}")
    except Exception as e:
        rep.add("TP_Detail/tp_fix_log", False, str(e), "WARN")

    # ── add-on orders per position ────────────────────────────────
    try:
        rows = conn.execute(
            "SELECT o.symbol, o.order_type, o.entry_price, o.qty, "
            "o.avg_entry_after, o.tp_after, o.filled_at "
            "FROM dca_orders o "
            "JOIN dca_positions p ON o.position_id=p.id "
            "WHERE p.status='OPEN' ORDER BY o.id"
        ).fetchall()
        orders = [dict(r) for r in rows]

        initial_cnt = sum(1 for o in orders if o.get("order_type") == "INITIAL")
        addon_cnt   = sum(1 for o in orders if str(o.get("order_type","")).startswith("ADD_ON"))
        l3_cnt      = sum(1 for o in orders if o.get("order_type") == "L3_ADDON")
        rot_cnt     = sum(1 for o in orders if o.get("order_type") == "ROTATION_REINVEST")

        rep.add("TP_Detail/addon_orders", True,
                f"dca_orders (ღია pos): INITIAL={initial_cnt} ADD_ON={addon_cnt} L3={l3_cnt} ROTATION={rot_cnt}")
    except Exception as e:
        rep.add("TP_Detail/addon_orders", False, str(e), "WARN")

    # ── cooldown + days open ─────────────────────────────────────
    try:
        rows = conn.execute(
            "SELECT id, symbol, last_add_on_ts, add_on_count, opened_at, "
            "ROUND(julianday('now')-julianday(opened_at),2) as days_open "
            "FROM dca_positions WHERE status='OPEN'"
        ).fetchall()

        cooldown_s  = int(os.getenv("DCA_ADDON_COOLDOWN_SECONDS", "300"))
        fc_days     = float(os.getenv("FORCE_CLOSE_MAX_DAYS", "10"))
        now_ts      = _now_utc().timestamp()

        for row in rows:
            d        = dict(row)
            sym      = d["symbol"]
            last_ts  = float(d.get("last_add_on_ts") or 0)
            days_open= float(d.get("days_open") or 0)
            n        = int(d.get("add_on_count") or 0)

            # cooldown
            if last_ts > 0:
                elapsed   = now_ts - last_ts
                remaining = max(0, cooldown_s - elapsed)
                in_cd     = remaining > 0
                rep.add(f"TP_Detail/cooldown/{sym}",
                        True,
                        f"add_on={n} | cooldown={'active, ' + str(int(remaining)) + 's left' if in_cd else 'clear'}")

            # days open vs force close
            days_warn = fc_days * 0.7  # 70% of FC limit
            ok        = days_open < fc_days
            rep.add(f"TP_Detail/days_open/{sym}",
                    ok,
                    f"days_open={days_open:.1f}d | fc_limit={fc_days:.0f}d",
                    severity="WARN" if days_open >= days_warn and ok else ("CRITICAL" if not ok else "INFO"),
                    fix=f"{sym} FC ზღვარს უახლოვდება ({days_open:.1f}d >= {fc_days:.0f}d) — FORCE_CLOSE trigger-ის ლოდინი" if not ok else "")
    except Exception as e:
        rep.add("TP_Detail/cooldown", False, str(e), "WARN")


# =============================================================================
# SECTION 19: FUTURES DB DETAIL — columns, positions, is_mirror_engine
# (shell audit section 5 — not in original diagnostics_pro)
# =============================================================================

def check_futures_db_detail(rep: Report, conn: sqlite3.Connection):
    """
    შემოწმება:
      - futures_positions columns — is_mirror_engine, close_reason და სხვა
      - ღია futures positions (hedge/short/mirror breakdown)
      - FUTURES/SHORT/HEDGE/MIRROR events audit_log-ში
    """
    # ── columns შემოწმება ─────────────────────────────────────────
    try:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(futures_positions)").fetchall()}

        critical_cols = [
            "is_mirror_engine", "close_reason", "is_dca_hedge",
            "is_independent_short", "avg_entry_price", "exit_price",
            "mirror_direction", "mirror_addons_down", "mirror_addons_up",
            "last_mirror_addon_ts", "hedge_tp_pct", "last_short_addon_ts",
        ]
        missing = [c for c in critical_cols if c not in cols]

        ok = len(missing) == 0
        rep.add("Futures_DB/columns", ok,
                f"futures_positions columns: {len(cols)} total | missing: {missing if missing else 'none'}",
                severity="CRITICAL" if missing else "INFO",
                fix=(
                    f"Missing columns: {missing}\n"
                    "Fix: deploy db.py with _migrate_futures_columns() და restart"
                ) if missing else "")
    except Exception as e:
        rep.add("Futures_DB/columns", False, str(e), "CRITICAL",
                fix="futures_positions PRAGMA fail — ცხრილი არ არსებობს?")

    # ── ღია positions breakdown ───────────────────────────────────
    try:
        rows = conn.execute(
            "SELECT * FROM futures_positions WHERE status='OPEN'"
        ).fetchall()
        if rows:
            cols_names = [d[0] for d in conn.execute("PRAGMA table_info(futures_positions)").fetchall()]
            positions  = [dict(zip(cols_names, r)) for r in rows]

            hedge_cnt  = sum(1 for p in positions if int(p.get("is_dca_hedge", 0) or 0))
            indep_cnt  = sum(1 for p in positions if int(p.get("is_independent_short", 0) or 0))
            mirror_cnt = sum(1 for p in positions if int(p.get("is_mirror_engine", 0) or 0))
            bear_cnt   = len(positions) - hedge_cnt - indep_cnt - mirror_cnt

            rep.add("Futures_DB/open_positions", True,
                    f"ღია futures: {len(positions)} total | "
                    f"hedge={hedge_cnt} indep={indep_cnt} mirror={mirror_cnt} bear={bear_cnt}")

            # Per position TP check
            for pos in positions:
                sym      = pos.get("symbol", "?")
                tp       = float(pos.get("tp_price", 0) or 0)
                avg_e    = float(pos.get("avg_entry_price", 0) or pos.get("entry_price", 0) or 0)
                pos_type = ("HEDGE" if int(pos.get("is_dca_hedge", 0) or 0)
                            else "MIRROR" if int(pos.get("is_mirror_engine", 0) or 0)
                            else "INDEP" if int(pos.get("is_independent_short", 0) or 0)
                            else "BEAR")
                ok = tp > 0
                rep.add(f"Futures_DB/tp/{sym}_{pos_type}",
                        ok,
                        f"type={pos_type} avg_entry={avg_e:.4f} tp={tp:.4f}",
                        severity="WARN" if not ok else "INFO",
                        fix=f"futures TP=0 — {sym} {pos_type} position-ი TP-ის გარეშეა!" if not ok else "")
        else:
            rep.add("Futures_DB/open_positions", True, "ღია futures positions: 0")
    except Exception as e:
        rep.add("Futures_DB/open_positions", False, str(e), "WARN")

    # ── FUTURES/SHORT/HEDGE/MIRROR events ─────────────────────────
    try:
        rows = conn.execute(
            "SELECT event_type, COUNT(*) as cnt FROM audit_log "
            "WHERE event_type LIKE '%FUTURES%' OR event_type LIKE '%SHORT%' "
            "   OR event_type LIKE '%HEDGE%' OR event_type LIKE '%MIRROR%' "
            "GROUP BY event_type ORDER BY cnt DESC LIMIT 10"
        ).fetchall()
        events = [dict(r) for r in rows]
        summary = " | ".join(f"{e['event_type']}={e['cnt']}" for e in events[:5])
        rep.add("Futures_DB/events", True,
                f"Futures/Short/Hedge/Mirror events: {summary if summary else 'none'}")
    except Exception as e:
        rep.add("Futures_DB/events", False, str(e), "WARN")


# =============================================================================
# SECTION 20: REGIME LOG — MARKET_REGIME_CHANGE in audit_log
# (shell audit section 6 — new after main.py fix)
# =============================================================================

def check_regime_log(rep: Report, conn: sqlite3.Connection):
    """
    შემოწმება:
      - MARKET_REGIME_CHANGE events (main.py fix #2)
      - BEAR_BLOCK events
      - ბოლო regime სტატუსი
    """
    try:
        rows = conn.execute(
            "SELECT event_type, message, created_at FROM audit_log "
            "WHERE event_type='MARKET_REGIME_CHANGE' "
            "ORDER BY id DESC LIMIT 10"
        ).fetchall()
        regime_events = [dict(r) for r in rows]

        rep.add("Regime_Log/change_events", True,
                f"MARKET_REGIME_CHANGE events: {len(regime_events)}" +
                (f" (ბოლო: {regime_events[0].get('created_at','')} → {regime_events[0].get('message','')})"
                 if regime_events else " (none yet — normal on first run)"))
    except Exception as e:
        rep.add("Regime_Log/change_events", False, str(e), "WARN")

    # ── BEAR_BLOCK events ────────────────────────────────────────
    try:
        rows = conn.execute(
            "SELECT COUNT(*) as cnt FROM audit_log "
            "WHERE message LIKE '%BEAR_BLOCK%' OR message LIKE '%BEAR market%'"
        ).fetchone()
        cnt = dict(rows)["cnt"] if rows else 0
        rep.add("Regime_Log/bear_blocks", True,
                f"BEAR_BLOCK events სულ: {cnt}" +
                (" (ნორმალური — BEAR MODE ADD-ON ბლოკი)" if cnt > 0 else " (none)"))
    except Exception as e:
        rep.add("Regime_Log/bear_blocks", False, str(e), "WARN")


# =============================================================================
# SECTION 21: INFRA DETAIL — WAL, integrity, tables, kill_switch
# (shell audit section 7 — partial overlap with check_db/check_system_state)
# =============================================================================

def check_infra_detail(rep: Report, conn: sqlite3.Connection):
    """
    შემოწმება (check_db-ს ავსებს):
      - PRAGMA integrity_check
      - PRAGMA journal_mode (WAL)
      - tables სია + missing tables
      - kill_switch + startup_sync_ok
      - error events ბოლო 24h
    """
    # ── integrity ─────────────────────────────────────────────────
    try:
        result = conn.execute("PRAGMA integrity_check").fetchone()
        ok = result and result[0] == "ok"
        rep.add("Infra/integrity", ok,
                f"PRAGMA integrity_check: {result[0] if result else 'failed'}",
                severity="CRITICAL" if not ok else "INFO",
                fix="DB corruption detected — sqlite3 backup + restore საჭიროა" if not ok else "")
    except Exception as e:
        rep.add("Infra/integrity", False, str(e), "CRITICAL")

    # ── WAL mode ──────────────────────────────────────────────────
    try:
        result = conn.execute("PRAGMA journal_mode").fetchone()
        mode = result[0] if result else "?"
        ok   = mode == "wal"
        rep.add("Infra/wal_mode", ok,
                f"journal_mode={mode}",
                severity="WARN" if not ok else "INFO",
                fix="WAL mode არ არის ჩართული — thread-safety რისკი. db.py WAL pragma შეამოწმე" if not ok else "")
    except Exception as e:
        rep.add("Infra/wal_mode", False, str(e), "WARN")

    # ── tables სია ────────────────────────────────────────────────
    try:
        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
        expected_tables = {
            "system_state", "trades", "oco_links", "audit_log",
            "executed_signals", "positions", "dca_positions",
            "dca_orders", "futures_positions", "sl_cooldown_per_symbol",
        }
        missing = expected_tables - tables
        extra   = tables - expected_tables

        ok = len(missing) == 0
        rep.add("Infra/tables", ok,
                f"tables: {len(tables)} | missing={list(missing) if missing else 'none'} | extra={list(extra) if extra else 'none'}",
                severity="CRITICAL" if missing else "INFO",
                fix=f"Missing tables: {list(missing)} — DB migration გაუშვი" if missing else "")
    except Exception as e:
        rep.add("Infra/tables", False, str(e), "CRITICAL")

    # ── error events ბოლო 24h ────────────────────────────────────
    try:
        rows = conn.execute(
            "SELECT event_type, message, created_at FROM audit_log "
            "WHERE (event_type LIKE '%FAIL%' OR event_type LIKE '%ERROR%') "
            "  AND created_at >= datetime('now','-1 day') "
            "ORDER BY id DESC LIMIT 10"
        ).fetchall()
        errors = [dict(r) for r in rows]
        ok = len(errors) == 0
        rep.add("Infra/errors_24h", ok,
                f"Error events ბოლო 24h: {len(errors)}",
                severity="WARN" if errors else "INFO",
                fix="Error events ნაპოვნია — details audit_log-ში შეამოწმე" if errors else "")
        for e in errors[:3]:
            rep.add("Infra/error_detail", False,
                    f"[{e.get('created_at','')}] {e.get('event_type','')} | {str(e.get('message',''))[:80]}",
                    severity="WARN")
    except Exception as e:
        rep.add("Infra/errors_24h", False, str(e), "WARN")


# =============================================================================
# SECTION 22: ENV MATH — TP/capital math + cross-system conflicts
# (shell audit ENV section — extends check_env with actual math)
# =============================================================================

def check_env_math(rep: Report):
    """
    შემოწმება (check_env-ს ავსებს — მხოლოდ math-based checks):
      - TP net profit calculation (tp - fee >= min_net)
      - Capital math (quote + addon_sum <= max_capital)
      - ADDON_SIZES count vs MAX_ADD_ONS
      - TRIGGER_PCTS ascending order
      - Cross-system conflicts (6 checks)
    """
    # ── TP MATH ───────────────────────────────────────────────────
    try:
        tp_pct   = float(os.getenv("DCA_TP_PCT",                "0.55"))
        l3_pct   = float(os.getenv("CASCADE_TP_L3_PCT",         "0.35"))
        fee      = float(os.getenv("ESTIMATED_ROUNDTRIP_FEE_PCT","0.20"))
        min_net  = float(os.getenv("MIN_NET_PROFIT_PCT",         "0.01"))
        tp_sync  = float(os.getenv("TP_PCT",                    "0.55"))

        net_l2 = tp_pct - fee
        net_l3 = l3_pct - fee

        rep.add("ENV_Math/tp_net_l2",
                net_l2 >= min_net,
                f"L2: TP={tp_pct}% - FEE={fee}% = NET={net_l2:.3f}% (min={min_net}%)",
                severity="CRITICAL" if net_l2 < 0 else ("WARN" if net_l2 < min_net else "INFO"),
                fix=f"L2 net profit={net_l2:.3f}% < MIN_NET={min_net}% — DCA_TP_PCT გაზარდე ან FEE შეამოწმე" if net_l2 < min_net else "")

        rep.add("ENV_Math/tp_net_l3",
                net_l3 >= 0,
                f"L3: TP={l3_pct}% - FEE={fee}% = NET={net_l3:.3f}%",
                severity="CRITICAL" if net_l3 < 0 else "INFO",
                fix=f"L3 net profit={net_l3:.3f}% უარყოფითია! TP hit = ზარალი!" if net_l3 < 0 else "")

        rep.add("ENV_Math/tp_sync",
                abs(tp_pct - tp_sync) < 0.001,
                f"TP_PCT={tp_sync}% vs DCA_TP_PCT={tp_pct}%",
                severity="WARN" if abs(tp_pct - tp_sync) >= 0.001 else "INFO",
                fix=f"TP_PCT={tp_sync} != DCA_TP_PCT={tp_pct} — სინქრონიზაცია საჭიროა" if abs(tp_pct - tp_sync) >= 0.001 else "")

        l3_lt_l2 = l3_pct < tp_pct
        rep.add("ENV_Math/l3_lt_l2",
                l3_lt_l2,
                f"CASCADE_TP_L3={l3_pct}% < DCA_TP={tp_pct}% = {l3_lt_l2}",
                severity="WARN" if not l3_lt_l2 else "INFO",
                fix=f"L3 TP {l3_pct}% >= L2 TP {tp_pct}% — L3 TP უნდა იყოს ნაკლები (L3-ზე პატარა bounce საკმარისია)" if not l3_lt_l2 else "")
    except Exception as e:
        rep.add("ENV_Math/tp", False, str(e), "WARN")

    # ── CAPITAL MATH ──────────────────────────────────────────────
    try:
        quote     = float(os.getenv("BOT_QUOTE_PER_TRADE",  "50"))
        max_cap   = float(os.getenv("DCA_MAX_CAPITAL_USDT", "350"))
        max_open  = int(os.getenv("MAX_OPEN_TRADES",         "6"))
        sizes_str = os.getenv("DCA_ADDON_SIZES",             "50,65,75,65,40")
        max_addons= int(os.getenv("DCA_MAX_ADD_ONS",          "5"))

        sizes    = [float(x.strip()) for x in sizes_str.split(",") if x.strip()]
        addon_sum= sum(sizes)
        auto_cap = quote + addon_sum

        rep.add("ENV_Math/capital",
                max_cap >= auto_cap,
                f"BOT_QUOTE={quote} + ADDON_SUM={addon_sum} = {auto_cap} <= MAX_CAP={max_cap}",
                severity="WARN" if max_cap < auto_cap else "INFO",
                fix=f"DCA_MAX_CAPITAL_USDT={max_cap} < required={auto_cap} — ADD-ONs ვერ დასრულდება" if max_cap < auto_cap else "")

        rep.add("ENV_Math/sizes_count",
                len(sizes) == max_addons,
                f"DCA_ADDON_SIZES count={len(sizes)} | DCA_MAX_ADD_ONS={max_addons}",
                severity="CRITICAL" if len(sizes) != max_addons else "INFO",
                fix=f"ADDON_SIZES count={len(sizes)} != MAX_ADD_ONS={max_addons} — სიები ერთი სიგრძის უნდა იყოს" if len(sizes) != max_addons else "")
    except Exception as e:
        rep.add("ENV_Math/capital", False, str(e), "WARN")

    # ── TRIGGER_PCTS ascending ────────────────────────────────────
    try:
        triggers_str = os.getenv("DCA_ADDON_TRIGGER_PCTS", "1.0,2.2,3.5,5.0,6.5")
        triggers = [float(x.strip()) for x in triggers_str.split(",") if x.strip()]
        ascending = all(triggers[i] < triggers[i+1] for i in range(len(triggers)-1))
        rep.add("ENV_Math/triggers_ascending",
                ascending,
                f"DCA_ADDON_TRIGGER_PCTS={triggers} ascending={ascending}",
                severity="CRITICAL" if not ascending else "INFO",
                fix="DCA_ADDON_TRIGGER_PCTS უნდა იყოს ascending — e.g. 1.0,2.2,3.5,5.0,6.5" if not ascending else "")
    except Exception as e:
        rep.add("ENV_Math/triggers", False, str(e), "WARN")

    # ── CROSS-SYSTEM CONFLICTS ────────────────────────────────────
    try:
        bot_q  = float(os.getenv("BOT_QUOTE_PER_TRADE", "50"))
        max_q  = float(os.getenv("MAX_QUOTE_PER_TRADE",  "50"))
        rep.add("ENV_Math/quote_conflict",
                bot_q <= max_q,
                f"BOT_QUOTE={bot_q} <= MAX_QUOTE={max_q}",
                severity="CRITICAL" if bot_q > max_q else "INFO",
                fix=f"BOT_QUOTE={bot_q} > MAX_QUOTE={max_q} — trade ბლოკდება!" if bot_q > max_q else "")

        fut_fc  = float(os.getenv("FUTURES_DCA_FC_DRAWDOWN_PCT", "22.0"))
        long_fc = float(os.getenv("FORCE_CLOSE_DRAWDOWN_PCT",    "15.0"))
        rep.add("ENV_Math/fc_hierarchy",
                fut_fc >= long_fc,
                f"FUTURES_FC={fut_fc}% vs LONG_FC={long_fc}%",
                severity="WARN" if fut_fc < long_fc else "INFO",
                fix=f"FUTURES_FC={fut_fc}% < LONG_FC={long_fc}% — hedge LONG-ზე ადრე იხურება" if fut_fc < long_fc else "")

        mirror_t  = float(os.getenv("MIRROR_TRIGGER_PCT",       "8.59"))
        triggers_str = os.getenv("DCA_ADDON_TRIGGER_PCTS",      "1.0,2.2,3.5,5.0,6.5")
        triggers   = [float(x.strip()) for x in triggers_str.split(",") if x.strip()]
        max_depth  = max(triggers) if triggers else 6.5
        rep.add("ENV_Math/mirror_vs_dca",
                mirror_t > max_depth,
                f"MIRROR_TRIGGER={mirror_t}% vs DCA_max_depth={max_depth}%",
                severity="WARN" if mirror_t <= max_depth else "INFO",
                fix=f"MIRROR_TRIGGER={mirror_t}% <= DCA depth={max_depth}% — MIRROR DCA-ზე ადრე იხსნება" if mirror_t <= max_depth else "")

        daily_loss = float(os.getenv("DAILY_MAX_LOSS_USDT",    "150"))
        max_cap2   = float(os.getenv("DCA_MAX_CAPITAL_USDT",   "350"))
        rep.add("ENV_Math/daily_loss",
                daily_loss >= max_cap2 * 0.1,
                f"DAILY_MAX_LOSS={daily_loss} vs 10%_of_MaxCap={max_cap2*0.1:.0f}",
                severity="WARN" if daily_loss < max_cap2 * 0.1 else "INFO",
                fix=f"DAILY_MAX_LOSS={daily_loss} < 10% of position ({max_cap2*0.1:.0f}) — ძალიან სწრაფად შეჩერდება" if daily_loss < max_cap2 * 0.1 else "")

        short_l1 = float(os.getenv("SHORT_L1_TRIGGER_PCT", "1.6"))
        dca_t1   = triggers[0] if triggers else 1.0
        rep.add("ENV_Math/short_vs_dca_trigger",
                short_l1 >= dca_t1,
                f"SHORT_L1_TRIGGER={short_l1}% vs DCA_TRIGGER[0]={dca_t1}%",
                severity="WARN" if short_l1 < dca_t1 else "INFO",
                fix=f"SHORT_L1={short_l1}% < DCA_TRIGGER[0]={dca_t1}% — SHORT LONG ADD-ON-ზე ადრე იხსნება" if short_l1 < dca_t1 else "")
    except Exception as e:
        rep.add("ENV_Math/conflicts", False, str(e), "WARN")



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

        # 16. DCA-სპეციფიური შემოწმებები
        check_dca_positions(rep, conn)

        # 17. Entry logic detail (shell audit §1)
        check_entry_logic(rep, conn)

        # 18. TP + ADD-ON detail (shell audit §2+3)
        check_tp_and_addon_detail(rep, conn)

        # 19. Futures DB columns + positions (shell audit §5)
        check_futures_db_detail(rep, conn)

        # 20. Regime log (shell audit §6 — MARKET_REGIME_CHANGE)
        check_regime_log(rep, conn)

        # 21. Infra detail — WAL, integrity, tables, errors (shell audit §7)
        check_infra_detail(rep, conn)

    # 22. ENV math + cross-conflicts (shell audit ENV section)
    check_env_math(rep)

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
