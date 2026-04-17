#!/bin/bash
# ============================================================
# GENIUS DCA BOT — სრული DEMO დიაგნოსტიკა
# გამოყენება: bash genius_demo_check.sh
# ============================================================

DB="/var/data/genius_bot_v2.db"
G='\033[0;32m'; Y='\033[1;33m'; R='\033[0;31m'; C='\033[0;36m'; B='\033[1m'; NC='\033[0m'

ok()   { echo -e "${G}[OK]${NC}   $1"; }
fail() { echo -e "${R}[FAIL]${NC} $1"; }
warn() { echo -e "${Y}[WARN]${NC} $1"; }
info() { echo -e "${C}[INFO]${NC} $1"; }
sec()  { echo; echo -e "${B}${C}━━━ $1 ━━━${NC}"; }

echo
echo -e "${B}${C}╔══════════════════════════════════════════════════════╗${NC}"
echo -e "${B}${C}║     GENIUS DCA BOT — სრული DEMO დიაგნოსტიკა        ║${NC}"
echo -e "${B}${C}║              $(date '+%Y-%m-%d %H:%M:%S')              ║${NC}"
echo -e "${B}${C}╚══════════════════════════════════════════════════════╝${NC}"

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
sec "1. DB ფაილი"
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if [ -f "$DB" ]; then
    ok "DB არსებობს: $DB"
    info "DB ზომა: $(du -sh $DB | cut -f1)"
else
    fail "DB არ არსებობს: $DB"
    exit 1
fi

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
sec "2. ცხრილები"
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
python3 - "$DB" << 'PYEOF'
import sqlite3, sys
db = sys.argv[1]
needed = ['system_state','trades','dca_positions','dca_orders',
          'futures_positions','audit_log','oco_links','executed_signals']
G='\033[0;32m'; R='\033[0;31m'; NC='\033[0m'
conn = sqlite3.connect(db)
tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
for t in needed:
    if t in tables:
        cnt = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        print(f"{G}[OK]{NC}   {t} ✅ ({cnt} rows)")
    else:
        print(f"{R}[FAIL]{NC} {t} — არ არსებობს!")
conn.close()
PYEOF

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
sec "3. System State"
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
python3 - "$DB" << 'PYEOF'
import sqlite3, sys
db = sys.argv[1]
G='\033[0;32m'; R='\033[0;31m'; Y='\033[1;33m'; NC='\033[0m'
conn = sqlite3.connect(db)
r = conn.execute("SELECT status, startup_sync_ok, kill_switch FROM system_state WHERE id=1").fetchone()
if not r:
    print(f"{R}[FAIL]{NC} system_state row არ არსებობს!")
else:
    status, sync, kill = str(r[0]).upper(), int(r[1] or 0), int(r[2] or 0)
    print(f"{G if status in ('ACTIVE','RUNNING') else Y}[{'OK' if status in ('ACTIVE','RUNNING') else 'WARN'}]{NC}   status={status}")
    print(f"{G if sync==1 else Y}[{'OK' if sync==1 else 'WARN'}]{NC}   startup_sync_ok={sync}")
    print(f"{G if kill==0 else R}[{'OK' if kill==0 else 'FAIL'}]{NC}   kill_switch={kill}")
conn.close()
PYEOF

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
sec "4. ღია DCA პოზიციები (LONG)"
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
python3 - "$DB" << 'PYEOF'
import sqlite3, sys
db = sys.argv[1]
G='\033[0;32m'; R='\033[0;31m'; Y='\033[1;33m'; C='\033[0;36m'; NC='\033[0m'
conn = sqlite3.connect(db)
rows = conn.execute("""
    SELECT symbol, avg_entry_price, current_tp_price, total_quote_spent, add_on_count, opened_at
    FROM dca_positions WHERE status='OPEN' ORDER BY opened_at
""").fetchall()
if not rows:
    print(f"{C}[INFO]{NC} ღია პოზიცია არ არის")
else:
    print(f"{C}[INFO]{NC} სულ ღია: {len(rows)} | ინვესტირებული: {sum(float(r[3]) for r in rows):.2f} USDT")
    for r in rows:
        sym, avg, tp, quote, addons, opened = r
        avg, tp, quote = float(avg or 0), float(tp or 0), float(quote or 0)
        tp_ok = tp > avg
        pct = (tp - avg) / avg * 100 if avg > 0 else 0
        status_icon = G if tp_ok else R
        print(f"  {status_icon}●{NC} {sym:<18} avg={avg:.2f}  tp={tp:.2f} ({pct:+.2f}%)  invested={quote:.1f}$  add-ons={addons}  opened={opened[:16]}")
        if not tp_ok:
            print(f"    {R}⚠️  TP <= avg — გასასწორებელია!{NC}")
conn.close()
PYEOF

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
sec "5. SHORT პოზიციები (Futures)"
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
python3 - "$DB" << 'PYEOF'
import sqlite3, sys
db = sys.argv[1]
G='\033[0;32m'; R='\033[0;31m'; Y='\033[1;33m'; C='\033[0;36m'; NC='\033[0m'
conn = sqlite3.connect(db)
# ღია SHORT-ები
open_s = conn.execute("""
    SELECT symbol, direction, entry_price, tp_price, sl_price, quote_in, leverage, mode, opened_at
    FROM futures_positions WHERE status='OPEN'
""").fetchall()
# დახურული SHORT-ები
closed_s = conn.execute("""
    SELECT symbol, outcome, pnl_quote, pnl_pct, closed_at
    FROM futures_positions WHERE status='CLOSED'
    ORDER BY closed_at DESC LIMIT 5
""").fetchall()

if not open_s:
    print(f"{C}[INFO]{NC} ღია SHORT არ არის (BEAR რეჟიმი საჭიროა)")
else:
    print(f"{Y}[WARN]{NC} ღია SHORT-ები: {len(open_s)}")
    for r in open_s:
        sym, direction, entry, tp, sl, quote, lev, mode, opened = r
        print(f"  ● {sym} {direction} | entry={float(entry):.2f} tp={float(tp):.2f} sl={float(sl):.2f} | {quote}$ x{lev} | {mode} | {opened[:16]}")

if not closed_s:
    print(f"{C}[INFO]{NC} დახურული SHORT არ არის ჯერ")
else:
    print(f"\n{C}[INFO]{NC} ბოლო დახურული SHORT-ები:")
    for r in closed_s:
        sym, outcome, pnl, pct, closed = r
        color = G if float(pnl or 0) >= 0 else R
        print(f"  {color}● {sym} {outcome} | pnl={float(pnl or 0):+.4f}$ ({float(pct or 0):+.2f}%) | {closed[:16]}{NC}")
conn.close()
PYEOF

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
sec "6. Performance სტატისტიკა"
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
python3 - "$DB" << 'PYEOF'
import sqlite3, sys
db = sys.argv[1]
G='\033[0;32m'; R='\033[0;31m'; C='\033[0;36m'; NC='\033[0m'
conn = sqlite3.connect(db)
r = conn.execute("""
    SELECT COUNT(*), 
           SUM(CASE WHEN outcome='TP' THEN 1 ELSE 0 END),
           SUM(CASE WHEN outcome NOT IN ('TP','CASCADE_EXCHANGE') AND pnl_quote < 0 THEN 1 ELSE 0 END),
           SUM(CASE WHEN outcome='CASCADE_EXCHANGE' THEN 1 ELSE 0 END),
           COALESCE(SUM(pnl_quote), 0)
    FROM trades WHERE outcome IS NOT NULL
""").fetchone()
total, wins, losses, cascades, pnl = r
winrate = wins/total*100 if total else 0
print(f"{C}[INFO]{NC} სულ დახურული: {total}")
print(f"{G}[OK]{NC}   TP wins:    {wins}")
print(f"{'[OK]' if losses==0 else '[WARN]'}   Losses:     {losses}")
print(f"{C}[INFO]{NC} CASCADE:     {cascades}")
print(f"{G if pnl>=0 else R}[{'OK' if pnl>=0 else 'WARN'}]{NC}   სულ PnL:    {pnl:+.4f} USDT")
print(f"{G if winrate>=90 else C}[INFO]{NC} Winrate:    {winrate:.2f}%")

# ღია პოზიციების კაპიტალი
open_r = conn.execute("""
    SELECT COUNT(*), COALESCE(SUM(total_quote_spent), 0)
    FROM dca_positions WHERE status='OPEN'
""").fetchone()
print(f"\n{C}[INFO]{NC} ღია პოზიციები: {open_r[0]} | კაპიტალი: {open_r[1]:.2f} USDT")
conn.close()
PYEOF

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
sec "7. FUTURES_ENABLED შემოწმება"
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
python3 - << 'PYEOF'
import os
G='\033[0;32m'; R='\033[0;31m'; Y='\033[1;33m'; C='\033[0;36m'; NC='\033[0m'
vals = {
    'FUTURES_ENABLED':  os.getenv('FUTURES_ENABLED', 'NOT SET'),
    'FUTURES_MODE':     os.getenv('FUTURES_MODE', 'NOT SET'),
    'FUTURES_QUOTE':    os.getenv('FUTURES_QUOTE', 'NOT SET'),
    'FUTURES_LEVERAGE': os.getenv('FUTURES_LEVERAGE', 'NOT SET'),
    'FUTURES_TP_PCT':   os.getenv('FUTURES_TP_PCT', 'NOT SET'),
    'FUTURES_SL_PCT':   os.getenv('FUTURES_SL_PCT', 'NOT SET'),
    'FUTURES_MAX_OPEN': os.getenv('FUTURES_MAX_OPEN', 'NOT SET'),
}
for k, v in vals.items():
    ok = v not in ('NOT SET', '', 'false', 'False')
    if k == 'FUTURES_ENABLED':
        color = G if v.lower() == 'true' else Y
        print(f"{color}[{'OK' if v.lower()=='true' else 'WARN'}]{NC}   {k}={v}")
    else:
        print(f"{C}[INFO]{NC} {k}={v}")
PYEOF

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
sec "8. futures_engine.py import ტესტი"
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
cd /opt/render/project/src
result=$(PYTHONPATH=. python3 -c "from execution.futures_engine import FuturesEngine; print('OK')" 2>&1)
if [ "$result" = "OK" ]; then
    ok "futures_engine import — OK ✅"
else
    fail "futures_engine import FAIL: $result"
fi

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
sec "9. Memory გამოყენება"
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
python3 - << 'PYEOF'
import resource
G='\033[0;32m'; Y='\033[1;33m'; R='\033[0;31m'; NC='\033[0m'
mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
color = G if mem < 400 else (Y if mem < 480 else R)
label = 'OK' if mem < 400 else ('WARN' if mem < 480 else 'FAIL')
print(f"{color}[{label}]{NC}   Memory: {mem:.0f}MB (limit≈512MB)")
PYEOF

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
sec "10. ბოლო 5 audit event"
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
python3 - "$DB" << 'PYEOF'
import sqlite3, sys
db = sys.argv[1]
C='\033[0;36m'; NC='\033[0m'
conn = sqlite3.connect(db)
rows = conn.execute("""
    SELECT event_type, message, created_at 
    FROM audit_log ORDER BY id DESC LIMIT 5
""").fetchall()
for r in rows:
    print(f"  {C}●{NC} [{r[2][:16]}] {r[0]} | {str(r[1])[:60]}")
conn.close()
PYEOF

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo
echo -e "${B}${G}✅ დიაგნოსტიკა დასრულდა!${NC}"
echo
