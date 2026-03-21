#!/bin/bash
# ============================================================
# GENIUS BOT — სრული დიაგნოსტიკა
# გამოყენება: bash /opt/render/project/src/execution/diagnose.sh
# ============================================================

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

PASS=0
FAIL_COUNT=0
WARN_COUNT=0

ok()   { echo -e "${GREEN}[OK]${NC}   $1"; ((PASS++)); }
fail() { echo -e "${RED}[FAIL]${NC} $1"; ((FAIL_COUNT++)); }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; ((WARN_COUNT++)); }
info() { echo -e "${CYAN}[INFO]${NC} $1"; }
section() { echo; echo -e "${BOLD}${CYAN}━━━ $1 ━━━${NC}"; }

echo
echo -e "${BOLD}╔══════════════════════════════════════════╗${NC}"
echo -e "${BOLD}║     GENIUS BOT — სრული დიაგნოსტიკა      ║${NC}"
echo -e "${BOLD}╚══════════════════════════════════════════╝${NC}"
echo -e "$(date '+%Y-%m-%d %H:%M:%S')"

# ─────────────────────────────────────────
section "1. Python ფაილები — სინტაქსი"
# ─────────────────────────────────────────
BASE="/opt/render/project/src/execution"
FILES=(
    "signal_generator.py"
    "execution_engine.py"
    "excel_live_core.py"
    "exchange_client.py"
    "kill_switch.py"
    "main.py"
    "telegram_notifier.py"
    "diagnostics_pro.py"
    "my_adapter.py"
)
for f in "${FILES[@]}"; do
    path="$BASE/$f"
    if [ -f "$path" ]; then
        err=$(python3 -c "import ast; ast.parse(open('$path').read())" 2>&1)
        if [ -z "$err" ]; then
            ok "$f — სინტაქსი OK"
        else
            fail "$f — SyntaxError: $(echo $err | head -c 80)"
        fi
    else
        fail "$f — ფაილი არ მოიძებნა"
    fi
done

# ─────────────────────────────────────────
section "2. DB კავშირი და ცხრილები"
# ─────────────────────────────────────────
DB_PATH="${DB_PATH:-/var/data/genius_bot_v2.db}"
if [ -f "$DB_PATH" ]; then
    ok "DB ფაილი: $DB_PATH"
    size=$(du -sh "$DB_PATH" 2>/dev/null | cut -f1)
    info "DB ზომა: $size"
    python3 - "$DB_PATH" << 'PYEOF'
import sqlite3, sys
db = sys.argv[1]
GREEN='\033[0;32m'; RED='\033[0;31m'; NC='\033[0m'
try:
    conn = sqlite3.connect(db)
    tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
    needed = ['system_state','trades','oco_links','audit_log','executed_signals']
    for t in needed:
        if t in tables:
            print(f'{GREEN}[OK]{NC}   ცხრილი {t} არსებობს')
        else:
            print(f'{RED}[FAIL]{NC} ცხრილი {t} არ არსებობს')
    conn.close()
except Exception as e:
    print(f'{RED}[FAIL]{NC} DB შეცდომა: {e}')
PYEOF
else
    fail "DB ფაილი არ მოიძებნა: $DB_PATH"
fi

# ─────────────────────────────────────────
section "3. system_state"
# ─────────────────────────────────────────
python3 - "$DB_PATH" << 'PYEOF'
import sqlite3, sys
db = sys.argv[1]
GREEN='\033[0;32m'; RED='\033[0;31m'; YELLOW='\033[1;33m'; NC='\033[0m'
try:
    conn = sqlite3.connect(db)
    r = conn.execute("SELECT status,startup_sync_ok,kill_switch FROM system_state WHERE id=1").fetchone()
    if not r:
        print(f'{RED}[FAIL]{NC} system_state row არ არსებობს')
    else:
        status, sync, kill = str(r[0]).upper(), int(r[1] or 0), int(r[2] or 0)
        if status in ('ACTIVE','RUNNING'):
            print(f'{GREEN}[OK]{NC}   status={status}')
        else:
            print(f'{YELLOW}[WARN]{NC} status={status} — ACTIVE/RUNNING უნდა იყოს')
        if sync == 1:
            print(f'{GREEN}[OK]{NC}   startup_sync_ok=1')
        else:
            print(f'{YELLOW}[WARN]{NC} startup_sync_ok=0 — ბოტი შესაძლოა PAUSED')
        if kill == 0:
            print(f'{GREEN}[OK]{NC}   kill_switch=OFF')
        else:
            print(f'{RED}[FAIL]{NC} kill_switch=ON — ბოტი სრულად BLOCKED!')
    conn.close()
except Exception as e:
    print(f'{RED}[FAIL]{NC} system_state: {e}')
PYEOF

# ─────────────────────────────────────────
section "4. ღია trade-ები"
# ─────────────────────────────────────────
python3 - "$DB_PATH" << 'PYEOF'
import sqlite3, sys
db = sys.argv[1]
GREEN='\033[0;32m'; CYAN='\033[0;36m'; NC='\033[0m'
try:
    conn = sqlite3.connect(db)
    rows = conn.execute("""
        SELECT symbol, qty, quote_in, entry_price, opened_at
        FROM trades WHERE closed_at IS NULL
        ORDER BY opened_at DESC
    """).fetchall()
    if rows:
        print(f'{GREEN}[OK]{NC}   ღია trade-ები: {len(rows)}')
        for r in rows:
            print(f'{CYAN}[INFO]{NC}   └─ {r[0]} qty={r[1]:.6f} invested={r[2]:.2f} USDT entry={r[3]:.4f} opened={r[4]}')
    else:
        print(f'{CYAN}[INFO]{NC} ღია trade-ები: 0')
    conn.close()
except Exception as e:
    print(f'\033[0;31m[FAIL]\033[0m trades: {e}')
PYEOF

# ─────────────────────────────────────────
section "5. ღია OCO links"
# ─────────────────────────────────────────
python3 - "$DB_PATH" << 'PYEOF'
import sqlite3, sys
db = sys.argv[1]
GREEN='\033[0;32m'; CYAN='\033[0;36m'; RED='\033[0;31m'; NC='\033[0m'
try:
    conn = sqlite3.connect(db)
    rows = conn.execute("""
        SELECT id, symbol, status, tp_price, sl_stop_price, amount, created_at
        FROM oco_links WHERE status IN ('ACTIVE','OPEN','ARMED')
        ORDER BY id DESC
    """).fetchall()
    if rows:
        print(f'{GREEN}[OK]{NC}   ღია OCO links: {len(rows)}')
        for r in rows:
            print(f'{CYAN}[INFO]{NC}   └─ link={r[0]} {r[1]} tp={r[3]:.4f} sl={r[4]:.4f} qty={r[5]:.6f}')
    else:
        print(f'{CYAN}[INFO]{NC} ღია OCO: 0')

    # DESYNC check
    broken = conn.execute("SELECT COUNT(*) FROM oco_links WHERE status='DESYNC'").fetchone()[0]
    broken2 = conn.execute("SELECT COUNT(*) FROM oco_links WHERE status='BROKEN'").fetchone()[0]
    if broken > 0:
        print(f'{RED}[FAIL]{NC} DESYNC OCO links: {broken} — ყურადღება!')
    if broken2 > 0:
        print(f'{RED}[FAIL]{NC} BROKEN OCO links: {broken2} — ყურადღება!')
    conn.close()
except Exception as e:
    print(f'\033[0;31m[FAIL]\033[0m oco_links: {e}')
PYEOF

# ─────────────────────────────────────────
section "6. Performance სტატისტიკა"
# ─────────────────────────────────────────
python3 - "$DB_PATH" << 'PYEOF'
import sqlite3, sys
db = sys.argv[1]
GREEN='\033[0;32m'; RED='\033[0;31m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
try:
    conn = sqlite3.connect(db)
    r = conn.execute("""
        SELECT COUNT(*),
               SUM(CASE WHEN pnl_quote>0 THEN 1 ELSE 0 END),
               SUM(CASE WHEN pnl_quote<=0 THEN 1 ELSE 0 END),
               COALESCE(SUM(pnl_quote),0),
               COALESCE(AVG(CASE WHEN pnl_quote>0 THEN pnl_quote END),0),
               COALESCE(ABS(AVG(CASE WHEN pnl_quote<0 THEN pnl_quote END)),0)
        FROM trades WHERE closed_at IS NOT NULL
    """).fetchone()
    total = r[0] or 0
    wins  = r[1] or 0
    losses= r[2] or 0
    pnl   = r[3] or 0.0
    avg_w = r[4] or 0.0
    avg_l = r[5] or 0.0
    wr = wins/total*100 if total else 0
    pf = avg_w/avg_l if avg_l else 0

    print(f'{CYAN}[INFO]{NC} სულ დახურული: {total} | Wins: {wins} | Losses: {losses}')

    if wr >= 40:
        print(f'{GREEN}[OK]{NC}   Winrate: {wr:.1f}%')
    elif wr >= 30:
        print(f'{YELLOW}[WARN]{NC} Winrate: {wr:.1f}% — დაბალია')
    else:
        print(f'{RED}[FAIL]{NC} Winrate: {wr:.1f}% — კრიტიკულად დაბალი')

    if pnl >= 0:
        print(f'{GREEN}[OK]{NC}   Total PnL: +{pnl:.4f} USDT')
    else:
        print(f'{RED}[FAIL]{NC} Total PnL: {pnl:.4f} USDT (ზარალი)')

    if pf >= 1.0:
        print(f'{GREEN}[OK]{NC}   Profit Factor: {pf:.2f}')
    elif pf > 0:
        print(f'{YELLOW}[WARN]{NC} Profit Factor: {pf:.2f} (< 1.0)')

    print(f'{CYAN}[INFO]{NC} Avg Win: +{avg_w:.4f} USDT | Avg Loss: -{avg_l:.4f} USDT')

    # open trades exposure
    r2 = conn.execute("SELECT COUNT(*), COALESCE(SUM(quote_in),0) FROM trades WHERE closed_at IS NULL").fetchone()
    open_n, open_q = r2[0], r2[1]
    print(f'{CYAN}[INFO]{NC} ღია: {open_n} trade | exposure: {open_q:.2f} USDT')
    conn.close()
except Exception as e:
    print(f'\033[0;31m[FAIL]\033[0m stats: {e}')
PYEOF

# ─────────────────────────────────────────
section "7. ENV ცვლადები"
# ─────────────────────────────────────────
check_env() {
    local key=$1 expected=$2
    val="${!key}"
    if [ -z "$val" ]; then
        echo -e "${RED}[FAIL]${NC} $key — დაყენებული არ არის"
        ((FAIL_COUNT++))
    elif [ -n "$expected" ] && [ "$val" != "$expected" ]; then
        echo -e "${YELLOW}[WARN]${NC} $key=$val (expected: $expected)"
        ((WARN_COUNT++))
    else
        echo -e "${GREEN}[OK]${NC}   $key=$val"
        ((PASS++))
    fi
}
check_env "MODE"               "LIVE"
check_env "KILL_SWITCH"        "false"
check_env "LIVE_CONFIRMATION"  "true"
check_env "ALLOW_LIVE_SIGNALS" "true"
check_env "BOT_QUOTE_PER_TRADE"
check_env "TP_PCT"
check_env "SL_PCT"
check_env "BOT_SYMBOLS"
check_env "WEIGHT_TREND"
check_env "THRESHOLD_CONF"
check_env "SL_COOLDOWN_AFTER_N"

# ─────────────────────────────────────────
section "8. SL Cooldown სტატუსი"
# ─────────────────────────────────────────
python3 - "$DB_PATH" << 'PYEOF'
import sqlite3, sys
db = sys.argv[1]
GREEN='\033[0;32m'; RED='\033[0;31m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
try:
    conn = sqlite3.connect(db)
    rows = conn.execute("""
        SELECT outcome, closed_at FROM trades
        WHERE closed_at IS NOT NULL
        AND closed_at >= datetime('now', '-1 hour')
        ORDER BY closed_at DESC
    """).fetchall()
    sl_count = sum(1 for r in rows if str(r[0]).upper() == 'SL')
    tp_count = sum(1 for r in rows if str(r[0]).upper() == 'TP')
    print(f'{CYAN}[INFO]{NC} ბოლო 1 სთ: {sl_count} SL | {tp_count} TP')
    limit = 2
    if sl_count >= limit:
        print(f'{YELLOW}[WARN]{NC} {sl_count} consecutive SL — Cooldown შეიძლება active იყოს (limit={limit})')
    else:
        print(f'{GREEN}[OK]{NC}   SL count OK ({sl_count}/{limit})')
    conn.close()
except Exception as e:
    print(f'\033[0;36m[INFO]\033[0m SL check: {e}')
PYEOF

# ─────────────────────────────────────────
section "9. signal_outbox"
# ─────────────────────────────────────────
OUTBOX="${SIGNAL_OUTBOX_PATH:-/var/data/signal_outbox.json}"
if [ -f "$OUTBOX" ]; then
    size=$(wc -c < "$OUTBOX" 2>/dev/null)
    age=$(( $(date +%s) - $(stat -c %Y "$OUTBOX" 2>/dev/null || echo 0) ))
    info "Outbox: ${size} bytes | ${age}s ago"
    if [ "$size" -gt 50000 ]; then
        warn "Outbox ძალიან დიდია (${size} bytes) — signal-ები შეიძლება დაგროვდა"
    else
        ok "Outbox ზომა OK (${size} bytes)"
    fi
else
    info "Outbox ჯერ არ შექმნილა — ნორმალურია"
fi

# ─────────────────────────────────────────
section "10. ბოლო audit events"
# ─────────────────────────────────────────
python3 - "$DB_PATH" << 'PYEOF'
import sqlite3, sys
db = sys.argv[1]
CYAN='\033[0;36m'; NC='\033[0m'
try:
    conn = sqlite3.connect(db)
    rows = conn.execute("""
        SELECT event_type, message, created_at FROM audit_log
        ORDER BY id DESC LIMIT 5
    """).fetchall()
    print(f'{CYAN}[INFO]{NC} ბოლო 5 event:')
    for r in rows:
        msg = str(r[1])[:70] if r[1] else ''
        print(f'{CYAN}[INFO]{NC}   └─ [{r[2]}] {r[0]}: {msg}')
    conn.close()
except Exception as e:
    print(f'\033[0;31m[FAIL]\033[0m audit_log: {e}')
PYEOF

# ─────────────────────────────────────────
section "საბოლოო ვერდიქტი"
# ─────────────────────────────────────────
TOTAL=$((PASS + FAIL_COUNT + WARN_COUNT))
echo
echo -e "  სულ შემოწმება : ${TOTAL}"
echo -e "  ${GREEN}OK   : ${PASS}${NC}"
echo -e "  ${YELLOW}WARN : ${WARN_COUNT}${NC}"
echo -e "  ${RED}FAIL : ${FAIL_COUNT}${NC}"
echo

if [ "$FAIL_COUNT" -eq 0 ] && [ "$WARN_COUNT" -eq 0 ]; then
    echo -e "${GREEN}${BOLD}  ✅ ბოტი სრულიად ჯანმრთელია!${NC}"
elif [ "$FAIL_COUNT" -eq 0 ]; then
    echo -e "${YELLOW}${BOLD}  ⚠️  ბოტი მუშაობს — ${WARN_COUNT} გაფრთხილება${NC}"
else
    echo -e "${RED}${BOLD}  ❌ ${FAIL_COUNT} კრიტიკული პრობლემა გამოვლინდა!${NC}"
fi
echo
