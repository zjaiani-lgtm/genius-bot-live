#!/bin/bash
# ============================================================
# GENIUS BOT — სრული დიაგნოსტიკა
# გამოყენება: bash diagnose.sh
# ============================================================

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

OK="${GREEN}[✓ OK]${NC}"
FAIL="${RED}[✗ FAIL]${NC}"
WARN="${YELLOW}[⚠ WARN]${NC}"
INFO="${CYAN}[ℹ INFO]${NC}"

PASS=0
FAIL_COUNT=0
WARN_COUNT=0

pass() { echo -e "$OK $1"; ((PASS++)); }
fail() { echo -e "$FAIL $1"; ((FAIL_COUNT++)); }
warn() { echo -e "$WARN $1"; ((WARN_COUNT++)); }
info() { echo -e "$INFO $1"; }
section() { echo; echo -e "${BOLD}${BLUE}━━━ $1 ━━━${NC}"; }

echo
echo -e "${BOLD}╔══════════════════════════════════════════╗${NC}"
echo -e "${BOLD}║     GENIUS BOT — სრული დიაგნოსტიკა      ║${NC}"
echo -e "${BOLD}╚══════════════════════════════════════════╝${NC}"
echo -e "$(date '+%Y-%m-%d %H:%M:%S')"

# ─────────────────────────────────────────
section "1. Python ფაილები"
# ─────────────────────────────────────────
FILES=(
    "execution/signal_generator.py"
    "execution/execution_engine.py"
    "execution/excel_live_core.py"
    "execution/exchange_client.py"
    "execution/kill_switch.py"
    "execution/main.py"
    "execution/telegram_notifier.py"
    "execution/diagnostics_pro.py"
)
BASE="/opt/render/project/src"
for f in "${FILES[@]}"; do
    path="$BASE/$f"
    if [ -f "$path" ]; then
        err=$(python3 -c "import ast; ast.parse(open('$path').read())" 2>&1)
        if [ -z "$err" ]; then
            pass "$f — სინტაქსი OK"
        else
            fail "$f — SyntaxError: $err"
        fi
    else
        fail "$f — ფაილი არ მოიძებნა"
    fi
done

# ─────────────────────────────────────────
section "2. DB კავშირი"
# ─────────────────────────────────────────
DB_PATH="${DB_PATH:-/var/data/genius_bot_v2.db}"
if [ -f "$DB_PATH" ]; then
    pass "DB ფაილი არსებობს: $DB_PATH"
    size=$(du -sh "$DB_PATH" 2>/dev/null | cut -f1)
    info "DB ზომა: $size"
    tables=$(python3 -c "
import sqlite3
try:
    conn = sqlite3.connect('$DB_PATH')
    cur = conn.cursor()
    cur.execute(\"SELECT name FROM sqlite_master WHERE type='table'\")
    print(' '.join([r[0] for r in cur.fetchall()]))
    conn.close()
except Exception as e:
    print('ERROR: ' + str(e))
" 2>/dev/null)
    if [[ "$tables" == *"ERROR"* ]]; then
        fail "DB კავშირი ვერ შედგა: $tables"
    else
        pass "DB ცხრილები: $tables"
    fi
else
    fail "DB ფაილი არ მოიძებნა: $DB_PATH"
fi

# ─────────────────────────────────────────
section "3. system_state შემოწმება"
# ─────────────────────────────────────────
python3 << 'PYEOF'
import sqlite3, os, sys
db = os.getenv("DB_PATH", "/var/data/genius_bot_v2.db")
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
try:
    conn = sqlite3.connect(db)
    row = conn.execute("SELECT * FROM system_state WHERE id=1").fetchone()
    if not row:
        print(f"\033[0;31m[✗ FAIL]\033[0m system_state row არ არსებობს")
        sys.exit(1)
    status = str(row[1]).upper()
    sync   = int(row[2] or 0)
    kill   = int(row[3] or 0)

    if status in ("ACTIVE", "RUNNING"):
        print(f"\033[0;32m[✓ OK]\033[0m   system status={status}")
    else:
        print(f"\033[1;33m[⚠ WARN]\033[0m system status={status} (ACTIVE/RUNNING უნდა იყოს)")

    if sync == 1:
        print(f"\033[0;32m[✓ OK]\033[0m   startup_sync_ok=1")
    else:
        print(f"\033[1;33m[⚠ WARN]\033[0m startup_sync_ok=0 (ბოტი შესაძლოა PAUSED-ში იყოს)")

    if kill == 0:
        print(f"\033[0;32m[✓ OK]\033[0m   kill_switch=0 (OFF)")
    else:
        print(f"\033[0;31m[✗ FAIL]\033[0m kill_switch=1 — ბოტი BLOCKED!")

    conn.close()
except Exception as e:
    print(f"\033[0;31m[✗ FAIL]\033[0m system_state შეცდომა: {e}")
PYEOF

# ─────────────────────────────────────────
section "4. ღია trade-ები"
# ─────────────────────────────────────────
python3 << 'PYEOF'
import sqlite3, os
db = os.getenv("DB_PATH", "/var/data/genius_bot_v2.db")
try:
    conn = sqlite3.connect(db)
    rows = conn.execute("""
        SELECT signal_id, symbol, qty, quote_in, entry_price, opened_at
        FROM trades WHERE closed_at IS NULL
    """).fetchall()
    if rows:
        print(f"\033[0;32m[✓ OK]\033[0m   ღია trade-ები: {len(rows)}")
        for r in rows:
            print(f"         └─ {r[1]} qty={r[2]:.6f} entry={r[4]:.4f} opened={r[5]}")
    else:
        print(f"\033[0;36m[ℹ INFO]\033[0m ღია trade-ები: 0")
    conn.close()
except Exception as e:
    print(f"\033[0;31m[✗ FAIL]\033[0m trades query: {e}")
PYEOF

# ─────────────────────────────────────────
section "5. ღია OCO links"
# ─────────────────────────────────────────
python3 << 'PYEOF'
import sqlite3, os
db = os.getenv("DB_PATH", "/var/data/genius_bot_v2.db")
try:
    conn = sqlite3.connect(db)
    rows = conn.execute("""
        SELECT id, symbol, status, tp_price, sl_stop_price, created_at
        FROM oco_links WHERE status IN ('ACTIVE','OPEN','ARMED')
    """).fetchall()
    if rows:
        print(f"\033[0;32m[✓ OK]\033[0m   ღია OCO links: {len(rows)}")
        for r in rows:
            print(f"         └─ link={r[0]} {r[1]} status={r[2]} tp={r[3]:.4f} sl={r[4]:.4f}")
    else:
        print(f"\033[0;36m[ℹ INFO]\033[0m ღია OCO links: 0")
    conn.close()
except Exception as e:
    print(f"\033[0;31m[✗ FAIL]\033[0m oco_links query: {e}")
PYEOF

# ─────────────────────────────────────────
section "6. Performance სტატისტიკა"
# ─────────────────────────────────────────
python3 << 'PYEOF'
import sqlite3, os
db = os.getenv("DB_PATH", "/var/data/genius_bot_v2.db")
try:
    conn = sqlite3.connect(db)
    r = conn.execute("""
        SELECT
            COUNT(*) as total,
            SUM(CASE WHEN pnl_quote > 0 THEN 1 ELSE 0 END) as wins,
            SUM(CASE WHEN pnl_quote <= 0 THEN 1 ELSE 0 END) as losses,
            COALESCE(SUM(pnl_quote),0) as total_pnl,
            COALESCE(AVG(CASE WHEN pnl_quote>0 THEN pnl_quote END),0) as avg_win,
            COALESCE(ABS(AVG(CASE WHEN pnl_quote<0 THEN pnl_quote END)),0) as avg_loss
        FROM trades WHERE closed_at IS NOT NULL
    """).fetchone()
    total,wins,losses,pnl,avg_win,avg_loss = r
    wins = wins or 0; losses = losses or 0
    winrate = (wins/total*100) if total else 0
    pf = (avg_win/avg_loss) if avg_loss else 0

    print(f"\033[0;36m[ℹ INFO]\033[0m დახურული trade-ები: {total}")
    print(f"\033[0;36m[ℹ INFO]\033[0m Wins: {wins} | Losses: {losses}")

    if winrate >= 40:
        print(f"\033[0;32m[✓ OK]\033[0m   Winrate: {winrate:.1f}%")
    elif winrate >= 30:
        print(f"\033[1;33m[⚠ WARN]\033[0m Winrate: {winrate:.1f}% (დაბალია)")
    else:
        print(f"\033[0;31m[✗ FAIL]\033[0m Winrate: {winrate:.1f}% (კრიტიკულად დაბალია)")

    if pnl >= 0:
        print(f"\033[0;32m[✓ OK]\033[0m   Total PnL: +{pnl:.4f} USDT")
    else:
        print(f"\033[0;31m[✗ FAIL]\033[0m Total PnL: {pnl:.4f} USDT (ზარალი)")

    if pf >= 1.0:
        print(f"\033[0;32m[✓ OK]\033[0m   Profit Factor: {pf:.2f}")
    else:
        print(f"\033[1;33m[⚠ WARN]\033[0m Profit Factor: {pf:.2f} (< 1.0)")

    print(f"\033[0;36m[ℹ INFO]\033[0m Avg Win: +{avg_win:.4f} | Avg Loss: -{avg_loss:.4f} USDT")
    conn.close()
except Exception as e:
    print(f"\033[0;31m[✗ FAIL]\033[0m stats query: {e}")
PYEOF

# ─────────────────────────────────────────
section "7. ENV ცვლადები"
# ─────────────────────────────────────────
check_env() {
    local key=$1 expected=$2
    val="${!key}"
    if [ -z "$val" ]; then
        fail "$key — დაყენებული არ არის"
    elif [ -n "$expected" ] && [ "$val" != "$expected" ]; then
        warn "$key=$val (expected=$expected)"
    else
        pass "$key=$val"
    fi
}
check_env "MODE" "LIVE"
check_env "KILL_SWITCH" "false"
check_env "LIVE_CONFIRMATION" "true"
check_env "ALLOW_LIVE_SIGNALS" "true"
check_env "BOT_QUOTE_PER_TRADE"
check_env "TP_PCT"
check_env "SL_PCT"
check_env "BOT_SYMBOLS"

# ─────────────────────────────────────────
section "8. signal_outbox შემოწმება"
# ─────────────────────────────────────────
OUTBOX="${SIGNAL_OUTBOX_PATH:-/var/data/signal_outbox.json}"
if [ -f "$OUTBOX" ]; then
    size=$(wc -c < "$OUTBOX")
    age=$(( $(date +%s) - $(stat -c %Y "$OUTBOX" 2>/dev/null || echo 0) ))
    info "Outbox: $OUTBOX (${size} bytes, ${age}s ago)"
    if [ "$size" -gt 10000 ]; then
        warn "Outbox ძალიან დიდია (${size} bytes) — შესაძლოა signal-ები დაგროვდა"
    else
        pass "Outbox ზომა OK"
    fi
else
    info "Outbox ფაილი ჯერ არ შექმნილა (ნორმალურია პირველ გაშვებაზე)"
fi

# ─────────────────────────────────────────
section "9. SL Cooldown სტატუსი"
# ─────────────────────────────────────────
python3 << 'PYEOF'
import sqlite3, os
from datetime import datetime, timedelta
db = os.getenv("DB_PATH", "/var/data/genius_bot_v2.db")
try:
    conn = sqlite3.connect(db)
    # ბოლო 1 საათის SL-ები
    rows = conn.execute("""
        SELECT outcome, closed_at FROM trades
        WHERE closed_at IS NOT NULL
        AND closed_at >= datetime('now', '-1 hour')
        ORDER BY closed_at DESC
    """).fetchall()

    sl_count = sum(1 for r in rows if r[0] == 'SL')
    tp_count = sum(1 for r in rows if r[0] == 'TP')

    print(f"\033[0;36m[ℹ INFO]\033[0m ბოლო 1 სთ: {sl_count} SL, {tp_count} TP")

    if sl_count >= 2:
        print(f"\033[1;33m[⚠ WARN]\033[0m {sl_count} SL ბოლო 1 სთ-ში — SL Cooldown შეიძლება active იყოს")
    else:
        print(f"\033[0;32m[✓ OK]\033[0m   SL count ნორმალურია ({sl_count}/2 limit)")
    conn.close()
except Exception as e:
    print(f"\033[0;36m[ℹ INFO]\033[0m SL check: {e}")
PYEOF

# ─────────────────────────────────────────
section "10. ბოლო audit_log"
# ─────────────────────────────────────────
python3 << 'PYEOF'
import sqlite3, os
db = os.getenv("DB_PATH", "/var/data/genius_bot_v2.db")
try:
    conn = sqlite3.connect(db)
    rows = conn.execute("""
        SELECT event_type, message, created_at FROM audit_log
        ORDER BY id DESC LIMIT 5
    """).fetchall()
    print(f"\033[0;36m[ℹ INFO]\033[0m ბოლო 5 audit event:")
    for r in rows:
        print(f"         └─ [{r[2]}] {r[0]}: {r[1][:60]}")
    conn.close()
except Exception as e:
    print(f"\033[0;31m[✗ FAIL]\033[0m audit_log: {e}")
PYEOF

# ─────────────────────────────────────────
section "საბოლოო ვერდიქტი"
# ─────────────────────────────────────────
echo
TOTAL=$((PASS + FAIL_COUNT + WARN_COUNT))
echo -e "  სულ შემოწმება: ${TOTAL}"
echo -e "  ${GREEN}OK: ${PASS}${NC}"
echo -e "  ${YELLOW}WARN: ${WARN_COUNT}${NC}"
echo -e "  ${RED}FAIL: ${FAIL_COUNT}${NC}"
echo

if [ "$FAIL_COUNT" -eq 0 ] && [ "$WARN_COUNT" -eq 0 ]; then
    echo -e "${GREEN}${BOLD}  ✅ ბოტი სრულიად ჯანმრთელია!${NC}"
elif [ "$FAIL_COUNT" -eq 0 ]; then
    echo -e "${YELLOW}${BOLD}  ⚠️  ბოტი მუშაობს, მაგრამ ${WARN_COUNT} გაფრთხილება${NC}"
else
    echo -e "${RED}${BOLD}  ❌ ${FAIL_COUNT} კრიტიკული პრობლემა გამოვლინდა!${NC}"
fi
echo
