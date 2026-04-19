#!/bin/bash
# ============================================================
# GENIUS BOT — სრული დიაგნოსტიკა (diagnose.sh + diagnostics_pro.py)
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
echo -e "${BOLD}╔══════════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}║           GENIUS BOT — სრული დიაგნოსტიკა (diagnose + pro)              ║${NC}"
echo -e "${BOLD}╚══════════════════════════════════════════════════════════════════════════╝${NC}"
echo -e "  $(date '+%Y-%m-%d %H:%M:%S')"
echo

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
    "regime_engine.py"
    "logger.py"
    "signal_client.py"
    "startup_sync.py"
    "performance_report.py"
    "portfolio_manager.py"
    "virtual_wallet.py"
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
        SELECT symbol, qty, quote_in, entry_price, opened_at, outcome
        FROM trades WHERE closed_at IS NULL
        ORDER BY opened_at DESC
    """).fetchall()
    if rows:
        print(f'{GREEN}[OK]{NC}   ღია trade-ები: {len(rows)}')
        for r in rows:
            outcome_str = f" outcome={r[5]}" if r[5] else " (open)"
            print(f'{CYAN}[INFO]{NC}   └─ {r[0]} qty={r[1]:.6f} invested={r[2]:.2f} USDT entry={r[3]:.4f} opened={r[4]}{outcome_str}')
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
    import os as _os
    limit = int(_os.getenv("SL_COOLDOWN_AFTER_N", "3"))
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
section "11. ლოგიკის აუდიტი — ATR / Regime / PnL"
# ─────────────────────────────────────────
BASE="${BASE_PATH:-/opt/render/project/src/execution}"
python3 - "$BASE" << 'PYEOF'
import os, sys, importlib.util
from datetime import datetime, timedelta

GREEN='\033[0;32m'; RED='\033[0;31m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
ok_c=0; fail_c=0; warn_c=0

def ok(m):   global ok_c;   print(f'{GREEN}[OK]{NC}   {m}'); ok_c+=1
def fail(m): global fail_c; print(f'{RED}[FAIL]{NC} {m}'); fail_c+=1
def warn(m): global warn_c; print(f'{YELLOW}[WARN]{NC} {m}'); warn_c+=1
def info(m): print(f'{CYAN}[INFO]{NC} {m}')

BASE = sys.argv[1] if len(sys.argv) > 1 else "/opt/render/project/src/execution"

# ══════════════════════════════════════════════════
# 1. regime_engine.py — ლოგიკის ტესტი
# ══════════════════════════════════════════════════
re_path = os.path.join(BASE, "regime_engine.py")
if not os.path.exists(re_path):
    fail(f"regime_engine.py ვერ მოიძებნა: {re_path}")
else:
    try:
        spec = importlib.util.spec_from_file_location("regime_engine_diag", re_path)
        mod  = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        # backward compat: ძველი stub-ი config=-ს მოითხოვს
        try:
            eng = mod.MarketRegimeEngine()
        except TypeError:
            eng = mod.MarketRegimeEngine(config={})

        is_new = hasattr(eng, 'notify_outcome') and hasattr(eng, 'is_paused')
        if is_new:
            ok("regime_engine.py — ახალი ვერსია (notify_outcome + is_paused) ✓")
        else:
            warn("regime_engine.py — ძველი stub ვერსია! SL Cooldown tracking არ მუშაობს engine-ში")
            warn("  → გამოსწორება: ჩაანაცვლე განახლებული regime_engine.py-ით")

        # რეჟიმების შემოწმება — 5 შემთხვევა
        stub_map = {"TREND_UP": "BULL", "TREND_DOWN": "BEAR"}  # ძველი stub aliases
        tests = [
            (0.6,  0.4,  "BULL"),
            (0.2,  0.2,  "SIDEWAYS"),
            (0.3,  0.5,  "UNCERTAIN"),
            (0.1,  2.0,  "VOLATILE"),
            (-0.2, 0.5,  "BEAR"),
        ]
        passed = 0
        for trend, atr, expected in tests:
            got = eng.detect_regime(trend, atr)
            norm = stub_map.get(got, got)
            if norm == expected:
                ok(f"detect_regime({trend:+.1f}, {atr}) → '{expected}' ✓")
                passed += 1
            else:
                fail(f"detect_regime({trend:+.1f}, {atr}) → '{got}' (expected '{expected}') ✗")

        if passed < 5:
            warn(f"  → {5-passed}/5 რეჟიმი არასწორია — ახალი regime_engine.py საჭიროა")

        # apply() API test
        for skip_r in ("BEAR", "VOLATILE", "SIDEWAYS"):
            try:
                r = eng.apply(skip_r, 0.3, "BTC/USDT")
                skip = r.get("SKIP_TRADING", False) if isinstance(r, dict) else False
                if skip:
                    ok(f"apply({skip_r}) → SKIP_TRADING=True ✓")
                else:
                    warn(f"apply({skip_r}) → SKIP_TRADING=False ან კლავიში არ არის (ძველი API?)")
            except TypeError:
                warn(f"apply({skip_r}) — ძველი API signature (< 2 args) — ახალი ვერსია საჭიროა")

        try:
            r_bull = eng.apply("BULL", 0.5, "BTC/USDT")
            if isinstance(r_bull, dict):
                tp = r_bull.get("TP_PCT", 0)
                sl = r_bull.get("SL_PCT", 0)
                if tp > 0 and sl > 0:
                    ok(f"apply(BULL, atr=0.5%) → TP={tp}%  SL={sl}% ✓")
                else:
                    warn(f"apply(BULL) → TP/SL კლავიშები არ არის — ახალი ვერსია საჭიროა")
        except TypeError:
            warn("apply(BULL, ...) — ძველი API signature")

        # SL Cooldown (მხოლოდ ახალ ვერსიაში)
        if is_new:
            import os as _os2
            sl_limit_test = int(_os2.getenv("SL_COOLDOWN_AFTER_N", "3"))
            t0 = datetime(2025, 1, 1, 10, 0, 0)
            # Fire SL sl_limit_test times to trigger pause
            for _ in range(sl_limit_test):
                try:
                    eng.notify_outcome("BTC/USDT", "SL", t0)
                except TypeError:
                    eng.notify_outcome("BTC/USDT", "SL")
            paused_check = False
            try:
                paused_check = eng.is_paused("BTC/USDT", t0 + timedelta(minutes=5))
            except TypeError:
                paused_check = eng.is_paused("BTC/USDT")
            if paused_check:
                ok(f"SL Cooldown — {sl_limit_test}×SL → pause ✓")
            else:
                fail(f"SL Cooldown — {sl_limit_test}×SL შემდეგ pause არ ავიდა!")
            try:
                eng.notify_outcome("BTC/USDT", "TP", t0)
            except TypeError:
                eng.notify_outcome("BTC/USDT", "TP")
            not_paused = True
            try:
                not_paused = not eng.is_paused("BTC/USDT")
            except TypeError:
                not_paused = not eng.is_paused("BTC/USDT")
            if not_paused:
                ok("SL Cooldown — TP reset ✓")
            else:
                fail("SL Cooldown — TP-ზე reset ვერ მოხდა!")

    except Exception as e:
        fail(f"regime_engine.py test error: {e}")

# ══════════════════════════════════════════════════
# 2. signal_generator.py — _atr_pct() OHLCV-based
# ══════════════════════════════════════════════════
sg_path = os.path.join(BASE, "signal_generator.py")
if os.path.exists(sg_path):
    try:
        src = open(sg_path).read()
        if "def _atr_pct" in src and "ohlcv" in src.lower() and ("high" in src or "ohlcv[i][2]" in src):
            ok("signal_generator._atr_pct() — real OHLCV ATR (high/low/prev_close) ✓")
        elif "def _atr_pct" in src:
            warn("signal_generator._atr_pct() — ფუნქცია არსებობს, OHLCV logic გადაამოწმე")
        else:
            fail("signal_generator._atr_pct() — ვერ მოიძებნა!")
    except Exception as e:
        warn(f"signal_generator.py read: {e}")
else:
    warn(f"signal_generator.py ვერ მოიძებნა: {sg_path}")

# ══════════════════════════════════════════════════
# 3. PnL კალკულაცია + Breakeven Winrate
# ══════════════════════════════════════════════════
try:
    fee_rt = float(os.getenv("ESTIMATED_ROUNDTRIP_FEE_PCT", "0.14"))
    slip   = float(os.getenv("ESTIMATED_SLIPPAGE_PCT",      "0.05"))
    tp_pct = float(os.getenv("TP_PCT",  "1.5"))
    sl_pct = float(os.getenv("SL_PCT",  "0.80"))
    quote  = float(os.getenv("BOT_QUOTE_PER_TRADE", "10.0"))
    cost   = (fee_rt + slip) / 100.0

    tp_net = quote * (tp_pct / 100.0) - quote * cost
    sl_net = -(quote * (sl_pct / 100.0) + quote * cost)
    rr     = abs(tp_net) / abs(sl_net) if sl_net != 0 else 0
    be_wr  = abs(sl_net) / (tp_net + abs(sl_net)) * 100 if (tp_net + abs(sl_net)) > 0 else 100

    info(f"TP({tp_pct}%) net=+{tp_net:.4f} USDT | SL({sl_pct}%) net={sl_net:.4f} USDT | R:R=1:{rr:.2f}")

    if tp_net > 0:
        ok(f"TP net profit: +{tp_net:.4f} USDT (fees={fee_rt+slip:.2f}% დაფარულია) ✓")
    else:
        fail(f"TP net: {tp_net:.4f} USDT — TP_PCT={tp_pct}% ძალიან დაბალია fees-ის გასაფარებლად!")

    if sl_net < 0:
        ok(f"SL net loss: {sl_net:.4f} USDT ✓")

    info(f"Breakeven winrate = {be_wr:.1f}% (ამაზე მეტი საჭიროა):")
    if be_wr <= 35:
        ok(f"Breakeven WR {be_wr:.1f}% — მარტივად მისაღწევია ✓")
    elif be_wr <= 45:
        warn(f"Breakeven WR {be_wr:.1f}% — winrate {be_wr:.0f}%+ საჭიროა — TP/SL-ის გადახედვა სასარგებლო იქნება")
    else:
        fail(f"Breakeven WR {be_wr:.1f}% — ძალიან მაღალი! TP_PCT ან SL_PCT გადასახედია")

except Exception as e:
    fail(f"PnL კალკულაცია: {e}")

# ══════════════════════════════════════════════════
# 4. Weight ჯამი — ai_score sanity
# ══════════════════════════════════════════════════
try:
    keys = ["WEIGHT_TREND","WEIGHT_STRUCTURE","WEIGHT_VOLUME","WEIGHT_RISK","WEIGHT_CONFIDENCE","WEIGHT_VOLATILITY"]
    vals = {k: float(os.getenv(k, "0")) for k in keys}
    total_w = sum(vals.values())
    if abs(total_w - 1.0) < 0.01:
        ok(f"WEIGHT ჯამი = {total_w:.3f} ≈ 1.0 — ai_score სწორ დიაპაზონშია ✓")
    else:
        warn(f"WEIGHT ჯამი = {total_w:.3f} (≠ 1.0) — ai_score შეიძლება >1 ან <1 გამოვიდეს")
        info(f"  " + "  ".join(f"{k.replace('WEIGHT_','')}={v}" for k,v in vals.items()))
except Exception as e:
    warn(f"WEIGHT check: {e}")

# ══════════════════════════════════════════════════
# 5. ENV threshold კონსისტენტობა
# ══════════════════════════════════════════════════
try:
    bull_min  = float(os.getenv("REGIME_BULL_TREND_MIN",   "0.30"))
    th_trend  = float(os.getenv("THRESHOLD_TREND",         "0.45"))
    th_conf   = float(os.getenv("THRESHOLD_CONF",          "0.55"))
    conf_min  = float(os.getenv("BUY_CONFIDENCE_MIN",      "0.55"))
    ai_thresh = 0.60  # ExcelLiveCore hardcoded

    if bull_min <= th_trend:
        ok(f"REGIME_BULL_TREND_MIN({bull_min}) ≤ THRESHOLD_TREND({th_trend}) ✓")
    else:
        warn(f"REGIME_BULL_TREND_MIN({bull_min}) > THRESHOLD_TREND({th_trend}) — კონფლიქტი!")

    if 0.40 <= th_conf <= 0.75:
        ok(f"THRESHOLD_CONF={th_conf} — ნორმალურ დიაპაზონშია ✓")
    else:
        warn(f"THRESHOLD_CONF={th_conf} — უჩვეულო მნიშვნელობა (0.40–0.75 მოსალოდნელია)")

    if conf_min >= th_conf:
        ok(f"BUY_CONFIDENCE_MIN({conf_min}) ≥ THRESHOLD_CONF({th_conf}) ✓")
    else:
        warn(f"BUY_CONFIDENCE_MIN({conf_min}) < THRESHOLD_CONF({th_conf}) — signal-ები conf gate-ს გვერდს ვერ აუვლის")

except Exception as e:
    warn(f"Threshold check: {e}")

# ══ შედეგი ══
print()
print(f"{CYAN}[INFO]{NC} ლოგიკის აუდიტი: {GREEN}OK={ok_c}{NC}  {YELLOW}WARN={warn_c}{NC}  {RED}FAIL={fail_c}{NC}")
PYEOF

echo
echo -e "${BOLD}╔══════════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}║                          საბოლოო ვერდიქტი                              ║${NC}"
echo -e "${BOLD}╚══════════════════════════════════════════════════════════════════════════╝${NC}"
TOTAL=$((PASS + FAIL_COUNT + WARN_COUNT))
echo
echo -e "  სულ შემოწმება : ${BOLD}${TOTAL}${NC}"
echo -e "  ${GREEN}${BOLD}OK   : ${PASS}${NC}"
echo -e "  ${YELLOW}${BOLD}WARN : ${WARN_COUNT}${NC}"
echo -e "  ${RED}${BOLD}FAIL : ${FAIL_COUNT}${NC}"
echo

if [ "$FAIL_COUNT" -eq 0 ] && [ "$WARN_COUNT" -eq 0 ]; then
    echo -e "${GREEN}${BOLD}  ✅ ბოტი სრულიად ჯანმრთელია!${NC}"
elif [ "$FAIL_COUNT" -eq 0 ]; then
    echo -e "${YELLOW}${BOLD}  ⚠️  ბოტი მუშაობს — ${WARN_COUNT} გაფრთხილება${NC}"
else
    echo -e "${RED}${BOLD}  ❌ ${FAIL_COUNT} კრიტიკული პრობლემა გამოვლინდა!${NC}"
fi
echo -e "  $(date '+%Y-%m-%d %H:%M:%S')"
echo
