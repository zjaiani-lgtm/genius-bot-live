#!/usr/bin/env python3
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# GENIUS DCA BOT — სრული სტატისტიკა
# გაშვება: python3 genius_stats.py
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
import sqlite3
import os
from datetime import datetime, timezone

DB_PATH = os.getenv("DB_PATH", "/var/data/genius_bot_v2.db")

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def fmt_num(n, decimals=4):
    if n is None: return "0"
    return f"{float(n):+.{decimals}f}" if float(n) != 0 else "0"

def fmt_pct(n):
    if n is None: return "0.00%"
    return f"{float(n):+.2f}%"

def print_section(title):
    print()
    print("━" * 55)
    print(f"  {title}")
    print("━" * 55)

def main():
    conn = get_db()
    now = datetime.now()

    print()
    print("╔═══════════════════════════════════════════════════════╗")
    print("║        🤖 GENIUS DCA BOT — სრული სტატისტიკა          ║")
    print(f"║  🕒 {now.strftime('%Y-%m-%d %H:%M:%S')}                          ║")
    print("╚═══════════════════════════════════════════════════════╝")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 1. სისტემის სტატუსი
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    print_section("⚙️  სისტემის სტატუსი")

    sys_row = conn.execute("SELECT * FROM system_state WHERE id=1").fetchone()
    if sys_row:
        print(f"  სტატუსი:       {sys_row['status']}")
        print(f"  Kill Switch:   {'🔴 ON' if sys_row['kill_switch'] else '🟢 OFF'}")
        print(f"  Startup Sync:  {'✅' if sys_row['startup_sync_ok'] else '❌'}")

    # პირველი trade-ის თარიღი
    first = conn.execute(
        "SELECT MIN(opened_at) as first FROM dca_positions"
    ).fetchone()
    if first and first["first"]:
        first_dt = datetime.fromisoformat(str(first["first"]))
        days = (now - first_dt).days
        print(f"  გაშვებულია:    {days} დღე ({first_dt.strftime('%Y-%m-%d')} -დან)")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 2. საერთო შედეგები
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    print_section("📊 საერთო სტატისტიკა")

    stats = conn.execute("""
        SELECT
            COUNT(*) as total,
            SUM(CASE WHEN outcome='TP' THEN 1 ELSE 0 END) as wins,
            SUM(CASE WHEN outcome IN ('FC','SL','FORCE_CLOSE') THEN 1 ELSE 0 END) as losses,
            ROUND(SUM(pnl_quote), 4) as total_pnl,
            ROUND(AVG(pnl_quote), 4) as avg_pnl,
            ROUND(MAX(pnl_quote), 4) as best_trade,
            ROUND(MIN(pnl_quote), 4) as worst_trade,
            ROUND(SUM(total_quote_spent), 2) as total_invested,
            ROUND(AVG(add_on_count), 2) as avg_addons
        FROM dca_positions
        WHERE outcome IS NOT NULL
    """).fetchone()

    if stats and stats["total"]:
        total = stats["total"]
        wins = stats["wins"] or 0
        losses = stats["losses"] or 0
        winrate = (wins / total * 100) if total > 0 else 0

        print(f"  📈 სულ Closed:    {total}")
        print(f"  🏆 Wins (TP):     {wins}")
        print(f"  ❌ Losses (FC/SL):{losses}")
        print(f"  🔥 Winrate:       {winrate:.2f}%")
        print(f"  💰 სულ PnL:       {stats['total_pnl']} USDT")
        print(f"  📉 avg PnL/trade: {stats['avg_pnl']} USDT")
        print(f"  🎯 Best trade:    {stats['best_trade']} USDT")
        print(f"  💸 Worst trade:   {stats['worst_trade']} USDT")
        print(f"  💼 სულ Invested:  {stats['total_invested']} USDT")
        print(f"  🔄 avg ADD-ONs:   {stats['avg_addons']}")
    else:
        print("  ჯერ closed trade არ არის")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 3. Symbol-ების breakdown
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    print_section("🪙 Symbol-ების სტატისტიკა")

    sym_stats = conn.execute("""
        SELECT
            symbol,
            COUNT(*) as trades,
            SUM(CASE WHEN outcome='TP' THEN 1 ELSE 0 END) as wins,
            ROUND(SUM(pnl_quote), 4) as pnl,
            ROUND(AVG(pnl_quote), 4) as avg_pnl,
            ROUND(AVG(add_on_count), 1) as avg_addons
        FROM dca_positions
        WHERE outcome IS NOT NULL
        GROUP BY symbol
        ORDER BY pnl DESC
    """).fetchall()

    for s in sym_stats:
        wr = (s["wins"] / s["trades"] * 100) if s["trades"] > 0 else 0
        print(f"  {s['symbol']:15s} | trades={s['trades']:4d} | WR={wr:.0f}% | pnl={s['pnl']:+.4f} | avg_addons={s['avg_addons']}")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 4. დღევანდელი შედეგები
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    print_section("📅 დღევანდელი სტატისტიკა")

    today = conn.execute("""
        SELECT
            COUNT(*) as trades,
            SUM(CASE WHEN outcome='TP' THEN 1 ELSE 0 END) as wins,
            ROUND(SUM(pnl_quote), 4) as pnl
        FROM dca_positions
        WHERE outcome IS NOT NULL
        AND date(closed_at) = date('now')
    """).fetchone()

    if today and today["trades"]:
        wr = (today["wins"] / today["trades"] * 100) if today["trades"] > 0 else 0
        print(f"  Closed today:  {today['trades']}")
        print(f"  Wins:          {today['wins']}")
        print(f"  Winrate:       {wr:.0f}%")
        print(f"  PnL today:     {today['pnl']:+.4f} USDT")
    else:
        print("  დღეს ჯერ closed trade არ არის")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 5. ღია პოზიციები
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    print_section("🟢 ღია პოზიციები")

    open_pos = conn.execute("""
        SELECT
            symbol,
            ROUND(avg_entry_price, 2) as avg,
            ROUND(current_tp_price, 2) as tp,
            add_on_count,
            max_add_ons,
            ROUND(total_quote_spent, 2) as invested,
            opened_at,
            ROUND(julianday('now') - julianday(opened_at), 1) as days_open
        FROM dca_positions
        WHERE status='OPEN'
        ORDER BY opened_at ASC
    """).fetchall()

    if open_pos:
        total_invested = sum(float(p["invested"]) for p in open_pos)
        for p in open_pos:
            tp_pct = ((float(p["tp"]) / float(p["avg"])) - 1) * 100
            print(f"  {p['symbol']:15s} | avg={p['avg']:>10} | tp={p['tp']:>10} (+{tp_pct:.2f}%) | addons={p['add_on_count']}/{p['max_add_ons']} | ${p['invested']} | {p['days_open']}d")
        print(f"\n  სულ Open:      {len(open_pos)} positions")
        print(f"  სულ Capital:   ${total_invested:.2f} USDT")
    else:
        print("  ღია პოზიცია არ არის")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 6. ბოლო 10 დახურული trade
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    print_section("📋 ბოლო 10 Closed Trade")

    recent = conn.execute("""
        SELECT
            symbol,
            outcome,
            ROUND(avg_entry_price, 2) as avg,
            ROUND(exit_price, 2) as exit,
            ROUND(pnl_quote, 4) as pnl,
            ROUND(pnl_pct, 3) as pct,
            add_on_count as addons,
            closed_at
        FROM dca_positions
        WHERE outcome IS NOT NULL
        ORDER BY closed_at DESC
        LIMIT 10
    """).fetchall()

    for t in recent:
        icon = "✅" if t["outcome"] == "TP" else "❌"
        closed_str = str(t["closed_at"])[:16] if t["closed_at"] else "?"
        print(f"  {icon} {t['symbol']:15s} | {t['outcome']:10s} | avg={t['avg']:>10} exit={t['exit']:>10} | pnl={t['pnl']:+.4f} ({t['pct']:+.3f}%) | addons={t['addons']} | {closed_str}")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 7. Futures სტატისტიკა
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    print_section("⚡ Futures / SHORT სტატისტიკა")

    fut = conn.execute("""
        SELECT
            COUNT(*) as total,
            SUM(CASE WHEN status='OPEN' THEN 1 ELSE 0 END) as open_cnt,
            SUM(CASE WHEN status='CLOSED' THEN 1 ELSE 0 END) as closed_cnt,
            ROUND(SUM(CASE WHEN status='CLOSED' THEN pnl_quote ELSE 0 END), 4) as total_pnl,
            SUM(CASE WHEN is_dca_hedge=1 THEN 1 ELSE 0 END) as hedges,
            SUM(CASE WHEN is_independent_short=1 THEN 1 ELSE 0 END) as ind_shorts,
            SUM(CASE WHEN is_mirror_engine=1 THEN 1 ELSE 0 END) as mirrors
        FROM futures_positions
    """).fetchone()

    if fut and fut["total"]:
        print(f"  სულ Futures:      {fut['total']}")
        print(f"  ღია:              {fut['open_cnt']}")
        print(f"  Closed:           {fut['closed_cnt']}")
        print(f"  Futures PnL:      {fut['total_pnl']} USDT")
        print(f"  DCA Hedges:       {fut['hedges']}")
        print(f"  Independent SHORT:{fut['ind_shorts']}")
        print(f"  Mirror Engine:    {fut['mirrors']}")
    else:
        print("  Futures trades ჯერ არ არის")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 8. კომბინირებული PnL
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    print_section("💎 კომბინირებული შედეგი")

    long_pnl = conn.execute(
        "SELECT ROUND(SUM(pnl_quote),4) FROM dca_positions WHERE outcome IS NOT NULL"
    ).fetchone()[0] or 0

    short_pnl = conn.execute(
        "SELECT ROUND(SUM(pnl_quote),4) FROM futures_positions WHERE status='CLOSED'"
    ).fetchone()[0] or 0

    total_pnl = float(long_pnl) + float(short_pnl)

    print(f"  LONG DCA PnL:    {long_pnl:+.4f} USDT")
    print(f"  SHORT/Futures:   {short_pnl:+.4f} USDT")
    print(f"  ════════════════════════════")
    print(f"  სულ PnL:         {total_pnl:+.4f} USDT")

    print()
    print("═" * 55)
    print(f"  ✅ სტატისტიკა დასრულდა | {now.strftime('%H:%M:%S')}")
    print("═" * 55)
    print()

    conn.close()


if __name__ == "__main__":
    main()
