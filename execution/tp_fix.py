# execution/tp_fix.py
# ============================================================
# TP FIX — Take Profit ავტომატური გასწორება
# ============================================================
# მსუბუქი ვერსია — Binance API არ სჭირდება!
# მხოლოდ DB-ს კითხულობს და TP-ს ასწორებს avg_entry-დან
#
# ლოგიკა:
#   L1-L2: TP = avg_entry × 1.0055  (+0.55%)
#   L3:    TP = avg_entry × 1.0065  (+0.65%)
#   (max_layers=3 — L4+ არ იხსნება)
#
# TIME_BASED_TP_ENABLED=true შემთხვევაში:
#   tp_fix-ი UTC session-ის მიხედვით TP-ს ასწორებს.
#   ASIA_FLAT: TP_BASE × 0.75, NY_OPEN: TP_BASE × 1.10, etc.
#   ანუ signal_generator-ის ლოგიკა tp_fix-შიც ასახულია.
#   თუ ეს გათიშულია — TP ყოველთვის 0.55%/0.65%.
#
# გამოყენება:
#   Shell-ში: python3 execution/tp_fix.py
#   ან main.py restart-ზე ავტომატურად
#
# ENV:
#   TP_FIX_ENABLED=true          ← ჩართვა/გამორთვა
#   TP_FIX_TOLERANCE=0.1         ← 0.1% სხვაობაზე გასწორება
#   DCA_TP_PCT=0.55              ← L1-L2 TP პროცენტი
#   CASCADE_TP_L3_PCT=0.65       ← L3 TP პროცენტი (max layer)
#   TIME_BASED_TP_ENABLED=false  ← Time-based TP (default: false)
# ============================================================

from __future__ import annotations

import os
import re
import sys
import logging
from datetime import datetime

logger = logging.getLogger("gbm")

TOLERANCE = float(os.getenv("TP_FIX_TOLERANCE", "0.1"))   # 0.1% სხვაობა
TP_BASE   = float(os.getenv("DCA_TP_PCT",        "0.55"))  # L1-L2
TP_L3     = float(os.getenv("CASCADE_TP_L3_PCT", "0.65"))  # L3 (max layer)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TIME_BASED_TP — signal_generator-თან სინქრონიზაცია.
# თუ TIME_BASED_TP_ENABLED=true, tp_fix-იც იგივე multiplier-ს
# იყენებს — TP-ს არ "გაასწორებს" სხვა მნიშვნელობაზე.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TIME_BASED_TP_ENABLED = os.getenv("TIME_BASED_TP_ENABLED", "false").strip().lower() == "true"
TIME_TP_ASIA_MULT     = float(os.getenv("TIME_TP_ASIA_MULT",    "0.75"))
TIME_TP_LONDON_MULT   = float(os.getenv("TIME_TP_LONDON_MULT",  "1.00"))
TIME_TP_NY_MULT       = float(os.getenv("TIME_TP_NY_MULT",      "1.10"))
TIME_TP_EVENING_MULT  = float(os.getenv("TIME_TP_EVENING_MULT", "0.85"))
TIME_TP_MIN_PCT       = float(os.getenv("TIME_TP_MIN_PCT",      "0.30"))
TIME_TP_MAX_PCT       = float(os.getenv("TIME_TP_MAX_PCT",      "1.00"))


def _session_mult() -> tuple:
    """signal_generator-ის _time_based_tp_mult()-ის იდენტური ლოგიკა."""
    if not TIME_BASED_TP_ENABLED:
        return 1.0, "DISABLED"
    h = datetime.utcnow().hour
    if 0 <= h < 8:
        return TIME_TP_ASIA_MULT, "ASIA_FLAT"
    elif 8 <= h < 13:
        return TIME_TP_LONDON_MULT, "LONDON"
    elif 13 <= h < 17:
        return TIME_TP_NY_MULT, "NY_OPEN"
    elif 17 <= h < 21:
        return TIME_TP_LONDON_MULT, "OVERLAP"
    else:
        return TIME_TP_EVENING_MULT, "EVENING"


def _layer_num(symbol: str) -> int:
    """BTC/USDT → 1, BTC/USDT_L2 → 2, BTC/USDT_L3 → 3 (max)"""
    m = re.search(r'_L(\d+)$', symbol)
    return int(m.group(1)) if m else 1


def run_tp_fix() -> dict:
    """
    DB-ის ყველა ღია პოზიციის TP შემოწმება და გასწორება.
    Binance API არ სჭირდება — memory-safe!

    Returns:
        dict — {"checked": N, "fixed": N, "skipped": N}
    """
    result = {"checked": 0, "fixed": 0, "skipped": 0}
    conn   = None

    try:
        from execution.db.db import get_connection

        conn = get_connection()
        rows = conn.execute("""
            SELECT id, symbol, avg_entry_price, current_tp_price
            FROM dca_positions
            WHERE status='OPEN'
            ORDER BY symbol
        """).fetchall()

        if not rows:
            logger.info("[TP_FIX] No open positions → skip")
            return result

        # TIME_BASED_TP: ახლანდელი session multiplier — ყველა position-ისთვის ერთი
        _mult, _session = _session_mult()
        if TIME_BASED_TP_ENABLED:
            logger.info(f"[TP_FIX] session={_session} mult={_mult}")

        for row in rows:
            pos_id, sym, avg, tp = row
            avg = float(avg or 0)
            tp  = float(tp  or 0)
            result["checked"] += 1

            if avg <= 0:
                logger.warning(f"[TP_FIX] SKIP {sym} | avg_entry=0")
                result["skipped"] += 1
                continue

            # Layer-ის მიხედვით base tp_pct
            layer    = _layer_num(sym)
            tp_pct   = TP_L3 if layer >= 3 else TP_BASE

            # TIME_BASED_TP: L1-L2-ზე session multiplier-ი.
            # L3+ CASCADE positions — TP cascade-ის ლოგიკით იმართება,
            # time-based override არ ეხება (CASCADE_TP_L3_PCT ფიქსირებულია).
            if TIME_BASED_TP_ENABLED and layer < 3 and _mult != 1.0:
                tp_pct = round(tp_pct * _mult, 4)
                tp_pct = max(TIME_TP_MIN_PCT, min(TIME_TP_MAX_PCT, tp_pct))

            correct = round(avg * (1 + tp_pct / 100.0), 6)

            # სხვაობა
            diff_pct = abs(tp - correct) / correct * 100 if correct > 0 else 100.0

            if diff_pct < TOLERANCE:
                logger.info(
                    f"[TP_FIX] OK    {sym} | "
                    f"layer={layer} tp={tp:.4f} ✅"
                )
                result["skipped"] += 1
                continue

            # გასწორება
            conn.execute(
                "UPDATE dca_positions SET current_tp_price=? WHERE id=? AND status='OPEN'",
                (correct, pos_id)
            )
            logger.warning(
                f"[TP_FIX] FIXED {sym} | "
                f"layer={layer} tp_pct={tp_pct:.2f}% | "
                f"{tp:.4f} → {correct:.4f} (diff={diff_pct:.2f}%)"
            )
            result["fixed"] += 1

        conn.commit()
        logger.info(
            f"[TP_FIX] DONE | "
            f"checked={result['checked']} "
            f"fixed={result['fixed']} "
            f"skipped={result['skipped']}"
        )

    except Exception as e:
        logger.error(f"[TP_FIX] ERROR | {e}")
        result["error"] = str(e)

    # FIX: conn.close() ამოღებულია!
    # thread-local connection-ს ვერ ვხურავთ — main loop-ი იყენებს
    # get_connection() thread-local-ს აბრუნებს → close() = main loop DB error!
    # commit() უკვე გაკეთებულია ზემოთ

    return result


def print_report(result: dict) -> None:
    print("\n" + "="*50)
    print("  TP FIX REPORT")
    print("="*50)
    print(f"  checked : {result.get('checked', 0)}")
    print(f"  fixed   : {result.get('fixed', 0)}")
    print(f"  skipped : {result.get('skipped', 0)}")
    if result.get("error"):
        print(f"  ERROR   : {result['error']}")
    print("="*50 + "\n")


# ── standalone Shell-იდან ────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(message)s"
    )
    sys.path.insert(0, "/opt/render/project/src")
    result = run_tp_fix()
    print_report(result)
