# execution/tp_fix.py
# ============================================================
# TP FIX — Take Profit ავტომატური გასწორება
# ============================================================
# მსუბუქი ვერსია — Binance API არ სჭირდება!
# მხოლოდ DB-ს კითხულობს და TP-ს ასწორებს avg_entry-დან
#
# ლოგიკა:
#   L1-L2:  TP = avg_entry × 1.0055  (+0.55%)
#   L3-L10: TP = avg_entry × 1.0065  (+0.65%)
#
# გამოყენება:
#   Shell-ში: python3 execution/tp_fix.py
#   ან main.py restart-ზე ავტომატურად
#
# ENV:
#   TP_FIX_ENABLED=true     ← ჩართვა/გამორთვა
#   TP_FIX_TOLERANCE=0.1    ← 0.1% სხვაობაზე გასწორება
#   DCA_TP_PCT=0.55         ← L1-L2 TP პროცენტი
#   CASCADE_TP_L3_PCT=0.65  ← L3+ TP პროცენტი
# ============================================================

from __future__ import annotations

import os
import re
import sys
import logging

logger = logging.getLogger("gbm")

TOLERANCE = float(os.getenv("TP_FIX_TOLERANCE", "0.1"))   # 0.1% სხვაობა
TP_BASE   = float(os.getenv("DCA_TP_PCT",        "0.55"))  # L1-L2
TP_L3     = float(os.getenv("CASCADE_TP_L3_PCT", "0.65"))  # L3-L10


def _layer_num(symbol: str) -> int:
    """BTC/USDT → 1, BTC/USDT_L2 → 2, BTC/USDT_L5 → 5"""
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

        for row in rows:
            pos_id, sym, avg, tp = row
            avg = float(avg or 0)
            tp  = float(tp  or 0)
            result["checked"] += 1

            if avg <= 0:
                logger.warning(f"[TP_FIX] SKIP {sym} | avg_entry=0")
                result["skipped"] += 1
                continue

            # Layer-ის მიხედვით tp_pct
            layer   = _layer_num(sym)
            tp_pct  = TP_L3 if layer >= 3 else TP_BASE
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
