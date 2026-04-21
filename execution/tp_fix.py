# execution/tp_fix.py
# ============================================================
# TP FIX — Take Profit ავტომატური გასწორება — ADDON CASCADE SYSTEM
# ============================================================
# Binance API არ სჭირდება — DB-ს კითხულობს მხოლოდ.
#
# Zone detection (add_on_count + l3_addon_done):
#   L2 zone: add_on_count < DCA_MAX_ADD_ONS, l3_addon_done=0 → 0.55%
#   L3 zone: add_on_count >= DCA_MAX_ADD_ONS OR l3_addon_done=1 → 0.35%
#
# ENV:
#   TP_FIX_ENABLED=true
#   TP_FIX_TOLERANCE=0.1
#   DCA_TP_PCT=0.55
#   CASCADE_TP_L3_PCT=0.35
#   DCA_MAX_ADD_ONS=5
#   TIME_BASED_TP_ENABLED=false
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
TP_BASE   = float(os.getenv("DCA_TP_PCT",        "0.55"))  # L2 zone (ADD-ON active)
TP_L3     = float(os.getenv("CASCADE_TP_L3_PCT", "0.35"))  # L3 zone (LIFO rotation)

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




def run_tp_fix() -> dict:
    """
    DB-ის ყველა ღია პოზიციის TP შემოწმება და გასწორება.
    Binance API არ სჭირდება — memory-safe!

    ADDON CASCADE SYSTEM zone detection:
      L2 zone: add_on_count < max_add_ons AND l3_addon_done=0 → TP_BASE (0.55%)
      L3 zone: add_on_count >= max_add_ons OR l3_addon_done=1  → TP_L3 (0.35%)

    Returns:
        dict — {"checked": N, "fixed": N, "skipped": N}
    """
    import os as _os
    _max_add_ons = int(_os.getenv("DCA_MAX_ADD_ONS", "5"))

    result = {"checked": 0, "fixed": 0, "skipped": 0}

    try:
        from execution.db.db import get_connection

        conn = get_connection()
        rows = conn.execute("""
            SELECT id, symbol, avg_entry_price, current_tp_price,
                   add_on_count, l3_addon_done
            FROM dca_positions
            WHERE status='OPEN'
            ORDER BY symbol
        """).fetchall()

        if not rows:
            logger.info("[TP_FIX] No open positions → skip")
            return result

        _mult, _session = _session_mult()
        if TIME_BASED_TP_ENABLED:
            logger.info(f"[TP_FIX] session={_session} mult={_mult}")

        for row in rows:
            pos_id = row[0]
            sym    = row[1]
            avg    = float(row[2] or 0)
            tp     = float(row[3] or 0)
            add_on_count  = int(row[4] or 0)
            l3_addon_done = int(row[5] or 0) if len(row) > 5 and row[5] is not None else 0

            result["checked"] += 1

            if avg <= 0:
                logger.warning(f"[TP_FIX] SKIP {sym} | avg_entry=0")
                result["skipped"] += 1
                continue

            # ADDON CASCADE SYSTEM: zone detection via add_on_count + l3_addon_done
            # L3 zone: ADD-ONs exhausted OR L3 ADD-ON გახსნილია
            in_l3_zone = (add_on_count >= _max_add_ons) or (l3_addon_done == 1)
            tp_pct = TP_L3 if in_l3_zone else TP_BASE

            # TIME_BASED_TP: L2 zone-ზე მხოლოდ
            if TIME_BASED_TP_ENABLED and not in_l3_zone and _mult != 1.0:
                tp_pct = round(tp_pct * _mult, 4)
                tp_pct = max(TIME_TP_MIN_PCT, min(TIME_TP_MAX_PCT, tp_pct))

            correct  = round(avg * (1 + tp_pct / 100.0), 6)
            diff_pct = abs(tp - correct) / correct * 100 if correct > 0 else 100.0

            if diff_pct < TOLERANCE:
                logger.info(
                    f"[TP_FIX] OK    {sym} | "
                    f"zone={'L3' if in_l3_zone else 'L2'} tp={tp:.4f} ✅"
                )
                result["skipped"] += 1
                continue

            conn.execute(
                "UPDATE dca_positions SET current_tp_price=? WHERE id=? AND status='OPEN'",
                (correct, pos_id)
            )
            logger.warning(
                f"[TP_FIX] FIXED {sym} | "
                f"zone={'L3' if in_l3_zone else 'L2'} tp_pct={tp_pct:.2f}% | "
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
