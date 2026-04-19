# execution/qty_sync.py
# ============================================================
# QTY SYNC — Binance vs DB ავტომატური სინქრონიზატორი
# ============================================================
# პრობლემა:
#   buy_qty = quote / price  ← slippage/fee იგნორირებული
#   Binance filled qty < DB qty → TP hit-ზე "Insufficient balance"
#
# გამოსავალი:
#   1. Binance Spot-დან რეალური coin ბალანსი
#   2. DB-ის total_qty-სთან შედარება
#   3. თუ სხვაობა > TOLERANCE → DB განახლება
#
# გამოყენება:
#   Shell-ში: python3 execution/qty_sync.py
#   ან main.py-დან: from execution.qty_sync import run_qty_sync
#
# ENV:
#   QTY_SYNC_TOLERANCE=0.005   ← 0.5% სხვაობაზე გასწორება
#   QTY_SYNC_DRY_RUN=false     ← true=მხოლოდ ბეჭდავს, არ ცვლის
# ============================================================

from __future__ import annotations

import os
import re
import sys
import logging

logger = logging.getLogger("gbm")

# ── tolerance: 0.5% სხვაობა → გასწორება ──────────────────
TOLERANCE = float(os.getenv("QTY_SYNC_TOLERANCE", "0.005"))
DRY_RUN   = os.getenv("QTY_SYNC_DRY_RUN", "false").strip().lower() in ("1", "true", "yes")


def _base_coin(symbol: str) -> str:
    """BTC/USDT → BTC, BTC/USDT_L2 → BTC"""
    clean = re.sub(r'_L\d+$', '', symbol)
    return clean.split("/")[0]


def run_qty_sync() -> dict:
    """
    Binance Spot ბალანსი vs DB total_qty შედარება და გასწორება.

    Returns:
        dict — {
            "checked": N,
            "fixed": N,
            "skipped": N,
            "details": [...]
        }
    """
    result = {"checked": 0, "fixed": 0, "skipped": 0, "details": []}
    conn = None
    ex   = None

    try:
        from execution.exchange_client import BinanceSpotClient
        from execution.db.repository import get_all_open_dca_positions
        from execution.db.db import get_connection

        # ── 1. ღია პოზიციები DB-დან ─────────────────────────
        positions = get_all_open_dca_positions()
        if not positions:
            logger.info("[QTY_SYNC] No open positions → skip")
            return result

        # ── 2. coin-ების აგრეგაცია DB-დან ───────────────────
        coin_db: dict[str, list] = {}
        for pos in positions:
            coin = _base_coin(pos["symbol"])
            coin_db.setdefault(coin, []).append(pos)

        # ── 3. Binance ბალანსი — coin-by-coin (memory safe) ──
        # FIX #1: ex ობიექტი try/finally-ში — მეხსიერება გათავისუფლდება
        # fetch_balance_free(coin) — ერთი coin-ი, არა მთელი ბალანსი
        ex = BinanceSpotClient()
        binance_total: dict[str, float] = {}
        logger.info(f"[QTY_SYNC] Fetching balances: {list(coin_db.keys())}")

        for coin in coin_db.keys():
            try:
                free = ex.fetch_balance_free(coin)
                binance_total[coin] = float(free or 0)
                logger.info(f"[QTY_SYNC] {coin} balance={binance_total[coin]:.8f}")
            except Exception as _be:
                logger.warning(f"[QTY_SYNC] {coin} fetch fail: {_be}")
                binance_total[coin] = 0.0

        # FIX #1: exchange client-ის გათავისუფლება მეხსიერებიდან
        del ex
        ex = None

        # FIX #2: conn try/finally-ში — ყოველთვის დაიხურება
        conn = get_connection()

        for coin, pos_list in coin_db.items():
            db_total_qty = sum(float(p.get("total_qty") or 0) for p in pos_list)
            binance_qty  = binance_total.get(coin, 0.0)

            if db_total_qty <= 0:
                continue

            result["checked"] += 1
            diff_pct = abs(db_total_qty - binance_qty) / db_total_qty

            detail = {
                "coin":        coin,
                "db_qty":      round(db_total_qty, 8),
                "binance_qty": round(binance_qty, 8),
                "diff_pct":    round(diff_pct * 100, 4),
                "action":      "ok",
                "positions":   [p["symbol"] for p in pos_list],
            }

            if diff_pct <= TOLERANCE:
                logger.info(
                    f"[QTY_SYNC] {coin} OK | "
                    f"db={db_total_qty:.8f} binance={binance_qty:.8f} "
                    f"diff={diff_pct*100:.3f}%"
                )
                detail["action"] = "ok"
                result["skipped"] += 1

            else:
                logger.warning(
                    f"[QTY_SYNC] {coin} MISMATCH | "
                    f"db={db_total_qty:.8f} binance={binance_qty:.8f} "
                    f"diff={diff_pct*100:.3f}% > tolerance={TOLERANCE*100:.1f}%"
                )

                if DRY_RUN:
                    logger.info(f"[QTY_SYNC] DRY_RUN — {coin} არ შეიცვლება")
                    detail["action"] = "dry_run"
                    result["skipped"] += 1
                else:
                    if len(pos_list) == 1:
                        # ── ერთი პოზიცია — პირდაპირ Binance qty-ს ვიყენებთ
                        pos = pos_list[0]
                        old_qty = float(pos.get("total_qty") or 0)
                        new_qty = round(binance_qty, 8)

                        conn.execute(
                            "UPDATE dca_positions SET total_qty=? WHERE id=? AND status='OPEN'",
                            (new_qty, pos["id"])
                        )
                        conn.commit()

                        logger.warning(
                            f"[QTY_SYNC] FIXED {pos['symbol']} | "
                            f"qty: {old_qty:.8f} → {new_qty:.8f}"
                        )
                        detail["action"]  = "fixed"
                        detail["old_qty"] = round(old_qty, 8)
                        detail["new_qty"] = new_qty
                        result["fixed"] += 1

                    else:
                        # FIX #3: რამდენიმე პოზიცია — ცალ-ცალკე Binance balance
                        # პროპორციული fix არასწორია — BTC და BTC_L2 ცალ-ცალკე ბალანსია
                        # ამიტომ: DB total vs Binance total-ის ratio გამოვიყენოთ
                        # მხოლოდ თუ Binance > 0 (არ გავანულოთ!)
                        if binance_qty <= 0:
                            logger.warning(
                                f"[QTY_SYNC] {coin} SKIP_MULTI | "
                                f"binance_qty=0 → საშიშია, skip"
                            )
                            detail["action"] = "skip_zero_binance"
                            result["skipped"] += 1
                        else:
                            ratio = binance_qty / db_total_qty
                            # safety: ratio > 1.1 ან < 0.5 → skip (უჩვეულო)
                            if ratio > 1.1 or ratio < 0.5:
                                logger.warning(
                                    f"[QTY_SYNC] {coin} SKIP_RATIO | "
                                    f"ratio={ratio:.4f} — უჩვეულო, manual check საჭიროა"
                                )
                                detail["action"] = "skip_unusual_ratio"
                                result["skipped"] += 1
                            else:
                                for pos in pos_list:
                                    old_qty = float(pos.get("total_qty") or 0)
                                    new_qty = round(old_qty * ratio, 8)
                                    conn.execute(
                                        "UPDATE dca_positions SET total_qty=? WHERE id=? AND status='OPEN'",
                                        (new_qty, pos["id"])
                                    )
                                    logger.warning(
                                        f"[QTY_SYNC] FIXED {pos['symbol']} | "
                                        f"qty: {old_qty:.8f} → {new_qty:.8f} "
                                        f"(ratio={ratio:.6f})"
                                    )
                                conn.commit()
                                detail["action"] = "fixed_proportional"
                                result["fixed"] += 1

            result["details"].append(detail)

        logger.info(
            f"[QTY_SYNC] DONE | "
            f"checked={result['checked']} "
            f"fixed={result['fixed']} "
            f"skipped={result['skipped']}"
        )

    except Exception as e:
        logger.error(f"[QTY_SYNC] ERROR | {e}")
        result["error"] = str(e)

    finally:
        # FIX #2: conn და ex ყოველთვის გათავისუფლდება
        if conn:
            try:
                conn.close()
            except Exception:
                pass
        if ex:
            try:
                del ex
            except Exception:
                pass

    return result


def print_report(result: dict) -> None:
    """Shell-ში ლამაზი ანგარიში."""
    print("\n" + "="*55)
    print("  QTY SYNC REPORT")
    print("="*55)
    print(f"  checked : {result.get('checked', 0)}")
    print(f"  fixed   : {result.get('fixed', 0)}")
    print(f"  skipped : {result.get('skipped', 0)}")
    if result.get("error"):
        print(f"  ERROR   : {result['error']}")
    print("-"*55)
    for d in result.get("details", []):
        action = d.get("action", "?").upper()
        icon   = "✅" if action == "OK" else "🔧" if "FIXED" in action else "⏭️"
        print(f"\n  {icon} {d['coin']}")
        print(f"     DB qty      : {d.get('db_qty', '?')}")
        print(f"     Binance qty : {d.get('binance_qty', '?')}")
        print(f"     diff        : {d.get('diff_pct', '?')}%")
        print(f"     action      : {action}")
        if d.get("old_qty"):
            print(f"     old → new   : {d['old_qty']} → {d.get('new_qty', '?')}")
        print(f"     positions   : {', '.join(d.get('positions', []))}")
    print("="*55 + "\n")


# ── standalone გაშვება Shell-იდან ───────────────────────────
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(asctime)s %(message)s"
    )

    # DRY_RUN override command line-დან
    if "--dry-run" in sys.argv:
        os.environ["QTY_SYNC_DRY_RUN"] = "true"
        print("🔍 DRY RUN MODE — მხოლოდ ბეჭდავს, DB-ს არ ცვლის\n")

    sys.path.insert(0, "/opt/render/project/src")

    result = run_qty_sync()
    print_report(result)
