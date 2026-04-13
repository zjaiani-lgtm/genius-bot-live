# execution/startup_sync.py
import os
import logging

from execution.db.repository import update_system_state, log_event

logger = logging.getLogger("gbm")


def _validate_and_fix_tp_prices() -> None:
    """
    STARTUP TP VALIDATION — FIX #5
    ================================
    რესტარტის შემდეგ ყველა ღია პოზიციის TP-ს ამოწმებს:
      - tp_price <= avg_entry_price  → TP entry-ზე დაბლა = ზარალი!
      - tp_price <= 0                → TP საერთოდ არ არის
    პრობლემის შემთხვევაში TP-ს ხელახლა ითვლის:
      new_tp = avg_entry * (1 + DCA_TP_PCT/100)

    ეს BNB/USDT TP=693 @ entry=699 ტიპის bug-ს ასწორებს რესტარტზე.
    """
    try:
        from execution.db.repository import (
            get_all_open_dca_positions,
        )
        from execution.db.db import get_connection

        tp_pct = float(os.getenv("DCA_TP_PCT", "0.55"))
        positions = get_all_open_dca_positions()

        if not positions:
            logger.info("STARTUP_SYNC: TP_VALIDATION | no open positions → skip")
            return

        fixed = 0
        conn = get_connection()

        for pos in positions:
            sym       = pos.get("symbol", "?")
            pos_id    = pos.get("id")
            avg_entry = float(pos.get("avg_entry_price") or 0.0)
            tp_price  = float(pos.get("current_tp_price") or 0.0)

            if avg_entry <= 0:
                continue

            correct_tp = round(avg_entry * (1.0 + tp_pct / 100.0), 6)

            # TP დაბლა entry-ზე ან 0 → გაასწორე
            if tp_price <= avg_entry or tp_price <= 0:
                logger.warning(
                    f"STARTUP_SYNC: TP_INVALID | {sym} "
                    f"avg={avg_entry:.4f} tp={tp_price:.4f} → fixing to {correct_tp:.6f}"
                )
                conn.execute(
                    "UPDATE dca_positions SET current_tp_price=? WHERE id=? AND status='OPEN'",
                    (correct_tp, pos_id),
                )
                conn.commit()
                log_event(
                    "STARTUP_TP_FIXED",
                    f"sym={sym} old_tp={tp_price:.4f} new_tp={correct_tp:.6f} avg={avg_entry:.4f}"
                )
                fixed += 1
            else:
                logger.info(
                    f"STARTUP_SYNC: TP_OK | {sym} "
                    f"avg={avg_entry:.4f} tp={tp_price:.4f} "
                    f"(+{(tp_price/avg_entry-1)*100:.3f}%)"
                )

        conn.close()
        logger.info(f"STARTUP_SYNC: TP_VALIDATION done | fixed={fixed}/{len(positions)}")

    except Exception as e:
        logger.warning(f"STARTUP_SYNC: TP_VALIDATION_FAIL | err={e}")


def run_startup_sync() -> bool:
    """
    Goal:
      - In LIVE/TESTNET: verify exchange connectivity (diagnostics)
      - Validate + fix all open position TP prices (FIX #5)
      - Mark system_state as ACTIVE + startup_sync_ok=1 if OK
      - Otherwise PAUSE + startup_sync_ok=0
    """
    mode = os.getenv("MODE", "DEMO").upper()

    try:
        if mode in ("LIVE", "TESTNET"):
            from execution.exchange_client import BinanceSpotClient

            ex = BinanceSpotClient()
            diag = ex.diagnostics()

            if not diag.get("ok"):
                err = diag.get("error", "unknown")
                logger.warning(f"STARTUP_SYNC: {mode} -> EXCHANGE_CONNECT_FAILED -> PAUSE | err={err}")
                update_system_state(status="PAUSED", startup_sync_ok=False)
                log_event("STARTUP_SYNC_FAILED", f"{mode} exchange_connect_failed err={err}")
                return False

            logger.info(
                f"STARTUP_SYNC: {mode} -> EXCHANGE_OK | "
                f"usdt_free={diag.get('usdt_free')} last={diag.get('last_price')}"
            )

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # FIX #5: TP VALIDATION — ყველა ღია პოზიციის TP-ს ამოწმებს
        # entry > tp → TP გამოსასწორებელია → new_tp = avg * 1.0055
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        _validate_and_fix_tp_prices()

        update_system_state(status="ACTIVE", startup_sync_ok=True)
        log_event("STARTUP_SYNC_OK", f"{mode} exchange_ok + tp_validated")
        return True

    except Exception as e:
        logger.warning(f"STARTUP_SYNC: ERROR -> PAUSE | err={e}")
        update_system_state(status="PAUSED", startup_sync_ok=False)
        log_event("STARTUP_SYNC_FAILED", f"{mode} err={e}")
        return False
