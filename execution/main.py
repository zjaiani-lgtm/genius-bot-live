import os
import time
import logging
from typing import Optional, Dict, Any

from execution.regime_engine import MarketRegimeEngine

from execution.db.db import init_db
from execution.db.repository import (
    get_system_state,
    update_system_state,
    log_event,
    get_trade_stats,
    get_closed_trades,
)
from execution.execution_engine import ExecutionEngine
from execution.signal_client import pop_next_signal
from execution.kill_switch import is_kill_switch_active
from execution.telegram_notifier import (
    notify_performance_snapshot,
    build_daily_stats_from_closed_trades,
    notify_daily_close_summary,
    _now_dt,
)

logger = logging.getLogger("gbm")

# SIGNAL_EXPIRATION_SECONDS — outbox-დან წამოღებული ძველი signal-ი → skip
_SIGNAL_EXPIRATION_SECONDS = int(os.getenv("SIGNAL_EXPIRATION_SECONDS", "0"))


def _bootstrap_state_if_needed() -> None:
    raw = get_system_state()
    if raw is None or len(raw) < 5:
        logger.warning("BOOTSTRAP_STATE | system_state row missing or invalid -> skip")
        return

    status = str(raw[1] or "").upper()
    startup_sync_ok = int(raw[2] or 0)
    kill_switch_db = int(raw[3] or 0)

    env_kill = os.getenv("KILL_SWITCH", "false").lower() == "true"

    logger.info(
        f"BOOTSTRAP_STATE | status={status} startup_sync_ok={startup_sync_ok} "
        f"kill_db={kill_switch_db} env_kill={env_kill}"
    )

    if env_kill or kill_switch_db == 1:
        logger.warning("BOOTSTRAP_STATE | kill switch ON -> skip overrides")
        return

    if status == "PAUSED" or startup_sync_ok == 0:
        logger.warning("BOOTSTRAP_STATE | applying self-heal -> status=RUNNING startup_sync_ok=1 kill_switch=0")
        update_system_state(status="RUNNING", startup_sync_ok=1, kill_switch=0)


def _try_import_generator():
    try:
        from execution.signal_generator import run_once as generate_once
        return generate_once
    except Exception as e:
        logger.error(f"GENERATOR_IMPORT_FAIL | err={e} -> generator disabled (consumer will still run)")
        try:
            log_event("GENERATOR_IMPORT_FAIL", f"err={e}")
        except Exception:
            pass
        return None


def _safe_pop_next_signal(outbox_path: str) -> Optional[Dict[str, Any]]:
    try:
        return pop_next_signal(outbox_path)
    except Exception as e:
        logger.exception(f"OUTBOX_POP_FAIL | path={outbox_path} err={e}")
        try:
            log_event("OUTBOX_POP_FAIL", f"path={outbox_path} err={e}")
        except Exception:
            pass
        return None


def _run_performance_report_safe(send_telegram: bool = False) -> None:
    try:
        s = get_trade_stats()
        logger.info(
            "PERF_REPORT | closed=%s wins=%s losses=%s winrate=%.2f%% roi=%.2f%% pnl=%.4f quote_in=%.4f pf=%.3f | open=%s open_quote_in=%.4f",
            s.get("closed_trades", 0),
            s.get("wins", 0),
            s.get("losses", 0),
            float(s.get("winrate_pct", 0.0)),
            float(s.get("roi_pct", 0.0)),
            float(s.get("pnl_quote_sum", 0.0)),
            float(s.get("quote_in_sum", 0.0)),
            float(s.get("profit_factor", 0.0)),
            s.get("open_trades", 0),
            float(s.get("open_quote_in_sum", 0.0)),
        )

        try:
            log_event(
                "PERF_REPORT",
                f"closed={s.get('closed_trades', 0)} "
                f"winrate={float(s.get('winrate_pct', 0.0)):.2f}% "
                f"roi={float(s.get('roi_pct', 0.0)):.2f}% "
                f"pnl={float(s.get('pnl_quote_sum', 0.0)):.4f} "
                f"open={s.get('open_trades', 0)} "
                f"open_quote_in={float(s.get('open_quote_in_sum', 0.0)):.4f}"
            )
        except Exception:
            pass

        if send_telegram:
            try:
                notify_performance_snapshot(s)
            except Exception as e:
                logger.warning(f"TG_NOTIFY_PERF_FAIL | err={e}")

    except Exception as e:
        logger.warning(f"PERF_REPORT_FAIL | err={e}")


def main():
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(asctime)s - %(message)s')

    mode = os.getenv("MODE", "DEMO").upper()
    outbox_path = os.getenv("SIGNAL_OUTBOX_PATH", "/var/data/signal_outbox.json")
    sleep_s = float(os.getenv("LOOP_SLEEP_SECONDS", "10"))

    report_every_s = int(os.getenv("REPORT_EVERY_SECONDS", "60"))
    telegram_report_every_s = int(os.getenv("TELEGRAM_REPORT_EVERY_SECONDS", "1800"))

    last_report_ts = 0.0
    last_tg_report_ts = 0.0
    last_daily_summary_date = None

    init_db()
    _bootstrap_state_if_needed()

    engine = ExecutionEngine()

    try:
        engine.reconcile_oco()
    except Exception as e:
        logger.warning(f"OCO_RECONCILE_START_WARN | err={e}")

    generate_once = _try_import_generator()

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # FIX #3: MarketRegimeEngine — loop-გარეთ, ᲔᲠᲗᲘ instance სამუდამოდ
    # ძველი კოდი ყოველ ტიკზე ახალ instance-ს ქმნიდა → state იკარგებოდა
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    regime_engine = MarketRegimeEngine()

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # FIX C-1: inject_regime_engine — execution_engine-ს regime_engine
    # გადაეცემა, რათა TP/SL close-ზე in-memory SL counter სწორად reset-ს.
    # გარეშე: _regime_engine=None → notify_outcome() არასოდეს გამოიძახება
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    engine.inject_regime_engine(regime_engine)

    logger.info(f"GENIUS BOT MAN worker starting | MODE={mode}")
    logger.info(f"OUTBOX_PATH={outbox_path}")
    logger.info(f"LOOP_SLEEP_SECONDS={sleep_s}")
    logger.info(f"REPORT_EVERY_SECONDS={report_every_s}")
    logger.info(f"TELEGRAM_REPORT_EVERY_SECONDS={telegram_report_every_s}")

    while True:
        try:
            if is_kill_switch_active():
                logger.warning("KILL_SWITCH_ACTIVE | worker will not generate/pop/execute signals")
                try:
                    log_event("WORKER_KILL_SWITCH_ACTIVE", "blocked before loop actions")
                except Exception:
                    pass
                time.sleep(sleep_s)
                continue

            try:
                engine.reconcile_oco()
            except Exception as e:
                logger.warning(f"OCO_RECONCILE_LOOP_WARN | err={e}")

            if generate_once is not None:
                try:
                    created = generate_once(outbox_path)
                    if created:
                        logger.info("SIGNAL_GENERATOR | signal created")
                except Exception as e:
                    logger.exception(f"SIGNAL_GENERATOR_FAIL | err={e}")
                    try:
                        log_event("SIGNAL_GENERATOR_FAIL", f"err={e}")
                    except Exception:
                        pass

                sig = _safe_pop_next_signal(outbox_path)

                if sig:
                    signal_id = sig.get("signal_id", "UNKNOWN")
                    verdict = str(sig.get("final_verdict", "")).upper()

                    logger.info(f"Signal received | id={signal_id} | verdict={verdict}")

                    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                    # SIGNAL_EXPIRATION_SECONDS — ძველი signal-ი → skip
                    # signal_generator-ი წერს sig["ts_utc"] (UTC ISO)
                    # თუ signal-ი outbox-ში SIGNAL_EXPIRATION_SECONDS-ზე
                    # მეტია → გამოტოვება (stale signal)
                    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                    if _SIGNAL_EXPIRATION_SECONDS > 0:
                        try:
                            from datetime import datetime, timezone
                            ts_raw = sig.get("ts_utc", "")
                            if ts_raw:
                                sig_dt = datetime.fromisoformat(
                                    str(ts_raw).replace("Z", "+00:00")
                                )
                                if sig_dt.tzinfo is None:
                                    sig_dt = sig_dt.replace(tzinfo=timezone.utc)
                                age_s = (datetime.now(timezone.utc) - sig_dt).total_seconds()
                                if age_s > _SIGNAL_EXPIRATION_SECONDS:
                                    logger.warning(
                                        f"[EXPIRED] signal skipped | id={signal_id} "
                                        f"age={age_s:.0f}s > limit={_SIGNAL_EXPIRATION_SECONDS}s"
                                    )
                                    try:
                                        log_event(
                                            "SIGNAL_EXPIRED",
                                            f"id={signal_id} age={age_s:.0f}s verdict={verdict}"
                                        )
                                    except Exception:
                                        pass
                                    continue
                        except Exception as e:
                            logger.warning(f"EXPIRY_CHECK_FAIL | id={signal_id} err={e} → skip check")

                    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                    # FIX #2: SELL signal — regime check bypass
                    # SELL (TREND_REVERSAL / PROTECTIVE_SELL) ყოველთვის
                    # სრულდება, SKIP_TRADING-ი მას ვერ ბლოკავს.
                    # ძველი კოდი SELL-საც ჩერდებოდა SIDEWAYS-ზე!
                    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                    if verdict == "SELL":
                        logger.info(
                            f"[AUTO] SELL signal → bypass regime check | "
                            f"id={signal_id} source={sig.get('meta', {}).get('source', 'UNKNOWN')}"
                        )
                        engine.execute_signal(sig)

                    elif verdict == "TRADE":
                        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                        # GAP-2 FIX: main.py-ი აღარ ახდენს TP/SL-ის recalc-ს.
                        # signal_generator-მა უკვე გაითვლა adaptive (TP/SL + MTF bonus)
                        # და sig["adaptive"]-ში ჩაწერა.
                        # main.py-ი მხოლოდ SKIP safety-net-ია:
                        # თუ სიგნალის emit-სა და execution-ს შორის (20 წამი)
                        # ბაზარი BEAR/VOLATILE/SIDEWAYS-ად გადაბრუნდა → ბლოკავს.
                        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                        trend   = float(sig.get("trend",     0) or 0)
                        atr_pct = float(sig.get("atr_pct",   0) or 0)
                        symbol  = str((sig.get("execution") or {}).get("symbol", ""))

                        regime  = regime_engine.detect_regime(trend=trend, atr_pct=atr_pct)

                        # SKIP safety-net only — sig["adaptive"] არ გადაიწერება
                        if regime in ("BEAR", "VOLATILE", "SIDEWAYS"):
                            logger.warning(
                                f"[AUTO] SKIP_SAFETY_NET | regime={regime} "
                                f"trend={trend:.3f} atr={atr_pct:.3f} | id={signal_id}"
                            )
                            continue

                        logger.info(
                            f"[AUTO] Regime={regime} trend={trend:.3f} "
                            f"atr_pct={atr_pct:.3f} symbol={symbol} "
                            f"TP={sig.get('adaptive', {}).get('TP_PCT', 'n/a')}% "
                            f"SL={sig.get('adaptive', {}).get('SL_PCT', 'n/a')}% "
                            f"mtf={sig.get('meta', {}).get('mtf_alignment', 'N/A')} "
                            f"| id={signal_id}"
                        )

                        engine.execute_signal(sig)

                    else:
                        # HOLD ან სხვა — უბრალოდ log
                        logger.info(f"[AUTO] Unsupported verdict={verdict} | id={signal_id} → skip")

                else:
                    logger.info("Worker alive, waiting for SIGNAL_OUTBOX...")

            now = time.time()

            if report_every_s > 0 and (now - last_report_ts) >= report_every_s:
                _run_performance_report_safe(send_telegram=False)
                last_report_ts = now

            if telegram_report_every_s > 0 and (now - last_tg_report_ts) >= telegram_report_every_s:
                _run_performance_report_safe(send_telegram=True)
                last_tg_report_ts = now

            try:
                now_local = _now_dt()
                today_str = now_local.date().isoformat()

                if (
                    now_local.hour == 23
                    and now_local.minute == 59
                    and last_daily_summary_date != today_str
                ):
                    closed_trades = get_closed_trades()
                    daily_stats = build_daily_stats_from_closed_trades(
                        closed_trades,
                        target_dt=now_local,
                    )
                    notify_daily_close_summary(daily_stats)
                    last_daily_summary_date = today_str

                    logger.info(
                        "DAILY_SUMMARY_SENT | date=%s closed=%s pnl=%.4f",
                        today_str,
                        daily_stats.get("closed_trades", 0),
                        float(daily_stats.get("pnl_quote_sum", 0.0)),
                    )

                    try:
                        log_event(
                            "DAILY_SUMMARY_SENT",
                            f"date={today_str} "
                            f"closed={daily_stats.get('closed_trades', 0)} "
                            f"wins={daily_stats.get('wins', 0)} "
                            f"losses={daily_stats.get('losses', 0)} "
                            f"pnl={float(daily_stats.get('pnl_quote_sum', 0.0)):.4f}"
                        )
                    except Exception:
                        pass

            except Exception as e:
                logger.warning(f"DAILY_SUMMARY_FAIL | err={e}")

        except Exception as e:
            logger.exception(f"WORKER_LOOP_ERROR | err={e}")
            try:
                log_event("WORKER_LOOP_ERROR", f"err={e}")
            except Exception:
                pass

        time.sleep(sleep_s)


if __name__ == "__main__":
    main()
