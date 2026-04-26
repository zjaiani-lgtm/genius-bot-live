import os
import time
import logging
from typing import Optional, Dict, Any

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# GENIUS-DCA-Bot — main.py
# CHANGELOG:
#
# FIX #1 — buy_qty slippage correction (3 ადგილი)
#   პრობლემა: buy_qty = quote / buy_price  → slippage იგნორირებული
#   გამოსწორება: buy.get("filled") → Binance-ის რეალური filled qty
#
# FIX #2 — TP/SL/FC trades lookup Layer სიმბოლოებისთვის
#   პრობლემა: BTC/USDT_L2 trade ვერ იპოვებოდა DB-ში
#   გამოსწორება: sym პირველი, exchange_sym fallback
#
# FIX #14 — TP_FIX ყოველ loop-ზე (2026-04-12)
#   პრობლემა: ADD-ON-ის შემდეგ TP < avg → პოზიცია არასოდეს იყიდება
#   გამოსწორება: run_tp_fix() ყოველ loop-ზე
#
# FIX #17 — DCA TP/FORCE_CLOSE → Hedge SHORT auto-close (BUG-1)
#   პრობლემა: DCA TP hit-ის შემდეგ hedge SHORT რჩებოდა ღია →
#             BTC ამაღლდებოდა → SHORT ზარალი (missing link)
#   გამოსწორება: close_dca_hedge_for_position(pos_id) TP და FC block-ებში
#   ფაილი: main.py (2 ადგილი) + futures_engine.py (ახალი მეთოდი)
#
# FIX #18 — MAX_OPEN_TRADES hardcoded fallback "8" → "6" (BUG-2)
#   პრობლემა: main.py os.getenv("MAX_OPEN_TRADES","8") — ENV=6, config=2
#             triple conflict: 3 სხვადასხვა მნიშვნელობა სამ ადგილში
#   გამოსწორება: fallback "8" → "6" (ENV-ს ემთხვევა)
#
# FIX #19 — INDEPENDENT SHORT DCA სისტემა
#   სარკე სტრატეგია: LONG L1 გახსნის შემდეგ SHORT იხსნება -1.6%-ზე
#   ADD-ONs ვარდნაზე (LONG-ის სარკე): -1.0%, -2.2%, -3.5%
#   TP: avg × 0.9945 — სავალდებულო დახურვა
#   FC: 10 დღე / +15% drawdown — ღია ტრეიდი არ რჩება
#   ENV: SHORT_DCA_ENABLED=true + SHORT_* params
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ENV LOADING — python-dotenv
# override=False → Render ENV პრიორიტეტულია .env-ზე
# ე.ი. Render-ზე დაყენებული ცვლადი ყოველთვის იმარჯვებს
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
try:
    from dotenv import load_dotenv
    load_dotenv("/opt/render/project/src/.env", override=False)
except ImportError:
    pass  # python-dotenv არ არის — Render ENV კმარა

from execution.regime_engine import MarketRegimeEngine

from execution.db.db import init_db
from execution.db.repository import (
    get_system_state,
    update_system_state,
    log_event,
    get_trade_stats,
    get_closed_trades,
    close_trade,
    get_open_trade_for_symbol,
    reset_consecutive_sl_per_symbol,
)
from execution.execution_engine import ExecutionEngine
from execution.signal_client import pop_next_signal
from execution.kill_switch import is_kill_switch_active
from execution.dca_position_manager import get_dca_manager
from execution.dca_tp_sl_manager import get_tp_sl_manager
from execution.dca_risk_manager import get_risk_manager
from execution.futures_engine import get_futures_engine
from execution.telegram_notifier import (
    notify_performance_snapshot,
    build_daily_stats_from_closed_trades,
    notify_daily_close_summary,
    notify_dca_addon,
    notify_dca_closed,
    notify_dca_breakeven,
    _now_dt,
)

logger = logging.getLogger("gbm")

# SIGNAL_EXPIRATION_SECONDS — outbox-დან წამოღებული ძველი signal-ი → skip
_SIGNAL_EXPIRATION_SECONDS = 0  # DCA: disabled


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



def _run_dca_loop(engine, dca_mgr, tp_sl_mgr, risk_mgr,
                  market_regime: str = "NEUTRAL",
                  futures_engine=None) -> None:
    """
    DCA monitoring loop — ყოველ main loop iteration-ზე გამოიძახება.

    შეამოწმებს:
      1. TP hit → close position
      2. Breakeven → SL გადაადგილება
      3. Force close → max drawdown ან max add-ons + SL
      4. SL confirmed → close position
      5. Add-on → drawdown trigger + recovery signals
         BEAR MODE: L2+L3 ADD-ON+rotation blocked
      6. DCA Hedge SHORT trigger → add_on_count == max_add_ons
    """
    from execution.db.repository import (
        get_all_open_dca_positions,
        close_dca_position,
        update_dca_position_after_addon,
        update_dca_sl_price,
        add_dca_order,
        open_dca_position,
    )
    from execution.dca_position_manager import recalculate_average, score_recovery_signals

    open_positions = get_all_open_dca_positions()
    if not open_positions:
        return

    for pos in open_positions:
        sym = pos["symbol"]
        pos_id = pos["id"]

        import re as _re_sym
        exchange_sym = _re_sym.sub(r'_L\d+$', '', sym)

        try:
            # current price — DEMO: price_feed, LIVE: exchange
            try:
                if engine.exchange is not None:
                    current_price = engine.exchange.fetch_last_price(exchange_sym)
                else:
                    # DEMO mode — public API, key არ სჭირდება
                    _ticker = engine.price_feed.fetch_ticker(exchange_sym)
                    current_price = float(_ticker.get("last") or 0.0)
                current_price = float(current_price) if current_price else 0.0
            except Exception as _pe:
                logger.warning(f"[DCA] PRICE_FETCH_ERR | {sym} err={_pe}")
                current_price = 0.0

            if current_price <= 0:
                logger.warning(f"[DCA] NO_PRICE | {sym}")
                continue

            avg_entry    = float(pos["avg_entry_price"] or 0)
            tp_price     = float(pos["current_tp_price"] or 0)
            sl_price     = 0.0  # DCA: SL გათიშულია — ყოველთვის 0
            total_qty    = float(pos["total_qty"] or 0)
            total_quote  = float(pos["total_quote_spent"] or 0)
            add_on_count = int(pos["add_on_count"] or 0)

            logger.info(
                f"[DCA] MONITOR | {sym} price={current_price:.4f} "
                f"avg={avg_entry:.4f} tp={tp_price:.4f} "
                f"qty={total_qty:.6f} add_ons={add_on_count}"
            )

            # ── 1. TP hit ────────────────────────────────────────────────
            if tp_price > 0 and current_price >= tp_price:
                logger.info(f"[DCA] TP_HIT | {sym} price={current_price:.4f} >= tp={tp_price:.4f}")
                try:
                    # DEMO: ვირტუალური გაყიდვა
                    if engine.exchange is None:
                        exit_price = current_price
                        pnl_quote = (exit_price - avg_entry) * total_qty
                        pnl_pct   = (exit_price / avg_entry - 1.0) * 100.0
                        sell = {"average": exit_price, "price": exit_price}
                    else:
                        sell = engine.exchange.place_market_sell(exchange_sym, total_qty)
                        exit_price = float(sell.get("average") or sell.get("price") or current_price)
                        pnl_quote = (exit_price - avg_entry) * total_qty
                        pnl_pct   = (exit_price / avg_entry - 1.0) * 100.0

                    # ── dca_positions დახურვა ──────────────────────────
                    close_dca_position(pos_id, exit_price, total_qty, pnl_quote, pnl_pct, "TP")

                    # ── DCA hedge SHORT auto-close (BUG-1 FIX) ─────────
                    # DCA TP hit → is_dca_hedge=1 SHORT-ი აღარ საჭიროა.
                    # თუ hedge ჯერ არ გახსნილა → no-op (empty fetch).
                    # თუ hedge უკვე დაიხურა futures TP-ზე → no-op (status!='OPEN').
                    # ᲛᲜᲘᲨᲕᲜᲔᲚᲝᲕᲐᲜᲘ: Independent SHORT (is_independent_short=1) აქ
                    # არ იხურება — მას საკუთარი TP/FC lifecycle აქვს futures_engine-ში.
                    if futures_engine is not None:
                        try:
                            futures_engine.close_dca_hedge_for_position(
                                pos_id, reason="DCA_TP_HIT"
                            )
                        except Exception as _hce:
                            logger.warning(f"[DCA] HEDGE_CLOSE_TP_FAIL | {sym} err={_hce}")

                    # trades ცხრილის დახურვა
                    _open_tr = get_open_trade_for_symbol(sym)
                    if not _open_tr:
                        _open_tr = get_open_trade_for_symbol(exchange_sym)
                    if _open_tr:
                        close_trade(_open_tr[0], exit_price, "TP", pnl_quote, pnl_pct)
                        logger.info(f"[DCA] TRADE_CLOSED_TP | {sym} signal_id={_open_tr[0]} pnl={pnl_quote:+.4f}")
                    else:
                        logger.warning(f"[DCA] TRADE_NOT_FOUND | {sym} — trades row missing on TP")

                    # ── SL cooldown reset (TP = recovery) ─────────────
                    try:
                        reset_consecutive_sl_per_symbol(sym)
                    except Exception as _e:
                        logger.warning(f"[DCA] SL_RESET_FAIL | {sym} err={_e}")

                    try:
                        log_event("DCA_CLOSED_TP", f"sym={sym} exit={exit_price:.4f} pnl={pnl_quote:+.4f} pct={pnl_pct:.3f}%")
                    except Exception:
                        pass

                    from execution.telegram_notifier import notify_dca_closed
                    from execution.db.repository import get_trade_stats
                    stats = get_trade_stats()
                    notify_dca_closed(
                        sym, avg_entry, exit_price, total_qty, total_quote,
                        pnl_quote, pnl_pct, "TP", add_on_count, stats
                    )
                    logger.info(f"[DCA] CLOSED_TP | {sym} pnl={pnl_quote:+.4f}")
                except Exception as e:
                    logger.error(f"[DCA] TP_SELL_FAIL | {sym} err={e}")
                continue

            # ── 2. Force close check ─────────────────────────────────────
            force_close, fc_reason = tp_sl_mgr.should_force_close(pos, current_price)
            if force_close:
                logger.warning(f"[DCA] FORCE_CLOSE | {sym} reason={fc_reason}")
                try:
                    if engine.exchange is None:
                        exit_price = current_price
                        sell = {"average": exit_price, "price": exit_price}
                    else:
                        sell = engine.exchange.place_market_sell(exchange_sym, total_qty)
                        exit_price = float(sell.get("average") or sell.get("price") or current_price)
                    pnl_quote = (exit_price - avg_entry) * total_qty
                    pnl_pct   = (exit_price / avg_entry - 1.0) * 100.0
                    # pnl_quote-ის ნიშანი FC reason-ზეა დამოკიდებული:
                    #   FC by drawdown (-15%): exit < avg_entry → pnl_quote უარყოფითია (ზარალი)
                    #   FC by time (10d):      exit შეიძლება avg-ზე მაღლა იყოს bounce-ის შემდეგ
                    #                          → pnl_quote დადებითი (მოგება) — არ არის გარანტირებული ზარალი

                    close_dca_position(pos_id, exit_price, total_qty, pnl_quote, pnl_pct, "FORCE_CLOSE")

                    # ── DCA hedge SHORT auto-close (BUG-1 FIX) ─────────
                    # FORCE_CLOSE → is_dca_hedge=1 SHORT-ი უნდა დაიხუროს.
                    # FORCE_CLOSE-ის დროს BTC კიდევ ეცემა → hedge SHORT სავარაუდოდ
                    # მოგებაშია, მაგრამ DCA position-ი აღარ არსებობს hedge-ის გასამართლებლად.
                    # ᲛᲜᲘᲨᲕᲜᲔᲚᲝᲕᲐᲜᲘ: close_dca_hedge_for_position() მხოლოდ is_dca_hedge=1-ს
                    # ხურავს (dca_pos_id match). Independent SHORT (is_independent_short=1)
                    # საკუთარი TP/FC lifecycle-ით იხურება — LONG FC-ზე არ იხურება.
                    if futures_engine is not None:
                        try:
                            futures_engine.close_dca_hedge_for_position(
                                pos_id, reason="DCA_FORCE_CLOSE"
                            )
                        except Exception as _hce:
                            logger.warning(f"[DCA] HEDGE_CLOSE_FC_FAIL | {sym} err={_hce}")

                    # trades ცხრილის დახურვა
                    _open_tr = get_open_trade_for_symbol(sym)
                    if not _open_tr:
                        _open_tr = get_open_trade_for_symbol(exchange_sym)
                    if _open_tr:
                        close_trade(_open_tr[0], exit_price, "FORCE_CLOSE", pnl_quote, pnl_pct)
                        logger.info(f"[DCA] TRADE_CLOSED_FC | {sym} signal_id={_open_tr[0]} pnl={pnl_quote:+.4f}")
                    else:
                        logger.warning(f"[DCA] TRADE_NOT_FOUND | {sym} — trades row missing on FORCE_CLOSE")

                    try:
                        log_event("DCA_FORCE_CLOSE", f"sym={sym} reason={fc_reason} exit={exit_price:.4f} pnl={pnl_quote:+.4f}")
                    except Exception:
                        pass

                    from execution.telegram_notifier import notify_dca_closed
                    notify_dca_closed(
                        sym, avg_entry, exit_price, total_qty, total_quote,
                        pnl_quote, pnl_pct, "FORCE_CLOSE", add_on_count
                    )
                except Exception as e:
                    logger.error(f"[DCA] FORCE_CLOSE_FAIL | {sym} err={e}")
                continue

            # ── 3. Fetch ohlcv for signal analysis ───────────────────────
            try:
                from execution.signal_generator import _fetch_ohlcv_direct
                tf = os.getenv("BOT_TIMEFRAME", "15m")
                ohlcv = _fetch_ohlcv_direct(exchange_sym, tf, 60)
            except Exception as e:
                logger.warning(f"[DCA] OHLCV_FAIL | {sym} err={e}")
                continue

            if not ohlcv or len(ohlcv) < 30:
                continue

            # ── 4. SL confirmed → close ──────────────────────────────────
            if sl_price > 0 and current_price < sl_price:
                sl_confirmed, sl_reason = tp_sl_mgr.is_sl_confirmed(sl_price, ohlcv)
                if sl_confirmed:
                    logger.info(f"[DCA] SL_CONFIRMED | {sym} reason={sl_reason}")
                    try:
                        if engine.exchange is None:
                            exit_price = current_price
                            sell = {"average": exit_price, "price": exit_price}
                        else:
                            sell = engine.exchange.place_market_sell(exchange_sym, total_qty)
                            exit_price = float(sell.get("average") or sell.get("price") or current_price)
                        pnl_quote = (exit_price - avg_entry) * total_qty
                        pnl_pct   = (exit_price / avg_entry - 1.0) * 100.0

                        close_dca_position(pos_id, exit_price, total_qty, pnl_quote, pnl_pct, "SL")

                            # trades ცხრილის დახურვა
                        _open_tr = get_open_trade_for_symbol(sym)
                        if not _open_tr:
                            _open_tr = get_open_trade_for_symbol(exchange_sym)
                        if _open_tr:
                            close_trade(_open_tr[0], exit_price, "SL", pnl_quote, pnl_pct)
                            logger.info(f"[DCA] TRADE_CLOSED_SL | {sym} signal_id={_open_tr[0]} pnl={pnl_quote:+.4f}")
                        else:
                            logger.warning(f"[DCA] TRADE_NOT_FOUND | {sym} — trades row missing on SL")

                        try:
                            from execution.db.repository import increment_consecutive_sl_per_symbol
                            increment_consecutive_sl_per_symbol(sym)
                        except Exception as _e:
                            logger.warning(f"[DCA] SL_INCREMENT_FAIL | {sym} err={_e}")

                        try:
                            log_event("DCA_CLOSED_SL", f"sym={sym} reason={sl_reason} exit={exit_price:.4f} pnl={pnl_quote:+.4f}")
                        except Exception:
                            pass

                        from execution.telegram_notifier import notify_dca_closed
                        notify_dca_closed(
                            sym, avg_entry, exit_price, total_qty, total_quote,
                            pnl_quote, pnl_pct, "SL", add_on_count
                        )
                    except Exception as e:
                        logger.error(f"[DCA] SL_SELL_FAIL | {sym} err={e}")
                    continue
                else:
                    logger.info(f"[DCA] SL_NOT_CONFIRMED | {sym} reason={sl_reason}")

            # ── 5. Breakeven check ───────────────────────────────────────
            be_update, new_sl = tp_sl_mgr.check_breakeven(avg_entry, current_price, sl_price)
            if be_update:
                update_dca_sl_price(pos_id, new_sl)
                from execution.telegram_notifier import notify_dca_breakeven
                notify_dca_breakeven(sym, avg_entry, sl_price, new_sl)
                sl_price = new_sl

            # ── 6. Add-on check ──────────────────────────────────────────

            all_positions = get_all_open_dca_positions()
            addon_ok, addon_reason = dca_mgr.should_add_on(pos, current_price, ohlcv)

            # BEAR MODE: L2 ADD-ON ბლოკილია (SHORT-ს ეწინააღმდეგება)
            # L3 zone: ბლოკი — ვარდნა გრძელდება, ADD-ON გაზრდის ზარალს
            if market_regime == "BEAR":
                logger.info(f"[DCA] BEAR_BLOCK | {sym} BEAR market → L2+L3 ADD-ON+rotation blocked")
                continue

            if not addon_ok:
                logger.debug(f"[DCA] NO_ADDON | {sym} reason={addon_reason}")

                # ── L3 ZONE ──────────────────────────────────────────────
                # ADD-ON ამოიწურა → L3 zone-ში ვართ
                n     = int(pos.get("add_on_count", 0))
                max_n = dca_mgr.max_add_ons

                if n >= max_n:
                    l3_done = int(pos.get("l3_addon_done", 0) or 0)

                    if not l3_done:
                        # ① L3 ADD-ON: ჯერ ერთხელ ყიდვა L2 resource-ით
                        # trigger: last_addon_price-დან drop >= rotation_trigger_pct
                        last_ap = float(pos.get("last_addon_price") or avg_entry)
                        if last_ap > 0:
                            drop_from_last = (last_ap - current_price) / last_ap * 100.0
                            rot_trigger    = dca_mgr.rotation_trigger_pct  # 1.5%
                            if drop_from_last >= rot_trigger:
                                logger.warning(
                                    f"[DCA] L3_ADDON_TRIGGER | {sym} "
                                    f"drop={drop_from_last:.2f}% >= {rot_trigger:.1f}% → L3 ADD-ON"
                                )
                                _execute_l3_addon(engine, pos, current_price, tp_sl_mgr)
                            else:
                                logger.debug(
                                    f"[DCA] L3_ADDON_WAIT | {sym} "
                                    f"drop={drop_from_last:.2f}% < {rot_trigger:.1f}%"
                                )
                    else:
                        # ② L3 ADD-ON გახსნილია → LIFO rotation
                        # trigger: last_addon_price (L3 ADD-ON price) -დან drop >= 1.5%
                        rotate_ok, rotate_reason = dca_mgr.should_rotate(pos, current_price)
                        if rotate_ok:
                            logger.warning(f"[DCA] L3_ROTATION_TRIGGER | {sym} → LIFO")
                            _execute_l3_rotation(engine, pos, current_price, tp_sl_mgr, dca_mgr)
                        else:
                            logger.debug(f"[DCA] NO_ROTATION | {sym} reason={rotate_reason}")

                continue

            risk_ok, risk_reason = risk_mgr.can_add_on(pos, dca_mgr.get_addon_size(add_on_count), all_positions)
            if not risk_ok:
                logger.info(f"[DCA] ADDON_RISK_BLOCK | {sym} reason={risk_reason}")
                continue

            # place add-on order
            addon_size = dca_mgr.get_addon_size(add_on_count)
            drawdown_pct = (avg_entry - current_price) / avg_entry * 100.0
            score, score_details = score_recovery_signals(ohlcv)

            logger.info(
                f"[DCA] PLACING_ADDON | {sym} level={add_on_count+1} "
                f"size={addon_size} drawdown={drawdown_pct:.2f}% score={score}/5"
            )

            try:
                # FIX #4: exchange_sym გამოიყენება (არა sym) — Binance "BTC/USDT_L2"-ს არ იცნობს
                # DEMO: ვირტუალური ყიდვა
                if engine.exchange is None:
                    buy_price = current_price
                    buy_qty   = addon_size / buy_price
                    buy = {"average": buy_price, "price": buy_price, "filled": buy_qty}
                else:
                    buy = engine.exchange.place_market_buy_by_quote(exchange_sym, addon_size)
                    buy_price = float(buy.get("average") or buy.get("price") or current_price)
                    # FIX #1: buy.get("filled") — Binance-ის რეალური დაფილვილი qty (slippage-გათვლილი)
                    buy_qty   = float(buy.get("filled") or buy.get("amount") or (addon_size / buy_price))

                avg_result = recalculate_average(total_qty, avg_entry, buy_qty, buy_price)
                new_avg    = avg_result["avg_entry_price"]
                new_qty    = avg_result["total_qty"]
                new_quote  = total_quote + addon_size

                # TP: position-ს ვაბარებთ — zone განსაზღვრისთვის
                # add_on_count+1 შემდეგ: L2 zone (< max_add_ons) → 0.55%
                #                        L3 zone (>= max_add_ons) → 0.35%
                _pos_after_addon = dict(pos)
                _pos_after_addon["add_on_count"] = add_on_count + 1
                tp_sl = tp_sl_mgr.calculate(new_avg, position=_pos_after_addon)
                new_tp = tp_sl["tp_price"]
                new_sl = tp_sl["sl_price"]

                # DB update
                update_dca_position_after_addon(
                    pos_id,
                    new_avg_entry=new_avg,
                    new_total_qty=new_qty,
                    new_total_quote=new_quote,
                    new_add_on_count=add_on_count + 1,
                    new_tp_price=new_tp,
                    new_sl_price=new_sl,
                    last_add_on_ts=time.time(),
                    last_addon_price=buy_price,  # L3 rotation trigger reference
                )

                # ── DCA HEDGE SHORT trigger ───────────────────────────
                # add_on_count+1 == max_add_ons → L2/L3 boundary
                # პირველი და ერთადერთი ჯერ: SHORT hedge გაიხსნება
                _new_add_on_count = add_on_count + 1
                _max_add_ons = dca_mgr.max_add_ons
                if _new_add_on_count == _max_add_ons and futures_engine is not None:
                    try:
                        _triggered = futures_engine.open_dca_hedge_short(
                            symbol=exchange_sym,
                            current_price=buy_price,
                            dca_pos_id=pos_id,
                            market_regime=market_regime,
                        )
                        if _triggered:
                            logger.warning(
                                f"[DCA] HEDGE_SHORT_TRIGGERED | {sym} "
                                f"add_on={_new_add_on_count}/{_max_add_ons} "
                                f"@ {buy_price:.4f}"
                            )
                    except Exception as _he:
                        logger.warning(f"[DCA] HEDGE_TRIGGER_FAIL | {sym} err={_he}")

                rsi_val = score_details.get("rsi", 0.0)
                atr_val = score_details.get("atr_pct", 0.0)

                add_dca_order(
                    position_id=pos_id,
                    symbol=sym,
                    order_type=f"ADD_ON_{add_on_count + 1}",
                    entry_price=buy_price,
                    qty=buy_qty,
                    quote_spent=addon_size,
                    avg_entry_after=new_avg,
                    tp_after=new_tp,
                    sl_after=new_sl,
                    trigger_drawdown_pct=drawdown_pct,
                    rsi_at_entry=rsi_val,
                    atr_pct_at_entry=atr_val,
                    recovery_score=score,
                    exchange_order_id=str(buy.get("id", "")),
                )

                notify_dca_addon(
                    symbol=sym,
                    addon_number=add_on_count + 1,
                    addon_price=buy_price,
                    addon_quote=addon_size,
                    new_avg_entry=new_avg,
                    total_quote_spent=new_quote,
                    new_tp_price=new_tp,
                    new_sl_price=new_sl,
                    drawdown_pct=drawdown_pct,
                    recovery_score=score,
                )

                logger.info(
                    f"[DCA] ADDON_PLACED | {sym} level={add_on_count+1} "
                    f"price={buy_price:.4f} new_avg={new_avg:.4f} "
                    f"tp={new_tp:.4f} sl={new_sl:.4f}"
                )

            except Exception as e:
                logger.error(f"[DCA] ADDON_PLACE_FAIL | {sym} err={e}")

        except Exception as e:
            logger.warning(f"[DCA] POSITION_LOOP_ERR | {sym} id={pos_id} err={e}")


def _execute_l3_addon(engine, pos: dict, current_price: float, tp_sl_mgr) -> None:
    """
    L3 ADD-ON — L2 resource ($10) გადადის L3-ზე.

    სრული flow:
      ADD-ONs exhausted + drop >= rotation_trigger_pct (1.5%) → L3 ADD-ON
      buy $10 @ current_price
      avg recalculate → TP = avg × 1.0035 (L3 zone 0.35%)
      l3_addon_done = 1 → შემდეგ iteration-ზე LIFO rotation იწყება

    მათემატიკა BTC @ $69,190 (ADD-ON #5) → drop -1.5% → $68,153:
      სულ invested: $80, avg=$71,742, qty=0.001170
      L3 ADD-ON: $10 @ $68,153 → qty=0.000147
      new_qty = 0.001317
      new_avg = ($80×$71,742 + $10×$68,153) / ($80+$10) per qty
             ≈ $71,365
      new_tp  = $71,365 × 1.0035 = $71,615 (0.35%)
    """
    from execution.db.repository import (
        add_dca_order,
        update_dca_position_after_l3_addon,
        log_event,
    )
    from execution.dca_position_manager import recalculate_average

    sym         = pos["symbol"]
    pos_id      = pos["id"]
    avg_entry   = float(pos["avg_entry_price"] or 0)
    total_qty   = float(pos["total_qty"] or 0)
    total_quote = float(pos["total_quote_spent"] or 0)

    import re as _re_l3
    exchange_sym = _re_l3.sub(r'_L\d+$', '', sym)

    try:
        l3_addon_quote = float(os.getenv("BOT_QUOTE_PER_TRADE", "12.0"))

        # RISK CHECK — balance საკმარისია?
        from execution.dca_risk_manager import get_risk_manager as _get_rm
        _l3_risk_ok, _l3_risk_reason = _get_rm().can_l3_operation(l3_addon_quote)
        if not _l3_risk_ok:
            logger.warning(f"[L3_ADDON] RISK_BLOCK | {sym} reason={_l3_risk_reason}")
            return

        if engine.exchange is None:
            buy_price = current_price
            buy_qty   = l3_addon_quote / buy_price
            buy = {"average": buy_price, "filled": buy_qty}
        else:
            buy = engine.exchange.place_market_buy_by_quote(exchange_sym, l3_addon_quote)
            buy_price = float(buy.get("average") or buy.get("price") or current_price)
            buy_qty   = float(buy.get("filled") or buy.get("amount") or (l3_addon_quote / buy_price))

        avg_result = recalculate_average(total_qty, avg_entry, buy_qty, buy_price)
        new_avg    = avg_result["avg_entry_price"]
        new_qty    = avg_result["total_qty"]
        new_quote  = total_quote + l3_addon_quote

        # L3 zone TP (0.35%)
        new_tp = tp_sl_mgr.calculate_rotation_tp(new_avg)

        drawdown_pct = (avg_entry - current_price) / avg_entry * 100.0 if avg_entry > 0 else 0.0

        logger.warning(
            f"[L3_ADDON] OPENED | {sym} @ {buy_price:.4f} "
            f"qty={buy_qty:.6f} quote={l3_addon_quote} | "
            f"old_avg={avg_entry:.4f} → new_avg={new_avg:.4f} "
            f"new_tp={new_tp:.4f} drawdown={drawdown_pct:.2f}%"
        )

        # DB: l3_addon_done=1, last_addon_price=buy_price (LIFO trigger reference)
        update_dca_position_after_l3_addon(
            position_id=pos_id,
            new_avg_entry=new_avg,
            new_total_qty=new_qty,
            new_total_quote=new_quote,
            new_tp_price=new_tp,
            last_addon_price=buy_price,
        )

        add_dca_order(
            position_id=pos_id,
            symbol=sym,
            order_type="L3_ADDON",
            entry_price=buy_price,
            qty=buy_qty,
            quote_spent=l3_addon_quote,
            avg_entry_after=new_avg,
            tp_after=new_tp,
            sl_after=0.0,
            trigger_drawdown_pct=drawdown_pct,
            exchange_order_id=str(buy.get("id", "")),
        )

        try:
            log_event(
                "L3_ADDON_OPENED",
                f"sym={sym} price={buy_price:.4f} quote={l3_addon_quote} "
                f"old_avg={avg_entry:.4f} new_avg={new_avg:.4f} "
                f"new_tp={new_tp:.4f} drawdown={drawdown_pct:.2f}%"
            )
        except Exception:
            pass

        try:
            from execution.telegram_notifier import send_telegram_message
            send_telegram_message(
                f"📥 <b>L3 ADD-ON გახსნა</b>\n\n"
                f"🪙 <b>Symbol:</b> <code>{sym}</code>\n"
                f"💰 <b>Entry:</b> <code>{buy_price:.2f}</code> "
                f"(<code>-{drawdown_pct:.2f}%</code>)\n"
                f"📊 <b>avg:</b> <code>{avg_entry:.2f} → {new_avg:.2f}</code> ↓\n"
                f"🎯 <b>TP:</b> <code>{new_tp:.2f}</code> (L3: 0.35%)\n"
                f"⚠️ <b>შემდეგი:</b> LIFO rotation კიდევ -1.5%-ზე\n"
                f"🕒 <code>{__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</code>"
            )
        except Exception as _tg:
            logger.warning(f"[L3_ADDON] TG_FAIL | err={_tg}")

    except Exception as e:
        logger.error(f"[L3_ADDON] FAIL | {sym} err={e}")


def _execute_l3_rotation(engine, pos: dict, current_price: float, tp_sl_mgr, dca_mgr) -> None:
    """
    LIFO Rotation — L3 zone: ADD-ONs exhausted + price კვლავ ეცემა.

    მათემატიკა:
      LIFO unit (ყველაზე ძვირი) @ entry_price=P_high, qty=Q_high
      sell @ current_price → proceeds = current_price × Q_high
      realized_loss = (current_price - P_high) × Q_high  (< 0)
      reinvest proceeds @ current_price → new_qty = Q_high (იგივე)
      new_avg = (total_value - P_high×Q_high + current×Q_high) / total_qty
             = old_avg - (P_high - current) × Q_high / total_qty
      → avg ეცემა, TP ახლოვდება

    vs FIFO: avg ამაღლდება (TP შორდება!) — LIFO სჯობია.
    """
    from execution.db.repository import (
        get_dca_orders,
        add_dca_order,
        update_dca_position_after_rotation,
        log_event,
    )
    from execution.dca_position_manager import recalculate_average

    sym      = pos["symbol"]
    pos_id   = pos["id"]
    avg_entry = float(pos["avg_entry_price"] or 0)
    total_qty = float(pos["total_qty"] or 0)
    total_quote = float(pos["total_quote_spent"] or 0)

    import re as _re_rot
    exchange_sym = _re_rot.sub(r'_L\d+$', '', sym)

    try:
        # 1. ყველა order-ი → LIFO unit
        dca_orders = get_dca_orders(pos_id)
        if not dca_orders:
            logger.warning(f"[L3_ROT] NO_ORDERS | {sym} → skip rotation")
            return

        lifo_unit = dca_mgr.get_lifo_unit(dca_orders)
        if not lifo_unit:
            logger.warning(f"[L3_ROT] NO_LIFO_UNIT | {sym} → skip rotation")
            return

        lifo_price = float(lifo_unit.get("entry_price", 0.0))
        lifo_qty   = float(lifo_unit.get("qty", 0.0))
        lifo_quote = float(lifo_unit.get("quote_spent", 0.0))

        if lifo_price <= 0 or lifo_qty <= 0:
            logger.warning(f"[L3_ROT] INVALID_LIFO | {sym} price={lifo_price} qty={lifo_qty} → skip")
            return

        # 2. ვირტუალური გაყიდვა — DEMO
        if engine.exchange is None:
            sell_price = current_price
        else:
            sell_result = engine.exchange.place_market_sell(exchange_sym, lifo_qty)
            sell_price  = float(sell_result.get("average") or sell_result.get("price") or current_price)

        proceeds     = sell_price * lifo_qty
        fee          = proceeds * 0.001   # 0.1% Binance fee
        net_proceeds = proceeds - fee

        # RISK CHECK — net_proceeds საკმარისია reinvest-ისთვის?
        from execution.dca_risk_manager import get_risk_manager as _get_rm
        _rot_risk_ok, _rot_risk_reason = _get_rm().can_l3_operation(net_proceeds)
        if not _rot_risk_ok:
            logger.warning(f"[L3_ROT] RISK_BLOCK | {sym} net_proceeds={net_proceeds:.2f} reason={_rot_risk_reason}")
            return

        realized_pnl = (sell_price - lifo_price) * lifo_qty - fee

        logger.warning(
            f"[L3_ROT] LIFO_SELL | {sym} "
            f"lifo_price={lifo_price:.4f} sell={sell_price:.4f} "
            f"qty={lifo_qty:.6f} pnl={realized_pnl:+.4f}"
        )

        # 3. reinvest @ current_price
        if engine.exchange is None:
            reinvest_price = current_price
            reinvest_qty   = net_proceeds / reinvest_price
        else:
            reinvest_result = engine.exchange.place_market_buy_by_quote(exchange_sym, net_proceeds)
            reinvest_price  = float(reinvest_result.get("average") or reinvest_result.get("price") or current_price)
            reinvest_qty    = float(reinvest_result.get("filled") or reinvest_result.get("amount") or (net_proceeds / reinvest_price))

        # 4. new_avg გამოთვლა — სწორი LIFO მათემატიკა
        # total position value-დან LIFO unit-ის ზუსტი value ამოვაკლოთ
        # (არა remaining_qty * avg_entry — ეს approximate-ია)
        remaining_qty   = total_qty - lifo_qty
        total_value     = total_qty * avg_entry
        remaining_value = total_value - lifo_qty * lifo_price  # ზუსტი: ვაკლებთ actual entry price

        new_qty   = remaining_qty + reinvest_qty
        new_value = remaining_value + reinvest_qty * reinvest_price
        new_avg   = round(new_value / new_qty, 8) if new_qty > 0 else avg_entry

        # TP — L3 zone (0.35%)
        new_tp = tp_sl_mgr.calculate_rotation_tp(new_avg)

        # total_quote update: LIFO unit-ის quote ამოვიღოთ, reinvest დავამატოთ
        new_total_quote = total_quote - lifo_quote + net_proceeds

        logger.warning(
            f"[L3_ROT] REINVEST | {sym} "
            f"@ {reinvest_price:.4f} qty={reinvest_qty:.6f} | "
            f"old_avg={avg_entry:.4f} → new_avg={new_avg:.4f} "
            f"new_tp={new_tp:.4f} (L3 zone 0.35%)"
        )

        # 5. DB განახლება
        update_dca_position_after_rotation(
            position_id=pos_id,
            new_avg_entry=new_avg,
            new_total_qty=new_qty,
            new_total_quote=new_total_quote,
            new_tp_price=new_tp,
            last_rotation_ts=time.time(),
            rotation_pnl=realized_pnl,
        )

        # rotation order-ი DB-ში
        add_dca_order(
            position_id=pos_id,
            symbol=sym,
            order_type="ROTATION_REINVEST",
            entry_price=reinvest_price,
            qty=reinvest_qty,
            quote_spent=net_proceeds,
            avg_entry_after=new_avg,
            tp_after=new_tp,
            sl_after=0.0,
            trigger_drawdown_pct=(avg_entry - current_price) / avg_entry * 100.0 if avg_entry > 0 else 0.0,
            exchange_order_id="",
        )

        try:
            log_event(
                "L3_ROTATION",
                f"sym={sym} lifo_price={lifo_price:.4f} sell={sell_price:.4f} "
                f"reinvest={reinvest_price:.4f} old_avg={avg_entry:.4f} "
                f"new_avg={new_avg:.4f} new_tp={new_tp:.4f} "
                f"pnl={realized_pnl:+.4f}"
            )
        except Exception:
            pass

        # Telegram
        try:
            from execution.telegram_notifier import send_telegram_message
            send_telegram_message(
                f"🔄 <b>L3 LIFO ROTATION</b>\n\n"
                f"🪙 <b>Symbol:</b> <code>{sym}</code>\n"
                f"💸 <b>LIFO sell:</b> <code>{lifo_price:.2f} → {sell_price:.2f}</code>\n"
                f"♻️ <b>Reinvest:</b> <code>{reinvest_price:.2f}</code>\n"
                f"📊 <b>avg:</b> <code>{avg_entry:.2f} → {new_avg:.2f}</code> ↓\n"
                f"🎯 <b>TP:</b> <code>{new_tp:.2f}</code> (L3: 0.35%)\n"
                f"💰 <b>Rotation PnL:</b> <code>{realized_pnl:+.4f} USDT</code>\n"
                f"🕒 <code>{__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</code>"
            )
        except Exception as _tg:
            logger.warning(f"[L3_ROT] TG_FAIL | err={_tg}")

    except Exception as e:
        logger.error(f"[L3_ROT] ROTATION_FAIL | {sym} err={e}")


def _start_bot_api_server() -> None:
    """
    Bot API Server — Dashboard-ისთვის DB data-ს აბრუნებს.
    GET /api/stats  → positions + trades + stats JSON
    GET /health     → liveness check

    ENV:
      BOT_API_ENABLED=true   ← ჩართვა/გამორთვა (default: true)
      BOT_API_PORT=5001      ← port (default: 5001)
    """
    if not os.getenv("BOT_API_ENABLED", "true").strip().lower() in ("1", "true", "yes"):
        return

    try:
        from flask import Flask as _Flask, jsonify as _jsonify
    except ImportError:
        logger.warning("[BOT_API] Flask not installed → API disabled")
        return

    import threading as _threading
    from datetime import datetime as _dt, timezone as _tz

    api_app = _Flask("bot_api")

    @api_app.route("/api/stats")
    def bot_api_stats():
        try:
            from execution.db.repository import (
                get_trade_stats,
                get_all_open_dca_positions,
                get_closed_trades,
            )
            stats     = get_trade_stats()
            positions = get_all_open_dca_positions()
            trades    = get_closed_trades()
            recent = sorted(
                [t for t in trades if t.get("outcome")],
                key=lambda x: str(x.get("closed_at", "")),
                reverse=True,
            )[:20]
            return _jsonify({
                "stats":         stats,
                "positions":     positions,
                "recent_trades": recent,
                "timestamp":     _dt.now(_tz.utc).isoformat(),
            })
        except Exception as e:
            logger.error(f"[BOT_API] stats error: {e}")
            return _jsonify({"error": str(e)}), 500

    @api_app.route("/health")
    def bot_api_health():
        return _jsonify({"status": "ok", "service": "GENIUS-DCA-Bot"})

    def _run():
        port = int(os.getenv("BOT_API_PORT", "5001"))
        logger.info(f"[BOT_API] Starting on port {port} → /api/stats")
        api_app.run(host="0.0.0.0", port=port, debug=False,
                    use_reloader=False, threaded=True)

    t = _threading.Thread(target=_run, daemon=True, name="bot_api")
    t.start()
    logger.info("[BOT_API] API server thread started")


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

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # E: HEARTBEAT — ყოველ N წამს Telegram-ზე "ბოტი ცოცხალია"
    # HEARTBEAT_INTERVAL_SECONDS=600 (default: 10 წუთი)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    heartbeat_every_s = int(os.getenv("HEARTBEAT_INTERVAL_SECONDS", "600"))
    last_heartbeat_ts = 0.0

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # C: DAILY LOSS LIMIT — დღიური ზარალის ჭერი
    # DAILY_MAX_LOSS_USDT=5.0 (default: $5 = 5% of $100)
    # თუ დღიური ზარალი > limit → PAUSE until next day
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    daily_max_loss = float(os.getenv("DAILY_MAX_LOSS_USDT", "5.0"))
    _daily_loss_date = ""    # რომელ დღეზე ვთვლით
    _daily_loss_total = 0.0  # დღის ზარალი

    init_db()
    _bootstrap_state_if_needed()

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # TP FIX — Take Profit ავტომატური გასწორება (memory-safe!)
    # Binance API არ სჭირდება — მხოლოდ DB-ს კითხულობს
    # L1-L2: TP=avg×1.0055  |  L3-L10: TP=avg×1.0065
    # 10s delay — kill_switch DB conflict-ის თავიდანაცილება
    # TP_FIX_ENABLED=true  ← ჩართვა/გამორთვა (default: true)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    if os.getenv("TP_FIX_ENABLED", "true").strip().lower() in ("1", "true", "yes"):
        try:
            import threading as _tp_thread
            import time as _tp_time

            def _run_tp_fix_delayed():
                _tp_time.sleep(10)
                try:
                    from execution.tp_fix import run_tp_fix
                    _r = run_tp_fix()
                    logger.info(
                        f"TP_FIX | checked={_r.get('checked',0)} "
                        f"fixed={_r.get('fixed',0)} "
                        f"skipped={_r.get('skipped',0)}"
                    )
                except Exception as _tpe2:
                    logger.warning(f"TP_FIX_FAIL | err={_tpe2}")

            _tp_thread.Thread(
                target=_run_tp_fix_delayed,
                daemon=True,
                name="tp_fix"
            ).start()
            logger.info("TP_FIX | scheduled in 10s (background thread)")
        except Exception as _tpe:
            logger.warning(f"TP_FIX_THREAD_FAIL | err={_tpe}")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # FIX #5: QTY SYNC — Binance vs DB qty სინქრონიზაცია
    # buy_qty bug-ის გამო DB qty > Binance qty → TP გაყიდვა ვერ ხდება
    # 20s delay — main loop DB init-ის შემდეგ გაეშვება (DB conflict თავიდანაცილება)
    # QTY_SYNC_ENABLED=true     ← ჩართვა/გამორთვა
    # QTY_SYNC_TOLERANCE=0.005  ← 0.5% სხვაობაზე გასწორება
    # QTY_SYNC_DELAY=20         ← delay წამებში (default: 20)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    if os.getenv("QTY_SYNC_ENABLED", "true").strip().lower() in ("1", "true", "yes"):
        try:
            import threading as _qty_thread

            def _run_qty_sync_delayed():
                import time as _t
                _delay = int(os.getenv("QTY_SYNC_DELAY", "20"))
                _t.sleep(_delay)
                try:
                    from execution.qty_sync import run_qty_sync
                    _r = run_qty_sync()
                    logger.info(
                        f"QTY_SYNC | checked={_r.get('checked',0)} "
                        f"fixed={_r.get('fixed',0)} "
                        f"skipped={_r.get('skipped',0)}"
                    )
                except Exception as _qe2:
                    logger.warning(f"QTY_SYNC_FAIL | err={_qe2}")

            _qty_thread.Thread(
                target=_run_qty_sync_delayed,
                daemon=True,
                name="qty_sync"
            ).start()
            logger.info("QTY_SYNC | scheduled in 20s (background thread)")
        except Exception as _qe:
            logger.warning(f"QTY_SYNC_THREAD_FAIL | err={_qe}")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # BOT API SERVER — Dashboard-ისთვის DB data /api/stats endpoint-ზე
    # BOT_API_ENABLED=true, BOT_API_PORT=5001
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    try:
        _start_bot_api_server()
    except Exception as _ae:
        logger.warning(f"BOT_API_START_FAIL | err={_ae}")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # LIVE DASHBOARD — background thread, port 8080
    # URL: https://your-render-url.onrender.com/dashboard
    # DASHBOARD_ENABLED=true ENV-ით ჩართვა/გამორთვა
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    if os.getenv("DASHBOARD_ENABLED", "true").strip().lower() in ("1", "true", "yes"):
        try:
            from execution.dashboard import start_dashboard
            _dash_port = int(os.getenv("DASHBOARD_PORT", "8080"))
            start_dashboard(port=_dash_port)
        except Exception as _de:
            logger.warning(f"DASHBOARD_START_FAIL | err={_de}")

    engine = ExecutionEngine()

    generate_once = _try_import_generator()

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # FIX C-2: MarketRegimeEngine — loop-გარეთ, ᲔᲠᲗᲘ instance სამუდამოდ
    # ძველი კოდი ყოველ ტიკზე ახალ instance-ს ქმნიდა → state იკარგებოდა
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    regime_engine = MarketRegimeEngine()

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # FIX C-1: inject_regime_engine — execution_engine-ს regime_engine
    # გადაეცემა, რათა TP/SL close-ზე in-memory SL counter სწორად reset-ს.
    # გარეშე: _regime_engine=None → notify_outcome() არასოდეს გამოიძახება
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    engine.inject_regime_engine(regime_engine)

    # DCA managers init
    # DCA MODE: DCA_ENABLED ENV-დან — default true (DCA ბოტია)
    _dca_enabled = os.getenv("DCA_ENABLED", "true").strip().lower() in ("1", "true", "yes")
    dca_mgr   = get_dca_manager()   if _dca_enabled else None
    tp_sl_mgr = get_tp_sl_manager() if _dca_enabled else None
    risk_mgr  = get_risk_manager()  if _dca_enabled else None
    if _dca_enabled:
        logger.info(f"DCA_ENABLED | max_add_ons={os.getenv('DCA_MAX_ADD_ONS', '5')} max_capital={os.getenv('DCA_MAX_CAPITAL_USDT', 'AUTO')}")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # SMART LONG + SHORT — Futures Engine init
    # FUTURES_ENABLED=true + FUTURES_MODE=DEMO → ვირტუალური SHORT
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    futures_engine = get_futures_engine()
    logger.info(
        f"FUTURES_ENGINE | enabled={futures_engine.enabled} "
        f"mode={futures_engine.mode} lev={futures_engine.leverage}x"
    )

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

            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # B: PRICE CACHE — ყოველ loop-ზე ერთხელ fetch, cache-დან კითხვა
            # 9 API call → 3 API call (rate limit risk ↓, სიჩქარე ↑)
            # _market_regime — default NEUTRAL (FUTURES block-ის წინ)
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            _market_regime: str = "NEUTRAL"  # ← safe default, FUTURES loop-ში განახლდება
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            _price_cache: dict = {}
            _symbols_to_cache = [s.strip() for s in os.getenv(
                "BOT_SYMBOLS", "BTC/USDT,BNB/USDT,ETH/USDT"
            ).split(",") if s.strip()]
            for _sym in _symbols_to_cache:
                try:
                    if engine.exchange is not None:
                        _price_cache[_sym] = float(
                            engine.exchange.fetch_last_price(_sym) or 0.0
                        )
                    else:
                        # DEMO: public API
                        _t = engine.price_feed.fetch_ticker(_sym)
                        _price_cache[_sym] = float(_t.get("last") or 0.0)
                except Exception as _pe:
                    logger.warning(f"PRICE_CACHE_FAIL | {_sym} err={_pe}")
                    _price_cache[_sym] = 0.0

            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # C: DAILY LOSS — დღის reset შემოწმება
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            _today = _now_dt().date().isoformat()
            if _today != _daily_loss_date:
                _daily_loss_date = _today
                _daily_loss_total = 0.0
                logger.info(f"DAILY_LOSS_RESET | date={_today} limit={daily_max_loss}")

            # L3_ROTATION_LOSS events DB-დან — daily loss tracking
            # L3 rotation realized loss-ები + force_close-ები
            try:
                from execution.db.repository import _fetchall
                import re as _re_loss
                _loss_rows = _fetchall(
                    "SELECT message FROM audit_log "
                    "WHERE event_type IN ('L3_ROTATION','DCA_FORCE_CLOSE') "
                    "AND created_at >= date('now') ORDER BY id DESC LIMIT 100"
                )
                _loss_total = 0.0
                for _row in (_loss_rows or []):
                    try:
                        _m = _re_loss.search(r'pnl=([+-]?\d+\.\d+)', str(_row[0]))
                        if _m:
                            _loss_total += float(_m.group(1))
                    except Exception:
                        pass
                if _loss_total < 0:
                    _daily_loss_total = _loss_total
            except Exception:
                pass

            # DAILY LOSS LIMIT — limit-ს გადაცდა → skip ვაჭრობა
            if daily_max_loss > 0 and _daily_loss_total <= -daily_max_loss:
                logger.warning(
                    f"DAILY_LOSS_LIMIT | loss={_daily_loss_total:.4f} >= limit={daily_max_loss} → skip"
                )
                try:
                    from execution.telegram_notifier import send_telegram_message
                    send_telegram_message(
                        f"⛔ <b>DAILY LOSS LIMIT</b>\n\n"
                        f"📉 დღის ზარალი: <code>{_daily_loss_total:.4f} USDT</code>\n"
                        f"🛡 Limit: <code>{daily_max_loss} USDT</code>\n"
                        f"⏸ ვაჭრობა შეჩერებულია დღეს\n"
                        f"🕒 <code>{_now_dt().strftime('%Y-%m-%d %H:%M')}</code>"
                    )
                except Exception:
                    pass
                time.sleep(sleep_s)
                continue

            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # FIX #14: TP_FIX — ყოველ loop-ზე TP სისწორის შემოწმება
            # ADD-ON-ის შემდეგ TP შეიძლება avg-ზე დაბლა დარჩეს → გასწორება
            # memory-safe: მხოლოდ DB-ს კითხულობს, Binance API არ სჭირდება
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            if _dca_enabled:
                try:
                    from execution.tp_fix import run_tp_fix
                    _tp_r = run_tp_fix()
                    if _tp_r.get("fixed", 0) > 0:
                        logger.warning(
                            f"TP_FIX_LOOP | fixed={_tp_r['fixed']} checked={_tp_r['checked']}"
                        )
                except Exception as _tfe:
                    logger.warning(f"TP_FIX_LOOP_WARN | err={_tfe}")

            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # FUTURES — TP/SL check + FC (ყველა regime-ზე)
            # BEAR regime SHORT (ძველი) — წაშლილია, ახალი Independent SHORT-ით
            # ჩანაცვლდა (price-level trigger, საკუთარი TP/FC lifecycle)
            # DCA hedge SHORT და Independent SHORT TP/FC — check_tp_sl() ფარავს
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            try:
                from execution.signal_generator import _detect_market_regime_24h
                _prev_regime = _market_regime
                _market_regime = _detect_market_regime_24h()
                futures_engine.check_tp_sl()

                # REGIME LOG — audit_log-ში ჩაწერა regime ცვლილებისას
                # ან ყოველ 10 loop-ზე (debug-ისთვის)
                if _market_regime != _prev_regime:
                    try:
                        log_event(
                            "MARKET_REGIME_CHANGE",
                            f"regime={_market_regime} prev={_prev_regime}"
                        )
                        logger.info(
                            f"MARKET_REGIME_CHANGE | {_prev_regime} → {_market_regime}"
                        )
                    except Exception:
                        pass

            except Exception as _fe:
                logger.warning(f"FUTURES_LOOP_WARN | err={_fe}")

            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # DCA LOOP — add-on check + TP/SL + breakeven + force close
            # BEAR MODE: ADD-ON ბლოკილია (_market_regime გადაეცემა)
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            if _dca_enabled:
                try:
                    _run_dca_loop(engine, dca_mgr, tp_sl_mgr, risk_mgr,
                                  market_regime=_market_regime,
                                  futures_engine=futures_engine)
                except Exception as e:
                    logger.warning(f"DCA_LOOP_WARN | err={e}")

            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # DCA HEDGE SHORT — ADD-ON + L3 EXCHANGE check
            # LONG-ის DCA ADD-ON-ების სიმეტრიული mirror SHORT-ში:
            #   ADD-ON: BTC bounce ↑ → avg_short ↑ → TP_short ↑
            #   L3:     ADD-ONs exhausted + ↑ → EXCHANGE (close+reopen ↑)
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            if _dca_enabled and futures_engine.enabled:
                try:
                    futures_engine.check_dca_hedge_addons()
                    futures_engine.check_dca_hedge_l3()
                except Exception as _he:
                    logger.warning(f"HEDGE_CHECK_WARN | err={_he}")

            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # FIX #19: INDEPENDENT SHORT DCA — სარკე სტრატეგია
            # LONG L1 გახსნის შემდეგ: SHORT იხსნება -1.6%-ზე (L2-L3 შუა)
            # ADD-ONs ვარდნაზე: -1.0%, -2.2%, -3.5% from SHORT L1
            # TP: avg × 0.9945 → სავალდებულო დახურვა
            # FC: 10 days / +15% drawdown → ღია ტრეიდი არ რჩება
            # SHORT_DCA_ENABLED=true ENV-ით ჩართვა
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            if _dca_enabled and futures_engine.enabled and futures_engine.short_dca_enabled:
                try:
                    futures_engine.check_independent_short_open()
                    futures_engine.check_independent_short_addons()
                except Exception as _se:
                    logger.warning(f"SHORT_DCA_LOOP_WARN | err={_se}")

            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # GENIUS MIRROR ENGINE — bilateral DCA (FIX-S7)
            # L2/L3 midpoint (-8.59% L1-დან) → SHORT იხსნება
            # ADD-ONs DOWN: ვარდნა გრძელდება → avg↓ TP↓
            # ADD-ONs UP:   bounce → avg_short↑ TP ახლოვდება
            # TP:  0.55% — scalp hybrid
            # FC:  drawdown only (-15%) — time FC გაუქმებულია
            # MIRROR_ENGINE_ENABLED=true ENV-ით ჩართვა
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            if _dca_enabled and futures_engine.enabled and futures_engine.mirror_enabled:
                try:
                    futures_engine.check_mirror_tp_sl()
                    futures_engine.check_mirror_engine_open()
                    futures_engine.check_mirror_addons()
                except Exception as _me:
                    logger.warning(f"MIRROR_ENGINE_LOOP_WARN | err={_me}")

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
                    # FIX C-3: SELL signal — regime check bypass
                    # SELL (TREND_REVERSAL / PROTECTIVE_SELL) ყოველთვის
                    # სრულდება, SKIP_TRADING-ი მას ვერ ბლოკავს.
                    # ძველი კოდი SELL-საც ჩერდებოდა SIDEWAYS-ზე!
                    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                    if verdict == "SELL":
                        source = sig.get("meta", {}).get("source", "UNKNOWN")
                        if source == "PROTECTIVE_SELL":
                            # crash guard — ATR EXTREME + KILL risk → გაყიდვა
                            logger.warning(
                                f"[AUTO] PROTECTIVE_SELL → executing | "
                                f"id={signal_id} source={source}"
                            )
                            engine.execute_signal(sig)
                        else:
                            # TREND_REVERSAL / RSI_OVERBOUGHT — DCA-ში ბლოკი
                            # DCA TP-ს ელოდება, არ გაყიდის ვარდნაზე
                            logger.info(
                                f"[AUTO] SELL blocked (DCA holds) | "
                                f"id={signal_id} source={source}"
                            )

                    elif verdict == "TRADE":
                        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                        # PAIRS_ADDON — BTC/ETH კორელაციური ADD-ON
                        # signal_type="PAIRS_ADDON" → ADD-ON ლოგიკა:
                        #   ახალი position კი არა, არსებული ETH position-ის
                        #   avg-ს ვამცირებთ $12 ADD-ON-ით.
                        #   dca_position_manager-ი ჩვეულებრივ ამუშავებს.
                        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                        if sig.get("signal_type") == "PAIRS_ADDON":
                            _pa_sym = str((sig.get("execution") or {}).get("symbol", ""))
                            logger.info(
                                f"[PAIRS_ADDON] trigger | lag={_pa_sym} "
                                f"z={sig.get('meta', {}).get('z_score', '?')} "
                                f"lead_move={sig.get('meta', {}).get('lead_move_pct', '?')}%"
                            )
                            if engine.exchange is None and _dca_enabled and _pa_sym:
                                # DEMO: ADD-ON ვირტუალური შესრულება
                                try:
                                    from execution.db.repository import (
                                        get_open_dca_position_for_symbol,
                                        update_dca_position_after_addon,
                                        add_dca_order,
                                    )
                                    from execution.dca_position_manager import recalculate_average
                                    _pa_pos = get_open_dca_position_for_symbol(_pa_sym)
                                    if _pa_pos:
                                        _pa_price = _price_cache.get(_pa_sym, 0.0)
                                        if _pa_price <= 0:
                                            _pa_t = engine.price_feed.fetch_ticker(_pa_sym)
                                            _pa_price = float(_pa_t.get("last") or 0.0)
                                        if _pa_price > 0:
                                            _pa_quote  = float(os.getenv("BOT_QUOTE_PER_TRADE", "12.0"))
                                            _pa_qty    = _pa_quote / _pa_price
                                            _pa_avg_old = float(_pa_pos["avg_entry_price"] or 0)
                                            _pa_tot_qty = float(_pa_pos["total_qty"] or 0)
                                            _pa_tot_q   = float(_pa_pos["total_quote_spent"] or 0)
                                            _pa_addons  = int(_pa_pos["add_on_count"] or 0)

                                            _pa_avg_res = recalculate_average(
                                                _pa_tot_qty, _pa_avg_old, _pa_qty, _pa_price
                                            )
                                            _pa_new_avg = _pa_avg_res["avg_entry_price"]
                                            _pa_new_qty = _pa_avg_res["total_qty"]
                                            _pa_tp_pct  = float(os.getenv("DCA_TP_PCT", "0.55"))
                                            _pa_new_tp  = round(_pa_new_avg * (1.0 + _pa_tp_pct / 100.0), 6)

                                            update_dca_position_after_addon(
                                                _pa_pos["id"],
                                                new_avg_entry=_pa_new_avg,
                                                new_total_qty=_pa_new_qty,
                                                new_total_quote=_pa_tot_q + _pa_quote,
                                                new_add_on_count=_pa_addons + 1,
                                                new_tp_price=_pa_new_tp,
                                                new_sl_price=0.0,
                                                last_add_on_ts=time.time(),
                                                last_addon_price=_pa_price,
                                            )
                                            add_dca_order(
                                                position_id=_pa_pos["id"],
                                                symbol=_pa_sym,
                                                order_type="PAIRS_ADDON",
                                                entry_price=_pa_price,
                                                qty=_pa_qty,
                                                quote_spent=_pa_quote,
                                                avg_entry_after=_pa_new_avg,
                                                tp_after=_pa_new_tp,
                                                sl_after=0.0,
                                                trigger_drawdown_pct=0.0,
                                                exchange_order_id="",
                                            )
                                            logger.info(
                                                f"[PAIRS_ADDON] DEMO_EXECUTED | {_pa_sym} "
                                                f"price={_pa_price:.4f} new_avg={_pa_new_avg:.4f} "
                                                f"new_tp={_pa_new_tp:.4f} addon#{_pa_addons+1}"
                                            )
                                    else:
                                        logger.info(
                                            f"[PAIRS_ADDON] NO_OPEN_POS | {_pa_sym} → skip"
                                        )
                                except Exception as _pa_err:
                                    logger.warning(f"[PAIRS_ADDON] EXEC_FAIL | err={_pa_err}")
                            # PAIRS_ADDON → signal handled, არ გავგრძელდეთ ჩვეულებრივ TRADE path-ზე
                            pass

                        else:
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

                          # DCA MODE: regime block გათიშულია — ვაჭრობა ყველა რეჟიმში
                          # if regime in ("BEAR", "VOLATILE", "SIDEWAYS"): → disabled
                          logger.info(f"[AUTO] regime={regime} trend={trend:.3f} atr={atr_pct:.3f} → DCA mode, no block")

                          logger.info(
                              f"[AUTO] Regime={regime} trend={trend:.3f} "
                              f"atr_pct={atr_pct:.3f} symbol={symbol} "
                              f"TP={sig.get('adaptive', {}).get('TP_PCT', 'n/a')}% "
                              f"SL={sig.get('adaptive', {}).get('SL_PCT', 'n/a')}% "
                              f"mtf={sig.get('meta', {}).get('mtf_alignment', 'N/A')} "
                              f"| id={signal_id}"
                          )

                          engine.execute_signal(sig)

                          # ── DEMO: DCA position გახსნა ──────────────────
                          # FIX: EXEC_REJECT check — თუ engine-მა reject-ი გააკეთა
                          # (ABOVE_MIN_OPEN_TRADES, MAX_OPEN_TRADES, და სხვა),
                          # mark_signal_id_executed() უკვე გამოიძახა → skip DEMO open.
                          # signal_id_already_executed() → True = rejected ან executed
                          if engine.exchange is None and _dca_enabled:
                              try:
                                  from execution.db.repository import (
                                      open_dca_position, add_dca_order, open_trade,
                                      get_open_dca_position_for_symbol,
                                      get_executed_signal_action,
                                      get_all_open_trades,
                                      get_all_open_dca_positions,
                                  )
                                  _sym = str((sig.get("execution") or {}).get("symbol", "BTC/USDT"))

                                  # ── REJECT CHECK ──────────────────────────────────────────
                                  # FIX: signal_id_already_executed() → True for BOTH:
                                  #   "TRADE_DEMO"  (ნორმალური შესრულება) — DCA MUST open
                                  #   "REJECT_*"    (ნამდვილი reject)     — DCA must NOT open
                                  # გამოსწორება: action სტრიქონით ვანსხვავებთ.
                                  # TRADE_DEMO engine-ში simulate_market_entry() = no-op,
                                  # ამიტომ DCA position-ი main.py-ში იხსნება.
                                  _exec_action = get_executed_signal_action(signal_id)
                                  _REAL_REJECTS = {
                                      "REJECT_MAX_OPEN_TRADES",
                                      "REJECT_ABOVE_MIN_OPEN",
                                      "REJECT_MAX_POSITIONS",
                                      "REJECT_ACTIVE_OCO",
                                      "REJECT_OPEN_TRADE_RACE",
                                      "REJECT_MIN_NOTIONAL",
                                      "REJECT_PORTFOLIO_EXPOSURE",
                                  }
                                  _is_real_reject = _exec_action in _REAL_REJECTS

                                  if _is_real_reject:
                                      logger.info(
                                          f"[DEMO] SKIP_REJECTED | {_sym} "
                                          f"action={_exec_action} id={signal_id}"
                                      )

                                  # FIX: count L1-only positions — _L2/_L3 CASCADE layers
                                  # არ ჩაითვლება (execution_engine-ის ლოგიკა მირორდება).
                                  # ძველი: len(get_all_open_dca_positions()) → _L2/_L3 ჩათვლით
                                  # = double-counting → DEMO-ში ახალი positions ვეღარ იხსნებოდა.
                                  import re as _re_main_cnt
                                  _all_dca_cnt = get_all_open_dca_positions() or []
                                  _l1_open_cnt = sum(
                                      1 for _p in _all_dca_cnt
                                      if not _re_main_cnt.search(r'_L\d+$', str(_p.get("symbol", "")))
                                  )
                                  _max_open_cnt = int(os.getenv("MAX_OPEN_TRADES", "6"))
                                  _at_max = _l1_open_cnt >= _max_open_cnt
                                  if _at_max:
                                      logger.info(
                                          f"[DEMO] SKIP_MAX_OPEN | {_sym} "
                                          f"l1_open={_l1_open_cnt} >= MAX_OPEN_TRADES={_max_open_cnt}"
                                      )

                                  # ALLOW_DCA_DUPLICATE: ერთი symbol-ი 2 L1 position
                                  # _existing = get_open_dca_position_for_symbol(_sym) → True/False
                                  # DUPLICATE mode: count vs MAX_DCA_PER_SYMBOL limit
                                  _allow_dup = os.getenv("ALLOW_DCA_DUPLICATE", "false").strip().lower() in ("1", "true", "yes")
                                  _max_dca_per_sym = int(os.getenv("MAX_DCA_PER_SYMBOL", "1"))
                                  try:
                                      from execution.db.repository import count_open_dca_positions_for_symbol
                                      _sym_dca_count = count_open_dca_positions_for_symbol(_sym)
                                  except Exception:
                                      _sym_dca_count = 1 if get_open_dca_position_for_symbol(_sym) else 0

                                  if _allow_dup:
                                      # duplicate mode: block only if count >= MAX_DCA_PER_SYMBOL
                                      _existing_blocked = _sym_dca_count >= _max_dca_per_sym
                                  else:
                                      # normal mode: block if ANY position open for symbol
                                      _existing_blocked = _sym_dca_count > 0

                                  # ── per-symbol ENTRY_COOLDOWN ────────────────────
                                  # ENV: ENTRY_COOLDOWN_SECONDS=600 (0=disabled)
                                  # იმავე symbol-ზე ახალი L1 position-ი მხოლოდ
                                  # cooldown-ის გასვლის შემდეგ გაიხსნება.
                                  # სხვა symbol-ები არ ბლოკდება → stagger effect.
                                  _entry_cd = int(os.getenv("ENTRY_COOLDOWN_SECONDS", "0"))
                                  if _entry_cd > 0 and not _existing_blocked:
                                      try:
                                          from execution.db.repository import get_last_entry_ts_for_symbol
                                          _last_e_ts = get_last_entry_ts_for_symbol(_sym) or 0.0
                                          _cd_elapsed = time.time() - _last_e_ts
                                          if _cd_elapsed < _entry_cd:
                                              logger.info(
                                                  f"[DEMO] ENTRY_COOLDOWN | {_sym} "
                                                  f"remaining={int(_entry_cd - _cd_elapsed)}s"
                                              )
                                              _existing_blocked = True
                                      except Exception:
                                          pass  # DB function არ არის → skip cooldown

                                  _rejected = _is_real_reject or _at_max or _existing_blocked
                                  if not _rejected:
                                      _quote = float(os.getenv("BOT_QUOTE_PER_TRADE", "12.0"))
                                      _price = _price_cache.get(_sym, 0.0)
                                      if _price <= 0:
                                          _t = engine.price_feed.fetch_ticker(_sym)
                                          _price = float(_t.get("last") or 0.0)
                                      if _price > 0:
                                          _qty = _quote / _price
                                          _tp_pct = float(os.getenv("DCA_TP_PCT", "0.55"))
                                          _tp = round(_price * (1.0 + _tp_pct / 100.0), 6)
                                          _quote_pt  = float(os.getenv("BOT_QUOTE_PER_TRADE", "12.0"))
                                          _sizes_str = os.getenv("DCA_ADDON_SIZES", "12,15,18,15,10")
                                          try:
                                              _addon_sum = sum(float(x.strip()) for x in _sizes_str.split(",") if x.strip())
                                          except Exception:
                                              _addon_sum = 70.0
                                          _auto_cap = _quote_pt + _addon_sum
                                          _max_cap  = float(os.getenv("DCA_MAX_CAPITAL_USDT") or _auto_cap)
                                          _pos_id = open_dca_position(
                                              symbol=_sym,
                                              initial_entry_price=_price,
                                              initial_qty=_qty,
                                              initial_quote_spent=_quote,
                                              tp_price=_tp,
                                              sl_price=0.0,
                                              tp_pct=_tp_pct,
                                              sl_pct=999.0,
                                              max_add_ons=int(os.getenv("DCA_MAX_ADD_ONS", "5")),
                                              max_capital=_max_cap,
                                              max_drawdown_pct=999.0,
                                          )
                                          # INITIAL order — dca_orders ცხრილი
                                          # საჭიროა get_lifo_unit()-ისთვის (L3 rotation)
                                          add_dca_order(
                                              position_id=_pos_id,
                                              symbol=_sym,
                                              order_type="INITIAL",
                                              entry_price=_price,
                                              qty=_qty,
                                              quote_spent=_quote,
                                              avg_entry_after=_price,
                                              tp_after=_tp,
                                              sl_after=0.0,
                                              trigger_drawdown_pct=0.0,
                                              exchange_order_id=signal_id,
                                          )
                                          open_trade(
                                              signal_id=signal_id,
                                              symbol=_sym,
                                              qty=_qty,
                                              quote_in=_quote,
                                              entry_price=_price,
                                          )
                                          logger.info(
                                              f"[DEMO] DCA_OPENED | {_sym} "
                                              f"price={_price:.4f} qty={_qty:.6f} "
                                              f"tp={_tp:.4f} quote={_quote}"
                                          )
                                          # ── TELEGRAM: 🚀 NEW SIGNAL OPENED ─────────────
                                          # FIX: DEMO mode-ში execution_engine `return`-ს
                                          # აკეთებს notify_signal_created-მდე (LIVE path).
                                          # DCA position გახსნის შემდეგ TG notification ხდება.
                                          try:
                                              from execution.telegram_notifier import notify_signal_created
                                              notify_signal_created(
                                                  symbol=_sym,
                                                  entry_price=_price,
                                                  quote_amount=_quote,
                                                  tp_price=_tp,
                                                  sl_price=0.0,
                                                  verdict=str(sig.get("final_verdict", "BUY")),
                                                  mode=os.getenv("MODE", "DEMO"),
                                              )
                                          except Exception as _tg_new:
                                              logger.warning(f"[DEMO] TG_NEW_SIGNAL_FAIL | {_sym} err={_tg_new}")
                              except Exception as _de:
                                  logger.warning(f"[DEMO] DCA_OPEN_FAIL | err={_de}")

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

            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # E: HEARTBEAT — Smart Schedule (Asia/Tbilisi):
            # 09:00-03:00 → ყოველ 1 საათში (3600s)
            # 03:00-09:00 → გაჩუმება 😴 (ძილის საათები)
            # 23:57-23:59 → Daily Summary-სთან ერთად (always)
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            try:
                _hb_now    = _now_dt()
                _hb_hour   = _hb_now.hour
                _hb_minute = _hb_now.minute

                # 03:00-09:00 → გაჩუმება (ძილი)
                # 03:00 ≤ hour < 09:00 → silent
                _hb_silent = (3 <= _hb_hour < 9)

                # 09:00-03:00 → ყოველ 1 საათში
                _hb_day_ok = not _hb_silent and (now - last_heartbeat_ts) >= 3600

                # 23:57-23:59 → Daily Summary-სთან ერთად (loop window fix)
                _hb_midnight_ok = (_hb_hour == 23 and _hb_minute >= 57)

                if _hb_day_ok or _hb_midnight_ok:
                    from execution.db.repository import get_all_open_dca_positions, get_trade_stats
                    from execution.telegram_notifier import notify_heartbeat
                    import resource as _res
                    _hb_positions = get_all_open_dca_positions()
                    _hb_capital = sum(
                        float(p.get("total_quote_spent", 0)) for p in _hb_positions
                    )
                    _hb_stats = get_trade_stats()
                    _hb_pnl_today = float(_hb_stats.get("pnl_quote_sum", 0.0))
                    _hb_mem = _res.getrusage(_res.RUSAGE_SELF).ru_maxrss / 1024
                    notify_heartbeat(
                        open_count=len(_hb_positions),
                        open_capital=_hb_capital,
                        prices=_price_cache,
                        memory_mb=_hb_mem,
                        pnl_today=_hb_pnl_today,
                        positions=_hb_positions,
                    )
                    last_heartbeat_ts = now
            except Exception as _hbe:
                logger.warning(f"HEARTBEAT_FAIL | err={_hbe}")

            try:
                now_local = _now_dt()
                today_str = now_local.date().isoformat()

                # FIX: LOOP_SLEEP=120s > 23:59 window=60s → 50% miss rate!
                # გაფართოება: 57,58,59 → 180s window > 120s loop → 100% guaranteed
                # last_daily_summary_date guard → double-send შეუძლებელია
                if (
                    now_local.hour == 23
                    and now_local.minute in (57, 58, 59)
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
