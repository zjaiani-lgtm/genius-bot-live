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
# FIX #3 — CASCADE symbol suffix regex
#   გამოსწორება: re.sub(r'_L\d+$', '') — ყველა suffix იფარება
#
# FIX #4 — ADD-ON exchange_sym
#   გამოსწორება: exchange_sym (suffix გარეშე) Binance-ისთვის
#
# FIX #9 — LAYER2 Cooldown 180s (2026-04-12)
#   პრობლემა: BTC/ETH/BNB ერთდროულად L2 → $36 ერთ წამში!
#   გამოსწორება: _LAST_L2_TS + LAYER2_COOLDOWN_SECONDS=180
#
# FIX #10 — CASCADE net_proceeds < $10 → skip (2026-04-12)
#   პრობლემა: max(net_proceeds, 10.0) → გარე ფულის დამატება
#   გამოსწორება: net_proceeds < $10 → skip
#
# FIX #11 — Layer პოზიციებზე ADD-ON skip (2026-04-12)
#   პრობლემა: BTC/USDT_L2-ზე ADD-ON → CASCADE conflict
#   გამოსწორება: is_layer2=True → SKIP_ADDON → continue
#
# FIX #13 — CASCADE buy_quote fixed $12 (2026-04-12)
#   პრობლემა: net_proceeds ცვალებადია → TP < avg → არასოდეს გაიყიდება
#   გამოსწორება: buy_quote = BOT_QUOTE_PER_TRADE (fixed $12)
#   ადგილი: _check_cascade_exchange() ახალი Layer გახსნა
#
# FIX #14 — TP_FIX ყოველ loop-ზე (2026-04-12)
#   პრობლემა: CASCADE-ის შემდეგ TP < avg → პოზიცია არასოდეს იყიდება
#   გამოსწორება: run_tp_fix() ყოველ 120s loop-ზე
#   ადგილი: main loop, DCA loop-მდე
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
from execution.dca_tp_sl_manager import get_tp_sl_manager, DCATpSlManager
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
                  market_regime: str = "NEUTRAL") -> None:
    """
    DCA monitoring loop — ყოველ main loop iteration-ზე გამოიძახება.

    შეამოწმებს:
      1. TP hit → close position
      2. Breakeven → SL გადაადგილება
      3. Force close → max drawdown ან max add-ons + SL
      4. SL confirmed → close position
      5. Add-on → drawdown trigger + recovery signals
         BEAR MODE: ADD-ON BLOCKED (SHORT-ს ეწინააღმდეგება!)
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

        # FIX #2 + #4: exchange_sym — ყველა Layer suffix ამოვიღოთ
        # DB-ში "BTC/USDT_L2", "BTC/USDT_L3"... ინახება
        # Binance-ს მხოლოდ "BTC/USDT" სჭირდება
        import re as _re_sym
        exchange_sym = _re_sym.sub(r'_L\d+$', '', sym)
        is_layer2 = sym != exchange_sym  # True თუ რაიმე suffix აქვს

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

                    # ── FIX #2: trades ცხრილის დახურვა ────────────────
                    # sym პირველი (DB-ში "BTC/USDT_L2"), exchange_sym fallback
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

                    close_dca_position(pos_id, exit_price, total_qty, pnl_quote, pnl_pct, "FORCE_CLOSE")

                    # ── FIX #2: trades ცხრილის დახურვა ────────────────
                    # sym პირველი (DB-ში "BTC/USDT_L2"), exchange_sym fallback
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
                    from execution.db.repository import get_trade_stats
                    stats = get_trade_stats()
                    notify_dca_closed(
                        sym, avg_entry, exit_price, total_qty, total_quote,
                        pnl_quote, pnl_pct, "FORCE_CLOSE", add_on_count, stats
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

                        # ── FIX #2: trades ცხრილის დახურვა ────────────
                        # sym პირველი (DB-ში "BTC/USDT_L2"), exchange_sym fallback
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
            # FIX #11: Layer პოზიციებზე ADD-ON არ უნდა მოხდეს!
            # BTC/USDT_L2, ETH/USDT_L2, _L3... → CASCADE მართავს
            # DCA ADD-ON მხოლოდ L1 (base) პოზიციებზე!
            if is_layer2:
                logger.debug(f"[DCA] SKIP_ADDON | {sym} is Layer position → CASCADE manages")
                continue

            # BEAR MODE: ADD-ON BLOCKED
            # SHORT-ი ბაზრის ვარდნაზე ფსონობს — ADD-ON საპირისპიროა!
            if market_regime == "BEAR":
                logger.info(f"[DCA] ADDON_BEAR_BLOCK | {sym} BEAR market → ADD-ON blocked")
                continue
            all_positions = get_all_open_dca_positions()
            addon_ok, addon_reason = dca_mgr.should_add_on(pos, current_price, ohlcv)

            if not addon_ok:
                logger.debug(f"[DCA] NO_ADDON | {sym} reason={addon_reason}")
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

                tp_sl = tp_sl_mgr.calculate(new_avg)
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
                )

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


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FIX #9 REAL: LAYER2 Cooldown — global timestamp (CHANGELOG-ში
# წერია "გამოსწორება: _LAST_L2_TS", მაგრამ ცვლადი არ არსებობდა).
# _LAST_L2_TS: ბოლო ნებისმიერი Layer2 გახსნის unix timestamp.
# LAYER2_COOLDOWN_SECONDS=180 → 3 წუთი BTC/ETH/BNB-ს შორის.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
_LAST_L2_TS: float = 0.0


def _check_and_open_layer2(engine, tp_sl_mgr) -> None:
    """
    Layer 2 — Crash Detection & Parallel Trading.

    ლოგიკა:
      1. თითო symbol-ისთვის 24h HIGH ამოიღე
      2. თუ current_price <= HIGH × (1 - LAYER2_DROP_PCT/100) → crash!
      3. Layer 2 პოზიცია უკვე ღიაა? → გამოტოვე
      4. USDT ბალანსი საკმარისია? → გახსენი Layer 2

    ENV:
      LAYER2_DROP_PCT=5.0      ← HIGH-დან რამდენი % ვარდნაზე გაიხსნოს
      LAYER2_ENABLED=true      ← ჩართვა/გამორთვა
      LAYER2_QUOTE=10.0        ← Layer 2-ის trade ზომა
      LAYER2_SYMBOLS=BTC/USDT,BNB/USDT,ETH/USDT
    """
    import os

    # Layer 2 ჩართულია?
    if not os.getenv("LAYER2_ENABLED", "true").strip().lower() in ("1", "true", "yes"):
        return

    # FIX #9 REAL: cooldown — BTC/ETH/BNB ერთდროულად trigger → $36 ერთ loop-ში
    global _LAST_L2_TS
    import time as _l2_time
    _l2_cooldown = int(os.getenv("LAYER2_COOLDOWN_SECONDS", "180"))
    _l2_elapsed  = _l2_time.time() - _LAST_L2_TS
    if _l2_elapsed < _l2_cooldown:
        logger.debug(
            f"[LAYER2] GLOBAL_COOLDOWN | remaining={int(_l2_cooldown - _l2_elapsed)}s → skip"
        )
        return

    from execution.db.repository import (
        get_open_dca_position_for_symbol,
        open_dca_position,
        add_dca_order,
        open_trade,
        log_event,
    )

    drop_pct    = float(os.getenv("LAYER2_DROP_PCT",  "5.0"))
    quote       = float(os.getenv("LAYER2_QUOTE",     "10.0"))
    symbols_raw = os.getenv("LAYER2_SYMBOLS", "BTC/USDT,BNB/USDT,ETH/USDT")
    symbols     = [s.strip() for s in symbols_raw.split(",") if s.strip()]
    tp_pct      = float(os.getenv("DCA_TP_PCT", "0.55"))
    buffer      = float(os.getenv("SMART_ADDON_BUFFER", "5.0"))

    # USDT ბალანსი — DEMO: ვირტუალური $120
    try:
        if engine.exchange is not None:
            free_usdt = float(engine.exchange.fetch_balance_free("USDT") or 0.0)
        else:
            free_usdt = float(os.getenv("DEMO_INITIAL_BALANCE", "120.0"))
    except Exception as _e:
        logger.warning(f"[LAYER2] balance_fetch_fail | err={_e}")
        free_usdt = float(os.getenv("DEMO_INITIAL_BALANCE", "120.0"))

    for sym in symbols:
        exchange_sym = sym  # Layer2: symbols სუფთაა (_L2 suffix არ აქვს)
        try:
            # current price — DEMO: price_feed
            if engine.exchange is not None:
                current_price = float(engine.exchange.fetch_last_price(exchange_sym) or 0.0)
            else:
                _t = engine.price_feed.fetch_ticker(exchange_sym)
                current_price = float(_t.get("last") or 0.0)
            if current_price <= 0:
                continue

            # 24h HIGH — ohlcv 1d candle
            try:
                ticker = engine.price_feed.fetch_ticker(sym)
                high_24h = float(ticker.get("high") or ticker.get("info", {}).get("highPrice") or 0.0)
            except Exception:
                high_24h = 0.0

            if high_24h <= 0:
                logger.debug(f"[LAYER2] NO_HIGH | {sym} → skip")
                continue

            drop_from_high = (high_24h - current_price) / high_24h * 100.0

            logger.info(
                f"[LAYER2] CHECK | {sym} price={current_price:.4f} "
                f"high24h={high_24h:.4f} drop={drop_from_high:.2f}% "
                f"trigger={drop_pct:.1f}%"
            )

            # crash trigger?
            if drop_from_high < drop_pct:
                logger.debug(f"[LAYER2] NO_CRASH | {sym} drop={drop_from_high:.2f}% < {drop_pct:.1f}%")
                continue

            # Layer 2 უკვე ღიაა ამ symbol-ზე?
            # Layer 2 პოზიციები tag-ით განვასხვავებთ: symbol = "BTC/USDT_L2"
            sym_l2 = f"{sym}_L2"
            existing_l2 = get_open_dca_position_for_symbol(sym_l2)
            if existing_l2:
                logger.debug(f"[LAYER2] ALREADY_OPEN | {sym_l2}")
                continue

            # ბალანსი საკმარისია?
            required = quote + buffer
            if free_usdt < required:
                logger.warning(
                    f"[LAYER2] INSUFFICIENT_BALANCE | {sym} "
                    f"free={free_usdt:.2f} < required={required:.2f}"
                )
                continue

            # Layer 2 გახსნა!
            logger.warning(
                f"[LAYER2] CRASH_DETECTED | {sym} "
                f"drop={drop_from_high:.2f}% >= {drop_pct:.1f}% → opening Layer 2"
            )

            # ყიდვა — DEMO: ვირტუალური
            if engine.exchange is None:
                buy_price = current_price
                buy_qty   = quote / buy_price
                buy = {"average": buy_price, "price": buy_price, "filled": buy_qty}
            else:
                buy = engine.exchange.place_market_buy_by_quote(sym, quote)
                buy_price = float(buy.get("average") or buy.get("price") or current_price)
                # FIX #1: buy.get("filled") — Binance-ის რეალური დაფილვილი qty (slippage-გათვლილი)
                buy_qty   = float(buy.get("filled") or buy.get("amount") or (quote / buy_price))

            tp_price = round(buy_price * (1.0 + tp_pct / 100.0), 6)

            # dca_positions — sym_l2 tag-ით
            pos_id = open_dca_position(
                symbol=sym_l2,
                initial_entry_price=buy_price,
                initial_qty=buy_qty,
                initial_quote_spent=quote,
                tp_price=tp_price,
                sl_price=0.0,
                tp_pct=tp_pct,
                sl_pct=999.0,
                max_add_ons=int(os.getenv("DCA_MAX_ADD_ONS", "1")),
                max_capital=float(os.getenv("DCA_MAX_CAPITAL_USDT", "20.0")),
                max_drawdown_pct=999.0,
            )

            add_dca_order(
                position_id=pos_id,
                symbol=sym_l2,
                order_type="LAYER2_INITIAL",
                entry_price=buy_price,
                qty=buy_qty,
                quote_spent=quote,
                avg_entry_after=buy_price,
                tp_after=tp_price,
                sl_after=0.0,
                trigger_drawdown_pct=drop_from_high,
                exchange_order_id=str(buy.get("id", "")),
            )

            # trades ცხრილი
            import uuid
            l2_signal_id = f"L2-{sym.replace('/', '')}-{uuid.uuid4().hex[:8]}"
            open_trade(
                signal_id=l2_signal_id,
                symbol=sym_l2,
                qty=buy_qty,
                quote_in=quote,
                entry_price=buy_price,
            )

            free_usdt -= quote  # ბალანსი განახლება in-memory

            try:
                log_event(
                    "LAYER2_OPENED",
                    f"sym={sym_l2} entry={buy_price:.4f} "
                    f"tp={tp_price:.4f} drop={drop_from_high:.2f}% "
                    f"pos_id={pos_id}"
                )
            except Exception:
                pass

            # Telegram
            try:
                from execution.telegram_notifier import notify_signal_created
                notify_signal_created(
                    symbol=sym_l2,
                    entry_price=buy_price,
                    quote_amount=quote,
                    tp_price=tp_price,
                    sl_price=0.0,
                    verdict="LAYER2_BUY",
                    mode=engine.mode,
                )
            except Exception as _tg:
                logger.warning(f"[LAYER2] TG_FAIL | {sym} err={_tg}")

            logger.warning(
                f"[LAYER2] OPENED | {sym_l2} entry={buy_price:.4f} "
                f"tp={tp_price:.4f} qty={buy_qty:.6f}"
            )
            # FIX #9 REAL: cooldown timestamp განახლება — შემდეგი symbol skip-ავს
            _LAST_L2_TS = _l2_time.time()

        except Exception as e:
            logger.error(f"[LAYER2] ERR | {sym} err={e}")


def _check_cascade_exchange(engine, tp_sl_mgr) -> None:
    """
    Cascade DCA — "Rolling Exchange" სტრატეგია.

    Layer-ის მიხედვით drop_pct და tp_pct იცვლება:
      L2-L3:  drop=1.5%  tp=0.55%
      L4-L7:  drop=2.0%  tp=0.65%
      L8-L10: drop=5.0%  tp=1.00%  ← განახლებული!
      L10+:   CASCADE_MAX_LAYERS=10 → გაჩერება

    ENV:
      CASCADE_ENABLED=true
      CASCADE_START_LAYER=2       ← L2-დან იწყება
      CASCADE_DROP_PCT=1.5        ← L2-L3 trigger
      CASCADE_DROP_L4_PCT=2.0     ← L4-L7 trigger
      CASCADE_DROP_L8_PCT=5.0     ← L8-L10 trigger (იყო 3.0%)
      CASCADE_TP_L3_PCT=0.65      ← L3-L7 TP პროცენტი
      CASCADE_TP_L8_PCT=1.00      ← L8-L10 TP პროცენტი (ახალი!)
      CASCADE_MAX_LAYERS=10       ← მე-10-ზე გაჩერება
      CASCADE_RESUME_LAYER=10     ← dead zone გაუქმება
      CASCADE_SYMBOLS=BTC/USDT,BNB/USDT,ETH/USDT
    """
    import os

    if not os.getenv("CASCADE_ENABLED", "true").strip().lower() in ("1", "true", "yes"):
        return

    from execution.db.repository import (
        get_all_open_dca_positions,
        close_dca_position,
        open_dca_position,
        add_dca_order,
        open_trade,
        get_open_trade_for_symbol,
        close_trade,
        log_event,
    )

    cascade_start  = int(os.getenv("CASCADE_START_LAYER",  "2"))
    drop_pct_base  = float(os.getenv("CASCADE_DROP_PCT",    "1.5"))  # L2-L3
    drop_pct_l4    = float(os.getenv("CASCADE_DROP_L4_PCT", "2.0"))  # L4-L7
    drop_pct_l8    = float(os.getenv("CASCADE_DROP_L8_PCT", "5.0"))  # L8-L10 (იყო 3.0%)
    tp_pct_base    = float(os.getenv("DCA_TP_PCT",          "0.55")) # L1-L2
    tp_pct_l3      = float(os.getenv("CASCADE_TP_L3_PCT",   "0.65")) # L3-L7
    tp_pct_l8      = float(os.getenv("CASCADE_TP_L8_PCT",   "1.00")) # L8-L10 (ახალი!)
    max_layers     = int(os.getenv("CASCADE_MAX_LAYERS",    "10"))
    resume_layer   = int(os.getenv("CASCADE_RESUME_LAYER",  "10"))   # dead zone გაუქმება
    symbols_raw    = os.getenv("CASCADE_SYMBOLS", "BTC/USDT,BNB/USDT,ETH/USDT")
    symbols        = [s.strip() for s in symbols_raw.split(",") if s.strip()]
    buffer         = float(os.getenv("SMART_ADDON_BUFFER", "5.0"))

    # ყველა ღია პოზიცია
    all_positions = get_all_open_dca_positions()

    # სულ რამდენი Layer გვაქვს (Layer1 + Layer2 + _L2 + _L3 ...)
    total_layers = len(all_positions)

    logger.info(
        f"[CASCADE] CHECK | total_layers={total_layers} "
        f"start_at={cascade_start} max={max_layers} resume_at={resume_layer}"
    )

    # Cascade ჯერ არ დაწყებულა?
    if total_layers < cascade_start:
        logger.debug(f"[CASCADE] NOT_YET | {total_layers} < {cascade_start}")
        return

    # მე-10-ზე გაჩერება — resume_layer-ს ვეცდინოთ
    if total_layers >= max_layers:
        # resume_layer-ზე მიაღწია? — გახსენი
        if total_layers < resume_layer:
            logger.info(f"[CASCADE] PAUSED | {total_layers} >= {max_layers}, waiting for {resume_layer}")
            return
        else:
            logger.warning(f"[CASCADE] RESUMING | total_layers={total_layers} >= {resume_layer}")

    for sym in symbols:
        try:
            exchange_sym = sym

            # current price — DEMO: price_feed
            if engine.exchange is not None:
                current_price = float(engine.exchange.fetch_last_price(exchange_sym) or 0.0)
            else:
                _t = engine.price_feed.fetch_ticker(exchange_sym)
                current_price = float(_t.get("last") or 0.0)
            if current_price <= 0:
                continue

            # ამ symbol-ის ყველა Layer — გახსნის დროის მიხედვით დავალაგოთ
            # FIX #3: regex — _L2, _L3 ... _L99+ ყველა suffix იფარება
            import re as _re
            sym_positions = [
                p for p in all_positions
                if _re.sub(r'_L\d+$', '', str(p.get("symbol", "")).upper()) == sym.upper()
            ]

            if len(sym_positions) < 2:
                logger.debug(f"[CASCADE] {sym} | only {len(sym_positions)} layer(s) → skip")
                continue

            # ყველაზე ძველი Layer — opened_at მიხედვით
            oldest = sorted(sym_positions, key=lambda p: str(p.get("opened_at", "")))[0]
            oldest_avg   = float(oldest.get("avg_entry_price", 0.0))
            oldest_qty   = float(oldest.get("total_qty", 0.0))
            oldest_quote = float(oldest.get("total_quote_spent", 0.0))
            oldest_id    = oldest["id"]
            oldest_sym   = oldest["symbol"]

            # Layer ნომერი განვსაზღვროთ (sym_positions.len = ახლანდელი layer count)
            layer_num = len(sym_positions)  # მაგ: 2 = L2, 4 = L4...

            # ── Layer-ის მიხედვით drop_pct ─────────────────────────
            # L2-L3: 1.5%  |  L4-L7: 2.0%  |  L8-L10: 5.0%
            if layer_num >= 8:
                drop_pct = drop_pct_l8
            elif layer_num >= 4:
                drop_pct = drop_pct_l4
            else:
                drop_pct = drop_pct_base

            # ── Layer-ის მიხედვით tp_pct ───────────────────────────
            # L1-L2: 0.55%  |  L3-L7: 0.65%  |  L8-L10: 1.00%
            if layer_num >= 8:
                tp_pct = tp_pct_l8
            elif layer_num >= 3:
                tp_pct = tp_pct_l3
            else:
                tp_pct = tp_pct_base

            logger.info(
                f"[CASCADE] {sym} | layer={layer_num} "
                f"drop_trigger={drop_pct:.1f}% tp={tp_pct:.2f}%"
            )

            if oldest_avg <= 0 or oldest_qty <= 0:
                continue

            # FIX: trigger ვზომავთ ყველაზე ახალი Layer-ის avg-დან
            # ანუ: Layer 2 გახსნიდან კიდევ -1.5% → CASCADE იწყება
            # (არა oldest-იდან — ის Layer 2-ის trigger-თან ემთხვეოდა)
            newest = sorted(sym_positions, key=lambda p: str(p.get("opened_at", "")))[-1]
            newest_avg = float(newest.get("avg_entry_price", 0.0))
            if newest_avg <= 0:
                newest_avg = oldest_avg

            drop_from_newest = (newest_avg - current_price) / newest_avg * 100.0

            logger.info(
                f"[CASCADE] {sym} | oldest={oldest_sym} avg={oldest_avg:.4f} "
                f"newest_avg={newest_avg:.4f} price={current_price:.4f} "
                f"drop_from_newest={drop_from_newest:.2f}% trigger={drop_pct:.1f}%"
            )

            if drop_from_newest < drop_pct:
                logger.debug(f"[CASCADE] {sym} | drop={drop_from_newest:.2f}% < {drop_pct:.1f}% → wait")
                continue

            # ბალანსი შემოწმება — DEMO: ვირტუალური $120
            if engine.exchange is not None:
                free_usdt = float(engine.exchange.fetch_balance_free("USDT") or 0.0)
            else:
                free_usdt = float(os.getenv("DEMO_INITIAL_BALANCE", "120.0"))
            if free_usdt < buffer:
                logger.warning(f"[CASCADE] {sym} | low_balance={free_usdt:.2f} < buffer={buffer:.1f}")
                continue

            # ── Exchange: ძველი Layer-ის გაყიდვა ──────────────────────
            logger.warning(
                f"[CASCADE] EXCHANGE | {oldest_sym} avg={oldest_avg:.4f} "
                f"qty={oldest_qty:.6f} drop={drop_from_newest:.2f}%"
            )

            try:
                # DEMO: ვირტუალური გაყიდვა
                if engine.exchange is None:
                    sell_price = current_price
                    sell = {"average": sell_price, "price": sell_price}
                else:
                    sell = engine.exchange.place_market_sell(exchange_sym, oldest_qty)
                    sell_price = float(sell.get("average") or sell.get("price") or current_price)
                proceeds = sell_price * oldest_qty
                fee = proceeds * 0.001  # 0.1% fee
                net_proceeds = round(proceeds - fee, 4)

                pnl_quote = (sell_price - oldest_avg) * oldest_qty
                pnl_pct   = (sell_price / oldest_avg - 1.0) * 100.0

                # dca_positions დახურვა
                close_dca_position(
                    oldest_id, sell_price, oldest_qty,
                    pnl_quote, pnl_pct, "CASCADE_EXCHANGE"
                )

                # trades დახურვა
                open_tr = get_open_trade_for_symbol(oldest_sym)
                if not open_tr:
                    # fallback: suffix-ის გარეშე ვეძებთ (BTC/USDT_L2 → BTC/USDT)
                    open_tr = get_open_trade_for_symbol(exchange_sym)
                if not open_tr:
                    # fallback2: base symbol-ის ყველა ვარიანტი (_L2 … _L10)
                    base = exchange_sym.replace("/USDT", "")
                    for suffix in ["", "_L2", "_L3", "_L4", "_L5", "_L6", "_L7", "_L8", "_L9", "_L10"]:
                        _tr = get_open_trade_for_symbol(f"{base}/USDT{suffix}")
                        if _tr:
                            open_tr = _tr
                            break
                if open_tr:
                    close_trade(open_tr[0], sell_price, "CASCADE_EXCHANGE", pnl_quote, pnl_pct)
                    logger.info(f"[CASCADE] TRADE_CLOSED | {oldest_sym} signal_id={open_tr[0]}")
                else:
                    logger.warning(f"[CASCADE] TRADE_NOT_FOUND | {oldest_sym} → DB may be out of sync")

                logger.warning(
                    f"[CASCADE] SOLD | {oldest_sym} price={sell_price:.4f} "
                    f"proceeds={net_proceeds:.4f} pnl={pnl_quote:+.4f}"
                )

                # ── Telegram — CASCADE გაყიდვის შეტყობინება (D: გაუმჯობესებული) ──
                try:
                    from execution.telegram_notifier import notify_cascade_exchange
                    # ახალი layer სახელი (ჯერ გამოვთვალოთ)
                    _new_layer_name = f"{sym}_L{layer_num + 1}"
                    notify_cascade_exchange(
                        symbol=sym,
                        old_avg=oldest_avg,
                        old_layer=oldest_sym,
                        new_avg=current_price,  # ახლანდელი ფასი = ახალი entry
                        new_layer=_new_layer_name,
                        sell_price=sell_price,
                        pnl_quote=pnl_quote,
                        drop_pct=drop_from_newest,
                        new_tp=round(current_price * (1.0 + tp_pct / 100.0), 6),
                    )
                except Exception as _tg_sell:
                    # fallback — ძველი ნოტიფიკაცია
                    try:
                        notify_dca_closed(
                            symbol=oldest_sym,
                            entry_price=oldest_avg,
                            exit_price=sell_price,
                            pnl_quote=pnl_quote,
                            pnl_pct=pnl_pct,
                            outcome="CASCADE_SELL",
                            add_on_count=0,
                            stats=None,
                        )
                    except Exception:
                        pass
                    logger.warning(f"[CASCADE] TG_SELL_FAIL | err={_tg_sell}")

                # ── C: DAILY LOSS TRACKING ────────────────────────────
                # FIX BUG#3: _daily_loss_total main()-ის scope-ში იყო →
                # UnboundLocalError inner function-ში.
                # გამოსწორება: DB audit_log-ში ვწერთ (restart-safe),
                # main() loop-ი ამ event-ებს კითხულობს daily sum-ისთვის.
                if pnl_quote < 0:
                    try:
                        log_event(
                            "CASCADE_LOSS",
                            f"sym={oldest_sym} pnl={pnl_quote:.4f}"
                        )
                        logger.info(
                            f"[DAILY_LOSS] CASCADE | {oldest_sym} "
                            f"pnl={pnl_quote:+.4f} (logged to DB)"
                        )
                    except Exception:
                        pass

            except Exception as _se:
                logger.error(f"[CASCADE] SELL_FAIL | {oldest_sym} err={_se}")
                continue

            # ── ახალი Layer გახსნა ───────────────────────────────────
            # FIX #13: ყოველთვის ENV-დან fixed quote — net_proceeds კი არა!
            # net_proceeds ცვალებადია (ზარალით გაყიდვა → ნაკლები თანხა)
            # fixed quote = ყოველთვის $12 → Binance minimum notional ✅
            if net_proceeds < 5.0:
                logger.warning(f"[CASCADE] LOW_PROCEEDS | {net_proceeds:.4f} < $5 → skip new layer")
                continue

            # Layer ნომერი განვსაზღვროთ
            layer_num = len(sym_positions)  # მიმდინარე + 1
            new_sym = f"{sym}_L{layer_num + 1}"

            try:
                # FIX #13: fixed quote ENV-დან (BOT_QUOTE_PER_TRADE=12)
                buy_quote = float(os.getenv("BOT_QUOTE_PER_TRADE", "12.0"))
                # DEMO: ვირტუალური ყიდვა
                if engine.exchange is None:
                    buy_price = current_price
                    buy_qty   = buy_quote / buy_price
                    buy = {"average": buy_price, "price": buy_price, "filled": buy_qty}
                else:
                    buy = engine.exchange.place_market_buy_by_quote(exchange_sym, buy_quote)
                    buy_price = float(buy.get("average") or buy.get("price") or current_price)
                    # FIX #1: buy.get("filled") — Binance-ის რეალური დაფილვილი qty
                    buy_qty   = float(buy.get("filled") or buy.get("amount") or (buy_quote / buy_price))
                tp_price  = round(buy_price * (1.0 + tp_pct / 100.0), 6)

                # dca_positions გახსნა
                pos_id = open_dca_position(
                    symbol=new_sym,
                    initial_entry_price=buy_price,
                    initial_qty=buy_qty,
                    initial_quote_spent=buy_quote,
                    tp_price=tp_price,
                    sl_price=0.0,
                    tp_pct=tp_pct,
                    sl_pct=999.0,
                    max_add_ons=int(os.getenv("DCA_MAX_ADD_ONS", "1")),
                    max_capital=float(os.getenv("DCA_MAX_CAPITAL_USDT", "20.0")),
                    max_drawdown_pct=999.0,
                )

                add_dca_order(
                    position_id=pos_id,
                    symbol=new_sym,
                    order_type="CASCADE_LAYER",
                    entry_price=buy_price,
                    qty=buy_qty,
                    quote_spent=buy_quote,  # FIX #13: fixed $12
                    avg_entry_after=buy_price,
                    tp_after=tp_price,
                    sl_after=0.0,
                    trigger_drawdown_pct=drop_from_newest,
                    exchange_order_id=str(buy.get("id", "")),
                )

                # trades გახსნა
                import uuid
                cascade_signal_id = f"CAS-{sym.replace('/', '')}-{uuid.uuid4().hex[:8]}"
                open_trade(
                    signal_id=cascade_signal_id,
                    symbol=new_sym,
                    qty=buy_qty,
                    quote_in=buy_quote,  # FIX #13: fixed $12
                    entry_price=buy_price,
                )

                try:
                    log_event(
                        "CASCADE_LAYER_OPENED",
                        f"sym={new_sym} entry={buy_price:.4f} tp={tp_price:.4f} "
                        f"quote={buy_quote:.4f} from={oldest_sym}"  # FIX #13
                    )
                except Exception:
                    pass

                # Telegram
                try:
                    from execution.telegram_notifier import notify_signal_created
                    notify_signal_created(
                        symbol=new_sym,
                        entry_price=buy_price,
                        quote_amount=buy_quote,  # FIX #13: fixed $12
                        tp_price=tp_price,
                        sl_price=0.0,
                        verdict="CASCADE_BUY",
                        mode=engine.mode,
                    )
                except Exception as _tg:
                    logger.warning(f"[CASCADE] TG_FAIL | err={_tg}")

                logger.warning(
                    f"[CASCADE] NEW_LAYER | {new_sym} entry={buy_price:.4f} "
                    f"tp={tp_price:.4f} quote={buy_quote:.4f}"  # FIX #13
                )

                # ── F: CASCADE DEPTH WARNING — L7+ გაფრთხილება ──────────
                new_layer_num = layer_num + 1
                _warn_from = int(os.getenv("CASCADE_WARN_FROM_LAYER", "7"))
                if new_layer_num >= _warn_from:
                    try:
                        from execution.telegram_notifier import notify_cascade_depth
                        # ბაზრის მიმართულება ბოლო 3 layer-ის avg-დან
                        if len(sym_positions) >= 2:
                            _sorted = sorted(sym_positions, key=lambda p: str(p.get("opened_at", "")))
                            _first_avg = float(_sorted[0].get("avg_entry_price", 0))
                            _last_avg = float(_sorted[-1].get("avg_entry_price", 0))
                            if _last_avg < _first_avg * 0.998:
                                _trend = "down"
                            elif _last_avg > _first_avg * 1.002:
                                _trend = "up"
                            else:
                                _trend = "sideways"
                        else:
                            _trend = "unknown"

                        # HIGH-დან ვარდნა
                        try:
                            _ticker = engine.price_feed.fetch_ticker(sym)
                            _high24 = float(_ticker.get("high") or 0.0)
                            _drop_h = ((_high24 - buy_price) / _high24 * 100.0) if _high24 > 0 else 0.0
                        except Exception:
                            _drop_h = drop_from_newest

                        notify_cascade_depth(
                            symbol=sym,
                            layer_num=new_layer_num,
                            max_layers=max_layers,
                            drop_from_high_pct=_drop_h,
                            current_price=buy_price,
                            avg_entry=buy_price,
                            price_trend=_trend,
                        )
                    except Exception as _cwe:
                        logger.warning(f"[CASCADE] DEPTH_WARN_FAIL | err={_cwe}")

            except Exception as _be:
                logger.error(f"[CASCADE] BUY_FAIL | {new_sym} err={_be}")

        except Exception as e:
            logger.error(f"[CASCADE] ERR | {sym} err={e}")


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
    # თუ CASCADE ზარალი > limit → PAUSE until next day
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
        logger.info(f"DCA_ENABLED | max_add_ons={os.getenv('DCA_MAX_ADD_ONS', '3')} max_capital={os.getenv('DCA_MAX_CAPITAL_USDT', '40')}")

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
            # FIX BUG#3: _daily_loss_total ახლა DB CASCADE_LOSS events-დანაც
            # ივსება (inner function scope bug გამოსწორება).
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            _today = _now_dt().date().isoformat()
            if _today != _daily_loss_date:
                _daily_loss_date = _today
                _daily_loss_total = 0.0
                logger.info(f"DAILY_LOSS_RESET | date={_today} limit={daily_max_loss}")

            # CASCADE_LOSS events DB-დან — inner function scope fix
            # FIX CRIT#1: _daily_loss_total = _cascade_total (არა +=!)
            # += იყო: ყოველ 2 წუთში -$0.88 ემატებოდა → -$40 double-counting!
            # = არის: DB-დან ყოველ loop-ზე სრულ ჯამს კითხულობს → ერთხელ ითვლება
            try:
                from execution.db.repository import _fetchall
                _cascade_losses = _fetchall(
                    "SELECT message FROM audit_log WHERE event_type='CASCADE_LOSS' "
                    "AND created_at >= date('now') ORDER BY id DESC LIMIT 50"
                )
                _cascade_total = 0.0
                for _row in (_cascade_losses or []):
                    try:
                        import re as _re_loss
                        _m = _re_loss.search(r'pnl=([+-]?\d+\.\d+)', str(_row[0]))
                        if _m:
                            _cascade_total += float(_m.group(1))
                    except Exception:
                        pass
                if _cascade_total < 0:
                    _daily_loss_total = _cascade_total  # ← FIX: = არა +=
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
            # CASCADE-ის შემდეგ TP შეიძლება avg-ზე დაბლა დარჩეს → არასოდეს გაიყიდება!
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
            # SMART LONG + SHORT — Market Regime Detection + Futures Hedge
            # BEAR  → SHORT გახსნა + ADD-ON/CASCADE/LAYER2 BLOCKED
            # BULL  → SHORT-ები დახურვა + ყველაფერი ნორმალური
            # NEUTRAL → TP/SL check + ყველაფერი ნორმალური
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            try:
                from execution.signal_generator import _detect_market_regime_24h
                _market_regime = _detect_market_regime_24h()

                if _market_regime == "BEAR":
                    futures_engine.check_and_open_short(_market_regime)
                    futures_engine.check_tp_sl()
                elif _market_regime == "BULL":
                    futures_engine.close_all_shorts(reason="BULL_MARKET")
                    futures_engine.check_tp_sl()
                else:
                    futures_engine.check_tp_sl()

            except Exception as _fe:
                logger.warning(f"FUTURES_LOOP_WARN | err={_fe}")

            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # DCA LOOP — add-on check + TP/SL + breakeven + force close
            # BEAR MODE: ADD-ON ბლოკილია (_market_regime გადაეცემა)
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            if _dca_enabled:
                try:
                    _run_dca_loop(engine, dca_mgr, tp_sl_mgr, risk_mgr,
                                  market_regime=_market_regime)
                except Exception as e:
                    logger.warning(f"DCA_LOOP_WARN | err={e}")

            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # LAYER 2 — Crash detection & parallel trading
            # BEAR MODE: BLOCKED — ახალი layer არ გაიხსნოს!
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            if _dca_enabled:
                if _market_regime == "BEAR":
                    logger.info("[LAYER2] BEAR_BLOCK | BEAR market → Layer2 open BLOCKED")
                else:
                    try:
                        _check_and_open_layer2(engine, tp_sl_mgr)
                    except Exception as e:
                        logger.warning(f"LAYER2_CHECK_WARN | err={e}")

            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # CASCADE DCA — Rolling Exchange სტრატეგია
            # BEAR MODE: BLOCKED — ახალი layer ყიდვა შეაჩერებს SHORT-ს!
            # არსებული CASCADE TP-ს ელოდება (TP hit = cascade დახურვა ✅)
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            if _dca_enabled:
                if _market_regime == "BEAR":
                    logger.info("[CASCADE] BEAR_BLOCK | BEAR market → new CASCADE layer BLOCKED")
                else:
                    try:
                        _check_cascade_exchange(engine, tp_sl_mgr)
                    except Exception as e:
                        logger.warning(f"CASCADE_CHECK_WARN | err={e}")

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
                          if engine.exchange is None and _dca_enabled:
                              try:
                                  from execution.db.repository import (
                                      open_dca_position, add_dca_order, open_trade,
                                      get_open_dca_position_for_symbol
                                  )
                                  _sym = str((sig.get("execution") or {}).get("symbol", "BTC/USDT"))
                                  _existing = get_open_dca_position_for_symbol(_sym)
                                  if not _existing:
                                      _quote = float(os.getenv("BOT_QUOTE_PER_TRADE", "12.0"))
                                      _price = _price_cache.get(_sym, 0.0)
                                      if _price <= 0:
                                          _t = engine.price_feed.fetch_ticker(_sym)
                                          _price = float(_t.get("last") or 0.0)
                                      if _price > 0:
                                          _qty = _quote / _price
                                          _tp_pct = float(os.getenv("DCA_TP_PCT", "0.55"))
                                          _tp = round(_price * (1.0 + _tp_pct / 100.0), 6)
                                          _pos_id = open_dca_position(
                                              symbol=_sym,
                                              initial_entry_price=_price,
                                              initial_qty=_qty,
                                              initial_quote_spent=_quote,
                                              tp_price=_tp,
                                              sl_price=0.0,
                                              tp_pct=_tp_pct,
                                              sl_pct=999.0,
                                              max_add_ons=int(os.getenv("DCA_MAX_ADD_ONS", "1")),
                                              max_capital=float(os.getenv("DCA_MAX_CAPITAL_USDT", "24.0")),
                                              max_drawdown_pct=999.0,
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
            # 08:00-02:00 → ყოველ 30 წუთს
            # 02:00-02:30 → ერთხელ ღამით (Daily Summary-ის შემდეგ)
            # 02:30-08:00 → გაჩუმება 😴
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            try:
                _hb_now    = _now_dt()
                _hb_hour   = _hb_now.hour
                _hb_minute = _hb_now.minute

                # 02:30-08:00 → გაჩუმება
                _hb_silent = (2 < _hb_hour < 8) or (_hb_hour == 2 and _hb_minute >= 30)

                # 02:00-02:30 → ერთხელ ღამით
                _hb_night_ok = (_hb_hour == 2 and 0 <= _hb_minute <= 30)

                # 08:00-02:00 → ყოველ 30 წუთს
                _hb_day_ok = not _hb_silent and (now - last_heartbeat_ts) >= 1800

                if not _hb_silent and (_hb_night_ok or _hb_day_ok):
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
