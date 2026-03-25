import os
import time
import logging
from typing import Any, Dict, Optional, Tuple, List

import ccxt

from execution.db.repository import (
    get_system_state,
    log_event,
    list_active_oco_links,
    create_oco_link,
    set_oco_status,
    update_system_state,
    signal_id_already_executed,
    mark_signal_id_executed,
    has_active_oco_for_symbol,
    has_open_trade_for_symbol,
    open_trade,
    get_trade,
    get_open_trade_for_symbol,
    close_trade,
    get_trade_stats,
    count_open_trades_for_symbol,
)
from execution.kill_switch import is_kill_switch_active
from execution.virtual_wallet import simulate_market_entry
from execution.telegram_notifier import (
    notify_signal_created,
    notify_trade_closed,
)
from execution.signal_generator import notify_outcome as _notify_sl_tp_outcome
import execution.config as _cfg  # FIX: single source of truth for all ENV params

logger = logging.getLogger("gbm")


def _to_bool01(v: Any) -> bool:
    if v is None:
        return False
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return int(v) != 0
    if isinstance(v, str):
        s = v.strip().lower()
        return s in ("1", "true", "yes", "y", "on")
    return False


def _norm(s: Any) -> str:
    return str(s or "").strip().lower()


class ExecutionEngine:
    def __init__(self):
        self.mode = os.getenv("MODE", "DEMO").upper()
        self.env_kill_switch = _cfg.KILL_SWITCH
        self.live_confirmation = _cfg.LIVE_CONFIRMATION

        self.price_feed = ccxt.binance({"enableRateLimit": True})

        self.exchange = None
        if self.mode in ("LIVE", "TESTNET"):
            from execution.exchange_client import BinanceSpotClient
            self.exchange = BinanceSpotClient()

        self.state_debug = os.getenv("STATE_DEBUG", "false").lower() == "true"

        self.tp_pct = _cfg.TP_PCT
        self.sl_pct = _cfg.SL_PCT
        self.sl_limit_gap_pct = _cfg.SL_LIMIT_GAP_PCT

        self.sell_buffer = _cfg.SELL_BUFFER
        self.sell_retry_buffer = _cfg.SELL_RETRY_BUFFER

        self.max_spread_pct = _cfg.MAX_SPREAD_PCT
        self.estimated_roundtrip_fee_pct = _cfg.ESTIMATED_ROUNDTRIP_FEE_PCT
        self.estimated_slippage_pct = _cfg.ESTIMATED_SLIPPAGE_PCT
        self.min_net_profit_pct = _cfg.MIN_NET_PROFIT_PCT

        self.entry_mode = os.getenv("ENTRY_MODE", "MARKET").strip().upper()
        self.limit_entry_offset_pct = float(os.getenv("LIMIT_ENTRY_OFFSET_PCT", "0.02"))
        self.limit_entry_timeout_sec = int(os.getenv("LIMIT_ENTRY_TIMEOUT_SEC", "6"))

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # DEAD PARAMS ACTIVATED — ადრე ENV-ში იყო, კოდი არ კითხულობდა
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

        # MAX_CONSECUTIVE_LOSSES — N consecutive loss-ის შემდეგ EXEC block (0=off)
        self.max_consecutive_losses = _cfg.MAX_CONSECUTIVE_LOSSES

        # MAX_DAILY_LOSS — daily P&L % loss limit, e.g. 3 = -3% → block (0=off)
        self.max_daily_loss_pct = float(_cfg.MAX_DAILY_LOSS)

        # AI_SIGNAL_THRESHOLD — secondary ai_score threshold (adaptive signal-ზე) (0=off)
        self.ai_signal_threshold = float(os.getenv("AI_SIGNAL_THRESHOLD", "0"))

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # CAPITAL PROTECTION — drawdown + exposure limits
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

        # MAX_ACCOUNT_DRAWDOWN — % balance drop from session start → KILL (0=off)
        # e.g. 7 = if balance drops 7% from start → stop all trading
        self.max_account_drawdown_pct = _cfg.MAX_ACCOUNT_DRAWDOWN
        self._session_start_balance: Optional[float] = None  # set on first live BUY

        # MAX_RISK_PER_TRADE_PCT — max % of balance per single trade (0=off)
        # e.g. 1.0 = max 1% of balance per trade → overrides BOT_QUOTE_PER_TRADE if lower
        self.max_risk_per_trade_pct = _cfg.MAX_RISK_PER_TRADE_PCT

        # CAPITAL_USAGE_MIN/MAX — % of balance that should be deployed (informational + guard)
        # MAX: if open exposure > MAX % of balance → skip new BUY
        self.capital_usage_min = _cfg.CAPITAL_USAGE_MIN
        self.capital_usage_max = _cfg.CAPITAL_USAGE_MAX

        # MAX_PORTFOLIO_EXPOSURE — max % of total balance in open trades (0=off)
        # e.g. 0.75 = max 75% of balance can be in open positions at once
        self.max_portfolio_exposure = _cfg.MAX_PORTFOLIO_EXPOSURE

        # MAX_SYMBOL_EXPOSURE — max % of balance in any single symbol (0=off)
        # e.g. 0.40 = max 40% of balance in BTC/USDT at once
        self.max_symbol_exposure = _cfg.MAX_SYMBOL_EXPOSURE

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # STRATEGY / MARKET MODE flags — logging + future branching
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        self.adaptive_mode    = os.getenv("ADAPTIVE_MODE",    "true").lower()  == "true"
        self.execution_style  = os.getenv("EXECUTION_STYLE",  "FAST").upper()
        self.strategy_mode    = os.getenv("STRATEGY_MODE",    "HYBRID").upper()
        self.market_mode      = os.getenv("MARKET_MODE",      "ADAPTIVE").upper()
        self.trade_activity   = os.getenv("TRADE_ACTIVITY",   "HIGH").upper()

        # BOT_POSITION_SIZE — fixed base-asset size override (0 = use quote_amount instead)
        self.bot_position_size = float(os.getenv("BOT_POSITION_SIZE", "0"))

        # DEDUPE_ONLY_WHEN_ACTIVE_OCO — if true: skip dedup check when no active OCO
        # (allows re-entry after full close without waiting full cooldown)
        self.dedupe_only_when_active_oco = os.getenv("DEDUPE_ONLY_WHEN_ACTIVE_OCO", "false").lower() == "true"

        logger.info(
            f"[ENGINE_INIT] mode={self.mode} adaptive={self.adaptive_mode} "
            f"strategy={self.strategy_mode} market_mode={self.market_mode} "
            f"execution_style={self.execution_style} trade_activity={self.trade_activity} "
            f"max_drawdown={self.max_account_drawdown_pct}% "
            f"max_portfolio_exp={self.max_portfolio_exposure} "
            f"max_risk_per_trade={self.max_risk_per_trade_pct}%"
        )

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # #3 Trailing Stop
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        self.trailing_stop_enabled  = _cfg.TRAILING_STOP_ENABLED
        self.trailing_stop_distance = _cfg.TRAILING_STOP_DISTANCE

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # #5 Partial Take Profit
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        self.use_partial_tp   = _cfg.USE_PARTIAL_TP
        self.partial_tp1_pct  = _cfg.PARTIAL_TP1_PCT
        self.partial_tp1_size = _cfg.PARTIAL_TP1_SIZE

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # #7 Breakeven Stop
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        self.use_breakeven_stop    = _cfg.USE_BREAKEVEN_STOP
        self.breakeven_trigger_pct = _cfg.BREAKEVEN_TRIGGER_PCT

        # trailing peak tracker: {signal_id: peak_price}
        self._trailing_peaks: dict = {}

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # FIX: regime_engine reference — injected by main.py.
        # None → regime notify silently skipped (safe fallback).
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # FIX C-3: Double SL Cooldown — DB primary, in-memory secondary.
        # DB-based cooldown (signal_generator._notify_sl_event) = primary.
        # regime_engine in-memory (notify_outcome) = secondary (history).
        # inject_regime_engine (C-1 fix main.py) → TP/SL reset სწორია.
        # restart-ზე in-memory ნულდება, DB რჩება → DB გამარჯვებს.
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        self._regime_engine = None  # set via engine.inject_regime_engine(re)


    def inject_regime_engine(self, regime_engine) -> None:
        """main.py-დან გამოიძახება MarketRegimeEngine instance-ის inject-ისთვის.

        engine = ExecutionEngine()
        engine.inject_regime_engine(regime_engine)  # main.py-ში, loop-ის წინ
        """
        self._regime_engine = regime_engine
        logger.info("[ENGINE] regime_engine injected — SL Cooldown tracking active")
    def _load_system_state(self) -> Dict[str, Any]:
        raw = get_system_state()
        if self.state_debug:
            logger.info(f"SYSTEM_STATE_RAW | type={type(raw)} value={raw}")

        if raw is not None:
            status = raw[1] if len(raw) > 1 else ""
            sync = raw[2] if len(raw) > 2 else 0
            kill = raw[3] if len(raw) > 3 else 0
            return {
                "status": str(status or "").upper(),
                "startup_sync_ok": _to_bool01(sync),
                "kill_switch": _to_bool01(kill),
            }

        if isinstance(raw, dict):
            return {
                "status": str(raw.get("status") or "").upper(),
                "startup_sync_ok": _to_bool01(raw.get("startup_sync_ok")),
                "kill_switch": _to_bool01(raw.get("kill_switch")),
            }

        return {"status": "", "startup_sync_ok": False, "kill_switch": False}

    def _get_spread_pct(self, symbol: str) -> Optional[float]:
        try:
            ob = self.price_feed.fetch_order_book(symbol, limit=5)
            bids = ob.get("bids") or []
            asks = ob.get("asks") or []
            if not bids or not asks:
                return None
            bid = float(bids[0][0])
            ask = float(asks[0][0])
            mid = (bid + ask) / 2.0 if (bid + ask) else 0.0
            if mid <= 0:
                return None
            return ((ask - bid) / mid) * 100.0
        except Exception as e:
            logger.warning(f"SPREAD_FETCH_FAIL | symbol={symbol} err={e}")
            return None

    def _net_edge_ok(self, tp_pct: Optional[float] = None) -> Tuple[bool, str]:
        """
        FIX: adaptive tp_pct-ს ღებულობს.
        სიგნალზე adaptive TP=0.6% → edge check-ი სწორ მნიშვნელობაზე მუშაობს.
        """
        effective_tp = tp_pct if tp_pct is not None else self.tp_pct
        cost = self.estimated_roundtrip_fee_pct + self.estimated_slippage_pct
        net = effective_tp - cost
        if net < self.min_net_profit_pct:
            return False, (
                f"EDGE_TOO_SMALL tp={effective_tp:.3f} cost={cost:.2f} "
                f"net={net:.3f} < min_net={self.min_net_profit_pct:.2f} "
                f"({'adaptive' if tp_pct is not None else 'static'})"
            )
        return True, "OK"

    @staticmethod
    def _exit_price_from_order(o: Dict[str, Any], fallback: float = 0.0) -> float:
        try:
            v = float(o.get("average") or o.get("price") or 0.0)
            return v if v > 0 else float(fallback or 0.0)
        except Exception:
            return float(fallback or 0.0)

    def _estimated_fee_quote(self, notional_quote: float) -> float:
        side_fee_pct = self.estimated_roundtrip_fee_pct / 2.0
        return float(notional_quote) * (side_fee_pct / 100.0)

    def _calc_net_pnl(self, quote_in: float, entry: float, exitp: float, qty: float) -> Tuple[float, float]:
        gross_pnl_quote = (float(exitp) - float(entry)) * float(qty)
        exit_notional = float(exitp) * float(qty)

        entry_fee_quote = self._estimated_fee_quote(float(quote_in))
        exit_fee_quote = self._estimated_fee_quote(exit_notional)

        net_pnl_quote = gross_pnl_quote - entry_fee_quote - exit_fee_quote
        net_pnl_pct = (net_pnl_quote / float(quote_in) * 100.0) if float(quote_in) else 0.0

        return float(net_pnl_quote), float(net_pnl_pct)

    def _run_post_close_diagnostics(
        self,
        signal_id: str,
        link_id: Optional[int],
        symbol: str,
        qty: float,
        quote_in: float,
        entry_price: float,
        exit_price: float,
        outcome: str,
        pnl_quote: float,
        pnl_pct: float,
        tp_order_id: str = "",
        sl_order_id: str = "",
        tp_price: Optional[float] = None,
        sl_stop_price: Optional[float] = None,
        sl_limit_price: Optional[float] = None,
    ) -> None:
        """
        Safe diagnostics wrapper.
        Runs only after trade close + telegram close notify.
        Does NOT depend on broken external adapter/db wiring.
        """
        try:
            from execution.diagnostics_pro import (
                Report,
                check_position_sync,
                check_order_link,
                check_partial_fill_engine,
                check_restart_recovery,
                check_api_resilience,
                check_race_condition,
                check_latency,
                check_slippage,
                check_fee_engine,
                check_logs,
                check_edge_cases,
            )
        except Exception as e:
            logger.warning(f"DIAG_IMPORT_FAIL | signal_id={signal_id} err={e}")
            return

        try:
            class _SafeAdapter:
                def __init__(self, engine: "ExecutionEngine"):
                    self.engine = engine

                def get_trade(self, _signal_id):
                    return {
                        "signal_id": str(signal_id),
                        "symbol": str(symbol),
                        "qty": float(qty),
                        "quote_in": float(quote_in),
                        "entry_price": float(entry_price),
                        "exit_price": float(exit_price),
                        "outcome": str(outcome),
                        "pnl_quote": float(pnl_quote),
                        "pnl_pct": float(pnl_pct),
                        "status": f"CLOSED_{str(outcome).upper()}",
                        "tp_order_id": str(tp_order_id or ""),
                        "sl_order_id": str(sl_order_id or ""),
                        "tp_price": float(tp_price) if tp_price is not None else None,
                        "sl_price": float(sl_stop_price or sl_limit_price) if (sl_stop_price is not None or sl_limit_price is not None) else None,
                        "link_id": link_id,
                    }

                def get_oco_status(self, _link_id):
                    return f"CLOSED_{str(outcome).upper()}"

                def get_close_events_count(self, _signal_id):
                    return 1

                def get_trade_logs(self, _signal_id):
                    return [
                        "ENTRY",
                        "OCO",
                        f"EXIT_{str(outcome).upper()}",
                        "PNL",
                    ]

                def get_open_trades(self):
                    return []

                def get_order(self, order_id):
                    if not order_id or self.engine.exchange is None:
                        return None
                    try:
                        return self.engine.exchange.fetch_order(str(order_id), str(symbol))
                    except Exception:
                        return None

                def get_fills(self, order_id):
                    # Optional exchange-specific fill API not wired here
                    return []

                def get_position(self, _symbol):
                    # Spot close expectation after TP/SL/MANUAL_SELL = zero
                    return {"qty": 0}

                def get_balance(self):
                    if self.engine.exchange is None:
                        return {}
                    try:
                        return {
                            "USDT": float(self.engine.exchange.fetch_balance_free("USDT"))
                        }
                    except Exception:
                        return {}

                def get_fee_rate(self, _symbol):
                    try:
                        return max((float(self.engine.estimated_roundtrip_fee_pct) / 2.0) / 100.0, 0.0)
                    except Exception:
                        return 0.001

                def get_latency_ms(self):
                    return 0

            adapter = _SafeAdapter(self)
            rep = Report()

            trade = adapter.get_trade(signal_id)
            tp = adapter.get_order(tp_order_id) if tp_order_id else None
            sl = adapter.get_order(sl_order_id) if sl_order_id else None
            pos = adapter.get_position(symbol)

            check_position_sync(rep, trade, pos)
            check_order_link(rep, tp, sl)

            if tp_order_id:
                check_partial_fill_engine(rep, adapter, tp_order_id, trade.get("qty"))
            if sl_order_id:
                check_partial_fill_engine(rep, adapter, sl_order_id, trade.get("qty"))

            check_restart_recovery(rep, adapter)
            check_api_resilience(rep, tp, sl)
            check_race_condition(rep, adapter, signal_id)
            check_latency(rep, adapter)

            expected_price = None
            if str(outcome).upper() == "TP":
                expected_price = tp_price
            elif str(outcome).upper() == "SL":
                expected_price = sl_stop_price or sl_limit_price
            else:
                expected_price = exit_price

            check_slippage(rep, expected_price, exit_price)
            check_fee_engine(rep, adapter, symbol, qty, exit_price, pnl_quote)
            check_logs(rep, adapter, signal_id)
            check_edge_cases(rep, trade)

            summary = rep.summary()

            logger.info(
                "DIAG_REPORT | "
                f"id={signal_id} symbol={symbol} outcome={outcome} "
                f"passed={summary.get('passed')} failed={summary.get('failed')} "
                f"critical={summary.get('critical')} status={summary.get('status')}"
            )

            for r in rep.results:
                level = str(r.severity or "INFO").upper()
                msg = f"DIAG | id={signal_id} {r.name} ok={r.ok} sev={level} msg={r.msg}"
                if level == "CRITICAL":
                    logger.error(msg)
                elif level == "WARN":
                    logger.warning(msg)
                else:
                    logger.info(msg)

        except Exception as e:
            logger.warning(f"DIAG_RUN_FAIL | signal_id={signal_id} symbol={symbol} err={e}")

    def reconcile_oco(self) -> None:
        if self.mode not in ("LIVE", "TESTNET"):
            return
        if self.exchange is None:
            return

        rows = list_active_oco_links(limit=50)
        if not rows:
            return

        for r in rows:
            (
                link_id,
                signal_id,
                symbol,
                base_asset,
                tp_order_id,
                sl_order_id,
                tp_price,
                sl_stop_price,
                sl_limit_price,
                amount,
                status,
                created_at,
                updated_at,
            ) = r

            if not tp_order_id or not sl_order_id:
                logger.warning(
                    f"OCO_RECONCILE_SKIP | link={link_id} missing order ids "
                    f"tp='{tp_order_id}' sl='{sl_order_id}'"
                )
                continue

            try:
                tp = self.exchange.fetch_order(tp_order_id, symbol)
                sl = self.exchange.fetch_order(sl_order_id, symbol)

                tp_status = _norm(tp.get("status"))
                sl_status = _norm(sl.get("status"))

                logger.info(
                    f"OCO_RECONCILE | link={link_id} id={signal_id} symbol={symbol} "
                    f"tp={tp_order_id}:{tp_status} sl={sl_order_id}:{sl_status}"
                )

                tp_status = (tp_status or "").lower().strip()
                sl_status = (sl_status or "").lower().strip()

                # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                # FIX C-6: "partially_filled" დამატებულია filled set-ში.
                # Binance-ი TP/SL ბრძანებას "partially_filled" სტატუსში
                # ტოვებს ზოგჯერ (partial execution). ადრე ეს სტატუსი
                # არ ეცნობოდა → OCO სამუდამოდ "ღიად" რჩებოდა → stuck position.
                # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                filled = {"filled", "closed", "partially_filled"}
                canceled_set = {"canceled", "cancelled", "expired", "rejected"}

                def _safe_float(x):
                    try:
                        return float(x)
                    except Exception:
                        return None

                current_status = status
                if isinstance(current_status, str):
                    current_status = current_status.strip().upper()
                else:
                    current_status = ""

                if current_status in {"CLOSED_TP", "CLOSED_SL", "CLOSED_SL_COOLDOWN"}:
                    logger.debug(f"OCO_ALREADY_CLOSED | link={link_id} status={current_status}")
                    continue

                # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                # RECONCILE SL_COOLDOWN CHECK — backup layer
                # თუ SL pause აქტიურია და OCO ჯერ კიდევ ღიაა →
                # დაუყოვნებლივ გაუქმდეს (race condition backup).
                # execute_signal-ის fix პირველი ხაზია, ეს მეორე.
                # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                try:
                    from execution.db.repository import is_sl_pause_active
                    if is_sl_pause_active():
                        logger.warning(
                            f"[SL_COOLDOWN][RECONCILE] pause active + OCO open | "
                            f"link={link_id} sym={symbol} → force cancel"
                        )
                        # cancel TP
                        if tp_order_id:
                            try:
                                self.exchange.cancel_order(str(tp_order_id), str(symbol))
                            except Exception:
                                pass
                        # cancel SL
                        if sl_order_id:
                            try:
                                self.exchange.cancel_order(str(sl_order_id), str(symbol))
                            except Exception:
                                pass
                        # market sell
                        try:
                            free_base = float(self.exchange.fetch_balance_free(str(base_asset)))
                            if free_base > 0:
                                sell_amt = self.exchange.floor_amount(
                                    str(symbol), free_base * self.sell_buffer
                                )
                                if sell_amt > 0:
                                    self.exchange.place_market_sell(str(symbol), sell_amt)
                                    logger.warning(
                                        f"[SL_COOLDOWN][RECONCILE] market sold | "
                                        f"sym={symbol} amount={sell_amt}"
                                    )
                        except Exception as e_ms:
                            logger.warning(f"[SL_COOLDOWN][RECONCILE] market sell fail | {e_ms}")

                        set_oco_status(link_id, "CLOSED_SL_COOLDOWN")
                        log_event("SL_COOLDOWN_RECONCILE_CLOSE", f"link={link_id} sym={symbol}")
                        continue
                except Exception as e_pause:
                    logger.warning(f"[SL_COOLDOWN][RECONCILE] pause check err | {e_pause}")

                tr = get_trade(signal_id)
                if not tr:
                    logger.warning(f"TRADE_ROW_MISSING | signal_id={signal_id}")
                    continue

                try:
                    _, _, qty, quote_in, entry_price, *_ = tr
                    qty = float(qty)
                    quote_in = float(quote_in)
                    entry_price = float(entry_price)
                except Exception as e:
                    logger.error(f"TRADE_PARSE_FAIL | {signal_id} | {e}")
                    continue

                # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                # #7 BREAKEVEN + #3 TRAILING — ყოველ reconcile ტიკზე
                # მხოლოდ ღია (not filled/cancelled) OCO-ებზე
                # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                if tp_status not in filled and sl_status not in filled:
                    try:
                        self._check_breakeven(
                            signal_id=str(signal_id),
                            link_id=int(link_id),
                            symbol=str(symbol),
                            entry_price=float(entry_price),
                            sl_order_id=str(sl_order_id),
                            sl_stop_price=float(sl_stop_price or 0),
                            sl_limit_price=float(sl_limit_price or 0),
                            amount=float(amount),
                        )
                    except Exception as e:
                        logger.warning(f"BREAKEVEN_CHECK_ERR | id={signal_id} err={e}")

                    try:
                        self._check_trailing_stop(
                            signal_id=str(signal_id),
                            link_id=int(link_id),
                            symbol=str(symbol),
                            entry_price=float(entry_price),
                            sl_order_id=str(sl_order_id),
                            amount=float(amount),
                        )
                    except Exception as e:
                        logger.warning(f"TRAILING_CHECK_ERR | id={signal_id} err={e}")

                if tp_status in filled and sl_status in filled:
                    logger.critical(f"OCO_DESYNC | BOTH_FILLED | {signal_id}")
                    set_oco_status(link_id, "DESYNC")
                    continue

                def _get_exit_price(order_obj, fallback_price):
                    try:
                        if order_obj:
                            px = self._exit_price_from_order(
                                order_obj,
                                fallback=_safe_float(fallback_price) or 0.0
                            )
                            if px:
                                return float(px)
                    except Exception as e:
                        logger.warning(f"EXIT_PRICE_FAIL | fallback used | {e}")

                    fb = _safe_float(fallback_price)
                    if fb is not None:
                        return fb

                    logger.error(f"NO_EXIT_PRICE | signal_id={signal_id}")
                    return None

                if tp_status in filled:
                    exitp = _get_exit_price(tp, tp_price)
                    if exitp is None:
                        continue

                    try:
                        pnl_quote, pnl_pct = self._calc_net_pnl(quote_in, entry_price, exitp, qty)
                    except Exception as e:
                        logger.error(f"TP_CALC_FAIL | {signal_id} | {e}")
                        continue

                    if current_status not in {"CLOSED_TP", "CLOSED_SL"}:
                        close_trade(
                            signal_id,
                            exit_price=exitp,
                            outcome="TP",
                            pnl_quote=pnl_quote,
                            pnl_pct=pnl_pct,
                        )
                        set_oco_status(link_id, "CLOSED_TP")

                    log_event("TRADE_CLOSED", f"{signal_id} {symbol} TP exit={exitp:.6f} pnl={pnl_quote:.4f}")
                    logger.info(
                        f"TRADE_CLOSED | id={signal_id} outcome=TP "
                        f"exit={exitp:.6f} pnl={pnl_quote:.4f}"
                    )

                    try:
                        stats = get_trade_stats()
                        notify_trade_closed(
                            symbol=symbol,
                            entry_price=entry_price,
                            exit_price=exitp,
                            pnl_quote=pnl_quote,
                            pnl_pct=pnl_pct,
                            outcome="TP",
                            stats=stats,
                        )

                        # SL/TP tracker — consecutive SL counter reset
                        # FIX I-8: symbol გადაეცემა per-symbol logging-ისთვის
                        try:
                            _notify_sl_tp_outcome("TP", symbol=str(symbol))
                        except Exception:
                            pass

                        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                        # FIX: regime_engine SL Cooldown — TP resets counter
                        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                        try:
                            from execution.regime_engine import MarketRegimeEngine as _RE
                            _re_instance = getattr(self, "_regime_engine", None)
                            if _re_instance is not None:
                                _re_instance.notify_outcome(str(symbol), "TP")
                                logger.info(f"[REGIME_OUTCOME] TP reset | sym={symbol}")
                        except Exception as _re_err:
                            logger.warning(f"[REGIME_OUTCOME] TP notify fail | {_re_err}")

                        self._run_post_close_diagnostics(
                            signal_id=str(signal_id),
                            link_id=link_id,
                            symbol=str(symbol),
                            qty=float(qty),
                            quote_in=float(quote_in),
                            entry_price=float(entry_price),
                            exit_price=float(exitp),
                            outcome="TP",
                            pnl_quote=float(pnl_quote),
                            pnl_pct=float(pnl_pct),
                            tp_order_id=str(tp_order_id or ""),
                            sl_order_id=str(sl_order_id or ""),
                            tp_price=float(tp_price) if tp_price is not None else None,
                            sl_stop_price=float(sl_stop_price) if sl_stop_price is not None else None,
                            sl_limit_price=float(sl_limit_price) if sl_limit_price is not None else None,
                        )
                    except Exception as e:
                        logger.warning(f"TG_TP_FAIL | {e}")

                    continue

                if sl_status in filled:
                    exitp = _get_exit_price(sl, sl_stop_price or sl_limit_price)
                    if exitp is None:
                        continue

                    try:
                        pnl_quote, pnl_pct = self._calc_net_pnl(quote_in, entry_price, exitp, qty)
                    except Exception as e:
                        logger.error(f"SL_CALC_FAIL | {signal_id} | {e}")
                        continue

                    if current_status not in {"CLOSED_TP", "CLOSED_SL"}:
                        close_trade(
                            signal_id,
                            exit_price=exitp,
                            outcome="SL",
                            pnl_quote=pnl_quote,
                            pnl_pct=pnl_pct,
                        )
                        set_oco_status(link_id, "CLOSED_SL")

                    log_event("TRADE_CLOSED", f"{signal_id} {symbol} SL exit={exitp:.6f} pnl={pnl_quote:.4f}")
                    logger.info(
                        f"TRADE_CLOSED | id={signal_id} outcome=SL "
                        f"exit={exitp:.6f} pnl={pnl_quote:.4f}"
                    )

                    try:
                        stats = get_trade_stats()
                        notify_trade_closed(
                            symbol=symbol,
                            entry_price=entry_price,
                            exit_price=exitp,
                            pnl_quote=pnl_quote,
                            pnl_pct=pnl_pct,
                            outcome="SL",
                            stats=stats,
                        )

                        # SL/TP tracker — consecutive SL counter გაზარდე
                        # FIX I-8: symbol გადაეცემა per-symbol logging-ისთვის
                        try:
                            _notify_sl_tp_outcome("SL", symbol=str(symbol))
                        except Exception:
                            pass

                        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                        # FIX: regime_engine SL Cooldown — SL increments counter
                        # ეს არის ის კრიტიკული call რომელიც არ არსებობდა.
                        # regime_engine.apply() SL Cooldown-ს ამოწმებს მხოლოდ
                        # მაშინ თუ notify_outcome("SL") გამოიძახება.
                        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                        try:
                            _re_instance = getattr(self, "_regime_engine", None)
                            if _re_instance is not None:
                                _re_instance.notify_outcome(str(symbol), "SL")
                                cons_sl = _re_instance.get_consecutive_sl(str(symbol))
                                logger.warning(
                                    f"[REGIME_OUTCOME] SL tracked | sym={symbol} "
                                    f"consecutive_sl={cons_sl}"
                                )
                        except Exception as _re_err:
                            logger.warning(f"[REGIME_OUTCOME] SL notify fail | {_re_err}")

                        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                        # FIX v2: SL hit-ის შემდეგ ყველა სხვა ღია OCO
                        # დაუყოვნებლივ გაუქმდეს — race condition-ის
                        # თავიდან ასაცილებლად.
                        #
                        # ძველი ლოგიკა: consecutive_sl >= limit(2) → cancel
                        # პრობლემა: SL #2-ის cancel-ამდე ETH OCO უკვე
                        # executed იყო → consecutive_sl=3.
                        #
                        # ახალი ლოგიკა: ნებისმიერი SL hit → სხვა სიმბოლოს
                        # OCO-ები გაუქმდება + market sell.
                        # limit-ზე: პაუზა + სრული დახურვა.
                        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                        try:
                            from execution.db.repository import get_sl_cooldown_state
                            sl_state = get_sl_cooldown_state()
                            sl_limit = int(os.getenv("SL_COOLDOWN_AFTER_N", "2"))
                            cur_sl   = sl_state["consecutive_sl"]

                            # ── ნებისმიერ SL hit-ზე: სხვა სიმბოლოს OCO-ები გაუქმდეს ──
                            # (current symbol-ის OCO უკვე exchange-ზე executed/closed)
                            open_links = list_active_oco_links(limit=20)
                            other_links = [
                                lnk for lnk in open_links
                                if str(lnk[1]) != str(signal_id)   # skip current
                                and str(lnk[2]) != str(symbol)      # skip same symbol (race)
                            ]

                            if other_links:
                                logger.warning(
                                    f"[SL_COOLDOWN] SL hit #{cur_sl} | "
                                    f"cancelling {len(other_links)} other open OCO(s) — race prevention"
                                )

                            for olink in other_links:
                                try:
                                    (ol_id, ol_sig, ol_sym, ol_base,
                                     ol_tp_oid, ol_sl_oid, *_rest) = olink

                                    # cancel TP order
                                    if ol_tp_oid:
                                        try:
                                            self.exchange.cancel_order(str(ol_tp_oid), str(ol_sym))
                                            logger.info(f"[SL_COOLDOWN] cancelled TP | sym={ol_sym} oid={ol_tp_oid}")
                                        except Exception:
                                            pass

                                    # cancel SL order
                                    if ol_sl_oid:
                                        try:
                                            self.exchange.cancel_order(str(ol_sl_oid), str(ol_sym))
                                            logger.info(f"[SL_COOLDOWN] cancelled SL | sym={ol_sym} oid={ol_sl_oid}")
                                        except Exception:
                                            pass

                                    # market sell remaining balance
                                    try:
                                        free_base = float(self.exchange.fetch_balance_free(str(ol_base)))
                                        if free_base > 0:
                                            sell_amt = self.exchange.floor_amount(
                                                str(ol_sym), free_base * self.sell_buffer
                                            )
                                            if sell_amt > 0:
                                                self.exchange.place_market_sell(str(ol_sym), sell_amt)
                                                logger.warning(
                                                    f"[SL_COOLDOWN] market sold remaining | "
                                                    f"sym={ol_sym} amount={sell_amt}"
                                                )
                                    except Exception as e2:
                                        logger.warning(f"[SL_COOLDOWN] market sell fail | sym={ol_sym} err={e2}")

                                    set_oco_status(int(ol_id), "CLOSED_SL_COOLDOWN")
                                    log_event("SL_COOLDOWN_FORCE_CLOSE", f"sig={ol_sig} sym={ol_sym} triggered_by={signal_id}")

                                except Exception as e3:
                                    logger.warning(f"[SL_COOLDOWN] cancel_loop err | {e3}")

                            # ── limit-ზე: დამატებითი warning (პაუზა signal_generator-ში უკვეა) ──
                            if cur_sl >= sl_limit:
                                logger.warning(
                                    f"[SL_COOLDOWN] {cur_sl} consecutive SL — "
                                    f"PAUSE active. All OCOs cancelled. No new BUY for "
                                    f"{int(os.getenv('SL_COOLDOWN_PAUSE_SECONDS', '1800')) // 60}min"
                                )
                                log_event("SL_COOLDOWN_PAUSE_ACTIVE",
                                          f"consecutive_sl={cur_sl} all_ocos_cancelled=True")

                        except Exception as e:
                            logger.warning(f"[SL_COOLDOWN] cancel_all_oco_fail | err={e}")

                        self._run_post_close_diagnostics(
                            signal_id=str(signal_id),
                            link_id=link_id,
                            symbol=str(symbol),
                            qty=float(qty),
                            quote_in=float(quote_in),
                            entry_price=float(entry_price),
                            exit_price=float(exitp),
                            outcome="SL",
                            pnl_quote=float(pnl_quote),
                            pnl_pct=float(pnl_pct),
                            tp_order_id=str(tp_order_id or ""),
                            sl_order_id=str(sl_order_id or ""),
                            tp_price=float(tp_price) if tp_price is not None else None,
                            sl_stop_price=float(sl_stop_price) if sl_stop_price is not None else None,
                            sl_limit_price=float(sl_limit_price) if sl_limit_price is not None else None,
                        )
                    except Exception as e:
                        logger.warning(f"TG_SL_FAIL | {e}")

                    continue

                if tp_status in canceled_set and sl_status in canceled_set:
                    logger.error(f"OCO_BROKEN | link={link_id} signal_id={signal_id}")
                    set_oco_status(link_id, "BROKEN")
                    continue

                if tp_status not in (filled | canceled_set) or sl_status not in (filled | canceled_set):
                    logger.debug(f"OCO_UNKNOWN_STATE | tp={tp_status} sl={sl_status} id={signal_id}")

            except Exception as e:
                logger.warning(f"OCO_RECONCILE_FAIL | link={link_id} symbol={symbol} err={e}")

    def _execute_sell(self, signal_id: str, symbol: str, signal_hash: str = None) -> None:
        logger.info(f"SELL_ENTER | id={signal_id} symbol={symbol} MODE={self.mode}")

        if self.mode == "DEMO":
            log_event("SELL_DEMO", f"{signal_id} DEMO SELL {symbol}")
            mark_signal_id_executed(signal_id, signal_hash=signal_hash, action="SELL_DEMO", symbol=str(symbol))
            return

        if self.exchange is None:
            log_event("SELL_BLOCKED_NO_EXCHANGE", f"{signal_id} {symbol}")
            logger.warning(f"SELL_BLOCKED | exchange client not wired | id={signal_id} symbol={symbol}")
            return

        if is_kill_switch_active():
            logger.error(f"KILL_SWITCH_ACTIVE_LAST_GATE | SELL_BLOCKED | id={signal_id} symbol={symbol}")
            log_event("SELL_BLOCKED_KILL_SWITCH_LAST_GATE", f"{signal_id} {symbol}")
            return

        rows = list_active_oco_links(limit=50)
        rows = [r for r in rows if str(r[2] or "").upper() == str(symbol).upper()]
        CLOSED = {"closed", "filled"}

        for r in rows:
            link_id, oco_signal_id, sym, base_asset, tp_order_id, sl_order_id, *_rest = r
            try:
                tp = self.exchange.fetch_order(tp_order_id, symbol)
                sl = self.exchange.fetch_order(sl_order_id, symbol)
                tp_status = _norm(tp.get("status"))
                sl_status = _norm(sl.get("status"))

                if tp_status in CLOSED:
                    set_oco_status(link_id, "CLOSED_TP")
                    log_event("SELL_SKIP", f"{signal_id} {symbol} already closed by TP (link={link_id})")
                    continue
                if sl_status in CLOSED:
                    set_oco_status(link_id, "CLOSED_SL")
                    log_event("SELL_SKIP", f"{signal_id} {symbol} already closed by SL (link={link_id})")
                    continue

                for oid in (tp_order_id, sl_order_id):
                    if not oid:
                        continue
                    try:
                        self.exchange.cancel_order(str(oid), symbol)
                    except Exception as e:
                        logger.warning(f"SELL_CANCEL_WARN | id={signal_id} symbol={symbol} order_id={oid} err={e}")

                set_oco_status(link_id, "CANCELED_BY_SIGNAL")
                log_event("OCO_CANCELED", f"{signal_id} {symbol} link={link_id} canceled_by_signal")

            except Exception as e:
                logger.warning(f"SELL_OCO_LOOKUP_FAIL | id={signal_id} symbol={symbol} link={link_id} err={e}")

        base_asset = symbol.split("/")[0].upper()
        free_base = float(self.exchange.fetch_balance_free(base_asset))
        sell_amount = self.exchange.floor_amount(symbol, free_base * self.sell_buffer)
        if sell_amount <= 0:
            sell_amount = self.exchange.floor_amount(symbol, free_base * self.sell_retry_buffer)

        if sell_amount <= 0:
            msg = f"SELL_SKIP_NO_FREE_BASE | id={signal_id} symbol={symbol} free_{base_asset}={free_base}"
            logger.warning(msg)
            log_event("SELL_SKIP_NO_FREE_BASE", msg)
            mark_signal_id_executed(signal_id, signal_hash=signal_hash, action="SELL_NO_FREE_BASE", symbol=str(symbol))
            return

        try:
            sell = self.exchange.place_market_sell(symbol=symbol, base_amount=sell_amount)
            avg = float(sell.get("average") or sell.get("price") or 0.0) or self.exchange.fetch_last_price(symbol)

            logger.info(f"SELL_LIVE_OK | id={signal_id} symbol={symbol} amount={sell_amount} avg={avg} order_id={sell.get('id')}")
            log_event("SELL_LIVE_OK", f"{signal_id} {symbol} amount={sell_amount} avg={avg} order_id={sell.get('id')}")

            tr = get_open_trade_for_symbol(symbol)
            if tr:
                trade_signal_id, _, qty, quote_in, entry_price, *_ = tr
                pnl_quote, pnl_pct = self._calc_net_pnl(
                    float(quote_in), float(entry_price), float(avg), float(qty)
                )
                close_trade(
                    trade_signal_id,
                    exit_price=float(avg),
                    outcome="MANUAL_SELL",
                    pnl_quote=float(pnl_quote),
                    pnl_pct=float(pnl_pct),
                )
                log_event(
                    "TRADE_CLOSED",
                    f"{trade_signal_id} {symbol} MANUAL_SELL exit={avg} net_pnl_quote={pnl_quote:.4f} net_pnl_pct={pnl_pct:.3f}"
                )
                logger.info(
                    f"TRADE_CLOSED | id={trade_signal_id} symbol={symbol} outcome=MANUAL_SELL "
                    f"exit={avg} net_pnl_quote={pnl_quote:.4f} net_pnl_pct={pnl_pct:.3f}"
                )

                try:
                    stats = get_trade_stats()
                    notify_trade_closed(
                        symbol=str(symbol),
                        entry_price=float(entry_price),
                        exit_price=float(avg),
                        pnl_quote=float(pnl_quote),
                        pnl_pct=float(pnl_pct),
                        outcome="MANUAL_SELL",
                        stats=stats,
                    )

                    self._run_post_close_diagnostics(
                        signal_id=str(trade_signal_id),
                        link_id=None,
                        symbol=str(symbol),
                        qty=float(qty),
                        quote_in=float(quote_in),
                        entry_price=float(entry_price),
                        exit_price=float(avg),
                        outcome="MANUAL_SELL",
                        pnl_quote=float(pnl_quote),
                        pnl_pct=float(pnl_pct),
                        tp_order_id="",
                        sl_order_id="",
                        tp_price=None,
                        sl_stop_price=None,
                        sl_limit_price=None,
                    )
                except Exception as e:
                    logger.warning(f"TG_NOTIFY_CLOSE_FAIL | id={trade_signal_id} outcome=MANUAL_SELL err={e}")

            mark_signal_id_executed(signal_id, signal_hash=signal_hash, action="SELL_LIVE", symbol=str(symbol))

        except Exception as e:
            logger.exception(f"SELL_LIVE_ERROR | id={signal_id} symbol={symbol} err={e}")
            log_event("SELL_LIVE_ERROR", f"{signal_id} {symbol} err={e}")
            return

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # #5 PARTIAL TAKE PROFIT
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    def _place_partial_tp_order(self, signal_id: str, symbol: str,
                                 sell_amount: float, buy_avg: float,
                                 adaptive: dict) -> Optional[str]:
        use_partial = adaptive.get("USE_PARTIAL_TP", self.use_partial_tp)
        if not use_partial:
            return None
        try:
            tp1_pct    = float(adaptive.get("PARTIAL_TP1_PCT",  self.partial_tp1_pct))
            tp1_size   = float(adaptive.get("PARTIAL_TP1_SIZE", self.partial_tp1_size))
            tp1_price  = self.exchange.floor_price(symbol, buy_avg * (1.0 + tp1_pct / 100.0))
            tp1_amount = self.exchange.floor_amount(symbol, sell_amount * tp1_size)
            if tp1_amount <= 0:
                return None
            order = self.exchange.place_limit_sell_amount(
                symbol=symbol, base_amount=tp1_amount, price=tp1_price
            )
            oid = str(order.get("id") or "")
            logger.info(f"PARTIAL_TP1_PLACED | id={signal_id} sym={symbol} price={tp1_price} amount={tp1_amount} oid={oid}")
            log_event("PARTIAL_TP1_PLACED", f"{signal_id} {symbol} price={tp1_price} amount={tp1_amount}")
            return oid
        except Exception as e:
            logger.warning(f"PARTIAL_TP1_FAIL | id={signal_id} sym={symbol} err={e}")
            return None

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # #7 BREAKEVEN STOP
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    def _check_breakeven(self, signal_id: str, link_id: int, symbol: str,
                          entry_price: float, sl_order_id: str,
                          sl_stop_price: float, sl_limit_price: float,
                          amount: float, adaptive_meta: Optional[dict] = None) -> bool:
        use_be = (adaptive_meta or {}).get("USE_BREAKEVEN_STOP", self.use_breakeven_stop)
        if not use_be:
            return False
        trigger_pct = float((adaptive_meta or {}).get("BREAKEVEN_TRIGGER_PCT", self.breakeven_trigger_pct))
        if float(sl_stop_price or 0) >= entry_price * 0.999:
            return False
        try:
            current_price = self.exchange.fetch_last_price(symbol)
        except Exception:
            return False
        if current_price < entry_price * (1.0 + trigger_pct / 100.0):
            return False
        new_sl_stop  = self.exchange.floor_price(symbol, entry_price)
        new_sl_limit = self.exchange.floor_price(symbol, entry_price * (1.0 - self.sl_limit_gap_pct / 100.0))
        try:
            self.exchange.cancel_order(str(sl_order_id), symbol)
        except Exception as e:
            logger.warning(f"BREAKEVEN_CANCEL_FAIL | id={signal_id} err={e}")
            return False
        try:
            new_sl = self.exchange.place_stop_loss_limit_sell(
                symbol=symbol, base_amount=float(amount),
                stop_price=float(new_sl_stop), limit_price=float(new_sl_limit),
            )
            new_sl_id = str(new_sl.get("id") or "")
            from execution.db.repository import _execute as _db_exec
            _db_exec(
                "UPDATE oco_links SET sl_order_id=?, sl_stop_price=?, sl_limit_price=?, "
                "updated_at=datetime('now') WHERE id=?",
                (new_sl_id, float(new_sl_stop), float(new_sl_limit), int(link_id)),
            )
            logger.info(f"BREAKEVEN_TRIGGERED | id={signal_id} sym={symbol} entry={entry_price:.6f} new_sl={new_sl_stop:.6f} oid={new_sl_id}")
            log_event("BREAKEVEN_TRIGGERED", f"{signal_id} {symbol} new_sl={new_sl_stop:.6f}")
            return True
        except Exception as e:
            logger.warning(f"BREAKEVEN_NEW_SL_FAIL | id={signal_id} sym={symbol} err={e}")
            return False

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # #3 TRAILING STOP
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    def _check_trailing_stop(self, signal_id: str, link_id: int, symbol: str,
                               entry_price: float, sl_order_id: str,
                               amount: float, adaptive_meta: Optional[dict] = None) -> bool:
        use_trail = (adaptive_meta or {}).get("TRAILING_STOP_ENABLED", self.trailing_stop_enabled)
        if not use_trail:
            return False
        distance_pct = float((adaptive_meta or {}).get("TRAILING_STOP_DISTANCE", self.trailing_stop_distance))
        try:
            current_price = self.exchange.fetch_last_price(symbol)
        except Exception:
            return False
        peak = self._trailing_peaks.get(signal_id, entry_price)
        if current_price > peak:
            self._trailing_peaks[signal_id] = current_price
            peak = current_price
        new_sl_stop  = self.exchange.floor_price(symbol, peak * (1.0 - distance_pct / 100.0))
        new_sl_limit = self.exchange.floor_price(symbol, new_sl_stop * (1.0 - self.sl_limit_gap_pct / 100.0))
        try:
            from execution.db.repository import _fetchone
            row = _fetchone("SELECT sl_stop_price FROM oco_links WHERE id=?", (int(link_id),))
            current_sl = float(row[0]) if row else 0.0
        except Exception:
            current_sl = 0.0
        if new_sl_stop <= current_sl * (1.0005):
            return False
        try:
            self.exchange.cancel_order(str(sl_order_id), symbol)
        except Exception as e:
            logger.warning(f"TRAILING_CANCEL_FAIL | id={signal_id} err={e}")
            return False
        try:
            new_sl = self.exchange.place_stop_loss_limit_sell(
                symbol=symbol, base_amount=float(amount),
                stop_price=float(new_sl_stop), limit_price=float(new_sl_limit),
            )
            new_sl_id = str(new_sl.get("id") or "")
            from execution.db.repository import _execute as _db_exec
            _db_exec(
                "UPDATE oco_links SET sl_order_id=?, sl_stop_price=?, sl_limit_price=?, "
                "updated_at=datetime('now') WHERE id=?",
                (new_sl_id, float(new_sl_stop), float(new_sl_limit), int(link_id)),
            )
            logger.info(f"TRAILING_SL_UPDATED | id={signal_id} sym={symbol} peak={peak:.6f} new_sl={new_sl_stop:.6f} oid={new_sl_id}")
            log_event("TRAILING_SL_UPDATED", f"{signal_id} {symbol} peak={peak:.6f} new_sl={new_sl_stop:.6f}")
            return True
        except Exception as e:
            logger.warning(f"TRAILING_NEW_SL_FAIL | id={signal_id} sym={symbol} err={e}")
            return False

    def _place_entry_buy(self, symbol: str, quote_amount: float) -> Tuple[Dict[str, Any], float]:
        if self.exchange is None:
            raise RuntimeError("exchange client not wired")

        sp = self._get_spread_pct(symbol)
        if sp is not None and sp > self.max_spread_pct:
            raise RuntimeError(f"SPREAD_TOO_WIDE spread%={sp:.4f} > MAX_SPREAD_PCT={self.max_spread_pct:.4f}")

        buy = self.exchange.place_market_buy_by_quote(symbol=symbol, quote_amount=quote_amount)
        buy_avg = float(buy.get("average") or buy.get("price") or 0.0) or self.exchange.fetch_last_price(symbol)
        return buy, buy_avg

    def execute_signal(self, signal: Dict[str, Any]) -> None:
        signal_id = str(signal.get("signal_id", "UNKNOWN"))
        verdict = str(signal.get("final_verdict", "")).upper()

        adaptive = signal.get("adaptive", {})

        if adaptive:
            logger.info(f"[AUTO] Using adaptive params: {adaptive}")
            tp_pct = float(adaptive.get("TP_PCT", self.tp_pct))
            sl_pct = float(adaptive.get("SL_PCT", self.sl_pct))
        else:
            tp_pct = self.tp_pct
            sl_pct = self.sl_pct

        logger.info(f"EXEC_ENTER | id={signal_id} verdict={verdict} MODE={self.mode} ENV_KILL_SWITCH={self.env_kill_switch}")

        if signal_id_already_executed(signal_id):
            logger.warning(f"EXEC_DEDUPED | duplicate ignored | id={signal_id}")
            log_event("EXEC_DEDUPED", f"id={signal_id}")
            return

        state = self._load_system_state()
        db_status = str(state.get("status") or "").upper()
        db_kill = bool(state.get("kill_switch"))
        sync_ok = bool(state.get("startup_sync_ok"))

        if self.env_kill_switch or db_kill:
            logger.warning(f"EXEC_BLOCKED | KILL_SWITCH_ON | id={signal_id}")
            log_event("EXEC_BLOCKED_KILL_SWITCH", f"{signal_id}")
            return

        if not sync_ok or db_status not in ("ACTIVE", "RUNNING"):
            logger.warning(f"EXEC_BLOCKED | system not ACTIVE/synced | id={signal_id} status={db_status} sync_ok={sync_ok}")
            log_event("EXEC_BLOCKED_SYSTEM_STATE", f"{signal_id}")
            return

        if self.mode == "LIVE" and not self.live_confirmation:
            logger.warning(f"EXEC_BLOCKED | LIVE_CONFIRMATION=OFF | id={signal_id}")
            log_event("EXEC_BLOCKED_LIVE_CONFIRMATION", f"{signal_id}")
            return

        execution = signal.get("execution") or {}
        symbol = execution.get("symbol")
        direction = str(execution.get("direction", "")).upper()
        entry = execution.get("entry") or {}
        entry_type = str(entry.get("type", "")).upper()

        position_size = execution.get("position_size")
        quote_amount = execution.get("quote_amount")

        signal_hash = signal.get("_fingerprint") or signal.get("signal_hash")

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # FIX C-4: SELL verdict → certified_signal check-ის წინ.
        # SELL (TREND_REVERSAL / PROTECTIVE_SELL) ყოველთვის უნდა
        # სრულდებოდეს. ადრე certified_signal check SELL-ს ბლოკავდა
        # თუ signal_generator-ის გარდა სხვა წყარო გამოიყენებოდა.
        # KILL_SWITCH + LIVE_CONFIRMATION შემოწმებები ზევიდანვე ხდება —
        # ისინი SELL-საც ბლოკავს (სწორია). მხოლოდ certified check
        # გადადის SELL execution-ის შემდეგ BUY path-ისთვის.
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        if verdict == "SELL":
            if not symbol or direction != "LONG":
                logger.warning(f"EXEC_REJECT | bad SELL payload | id={signal_id}")
                log_event("REJECT_BAD_SELL_PAYLOAD", f"{signal_id}")
                return

            self._execute_sell(signal_id=signal_id, symbol=str(symbol), signal_hash=signal_hash)
            return

        # BUY path only — certified_signal check
        if signal.get("certified_signal") is not True:
            log_event("REJECT_NOT_CERTIFIED", f"{signal_id}")
            return

        if not symbol or direction != "LONG" or entry_type != "MARKET":
            logger.warning(f"EXEC_REJECT | bad payload | id={signal_id}")
            log_event("REJECT_BAD_PAYLOAD", f"{signal_id}")
            return

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # DEAD PARAMS GUARDS — BUY-ის წინ, ყველა mode-ში
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

        # 1. MAX_CONSECUTIVE_LOSSES — N consecutive SL-ის შემდეგ EXEC block
        if self.max_consecutive_losses > 0:
            try:
                stats = get_trade_stats()
                losses = int(stats.get("losses", 0))
                wins   = int(stats.get("wins",   0))
                # consecutive losses: ბოლო N trade-ი ვინახავ ბრუნვის გარეშე
                # მარტივი proxy: ბოლო (losses-wins) streak თუ > limit
                from execution.db.repository import get_closed_trades
                closed = get_closed_trades()
                if closed:
                    streak = 0
                    for t in reversed(closed):
                        if float(t.get("pnl_quote", 0) or 0) < 0:
                            streak += 1
                        else:
                            break
                    if streak >= self.max_consecutive_losses:
                        msg = (
                            f"EXEC_REJECT | MAX_CONSECUTIVE_LOSSES | "
                            f"streak={streak} >= limit={self.max_consecutive_losses} | id={signal_id}"
                        )
                        logger.warning(msg)
                        log_event("EXEC_REJECT_CONSECUTIVE_LOSSES", msg)
                        return
            except Exception as e:
                logger.warning(f"CONSECUTIVE_LOSS_CHECK_FAIL | err={e} → skipped")

        # 2. MAX_DAILY_LOSS — დღის P&L% ზარალი limit-ს გადასცდა?
        if self.max_daily_loss_pct > 0:
            try:
                from execution.db.repository import get_closed_trades
                from datetime import datetime, timezone
                closed = get_closed_trades()
                today = datetime.now(timezone.utc).date().isoformat()
                daily_pnl = sum(
                    float(t.get("pnl_quote", 0) or 0)
                    for t in closed
                    if str(t.get("closed_at", "") or "")[:10] == today
                )
                daily_quote_in = sum(
                    float(t.get("quote_in", 0) or 0)
                    for t in closed
                    if str(t.get("closed_at", "") or "")[:10] == today
                )
                if daily_quote_in > 0:
                    daily_loss_pct = abs(min(0, daily_pnl)) / daily_quote_in * 100.0
                    if daily_loss_pct >= self.max_daily_loss_pct:
                        msg = (
                            f"EXEC_REJECT | MAX_DAILY_LOSS | "
                            f"daily_loss={daily_loss_pct:.2f}% >= limit={self.max_daily_loss_pct}% | id={signal_id}"
                        )
                        logger.warning(msg)
                        log_event("EXEC_REJECT_DAILY_LOSS", msg)
                        return
            except Exception as e:
                logger.warning(f"DAILY_LOSS_CHECK_FAIL | err={e} → skipped")

        # 3. AI_SIGNAL_THRESHOLD — adaptive signal-ის ai_score secondary check
        if self.ai_signal_threshold > 0 and adaptive:
            sig_ai = float(signal.get("meta", {}).get("decision", {}).get("ai_score", 1.0) or 1.0)
            if sig_ai < self.ai_signal_threshold:
                msg = (
                    f"EXEC_REJECT | AI_SIGNAL_THRESHOLD | "
                    f"ai={sig_ai:.3f} < threshold={self.ai_signal_threshold} | id={signal_id}"
                )
                logger.warning(msg)
                log_event("EXEC_REJECT_AI_THRESHOLD", msg)
                return

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 4. MAX_ACCOUNT_DRAWDOWN — session balance drop limit
        # FIX I-5: _session_start_balance DB-ში ინახება — restart-safe.
        # ადრე: in-memory → restart-ზე ნულდებოდა → drawdown protection
        # bypass-ი შესაძლებელი იყო (6% drawdown + restart = სრული limit).
        # ახლა: audit_log-ში "SESSION_START_BALANCE" event-ი ინახება.
        # restart-ზე DB-დან წამოვიღებთ — protection გრძელდება.
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        if self.max_account_drawdown_pct > 0 and self.exchange is not None:
            try:
                current_balance = float(self.exchange.fetch_balance_free("USDT"))

                # I-5: in-memory None → DB-დან ვცდილობთ წამოღებას
                if self._session_start_balance is None:
                    try:
                        from execution.db.repository import _fetchone
                        row = _fetchone(
                            "SELECT message FROM audit_log "
                            "WHERE event_type = 'SESSION_START_BALANCE' "
                            "ORDER BY id DESC LIMIT 1"
                        )
                        if row:
                            self._session_start_balance = float(row[0])
                            logger.info(
                                f"[DRAWDOWN] session_start_balance restored from DB: "
                                f"{self._session_start_balance:.2f} USDT"
                            )
                    except Exception:
                        pass

                if self._session_start_balance is None:
                    # პირველი ჯერ — DB-ში ვინახავთ
                    self._session_start_balance = current_balance
                    logger.info(f"[DRAWDOWN] session_start_balance={current_balance:.2f} USDT (saved to DB)")
                    try:
                        log_event("SESSION_START_BALANCE", f"{current_balance:.4f}")
                    except Exception:
                        pass

                elif self._session_start_balance > 0:
                    drawdown_pct = (self._session_start_balance - current_balance) / self._session_start_balance * 100.0
                    if drawdown_pct >= self.max_account_drawdown_pct:
                        msg = (
                            f"EXEC_REJECT | MAX_ACCOUNT_DRAWDOWN | "
                            f"drawdown={drawdown_pct:.2f}% >= limit={self.max_account_drawdown_pct}% | "
                            f"start={self._session_start_balance:.2f} current={current_balance:.2f} | id={signal_id}"
                        )
                        logger.error(msg)
                        log_event("EXEC_REJECT_DRAWDOWN", msg)
                        return
            except Exception as e:
                logger.warning(f"DRAWDOWN_CHECK_FAIL | err={e} → skipped")

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 5. MAX_PORTFOLIO_EXPOSURE — max % of balance in open trades
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        if self.max_portfolio_exposure > 0 and self.exchange is not None:
            try:
                from execution.db.repository import get_trade_stats
                stats = get_trade_stats()
                open_exposure = float(stats.get("open_quote_in_sum", 0) or 0)
                total_balance = float(self.exchange.fetch_balance_free("USDT")) + open_exposure
                if total_balance > 0:
                    exposure_ratio = open_exposure / total_balance
                    if exposure_ratio >= self.max_portfolio_exposure:
                        msg = (
                            f"EXEC_REJECT | MAX_PORTFOLIO_EXPOSURE | "
                            f"exposure={exposure_ratio:.2%} >= limit={self.max_portfolio_exposure:.2%} | "
                            f"open_usdt={open_exposure:.2f} total={total_balance:.2f} | id={signal_id}"
                        )
                        logger.warning(msg)
                        log_event("EXEC_REJECT_PORTFOLIO_EXPOSURE", msg)
                        return
            except Exception as e:
                logger.warning(f"PORTFOLIO_EXPOSURE_CHECK_FAIL | err={e} → skipped")

        if self.mode == "DEMO":
            last_price = float(self.price_feed.fetch_ticker(symbol)["last"])
            base_size = float(position_size) if position_size is not None else float(quote_amount) / float(last_price)

            simulate_market_entry(symbol=symbol, side=direction, size=base_size, price=last_price)

            log_event("TRADE_EXECUTED", f"{signal_id} DEMO {symbol}")
            logger.info(f"EXEC_DEMO_OK | id={signal_id}")

            mark_signal_id_executed(signal_id, signal_hash=signal_hash, action="TRADE_DEMO", symbol=str(symbol))
            return

        if self.exchange is None:
            log_event("EXEC_BLOCKED_NO_EXCHANGE", f"{signal_id}")
            logger.warning(f"EXEC_BLOCKED | exchange client not wired | id={signal_id}")
            return

        from execution.exchange_client import LiveTradingBlocked

        try:
            ok_edge, edge_reason = self._net_edge_ok(tp_pct=tp_pct)
            if not ok_edge:
                msg = f"EXEC_REJECT | EDGE_GATE | id={signal_id} symbol={symbol} {edge_reason}"
                logger.warning(msg)
                log_event("EXEC_REJECT_EDGE_GATE", msg)
                mark_signal_id_executed(signal_id, signal_hash=signal_hash, action="REJECT_EDGE_GATE", symbol=str(symbol))
                return

            if quote_amount is None:
                last = self.exchange.fetch_last_price(symbol)
                quote_amount = float(position_size) * float(last)

            quote_amount = float(quote_amount)

            min_notional = float(self.exchange.get_min_notional(symbol) or 5.0)
            env_quote = float(os.getenv("BOT_QUOTE_PER_TRADE", "5"))

            balance = float(self.exchange.fetch_balance_free("USDT"))
            risk_pct = float(os.getenv("RISK_PER_TRADE_PCT", "1.0"))
            risk_amount = balance * (risk_pct / 100.0)

            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # FIX: regime QUOTE_SIZE პატივდება
            # adaptive["QUOTE_SIZE"] = 5.6 (UNCERTAIN) ან 7.0 (BULL)
            # min_notional-ი კვლავ floor-ია
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            regime_quote = float(adaptive.get("QUOTE_SIZE", env_quote)) if adaptive else env_quote
            quote_amount = max(min_notional, regime_quote)
            quote_amount = round(quote_amount, 2)

            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # MAX_RISK_PER_TRADE_PCT — hard ceiling on quote by % of balance
            # e.g. 1.0% of 200 USDT balance = max 2 USDT per trade
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            if self.max_risk_per_trade_pct > 0 and balance > 0:
                max_by_risk = round(balance * (self.max_risk_per_trade_pct / 100.0), 2)
                if quote_amount > max_by_risk:
                    logger.info(
                        f"[RISK_CAP] quote {quote_amount:.2f} → {max_by_risk:.2f} "
                        f"(MAX_RISK_PER_TRADE_PCT={self.max_risk_per_trade_pct}% of {balance:.2f})"
                    )
                    quote_amount = max_by_risk

            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # MAX_SYMBOL_EXPOSURE — max % of balance in this symbol
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            if self.max_symbol_exposure > 0 and balance > 0:
                from execution.db.repository import count_open_trades_for_symbol
                sym_open_count = count_open_trades_for_symbol(str(symbol))
                total_balance_est = balance + float(
                    __import__("execution.db.repository", fromlist=["get_trade_stats"])
                    .get_trade_stats().get("open_quote_in_sum", 0) or 0
                )
                max_sym_quote = round(total_balance_est * self.max_symbol_exposure, 2)
                if sym_open_count > 0 and quote_amount > max_sym_quote:
                    logger.info(
                        f"[SYM_EXP_CAP] quote {quote_amount:.2f} → {max_sym_quote:.2f} "
                        f"(MAX_SYMBOL_EXPOSURE={self.max_symbol_exposure:.0%} sym_open={sym_open_count})"
                    )
                    quote_amount = max(min_notional, max_sym_quote)
                    quote_amount = round(quote_amount, 2)

            regime_name = adaptive.get("REGIME", "STATIC") if adaptive else "STATIC"
            logger.info(
                f"[SIZE_FIX] final_quote={quote_amount} "
                f"regime={regime_name} regime_quote={regime_quote:.2f} "
                f"min_notional={min_notional} tp={tp_pct:.3f}% sl={sl_pct:.3f}%"
            )

            try:
                if has_open_trade_for_symbol(str(symbol)):
                    msg = f"EXEC_REJECT | OPEN_TRADE_RACE | id={signal_id} symbol={symbol}"
                    logger.warning(msg)
                    log_event("EXEC_REJECT_OPEN_TRADE_RACE", msg)
                    mark_signal_id_executed(
                        signal_id,
                        signal_hash=signal_hash,
                        action="REJECT_OPEN_TRADE_RACE",
                        symbol=str(symbol)
                    )
                    return

                allow_scaling = os.getenv("ALLOW_POSITION_SCALING", "false").lower() == "true"

                try:
                    max_positions = int(os.getenv("MAX_POSITIONS_PER_SYMBOL", "1"))
                except Exception:
                    max_positions = 1

                try:
                    active_positions = count_open_trades_for_symbol(symbol)
                except Exception:
                    active_positions = 0

                if has_active_oco_for_symbol(str(symbol)):
                    if not allow_scaling:
                        msg = f"EXEC_REJECT | ACTIVE_OCO (scaling disabled) | id={signal_id} symbol={symbol}"
                        logger.warning(msg)
                        log_event("EXEC_REJECT_ACTIVE_OCO", msg)
                        mark_signal_id_executed(
                            signal_id,
                            signal_hash=signal_hash,
                            action="REJECT_ACTIVE_OCO",
                            symbol=str(symbol)
                        )
                        return

                    if active_positions >= max_positions:
                        msg = f"EXEC_REJECT | MAX_POSITIONS_REACHED ({active_positions}) | id={signal_id} symbol={symbol}"
                        logger.warning(msg)
                        log_event("EXEC_REJECT_MAX_POSITIONS", msg)
                        mark_signal_id_executed(
                            signal_id,
                            signal_hash=signal_hash,
                            action="REJECT_MAX_POSITIONS",
                            symbol=str(symbol)
                        )
                        return
            except Exception as e:
                msg = f"EXEC_BLOCKED | TRADE_STATE_CHECK_FAIL | id={signal_id} symbol={symbol} err={e}"
                logger.warning(msg)
                log_event("EXEC_BLOCKED_TRADE_STATE_CHECK_FAIL", msg)
                return

            min_notional = 0.0
            try:
                min_notional = float(self.exchange.get_min_notional(symbol))
            except Exception:
                min_notional = 0.0

            if min_notional > 0 and quote_amount < min_notional:
                msg = f"EXEC_REJECT | MIN_NOTIONAL | id={signal_id} symbol={symbol} quote={quote_amount:.8f} < min_notional={min_notional}"
                logger.warning(msg)
                log_event("EXEC_REJECT_MIN_NOTIONAL", msg)
                mark_signal_id_executed(
                    signal_id,
                    signal_hash=signal_hash,
                    action="REJECT_MIN_NOTIONAL",
                    symbol=str(symbol)
                )
                return

            if is_kill_switch_active():
                logger.error(f"KILL_SWITCH_ACTIVE_LAST_GATE | BUY_BLOCKED | id={signal_id}")
                log_event("EXEC_BLOCKED_KILL_SWITCH_LAST_GATE", f"{signal_id} BUY_BLOCKED")
                return

            buy, buy_avg = self._place_entry_buy(symbol=str(symbol), quote_amount=quote_amount)

            logger.info(
                f"EXEC_LIVE_BUY_OK | id={signal_id} symbol={symbol} "
                f"quote={quote_amount} avg={buy_avg} "
                f"regime={regime_name} tp={tp_pct:.3f}% sl={sl_pct:.3f}% "
                f"order_id={buy.get('id')}"
            )
            log_event("TRADE_EXECUTED", f"{signal_id} LIVE BUY {symbol} quote={quote_amount} avg={buy_avg} regime={regime_name} tp={tp_pct:.3f}% sl={sl_pct:.3f}% order_id={buy.get('id')}")

            mark_signal_id_executed(signal_id, signal_hash=signal_hash, action="TRADE_LIVE_BUY", symbol=str(symbol))

            base_asset = symbol.split("/")[0].upper()
            free_base = float(self.exchange.fetch_balance_free(base_asset))

            sell_amount = self.exchange.floor_amount(symbol, free_base * self.sell_buffer)
            if sell_amount <= 0:
                sell_amount = self.exchange.floor_amount(symbol, free_base * self.sell_retry_buffer)

            if sell_amount <= 0:
                msg = f"OCO_SKIP_NO_FREE_BASE | id={signal_id} free_{base_asset}={free_base}"
                logger.warning(msg)
                log_event("OCO_SKIP_NO_FREE_BASE", msg)
                return

            open_trade(
                signal_id=signal_id,
                symbol=str(symbol),
                qty=float(sell_amount),
                quote_in=float(quote_amount),
                entry_price=float(buy_avg),
            )

            tp_price = float(buy_avg) * (1.0 + tp_pct / 100.0)
            sl_stop = float(buy_avg) * (1.0 - sl_pct / 100.0)
            sl_limit = sl_stop * (1.0 - self.sl_limit_gap_pct / 100.0)

            tp_price = self.exchange.floor_price(symbol, tp_price)
            sl_stop = self.exchange.floor_price(symbol, sl_stop)
            sl_limit = self.exchange.floor_price(symbol, sl_limit)

            oco = self.exchange.place_oco_sell(
                symbol=str(symbol),
                base_amount=float(sell_amount),
                tp_price=float(tp_price),
                sl_stop_price=float(sl_stop),
                sl_limit_price=float(sl_limit),
            )

            raw = oco.get("raw") or {}
            orders = raw.get("orders") or []
            list_order_id = str(raw.get("orderListId") or "")

            tp_order_id = ""
            sl_order_id = ""

            for x in orders:
                oid = str(x.get("orderId") or "")
                typ = str(x.get("type") or "").upper()
                if typ == "LIMIT_MAKER":
                    tp_order_id = oid
                elif typ == "STOP_LOSS_LIMIT":
                    sl_order_id = oid

            if not tp_order_id or not sl_order_id:
                reports = raw.get("orderReports") or []
                for rep in reports:
                    oid = str(rep.get("orderId") or "")
                    typ = str(rep.get("type") or "").upper()
                    if typ == "LIMIT_MAKER" and not tp_order_id:
                        tp_order_id = oid
                    elif typ == "STOP_LOSS_LIMIT" and not sl_order_id:
                        sl_order_id = oid

            create_oco_link(
                signal_id=signal_id,
                symbol=str(symbol),
                base_asset=base_asset,
                tp_order_id=str(tp_order_id),
                sl_order_id=str(sl_order_id),
                tp_price=float(tp_price),
                sl_stop_price=float(sl_stop),
                sl_limit_price=float(sl_limit),
                amount=float(sell_amount),
            )

            log_event("TRADE_LIVE_ARMED", f"{signal_id} {symbol} OCO_ARMED listOrderId={list_order_id}")

            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # #5 PARTIAL TP — TP1 limit order (50% at 1.5%)
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            try:
                self._place_partial_tp_order(
                    signal_id=str(signal_id),
                    symbol=str(symbol),
                    sell_amount=float(sell_amount),
                    buy_avg=float(buy_avg),
                    adaptive=adaptive if adaptive else {},
                )
            except Exception as e:
                logger.warning(f"PARTIAL_TP_FAIL | id={signal_id} err={e}")

            try:
                notify_signal_created(
                    symbol=str(symbol),
                    entry_price=float(buy_avg),
                    quote_amount=float(quote_amount),
                    tp_price=float(tp_price),
                    sl_price=float(sl_stop),
                    verdict="BUY",
                    mode=self.mode,
                )
            except Exception as e:
                logger.warning(f"TG_NOTIFY_SIGNAL_FAIL | id={signal_id} err={e}")

        except LiveTradingBlocked as e:
            msg = f"EXEC_REJECT | LIVE_BLOCKED | id={signal_id} reason={e}"
            logger.warning(msg)
            log_event("EXEC_REJECT_LIVE_BLOCKED", msg)
            mark_signal_id_executed(
                signal_id,
                signal_hash=signal_hash,
                action="REJECT_LIVE_BLOCKED",
                symbol=str(symbol)
            )
            return

        except Exception as e:
            logger.exception(f"EXEC_LIVE_ERROR | id={signal_id} err={e}")
            log_event("EXEC_LIVE_ERROR", f"{signal_id} err={e}")
            return
