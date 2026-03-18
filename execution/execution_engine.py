def execute_signal(self, signal: Dict[str, Any]) -> None:
    signal_id = str(signal.get("signal_id", "UNKNOWN"))
    verdict = str(signal.get("final_verdict", "")).upper()

    # 🔥 AUTO ADAPTIVE CONFIG
    adaptive = signal.get("adaptive", {})

    if adaptive:
        logger.info(f"[AUTO] Using adaptive params: {adaptive}")
        tp_pct = float(adaptive.get("TP_PCT", self.tp_pct))
        sl_pct = float(adaptive.get("SL_PCT", self.sl_pct))
    else:
        tp_pct = self.tp_pct
        sl_pct = self.sl_pct

    logger.info(f"EXEC_ENTER | id={signal_id} verdict={verdict} MODE={self.mode} ENV_KILL_SWITCH={self.env_kill_switch}")

    # DEDUP CHECK
    if signal_id_already_executed(signal_id):
        logger.warning(f"EXEC_DEDUPED | duplicate ignored | id={signal_id}")
        log_event("EXEC_DEDUPED", f"id={signal_id}")
        return

    # SYSTEM STATE
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

    if signal.get("certified_signal") is not True:
        log_event("REJECT_NOT_CERTIFIED", f"{signal_id}")
        return

    execution = signal.get("execution") or {}
    symbol = execution.get("symbol")
    direction = str(execution.get("direction", "")).upper()
    entry = execution.get("entry") or {}
    entry_type = str(entry.get("type", "")).upper()

    position_size = execution.get("position_size")
    quote_amount = execution.get("quote_amount")

    signal_hash = signal.get("_fingerprint") or signal.get("signal_hash")

    # SELL FLOW
    if verdict == "SELL":
        if not symbol or direction != "LONG":
            logger.warning(f"EXEC_REJECT | bad SELL payload | id={signal_id}")
            log_event("REJECT_BAD_SELL_PAYLOAD", f"{signal_id}")
            return

        self._execute_sell(signal_id=signal_id, symbol=str(symbol), signal_hash=signal_hash)
        return

    # VALIDATION
    if not symbol or direction != "LONG" or entry_type != "MARKET":
        logger.warning(f"EXEC_REJECT | bad payload | id={signal_id}")
        log_event("REJECT_BAD_PAYLOAD", f"{signal_id}")
        return

    # DEMO MODE
    if self.mode == "DEMO":
        last_price = float(self.price_feed.fetch_ticker(symbol)["last"])
        base_size = float(position_size) if position_size is not None else float(quote_amount) / float(last_price)

        simulate_market_entry(symbol=symbol, side=direction, size=base_size, price=last_price)

        log_event("TRADE_EXECUTED", f"{signal_id} DEMO {symbol}")
        logger.info(f"EXEC_DEMO_OK | id={signal_id}")

        mark_signal_id_executed(signal_id, signal_hash=signal_hash, action="TRADE_DEMO", symbol=str(symbol))
        return

    # EXCHANGE CHECK
    if self.exchange is None:
        log_event("EXEC_BLOCKED_NO_EXCHANGE", f"{signal_id}")
        logger.warning(f"EXEC_BLOCKED | exchange client not wired | id={signal_id}")
        return

    from execution.exchange_client import LiveTradingBlocked

    try:
        ok_edge, edge_reason = self._net_edge_ok()
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

        # ADAPTIVE SIZE
        if adaptive:
            quote_amount = float(adaptive.get("QUOTE_SIZE", quote_amount))
            logger.info(f"[AUTO] Using adaptive quote: {quote_amount}")

        # STATE CHECKS
        if has_open_trade_for_symbol(str(symbol)):
            msg = f"EXEC_REJECT | OPEN_TRADE_RACE | id={signal_id} symbol={symbol}"
            logger.warning(msg)
            log_event("EXEC_REJECT_OPEN_TRADE_RACE", msg)
            return

        if has_active_oco_for_symbol(str(symbol)):
            msg = f"EXEC_REJECT | ACTIVE_OCO_RACE | id={signal_id} symbol={symbol}"
            logger.warning(msg)
            log_event("EXEC_REJECT_ACTIVE_OCO_RACE", msg)
            return

        if is_kill_switch_active():
            logger.error(f"KILL_SWITCH_ACTIVE_LAST_GATE | BUY_BLOCKED | id={signal_id}")
            log_event("EXEC_BLOCKED_KILL_SWITCH_LAST_GATE", f"{signal_id}")
            return

        # BUY
        buy, buy_avg = self._place_entry_buy(symbol=str(symbol), quote_amount=quote_amount)

        logger.info(f"EXEC_LIVE_BUY_OK | id={signal_id} symbol={symbol} avg={buy_avg}")
        log_event("TRADE_EXECUTED", f"{signal_id} LIVE BUY {symbol}")

        mark_signal_id_executed(signal_id, signal_hash=signal_hash, action="TRADE_LIVE_BUY", symbol=str(symbol))

    except LiveTradingBlocked as e:
        log_event("EXEC_REJECT_LIVE_BLOCKED", f"{signal_id}")
        return

    except Exception as e:
        logger.exception(f"EXEC_LIVE_ERROR | id={signal_id} err={e}")
        log_event("EXEC_LIVE_ERROR", f"{signal_id}")
        return
