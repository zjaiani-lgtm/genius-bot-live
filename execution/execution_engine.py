# (შემოკლებული header არ შევცვალე — შენთან სწორია)

class ExecutionEngine:
    def __init__(self):
        self.mode = os.getenv("MODE", "DEMO").upper()
        self.env_kill_switch = os.getenv("KILL_SWITCH", "false").lower() == "true"
        self.live_confirmation = os.getenv("LIVE_CONFIRMATION", "false").lower() == "true"

        self.price_feed = ccxt.binance({"enableRateLimit": True})

        self.exchange = None
        if self.mode in ("LIVE", "TESTNET"):
            from execution.exchange_client import BinanceSpotClient
            self.exchange = BinanceSpotClient()

        self.tp_pct = float(os.getenv("TP_PCT", "1.30"))
        self.sl_pct = float(os.getenv("SL_PCT", "0.70"))
        self.sl_limit_gap_pct = float(os.getenv("SL_LIMIT_GAP_PCT", "0.15"))

    # ==============================
    # ✅ MAIN EXECUTION FUNCTION FIXED
    # ==============================

    def execute_signal(self, signal: Dict[str, Any]) -> None:
        signal_id = str(signal.get("signal_id", "UNKNOWN"))
        verdict = str(signal.get("final_verdict", "")).upper()

        # 🔥 ADAPTIVE CONFIG (FIXED)
        adaptive = signal.get("adaptive", {})

        if adaptive:
            logger.info(f"[AUTO] Using adaptive params: {adaptive}")
            tp_pct = float(adaptive.get("TP_PCT", self.tp_pct))
            sl_pct = float(adaptive.get("SL_PCT", self.sl_pct))
        else:
            tp_pct = self.tp_pct
            sl_pct = self.sl_pct

        logger.info(
            f"EXEC_ENTER | id={signal_id} verdict={verdict} MODE={self.mode}"
        )

        # =========================
        # ✅ DEDUP CHECK
        # =========================
        if signal_id_already_executed(signal_id):
            logger.warning(f"EXEC_DEDUPED | id={signal_id}")
            log_event("EXEC_DEDUPED", f"{signal_id}")
            return

        # =========================
        # ✅ SYSTEM STATE
        # =========================
        state = self._load_system_state()
        db_status = str(state.get("status") or "").upper()
        db_kill = bool(state.get("kill_switch"))
        sync_ok = bool(state.get("startup_sync_ok"))

        if self.env_kill_switch or db_kill:
            logger.warning(f"KILL_SWITCH | id={signal_id}")
            log_event("BLOCK_KILL_SWITCH", f"{signal_id}")
            return

        if not sync_ok or db_status not in ("ACTIVE", "RUNNING"):
            logger.warning(f"SYSTEM_BLOCK | id={signal_id}")
            log_event("BLOCK_SYSTEM", f"{signal_id}")
            return

        if self.mode == "LIVE" and not self.live_confirmation:
            log_event("BLOCK_LIVE_CONFIRMATION", f"{signal_id}")
            return

        if signal.get("certified_signal") is not True:
            log_event("REJECT_NOT_CERTIFIED", f"{signal_id}")
            return

        # =========================
        # ✅ PARSE SIGNAL
        # =========================
        execution = signal.get("execution") or {}
        symbol = execution.get("symbol")
        direction = str(execution.get("direction", "")).upper()
        entry = execution.get("entry") or {}
        entry_type = str(entry.get("type", "")).upper()

        position_size = execution.get("position_size")
        quote_amount = execution.get("quote_amount")

        signal_hash = signal.get("_fingerprint") or signal.get("signal_hash")

        # =========================
        # ✅ SELL FLOW
        # =========================
        if verdict == "SELL":
            if not symbol or direction != "LONG":
                log_event("BAD_SELL", f"{signal_id}")
                return

            self._execute_sell(signal_id, symbol, signal_hash)
            return

        # =========================
        # ✅ VALIDATION
        # =========================
        if not symbol or direction != "LONG" or entry_type != "MARKET":
            log_event("BAD_PAYLOAD", f"{signal_id}")
            return

        # =========================
        # ✅ DEMO MODE
        # =========================
        if self.mode == "DEMO":
            last_price = float(self.price_feed.fetch_ticker(symbol)["last"])

            if position_size:
                base_size = float(position_size)
            else:
                base_size = float(quote_amount) / last_price

            simulate_market_entry(symbol, direction, base_size, last_price)

            log_event("DEMO_TRADE", f"{signal_id}")
            mark_signal_id_executed(signal_id, signal_hash=signal_hash)
            return

        # =========================
        # ✅ EXCHANGE CHECK
        # =========================
        if self.exchange is None:
            log_event("NO_EXCHANGE", f"{signal_id}")
            return

        try:
            # QUOTE SIZE
            if quote_amount is None:
                last = self.exchange.fetch_last_price(symbol)
                quote_amount = float(position_size) * float(last)

            quote_amount = float(quote_amount)

            if adaptive:
                quote_amount = float(adaptive.get("QUOTE_SIZE", quote_amount))

            # BUY
            buy, buy_avg = self._place_entry_buy(symbol, quote_amount)

            log_event("BUY_OK", f"{signal_id}")

            mark_signal_id_executed(signal_id, signal_hash=signal_hash)

            # TP / SL
            tp_price = buy_avg * (1 + tp_pct / 100)
            sl_price = buy_avg * (1 - sl_pct / 100)

            logger.info(
                f"TRADE_OPENED | {symbol} entry={buy_avg} tp={tp_price} sl={sl_price}"
            )

        except Exception as e:
            logger.exception(f"EXEC_ERROR | id={signal_id} err={e}")
            log_event("EXEC_ERROR", f"{signal_id}")
            return
