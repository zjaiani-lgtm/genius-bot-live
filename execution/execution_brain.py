
import time
import logging

logger = logging.getLogger("execution_brain")


class ExecutionBrain:
    """
    Adaptive Market Matrix / Execution Brain

    Responsibilities:
    - Trade rate limiting
    - Symbol cooldown protection
    - Exposure-aware trade decisions
    - Regime-aware position sizing
    """

    def __init__(self, config, portfolio):

        self.config = config
        self.portfolio = portfolio

        # symbol -> last trade timestamp
        self.symbol_cooldowns = {}

        # list of timestamps of recent trades
        self.trade_timestamps = []

    # ------------------------------------------------
    # MAIN ENTRY POINT
    # ------------------------------------------------

    def evaluate_trade(
        self,
        symbol,
        signal,
        signal_score,
        regime,
    ):

        now = time.time()

        if signal != "BUY":
            return None

        # 1️⃣ Global trade throttle
        if not self._check_trade_rate(now):
            logger.info("EXEC_BRAIN_THROTTLE")
            return None

        # 2️⃣ Symbol cooldown
        if self._symbol_cooldown_active(symbol, now):
            logger.info(f"EXEC_BRAIN_SYMBOL_COOLDOWN {symbol}")
            return None

        # 3️⃣ Portfolio exposure
        exposure = self._get_exposure()

        # 4️⃣ Decision matrix
        size_multiplier = self._matrix_decision(
            regime,
            signal_score,
            exposure
        )

        if size_multiplier is None:
            logger.info("EXEC_BRAIN_REJECTED")
            return None

        # 5️⃣ Register trade
        self._register_trade(symbol, now)

        logger.info(
            f"EXEC_BRAIN_APPROVED symbol={symbol} "
            f"size_mult={size_multiplier} "
            f"regime={regime} "
            f"score={signal_score:.2f} "
            f"exposure={exposure:.2f}"
        )

        return {
            "symbol": symbol,
            "size_multiplier": size_multiplier,
        }

    # ------------------------------------------------
    # TRADE RATE CONTROL
    # ------------------------------------------------

    def _check_trade_rate(self, now):

        window = self.config.MAX_TRADES_WINDOW
        limit = self.config.MAX_TRADES_PER_WINDOW

        self.trade_timestamps = [
            t for t in self.trade_timestamps
            if now - t < window
        ]

        if len(self.trade_timestamps) >= limit:
            return False

        return True

    # ------------------------------------------------
    # SYMBOL COOLDOWN
    # ------------------------------------------------

    def _symbol_cooldown_active(self, symbol, now):

        cooldown = self.config.SYMBOL_COOLDOWN

        last_trade = self.symbol_cooldowns.get(symbol)

        if last_trade is None:
            return False

        return (now - last_trade) < cooldown

    # ------------------------------------------------
    # PORTFOLIO EXPOSURE
    # ------------------------------------------------

    def _get_exposure(self):

        try:
            return self.portfolio.current_exposure()
        except Exception:
            # fallback if portfolio object doesn't yet implement exposure
            return 0.0

    # ------------------------------------------------
    # DECISION MATRIX
    # ------------------------------------------------

def _matrix_decision(self, regime, score, exposure):

    # Bull market
    if regime == "BULL":

        if score > 75 and exposure < 0.4:
            return 1.0

        if score > 70 and exposure < 0.6:
            return 0.6


    # Neutral market
    if regime == "NEUTRAL":

        if score > 65 and exposure < 0.5:
            return 0.5


    # Range market
    if regime == "RANGE":

        if score > 80 and exposure < 0.5:
            return 0.5


    # Bear market
    if regime == "BEAR":

        if score > 85 and exposure < 0.2:
            return 0.3


    return None

    # ------------------------------------------------
    # REGISTER TRADE
    # ------------------------------------------------

    def _register_trade(self, symbol, now):

        self.trade_timestamps.append(now)
        self.symbol_cooldowns[symbol] = now
