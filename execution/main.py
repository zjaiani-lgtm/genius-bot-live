from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
import os

import pandas as pd

from execution.config import Settings
from execution.database import TradeDB
from execution.exchange.base import TokenBucket
from execution.exchange.binance_rest import BinanceSpot
from execution.exchange.bybit_rest import BybitSpot
from execution.exchange.binance_ws import BinanceWS
from execution.exchange.bybit_ws import BybitWS
from execution.ml.signal_model import MLSignalFilter
from execution.portfolio import Portfolio
from execution.risk.manager import RiskManager
from execution.smart_router import SmartRouter
from execution.execution_brain import ExecutionBrain
from execution.position_manager import PositionManager
from execution.strategy.orderbook_alpha import compute_long_signal
from ui.env_override import EnvOverrideBridge

logging.basicConfig(
    level=Settings().LOG_LEVEL,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

log = logging.getLogger("main")


def _ms_to_dt(ms: int) -> datetime:
    return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc)


class Engine:

    def __init__(self, s: Settings) -> None:

        self.s = s
        self.db = TradeDB(s.DB_PATH)
        self.portfolio = Portfolio()

        self.risk = RiskManager(
            position_pct=s.POSITION_PCT,
            stop_atr_mult=s.STOP_ATR_MULT,
            tp_atr_mult=s.TP_ATR_MULT,
            taker_fee=s.TAKER_FEE,
            maker_fee=s.MAKER_FEE,
            slippage_bps=s.SLIPPAGE_BPS,
            partial_tp_pct=s.PARTIAL_TP_PCT,
        )

        self.ml = MLSignalFilter(enabled=s.ML_ENABLED, min_proba=s.ML_MIN_PROBA)
        self.router = SmartRouter()
        self.override = EnvOverrideBridge()

        self.execution_brain = ExecutionBrain(s, self.portfolio)

        self.position_manager = PositionManager(
            tp_pct=0.002,
            sl_pct=0.01,
            max_bars=30,
        )

        self._idx: dict[str, int] = {sym: 0 for sym in s.SYMBOLS}
        self._df15: dict[str, pd.DataFrame] = {}

        self._execution_lock: set[str] = set()

        limiter = TokenBucket(rate_per_sec=s.REST_RATE_PER_SEC, burst=s.REST_BURST)

        if s.EXCHANGE == "binance":

            self.ex = BinanceSpot(
                s.BINANCE_BASE_URL,
                s.BINANCE_API_KEY,
                s.BINANCE_API_SECRET,
                limiter,
            )

            self.ws = BinanceWS(s.BINANCE_WS_URL)

        else:

            self.ex = BybitSpot(
                s.BYBIT_API_KEY,
                s.BYBIT_API_SECRET,
            )

            self.ws = BybitWS(s.BYBIT_WS_URL)

    async def seed_history(self, symbol: str) -> None:

        log.info(f"FETCH_OHLCV_START {symbol}")

        candles = await asyncio.wait_for(
            self.ex.fetch_ohlcv(symbol, self.s.PRIMARY_TF, limit=600),
            timeout=15,
        )

        log.info(f"FETCH_OHLCV_DONE {symbol}")

        df = pd.DataFrame(
            [
                {
                    "ts": _ms_to_dt(c["ts"]),
                    "open": c["open"],
                    "high": c["high"],
                    "low": c["low"],
                    "close": c["close"],
                    "volume": c["volume"],
                }
                for c in candles
            ]
        ).set_index("ts")

        self._df15[symbol] = df

    # ===============================
    # FIXED POSITION SYNC
    # ===============================

    async def sync_positions(self):

        try:

            balances = await self.ex.fetch_balances()

            for asset, amount in balances.items():

                if amount <= 0:
                    continue

                symbol = f"{asset}USDT"

                if symbol not in self.s.SYMBOLS:
                    continue

                price = await self.ex.get_last_price(symbol)

                self.portfolio.sync_position(
                    symbol=symbol,
                    qty=amount,
                    entry_price=price
                )

                log.info(f"SYNC_POSITION {symbol} qty={amount} entry_price={price}")

        except Exception as e:

            log.warning(f"POSITION_SYNC_FAILED {e}")

    async def maybe_open_position(self, symbol: str, idx: int) -> None:

        if symbol in self._execution_lock:
            return

        if self.portfolio.has_position(symbol):
            return

        if self.portfolio.in_cooldown(symbol, idx):
            return

        df15 = self._df15[symbol]

        if len(df15) < 50:
            return

        sig = compute_long_signal(
            df15,
            df15,
            df15,
            self.s.EMA_FAST,
            self.s.EMA_SLOW,
            self.s.RSI_PERIOD,
            self.s.RSI_LONG_MIN,
            self.s.ATR_PERIOD,
        )

        if sig is None or sig.action != "BUY":
            return

        if self.s.ML_ENABLED and not self.ml.allow(sig.features):
            return

        open_positions = self.portfolio.count_open_positions()

        if open_positions >= 5:
            log.info("MAX_POSITIONS_GUARD")
            return

        capital = await self.ex.fetch_usdt_balance()

        log.info(f"DEBUG_BALANCE={capital}")

        if capital < 3:
            log.warning("INSUFFICIENT_CAPITAL")
            return

        max_positions = 2
        remaining_slots = max_positions - open_positions

        if remaining_slots <= 0:
            return

        position_size = capital / remaining_slots

        self._execution_lock.add(symbol)

        try:

            log.info(f"EXECUTION_START {symbol} size={position_size}")

            order = await self.router.open_long(
                self.ex,
                symbol,
                position_size
            )

            if not order:
                log.warning("ORDER_FAILED")
                return

            qty = getattr(order, "executed_qty", None)
            price = getattr(order, "avg_price", None)

            if not qty or not price:
                log.warning("INVALID_ORDER_RESPONSE")
                return

            log.info(f"BUY_EXECUTED {symbol} qty={qty} price={price}")

            self.portfolio.open_position(
                symbol=symbol,
                qty=qty,
                entry_price=price,
                entry_idx=idx,
            )

            oco_ok = False

            for attempt in range(3):

                try:

                    await self.router.place_oco_tp_sl(
                        self.ex,
                        symbol,
                        qty,
                        price
                    )

                    await asyncio.sleep(1)

                    if await self.router.verify_oco(self.ex, symbol):

                        oco_ok = True
                        log.info(f"OCO_VERIFIED {symbol}")
                        break

                except Exception as e:

                    log.warning(f"OCO_ATTEMPT_{attempt} failed {e}")

            if not oco_ok:

                log.critical(f"OCO_FAILED_PROTECTION {symbol}")

                try:

                    await self.router.close_position(
                        self.ex,
                        symbol,
                        qty
                    )

                    log.critical(f"EMERGENCY_CLOSE_EXECUTED {symbol}")

                except Exception as e:

                    log.critical(f"FAILED_EMERGENCY_CLOSE {e}")

        except Exception as e:

            log.exception(f"EXECUTION_FAILED {symbol} {e}")

        finally:

            self._execution_lock.discard(symbol)

    async def run_live(self) -> None:

        await self.db.init()

        await self.sync_positions()

        for sym in self.s.SYMBOLS:
            await self.seed_history(sym)

        while True:

            try:

                async for msg in self.ws.stream_klines(
                    list(self.s.SYMBOLS),
                    self.s.PRIMARY_TF
                ):

                    if not msg.is_closed:
                        continue

                    if msg.symbol not in self._df15:
                        continue

                    df = self._df15[msg.symbol]

                    df.loc[_ms_to_dt(msg.ts)] = {
                        "open": msg.kline.open,
                        "high": msg.kline.high,
                        "low": msg.kline.low,
                        "close": msg.kline.close,
                        "volume": msg.kline.volume,
                    }

                    if len(df) > 1000:
                        df = df.tail(1000)

                    self._df15[msg.symbol] = df

                    self._idx[msg.symbol] += 1

                    override = self.override.read_override()

                    if override.enabled and override.kill_switch:
                        log.warning("GLOBAL KILL SWITCH ACTIVE")
                        continue

                    await self.maybe_open_position(msg.symbol, self._idx[msg.symbol])

                    price = msg.kline.close

                    await self.position_manager.maybe_close_position(
                        self.router,
                        self.ex,
                        self.portfolio,
                        msg.symbol,
                        price,
                        self._idx[msg.symbol],
                    )

            except Exception as e:

                log.error(f"WS_STREAM_CRASH {e}")

                await asyncio.sleep(5)


async def main() -> None:

    s = Settings()
    engine = Engine(s)

    try:

        if (os.getenv("RUN_BACKTEST") or "").strip() == "1":
            return

        await engine.run_live()

    finally:

        try:
            if engine.ws and hasattr(engine.ws, "close"):
                await engine.ws.close()
        except Exception:
            pass

        try:
            if engine.ex and hasattr(engine.ex, "close"):
                await engine.ex.close()
        except Exception:
            pass


if __name__ == "__main__":
    asyncio.run(main())
