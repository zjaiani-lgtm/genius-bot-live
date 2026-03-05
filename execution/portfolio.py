from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

log = logging.getLogger("portfolio")


# ==========================================
# POSITION OBJECT
# ==========================================

@dataclass
class Position:

    symbol: str
    qty: float
    entry_price: float

    entry_time: datetime
    entry_index: int

    best_price: float = 0.0
    trade_id: int = 0
    partial_done: bool = False


# ==========================================
# PORTFOLIO MANAGER
# ==========================================

@dataclass
class Portfolio:

    positions: dict[str, Position] = field(default_factory=dict)

    cooldown_until_ts: dict[str, float] = field(default_factory=dict)

    cooldown_seconds: int = 900   # 15 minutes


    # ======================================
    # POSITION QUERIES
    # ======================================

    def has_position(self, symbol: str) -> bool:
        return symbol in self.positions


    def get(self, symbol: str) -> Optional[Position]:
        return self.positions.get(symbol)


    def count_open_positions(self) -> int:
        return len(self.positions)


    # ======================================
    # OPEN POSITION (BUY EXECUTION)
    # ======================================

    def open_position(
        self,
        symbol: str,
        qty: float,
        entry_price: float,
        entry_idx: int,
    ) -> None:

        if symbol in self.positions:

            log.warning(f"DUPLICATE_POSITION_BLOCKED {symbol}")
            return

        p = Position(
            symbol=symbol,
            qty=qty,
            entry_price=entry_price,
            entry_time=self.now(),
            entry_index=entry_idx,
            best_price=entry_price,
        )

        self.positions[symbol] = p

        log.info(
            f"POSITION_OPENED "
            f"{symbol} qty={qty} entry={entry_price}"
        )


    # ======================================
    # CLOSE POSITION (SELL EXECUTION)
    # ======================================

    def close(self, symbol: str) -> None:

        if symbol not in self.positions:
            return

        del self.positions[symbol]

        self.cooldown_until_ts[symbol] = (
            time.time() + self.cooldown_seconds
        )

        log.info(f"POSITION_REMOVED {symbol}")


    # ======================================
    # COOLDOWN CHECK
    # ======================================

    def in_cooldown(self, symbol: str, idx: int) -> bool:

        until = self.cooldown_until_ts.get(symbol, 0.0)

        if time.time() < until:

            remaining = int(until - time.time())

            log.info(
                f"COOLDOWN_ACTIVE {symbol} "
                f"remaining={remaining}s"
            )

            return True

        return False


    # ======================================
    # SYNC POSITION FROM EXCHANGE
    # ======================================

    def sync_position(
        self,
        symbol: str,
        qty: float,
        entry_price: float,
    ) -> None:

        if symbol in self.positions:
            return

        p = Position(
            symbol=symbol,
            qty=qty,
            entry_price=entry_price,
            entry_time=self.now(),
            entry_index=0,
            best_price=entry_price,
        )

        self.positions[symbol] = p

        log.info(
            f"SYNC_POSITION "
            f"{symbol} qty={qty} entry={entry_price}"
        )


    # ======================================
    # TIME UTILITY
    # ======================================

    @staticmethod
    def now() -> datetime:
        return datetime.now(timezone.utc)
