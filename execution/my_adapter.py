# execution/my_adapter.py
# ============================================================
# გასწორებული adapter — execution_engine.py-ის _SafeAdapter-ის
# მსგავსად, რეალური repository და BinanceSpotClient methods-ებით
# ============================================================

from execution.diagnostics_pro import Adapter
from execution.db.repository import (
    get_trade,
    list_active_oco_links,
    get_closed_trades,
    get_trade_stats,
)
from execution.db.db import get_connection

import logging
import time

logger = logging.getLogger("gbm")


class MyAdapter(Adapter):
    def __init__(self, exchange=None, signal_id: str = "", symbol: str = ""):
        self.exchange = exchange      # BinanceSpotClient ან None (DEMO)
        self.signal_id = signal_id
        self.symbol = symbol

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # DB LAYER
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def get_trade(self, signal_id) -> dict:
        """repository.get_trade() → tuple → dict"""
        row = get_trade(str(signal_id))
        if not row:
            return {}
        return {
            "signal_id":   row[0],
            "symbol":      row[1],
            "qty":         row[2],
            "quote_in":    row[3],
            "entry_price": row[4],
            "opened_at":   row[5],
            "exit_price":  row[6],
            "closed_at":   row[7],
            "outcome":     row[8],
            "pnl_quote":   row[9],
            "pnl_pct":     row[10],
            "status": f"CLOSED_{str(row[8]).upper()}" if row[7] else "OPEN",
        }

    def get_oco_status(self, link_id) -> str:
        """oco_links ცხრილიდან status-ის წაკითხვა"""
        if link_id is None:
            return ""
        try:
            conn = get_connection()
            row = conn.execute(
                "SELECT status FROM oco_links WHERE id=?", (int(link_id),)
            ).fetchone()
            conn.close()
            return str(row[0]) if row else ""
        except Exception as e:
            logger.warning(f"get_oco_status err: {e}")
            return ""

    def get_close_events_count(self, signal_id) -> int:
        """რამდენჯერ დაიხურა ეს trade — race condition check"""
        try:
            conn = get_connection()
            row = conn.execute(
                "SELECT COUNT(*) FROM trades WHERE signal_id=? AND closed_at IS NOT NULL",
                (str(signal_id),)
            ).fetchone()
            conn.close()
            return int(row[0]) if row else 0
        except Exception as e:
            logger.warning(f"get_close_events_count err: {e}")
            return 0

    def get_trade_logs(self, signal_id) -> list:
        """audit_log-დან ამ signal_id-ის ჩანაწერები"""
        try:
            conn = get_connection()
            rows = conn.execute(
                "SELECT event_type FROM audit_log WHERE message LIKE ? ORDER BY id",
                (f"%{signal_id}%",)
            ).fetchall()
            conn.close()
            return [r[0] for r in rows]
        except Exception as e:
            logger.warning(f"get_trade_logs err: {e}")
            return []

    def get_open_trades(self) -> list:
        """ყველა ღია trade + oco link_id"""
        try:
            conn = get_connection()
            rows = conn.execute("""
                SELECT t.signal_id, t.symbol, t.qty, t.quote_in,
                       t.entry_price, o.id as link_id
                FROM trades t
                LEFT JOIN oco_links o
                  ON t.signal_id = o.signal_id
                 AND o.status IN ('ACTIVE','OPEN','ARMED')
                WHERE t.closed_at IS NULL
            """).fetchall()
            conn.close()
            return [
                {
                    "signal_id":   r[0],
                    "symbol":      r[1],
                    "qty":         r[2],
                    "quote_in":    r[3],
                    "entry_price": r[4],
                    "link_id":     r[5],
                }
                for r in rows
            ]
        except Exception as e:
            logger.warning(f"get_open_trades err: {e}")
            return []

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # EXCHANGE LAYER
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def get_order(self, order_id) -> dict:
        """BinanceSpotClient.fetch_order()"""
        if not order_id or self.exchange is None:
            return None
        try:
            return self.exchange.fetch_order(str(order_id), str(self.symbol))
        except Exception as e:
            logger.warning(f"get_order err: {e}")
            return None

    def get_fills(self, order_id) -> list:
        """Spot-ზე individual fills API არ არის — ცარიელი list"""
        return []

    def get_position(self, symbol) -> dict:
        """Spot trading — position ყოველთვის 0 (no margin)"""
        return {"qty": 0, "positionAmt": 0}

    def get_balance(self) -> dict:
        """USDT თავისუფალი ბალანსი"""
        if self.exchange is None:
            return {}
        try:
            usdt = self.exchange.fetch_balance_free("USDT")
            return {"USDT": float(usdt)}
        except Exception as e:
            logger.warning(f"get_balance err: {e}")
            return {}

    def get_fee_rate(self, symbol) -> float:
        """Binance Spot taker fee ~0.1%"""
        return 0.001

    def get_latency_ms(self) -> int:
        """Ping test — exchange არ გვაქვს, 0 ვაბრუნებთ"""
        return 0
