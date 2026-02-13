
import sqlite3
from datetime import datetime

class TradeLogger:
    def __init__(self, db_path="trades.db"):
        self.conn = sqlite3.connect(db_path)
        self._create()

    def _create(self):
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS trades ("
            "time TEXT, symbol TEXT, side TEXT, amount REAL, status TEXT)"
        )
        self.conn.commit()

    def log(self, order):
        self.conn.execute(
            "INSERT INTO trades VALUES (?,?,?,?,?)",
            (datetime.utcnow().isoformat(),
             order.get("symbol"),
             order.get("side"),
             order.get("amount"),
             order.get("status"))
        )
        self.conn.commit()
