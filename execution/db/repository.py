from datetime import datetime
from typing import Optional, Dict, Any, Tuple
from execution.db.db import get_connection

# ---------------- SYSTEM STATE ----------------

def get_system_state():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM system_state WHERE id = 1")
    row = cur.fetchone()
    conn.close()
    return row

def update_system_state(status=None, startup_sync_ok=None, kill_switch=None):
    conn = get_connection()
    cur = conn.cursor()

    fields = []
    values = []

    if status is not None:
        fields.append("status = ?")
        values.append(status)

    if startup_sync_ok is not None:
        fields.append("startup_sync_ok = ?")
        values.append(int(startup_sync_ok))

    if kill_switch is not None:
        fields.append("kill_switch = ?")
        values.append(int(kill_switch))

    fields.append("updated_at = ?")
    values.append(datetime.utcnow().isoformat())

    sql = f"UPDATE system_state SET {', '.join(fields)} WHERE id = 1"
    cur.execute(sql, values)

    conn.commit()
    conn.close()

# ---------------- POSITIONS (legacy) ----------------

def get_open_positions():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM positions WHERE status = 'OPEN'")
    rows = cur.fetchall()
    conn.close()
    return rows

def get_latest_open_position(symbol: str):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT id, symbol, side, size, entry_price, status, opened_at, closed_at, pnl
        FROM positions
        WHERE status = 'OPEN' AND symbol = ?
        ORDER BY id DESC
        LIMIT 1
        """,
        (symbol,)
    )
    row = cur.fetchone()
    conn.close()
    return row

def open_position(symbol, side, size, entry_price):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO positions
        (symbol, side, size, entry_price, status, opened_at)
        VALUES (?, ?, ?, ?, 'OPEN', ?)
        """,
        (symbol, side, float(size), float(entry_price), datetime.utcnow().isoformat())
    )
    conn.commit()
    conn.close()

def close_position(position_id: int, close_price: float, pnl: float):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        UPDATE positions
        SET status='CLOSED', closed_at=?, pnl=?
        WHERE id=?
        """,
        (datetime.utcnow().isoformat(), float(pnl), int(position_id))
    )
    conn.commit()
    conn.close()

# ---------------- AUDIT LOG ----------------

def log_event(event_type, message):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO audit_log (event_type, message, created_at)
        VALUES (?, ?, ?)
        """,
        (event_type, message, datetime.utcnow().isoformat())
    )
    conn.commit()
    conn.close()

# ---------------- OCO LINKS ----------------

def create_oco_link(
    signal_id: str,
    symbol: str,
    base_asset: str,
    tp_order_id: str,
    sl_order_id: str,
    tp_price: float,
    sl_stop_price: float,
    sl_limit_price: float,
    amount: float,
):
    conn = get_connection()
    cur = conn.cursor()
    now = datetime.utcnow().isoformat()
    cur.execute(
        """
        INSERT INTO oco_links
        (signal_id, symbol, base_asset, tp_order_id, sl_order_id, tp_price, sl_stop_price, sl_limit_price, amount, status, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'ACTIVE', ?, ?)
        """,
        (
            signal_id, symbol, base_asset,
            str(tp_order_id), str(sl_order_id),
            float(tp_price), float(sl_stop_price), float(sl_limit_price),
            float(amount),
            now, now
        )
    )
    conn.commit()
    conn.close()

def set_oco_status(link_id: int, status: str):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        UPDATE oco_links
        SET status=?, updated_at=?
        WHERE id=?
        """,
        (status, datetime.utcnow().isoformat(), int(link_id))
    )
    conn.commit()
    conn.close()

def list_active_oco_links(limit: int = 50):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT id, signal_id, symbol, base_asset, tp_order_id, sl_order_id, tp_price, sl_stop_price, sl_limit_price, amount, status, created_at, updated_at
        FROM oco_links
        WHERE status='ACTIVE'
        ORDER BY id DESC
        LIMIT ?
        """,
        (int(limit),)
    )
    rows = cur.fetchall()
    conn.close()
    return rows

def has_active_oco_for_symbol(symbol: str) -> bool:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT 1
        FROM oco_links
        WHERE status='ACTIVE' AND UPPER(symbol)=UPPER(?)
        LIMIT 1
        """,
        (str(symbol),)
    )
    row = cur.fetchone()
    conn.close()
    return row is not None

# ---------------- EXECUTED SIGNALS (IDEMPOTENCY) ----------------

def signal_id_already_executed(signal_id: str) -> bool:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT 1 FROM executed_signals WHERE signal_id = ? LIMIT 1",
        (str(signal_id),)
    )
    row = cur.fetchone()
    conn.close()
    return row is not None

def mark_signal_id_executed(signal_id: str, signal_hash: str = None, action: str = None, symbol: str = None):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT OR IGNORE INTO executed_signals (signal_id, signal_hash, action, symbol, executed_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            str(signal_id),
            str(signal_hash) if signal_hash is not None else None,
            str(action) if action is not None else None,
            str(symbol) if symbol is not None else None,
            datetime.utcnow().isoformat()
        )
    )
    conn.commit()
    conn.close()

# ---------------- TRADES (NEW) ----------------

def open_trade(signal_id: str, symbol: str, qty: float, quote_in: float, entry_price: float):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT OR REPLACE INTO trades
        (signal_id, symbol, qty, quote_in, entry_price, opened_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            str(signal_id),
            str(symbol),
            float(qty),
            float(quote_in),
            float(entry_price),
            datetime.utcnow().isoformat()
        )
    )
    conn.commit()
    conn.close()

def get_trade(signal_id: str) -> Optional[Tuple]:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT signal_id, symbol, qty, quote_in, entry_price, opened_at, exit_price, closed_at, outcome, pnl_quote, pnl_pct
        FROM trades
        WHERE signal_id = ?
        LIMIT 1
        """,
        (str(signal_id),)
    )
    row = cur.fetchone()
    conn.close()
    return row

def close_trade(signal_id: str, exit_price: float, outcome: str, pnl_quote: float, pnl_pct: float):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        UPDATE trades
        SET exit_price=?, closed_at=?, outcome=?, pnl_quote=?, pnl_pct=?
        WHERE signal_id=?
        """,
        (
            float(exit_price),
            datetime.utcnow().isoformat(),
            str(outcome),
            float(pnl_quote),
            float(pnl_pct),
            str(signal_id),
        )
    )
    conn.commit()
    conn.close()

def get_trade_stats() -> Dict[str, Any]:
    conn = get_connection()
    cur = conn.cursor()

    # only closed trades
    cur.execute("SELECT COUNT(*), COALESCE(SUM(pnl_quote),0), COALESCE(SUM(quote_in),0) FROM trades WHERE closed_at IS NOT NULL")
    n, pnl_sum, quote_sum = cur.fetchone()

    cur.execute("SELECT COUNT(*) FROM trades WHERE closed_at IS NOT NULL AND pnl_quote > 0")
    wins = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM trades WHERE closed_at IS NOT NULL AND pnl_quote < 0")
    losses = cur.fetchone()[0]

    cur.execute("SELECT COALESCE(SUM(pnl_quote),0) FROM trades WHERE closed_at IS NOT NULL AND pnl_quote > 0")
    gross_profit = cur.fetchone()[0]

    cur.execute("SELECT COALESCE(ABS(SUM(pnl_quote)),0) FROM trades WHERE closed_at IS NOT NULL AND pnl_quote < 0")
    gross_loss = cur.fetchone()[0]

    conn.close()

    roi_pct = (float(pnl_sum) / float(quote_sum) * 100.0) if quote_sum else 0.0
    winrate = (float(wins) / float(n) * 100.0) if n else 0.0
    profit_factor = (float(gross_profit) / float(gross_loss)) if gross_loss else (float('inf') if gross_profit > 0 else 0.0)

    return {
        "closed_trades": int(n),
        "wins": int(wins),
        "losses": int(losses),
        "winrate_pct": float(winrate),
        "pnl_quote_sum": float(pnl_sum),
        "quote_in_sum": float(quote_sum),
        "roi_pct": float(roi_pct),
        "gross_profit": float(gross_profit),
        "gross_loss": float(gross_loss),
        "profit_factor": float(profit_factor),
    }
