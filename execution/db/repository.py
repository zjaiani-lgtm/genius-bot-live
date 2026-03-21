import sqlite3
import time
from typing import Any, Dict, List, Optional, Tuple

from execution.db.db import get_connection


def _fetchone(query: str, params: Tuple = ()) -> Optional[Tuple]:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(query, params)
    row = cur.fetchone()
    conn.close()
    return row


def _fetchall(query: str, params: Tuple = ()) -> List[Tuple]:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(query, params)
    rows = cur.fetchall()
    conn.close()
    return rows


def _execute(query: str, params: Tuple = ()) -> None:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(query, params)
    conn.commit()
    conn.close()


# -----------------------
# audit log
# -----------------------
def log_event(event_type: str, message: str) -> None:
    _execute(
        "INSERT INTO audit_log (event_type, message, created_at) VALUES (?, ?, datetime('now'))",
        (str(event_type), str(message)),
    )


# -----------------------
# system state
# -----------------------
def get_system_state():
    return _fetchone("SELECT * FROM system_state WHERE id = 1")


def update_system_state(
    status: Optional[str] = None,
    startup_sync_ok: Optional[int] = None,
    kill_switch: Optional[int] = None
) -> None:
    fields = []
    params = []

    if status is not None:
        fields.append("status = ?")
        params.append(str(status))
    if startup_sync_ok is not None:
        fields.append("startup_sync_ok = ?")
        params.append(int(startup_sync_ok))
    if kill_switch is not None:
        fields.append("kill_switch = ?")
        params.append(int(kill_switch))

    if not fields:
        return

    fields.append("updated_at = datetime('now')")
    q = "UPDATE system_state SET " + ", ".join(fields) + " WHERE id = 1"
    _execute(q, tuple(params))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SL COOLDOWN — DB-based (restart-safe)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def get_sl_cooldown_state() -> Dict[str, Any]:
    """
    DB-დან წაიკითხავს consecutive_sl და sl_pause_until.
    restart-ზეც შენარჩუნდება!

    Returns:
        {
            "consecutive_sl": int,
            "sl_pause_until": float | None  (unix timestamp)
        }
    """
    row = _fetchone(
        "SELECT consecutive_sl, sl_pause_until FROM system_state WHERE id = 1"
    )
    if not row:
        return {"consecutive_sl": 0, "sl_pause_until": None}

    consecutive_sl = int(row[0] or 0)
    sl_pause_raw = row[1]

    # sl_pause_until — UTC ISO string ან None
    sl_pause_until: Optional[float] = None
    if sl_pause_raw:
        try:
            from datetime import datetime, timezone
            dt = datetime.fromisoformat(str(sl_pause_raw).replace("Z", "+00:00"))
            sl_pause_until = dt.timestamp()
        except Exception:
            sl_pause_until = None

    return {
        "consecutive_sl": consecutive_sl,
        "sl_pause_until": sl_pause_until,
    }


def increment_consecutive_sl(pause_seconds: int = 1800) -> int:
    """
    SL hit-ზე გამოიძახება.
    consecutive_sl += 1
    თუ limit-ს მიაღწია → sl_pause_until = now + pause_seconds

    Returns: ახალი consecutive_sl მნიშვნელობა
    """
    state = get_sl_cooldown_state()
    new_count = state["consecutive_sl"] + 1

    from datetime import datetime, timezone, timedelta
    pause_until_iso: Optional[str] = None

    # SL limit-ი signal_generator.py-ის ENV-დან (default=2)
    import os
    limit = int(os.getenv("SL_COOLDOWN_AFTER_N", "2"))

    if new_count >= limit:
        pause_dt = datetime.now(timezone.utc) + timedelta(seconds=pause_seconds)
        pause_until_iso = pause_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    _execute(
        """
        UPDATE system_state
        SET consecutive_sl   = ?,
            sl_pause_until   = ?,
            updated_at       = datetime('now')
        WHERE id = 1
        """,
        (new_count, pause_until_iso),
    )
    return new_count


def reset_consecutive_sl() -> None:
    """
    TP hit ან recovery pass-ზე გამოიძახება.
    consecutive_sl = 0, sl_pause_until = NULL
    """
    _execute(
        """
        UPDATE system_state
        SET consecutive_sl = 0,
            sl_pause_until = NULL,
            updated_at     = datetime('now')
        WHERE id = 1
        """
    )


def is_sl_pause_active() -> bool:
    """
    True თუ SL პაუზა ჯერ კიდევ აქტიურია.
    signal_generator.py-ი ამ ფუნქციას იყენებს cooldown-ის ნაცვლად.
    """
    state = get_sl_cooldown_state()
    pause_until = state.get("sl_pause_until")
    if pause_until is None:
        return False
    return time.time() < pause_until


# -----------------------
# executed signals
# -----------------------
def signal_id_already_executed(signal_id: str) -> bool:
    row = _fetchone(
        "SELECT signal_id FROM executed_signals WHERE signal_id = ?",
        (str(signal_id),)
    )
    return row is not None


def mark_signal_id_executed(
    signal_id: str,
    signal_hash: Optional[str] = None,
    action: str = "",
    symbol: str = ""
) -> None:
    _execute(
        """
        INSERT OR REPLACE INTO executed_signals
        (signal_id, signal_hash, action, symbol, executed_at)
        VALUES (?, ?, ?, ?, datetime('now'))
        """,
        (
            str(signal_id),
            str(signal_hash) if signal_hash else None,
            str(action),
            str(symbol),
        ),
    )


# -----------------------
# OCO links
# -----------------------
def list_active_oco_links(limit: int = 50) -> List[Tuple]:
    return _fetchall(
        """
        SELECT id, signal_id, symbol, base_asset, tp_order_id, sl_order_id,
               tp_price, sl_stop_price, sl_limit_price, amount, status, created_at, updated_at
        FROM oco_links
        WHERE status IN ('ACTIVE', 'OPEN', 'ARMED')
        ORDER BY id DESC
        LIMIT ?
        """,
        (int(limit),),
    )


def set_oco_status(link_id: int, status: str) -> None:
    _execute(
        "UPDATE oco_links SET status = ?, updated_at = datetime('now') WHERE id = ?",
        (str(status), int(link_id)),
    )


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
) -> None:
    _execute(
        """
        INSERT INTO oco_links (
            signal_id, symbol, base_asset, tp_order_id, sl_order_id,
            tp_price, sl_stop_price, sl_limit_price, amount,
            status, created_at, updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'ACTIVE', datetime('now'), datetime('now'))
        """,
        (
            str(signal_id),
            str(symbol),
            str(base_asset),
            str(tp_order_id),
            str(sl_order_id),
            float(tp_price),
            float(sl_stop_price),
            float(sl_limit_price),
            float(amount),
        ),
    )


def has_active_oco_for_symbol(symbol: str) -> bool:
    row = _fetchone(
        """
        SELECT id FROM oco_links
        WHERE UPPER(symbol) = UPPER(?)
          AND status IN ('ACTIVE', 'OPEN', 'ARMED')
        LIMIT 1
        """,
        (str(symbol),),
    )
    return row is not None


# -----------------------
# trades
# -----------------------
def has_open_trade_for_symbol(symbol: str) -> bool:
    row = _fetchone(
        """
        SELECT signal_id
        FROM trades
        WHERE UPPER(symbol) = UPPER(?)
          AND closed_at IS NULL
        LIMIT 1
        """,
        (str(symbol),),
    )
    return row is not None


def count_open_trades_for_symbol(symbol: str) -> int:
    row = _fetchone(
        """
        SELECT COUNT(*)
        FROM trades
        WHERE UPPER(symbol) = UPPER(?)
          AND closed_at IS NULL
        """,
        (str(symbol),),
    )
    return int(row[0] or 0) if row else 0


def get_open_trade_for_symbol(symbol: str):
    return _fetchone(
        """
        SELECT signal_id, symbol, qty, quote_in, entry_price, opened_at,
               exit_price, closed_at, outcome, pnl_quote, pnl_pct
        FROM trades
        WHERE UPPER(symbol) = UPPER(?)
          AND closed_at IS NULL
        ORDER BY opened_at DESC
        LIMIT 1
        """,
        (str(symbol),),
    )


def open_trade(signal_id: str, symbol: str, qty: float, quote_in: float, entry_price: float) -> None:
    _execute(
        """
        INSERT OR REPLACE INTO trades (
            signal_id, symbol, qty, quote_in, entry_price, opened_at,
            exit_price, closed_at, outcome, pnl_quote, pnl_pct
        )
        VALUES (?, ?, ?, ?, ?, datetime('now'), NULL, NULL, NULL, NULL, NULL)
        """,
        (str(signal_id), str(symbol), float(qty), float(quote_in), float(entry_price)),
    )


def get_trade(signal_id: str):
    return _fetchone(
        """
        SELECT signal_id, symbol, qty, quote_in, entry_price, opened_at,
               exit_price, closed_at, outcome, pnl_quote, pnl_pct
        FROM trades
        WHERE signal_id = ?
        """,
        (str(signal_id),),
    )


def close_trade(signal_id: str, exit_price: float, outcome: str, pnl_quote: float, pnl_pct: float) -> None:
    _execute(
        """
        UPDATE trades
        SET exit_price = ?,
            closed_at = datetime('now'),
            outcome = ?,
            pnl_quote = ?,
            pnl_pct = ?
        WHERE signal_id = ?
        """,
        (float(exit_price), str(outcome), float(pnl_quote), float(pnl_pct), str(signal_id)),
    )


def get_closed_trades() -> List[Dict[str, Any]]:
    rows = _fetchall(
        """
        SELECT signal_id, symbol, qty, quote_in, entry_price, opened_at,
               exit_price, closed_at, outcome, pnl_quote, pnl_pct
        FROM trades
        WHERE closed_at IS NOT NULL
        ORDER BY closed_at DESC
        """
    )

    result: List[Dict[str, Any]] = []
    for row in rows:
        result.append({
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
        })
    return result


def get_trade_stats() -> Dict[str, Any]:
    row = _fetchone(
        """
        SELECT
            COUNT(*) AS closed_trades,
            SUM(CASE WHEN pnl_quote > 0 THEN 1 ELSE 0 END) AS wins,
            SUM(CASE WHEN pnl_quote <= 0 THEN 1 ELSE 0 END) AS losses,
            COALESCE(SUM(pnl_quote), 0) AS pnl_quote_sum,
            COALESCE(SUM(quote_in), 0) AS quote_in_sum,
            COALESCE(SUM(CASE WHEN pnl_quote > 0 THEN pnl_quote ELSE 0 END), 0) AS gross_profit,
            COALESCE(ABS(SUM(CASE WHEN pnl_quote < 0 THEN pnl_quote ELSE 0 END)), 0) AS gross_loss,
            COALESCE(AVG(CASE WHEN pnl_quote > 0 THEN pnl_quote END), 0) AS avg_win,
            COALESCE(AVG(CASE WHEN pnl_quote < 0 THEN pnl_quote END), 0) AS avg_loss,
            COALESCE(AVG(pnl_quote), 0) AS expectancy_quote
        FROM trades
        WHERE closed_at IS NOT NULL
        """
    ) or (0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    closed_trades     = int(row[0] or 0)
    wins              = int(row[1] or 0)
    losses            = int(row[2] or 0)
    pnl_quote_sum     = float(row[3] or 0.0)
    quote_in_sum      = float(row[4] or 0.0)
    gross_profit      = float(row[5] or 0.0)
    gross_loss        = float(row[6] or 0.0)
    avg_win           = float(row[7] or 0.0)
    avg_loss          = float(row[8] or 0.0)
    expectancy_quote  = float(row[9] or 0.0)

    winrate_pct    = (wins / closed_trades * 100.0) if closed_trades else 0.0
    roi_pct        = (pnl_quote_sum / quote_in_sum * 100.0) if quote_in_sum else 0.0
    profit_factor  = (gross_profit / gross_loss) if gross_loss > 0 else (gross_profit if gross_profit > 0 else 0.0)

    row2 = _fetchone(
        """
        SELECT
            COUNT(*) AS open_trades,
            COALESCE(SUM(quote_in), 0) AS open_quote_in_sum
        FROM trades
        WHERE closed_at IS NULL
        """
    ) or (0, 0.0)

    open_trades        = int(row2[0] or 0)
    open_quote_in_sum  = float(row2[1] or 0.0)

    return {
        "closed_trades":    closed_trades,
        "wins":             wins,
        "losses":           losses,
        "winrate_pct":      winrate_pct,
        "roi_pct":          roi_pct,
        "pnl_quote_sum":    pnl_quote_sum,
        "quote_in_sum":     quote_in_sum,
        "profit_factor":    profit_factor,
        "gross_profit":     gross_profit,
        "gross_loss":       gross_loss,
        "avg_win":          avg_win,
        "avg_loss":         avg_loss,
        "expectancy_quote": expectancy_quote,
        "open_trades":      open_trades,
        "open_quote_in_sum":open_quote_in_sum,
    }
