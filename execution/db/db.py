# execution/db/db.py
# ============================================================
# FIX #3: Thread-local connection pool
# ─────────────────────────────────────────────────────────────
# ძველი კოდი: get_connection() ყოველ query-ზე ახალ sqlite3.connect()-ს
# ხსნიდა და conn.close()-ს ეძახებოდა — overhead + lock risk.
#
# ახალი კოდი:
#   • threading.local() — თითოეულ thread-ს საკუთარი connection აქვს
#   • connection იხსნება ერთხელ thread-ის სიცოცხლეში
#   • init_db() ახლა get_connection()-ს იყენებს — კონსისტენტური
#   • close_all_connections() — graceful shutdown-ისთვის
#   • WAL mode + busy_timeout — SQLite concurrent write-ების დასაცავად
# ============================================================
import sqlite3
import threading
import logging
from execution.config import DB_PATH

logger = logging.getLogger("gbm")

# thread-local storage: თითოეულ thread-ს _local.conn აქვს
_local = threading.local()

# ─────────────────────────────────────────────────────────────
# WAL + busy timeout — production SQLite defaults
# ─────────────────────────────────────────────────────────────
_WAL_PRAGMAS = [
    "PRAGMA journal_mode=WAL",      # concurrent readers + writer
    "PRAGMA synchronous=NORMAL",    # WAL-თან უსაფრთხოა, სწრაფია
    "PRAGMA busy_timeout=3000",     # 3s wait on locked DB instead of crash
    "PRAGMA foreign_keys=ON",
]


def get_connection() -> sqlite3.Connection:
    """
    Thread-local connection — ერთი connection per thread, სამუდამოდ.

    პირველ გამოძახებაზე thread-ისთვის ახალ connection-ს ხსნის,
    შემდგომ გამოძახებებზე იმავეს აბრუნებს.
    """
    conn = getattr(_local, "conn", None)

    # connection-ი closed ან არარსებული → ახლიდან გახსნა
    if conn is None:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(
            str(DB_PATH),
            check_same_thread=False,   # thread-local-ით ვმართავთ thread-safety-ს
            timeout=10,                # fallback timeout
        )
        conn.row_factory = sqlite3.Row  # dict-like rows — optional, backward-compat off by default

        # production pragmas
        for pragma in _WAL_PRAGMAS:
            try:
                conn.execute(pragma)
            except Exception as e:
                logger.warning(f"[DB] pragma fail | {pragma} | {e}")

        _local.conn = conn
        logger.debug(f"[DB] new connection opened | thread={threading.current_thread().name}")

    return conn


def close_thread_connection() -> None:
    """
    მიმდინარე thread-ის connection-ს ხურავს.
    გამოიყენე graceful shutdown-ზე ან test teardown-ზე.
    """
    conn = getattr(_local, "conn", None)
    if conn is not None:
        try:
            conn.close()
        except Exception:
            pass
        _local.conn = None
        logger.debug(f"[DB] connection closed | thread={threading.current_thread().name}")


def init_db() -> None:
    """
    Schema initialization — ერთხელ გამოიძახება main.py-ში.
    get_connection()-ის connection-ს იყენებს (WAL უკვე ჩართულია).
    """
    conn = get_connection()
    cur = conn.cursor()

    # positions (legacy)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS positions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol TEXT NOT NULL,
        side TEXT NOT NULL,
        size REAL NOT NULL,
        entry_price REAL NOT NULL,
        status TEXT NOT NULL,
        opened_at TEXT NOT NULL,
        closed_at TEXT,
        pnl REAL
    )
    """)

    # audit log
    cur.execute("""
    CREATE TABLE IF NOT EXISTS audit_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        event_type TEXT NOT NULL,
        message TEXT NOT NULL,
        created_at TEXT NOT NULL
    )
    """)

    # system state
    cur.execute("""
    CREATE TABLE IF NOT EXISTS system_state (
        id INTEGER PRIMARY KEY CHECK (id = 1),
        status TEXT NOT NULL DEFAULT 'RUNNING',
        startup_sync_ok INTEGER NOT NULL DEFAULT 0,
        kill_switch INTEGER NOT NULL DEFAULT 0,
        updated_at TEXT NOT NULL
    )
    """)

    cur.execute("""
    INSERT OR IGNORE INTO system_state (id, status, startup_sync_ok, kill_switch, updated_at)
    VALUES (1, 'RUNNING', 0, 0, datetime('now'))
    """)

    # oco links
    cur.execute("""
    CREATE TABLE IF NOT EXISTS oco_links (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        signal_id TEXT NOT NULL,
        symbol TEXT NOT NULL,
        base_asset TEXT NOT NULL,
        tp_order_id TEXT NOT NULL,
        sl_order_id TEXT NOT NULL,
        tp_price REAL NOT NULL,
        sl_stop_price REAL NOT NULL,
        sl_limit_price REAL NOT NULL,
        amount REAL NOT NULL,
        status TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """)

    # executed signals (idempotency)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS executed_signals (
        signal_id TEXT PRIMARY KEY,
        signal_hash TEXT,
        action TEXT,
        symbol TEXT,
        executed_at TEXT NOT NULL
    )
    """)

    # trades (realized performance)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS trades (
        signal_id TEXT PRIMARY KEY,
        symbol TEXT NOT NULL,
        qty REAL NOT NULL,
        quote_in REAL NOT NULL,
        entry_price REAL NOT NULL,
        opened_at TEXT NOT NULL,

        exit_price REAL,
        closed_at TEXT,
        outcome TEXT,
        pnl_quote REAL,
        pnl_pct REAL
    )
    """)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # MIGRATION: SL Cooldown columns
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    _migrate_sl_cooldown_columns(cur)

    conn.commit()
    # კავშირი აღარ იხურება — thread-local-ში რჩება


def _migrate_sl_cooldown_columns(cur) -> None:
    """
    system_state ცხრილში ამატებს SL Cooldown columns-ებს.
    იდემპოტენტურია.
    """
    cur.execute("PRAGMA table_info(system_state)")
    existing = {row[1] for row in cur.fetchall()}

    if "consecutive_sl" not in existing:
        cur.execute(
            "ALTER TABLE system_state ADD COLUMN consecutive_sl INTEGER NOT NULL DEFAULT 0"
        )

    if "sl_pause_until" not in existing:
        cur.execute(
            "ALTER TABLE system_state ADD COLUMN sl_pause_until TEXT DEFAULT NULL"
        )
