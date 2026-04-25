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

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # DCA: dca_positions + dca_orders ცხრილები
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    # DCA პოზიცია — ერთი position per symbol
    cur.execute("""
    CREATE TABLE IF NOT EXISTS dca_positions (
        id                   INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol               TEXT NOT NULL,
        status               TEXT NOT NULL DEFAULT 'OPEN',

        -- initial entry
        initial_entry_price  REAL NOT NULL,
        initial_qty          REAL NOT NULL,
        initial_quote_spent  REAL NOT NULL,

        -- running average (განახლდება ყოველ add-on-ზე)
        avg_entry_price      REAL NOT NULL,
        total_qty            REAL NOT NULL,
        total_quote_spent    REAL NOT NULL,

        -- add-on tracking
        add_on_count         INTEGER NOT NULL DEFAULT 0,
        max_add_ons          INTEGER NOT NULL DEFAULT 3,
        last_add_on_ts       REAL    DEFAULT NULL,
        last_addon_price     REAL    DEFAULT NULL,  -- ბოლო ADD-ON entry price (L3 rotation trigger-ისთვის)
        last_rotation_ts     REAL    DEFAULT NULL,  -- ბოლო LIFO rotation timestamp (cooldown)
        l3_addon_done        INTEGER DEFAULT 0,     -- 0=L3 ADD-ON ჯერ არ გახსნილა, 1=გახსნილა

        -- current TP/SL (avg_entry-ით გამოთვლილი)
        current_tp_price     REAL,
        current_sl_price     REAL,
        current_tp_pct       REAL,
        current_sl_pct       REAL,

        -- risk limits
        max_capital          REAL,
        max_drawdown_pct     REAL,

        -- outcome
        exit_price           REAL,
        exit_qty             REAL,
        pnl_quote            REAL,
        pnl_pct              REAL,
        outcome              TEXT,

        opened_at            TEXT NOT NULL DEFAULT (datetime('now')),
        closed_at            TEXT,
        updated_at           TEXT NOT NULL DEFAULT (datetime('now'))
    )
    """)

    cur.execute(
        "CREATE INDEX IF NOT EXISTS ix_dca_pos_symbol_status "
        "ON dca_positions(symbol, status)"
    )

    # DCA individual orders (initial + add-ons)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS dca_orders (
        id                     INTEGER PRIMARY KEY AUTOINCREMENT,
        position_id            INTEGER NOT NULL REFERENCES dca_positions(id),
        symbol                 TEXT NOT NULL,
        order_type             TEXT NOT NULL,  -- INITIAL | ADD_ON_1 | ADD_ON_2 | ADD_ON_3

        -- fill data
        entry_price            REAL NOT NULL,
        qty                    REAL NOT NULL,
        quote_spent            REAL NOT NULL,

        -- context at time of add-on
        trigger_drawdown_pct   REAL,
        rsi_at_entry           REAL,
        atr_pct_at_entry       REAL,
        recovery_score         INTEGER,

        -- running avg after this order
        avg_entry_after        REAL,
        tp_after               REAL,
        sl_after               REAL,

        exchange_order_id      TEXT,
        filled_at              TEXT NOT NULL DEFAULT (datetime('now'))
    )
    """)

    cur.execute(
        "CREATE INDEX IF NOT EXISTS ix_dca_orders_position "
        "ON dca_orders(position_id)"
    )

    conn.commit()
    # კავშირი აღარ იხურება — thread-local-ში რჩება

    # ADDON CASCADE SYSTEM migration — არსებული DB-სთვის
    _migrate_dca_rotation_columns(cur)
    conn.commit()

    # FUTURES ENGINE migration — is_mirror + ყველა missing column
    _migrate_futures_columns(cur)
    conn.commit()


def _migrate_dca_rotation_columns(cur) -> None:
    """
    dca_positions ცხრილში LIFO rotation columns-ების დამატება.
    იდემპოტენტურია — CREATE TABLE IF NOT EXISTS-ს გვერდი:
    არსებული DB-ები schema ცვლილებას ALTER TABLE-ით იღებენ.

    ახალი columns:
      last_addon_price  REAL — ბოლო ADD-ON entry price (L3 trigger reference)
      last_rotation_ts  REAL — ბოლო LIFO rotation unix timestamp (cooldown)
    """
    try:
        cur.execute("PRAGMA table_info(dca_positions)")
        existing = {row[1] for row in cur.fetchall()}

        if "last_addon_price" not in existing:
            cur.execute(
                "ALTER TABLE dca_positions ADD COLUMN "
                "last_addon_price REAL DEFAULT NULL"
            )
            logger.info("[DB_MIGRATE] dca_positions.last_addon_price column added")

        if "last_rotation_ts" not in existing:
            cur.execute(
                "ALTER TABLE dca_positions ADD COLUMN "
                "last_rotation_ts REAL DEFAULT NULL"
            )
            logger.info("[DB_MIGRATE] dca_positions.last_rotation_ts column added")

        if "l3_addon_done" not in existing:
            cur.execute(
                "ALTER TABLE dca_positions ADD COLUMN "
                "l3_addon_done INTEGER DEFAULT 0"
            )
            logger.info("[DB_MIGRATE] dca_positions.l3_addon_done column added")

    except Exception as e:
        logger.warning(f"[DB_MIGRATE] dca_rotation_columns fail | err={e}")


def _migrate_futures_columns(cur) -> None:
    """
    futures_positions ცხრილში ყველა missing column-ის დამატება.
    იდემპოტენტურია — ყველა ALTER TABLE idempotent pattern.

    FIX: is_mirror_engine column — Mirror Engine crash-ის თავიდანაცილება.
    FIX: close_reason column — _close_mirror() UPDATE crash-ის თავიდანაცილება.
    სრული სია — FuturesEngine._ensure_addon_cols() + _close_mirror() columns.
    """
    try:
        cur.execute("PRAGMA table_info(futures_positions)")
        existing = {row[1] for row in cur.fetchall()}

        migrations = [
            ("add_on_count",          "INTEGER DEFAULT 0"),
            ("add_on_quote",          "REAL DEFAULT 0.0"),
            ("avg_entry_price",       "REAL DEFAULT 0.0"),
            ("sl_price_addon",        "REAL DEFAULT 0.0"),
            ("is_dca_hedge",          "INTEGER DEFAULT 0"),
            ("dca_pos_id",            "INTEGER DEFAULT 0"),
            ("is_independent_short",  "INTEGER DEFAULT 0"),
            ("long_ref_price",        "REAL DEFAULT 0.0"),
            ("last_short_addon_ts",   "REAL DEFAULT 0.0"),
            ("hedge_tp_pct",          "REAL DEFAULT 3.5"),
            ("exit_price",            "REAL DEFAULT 0.0"),
            ("is_mirror_engine",      "INTEGER DEFAULT 0"),
            ("mirror_direction",      "TEXT DEFAULT ''"),
            ("mirror_addons_down",    "INTEGER DEFAULT 0"),
            ("mirror_addons_up",      "INTEGER DEFAULT 0"),
            ("mirror_long_ref_price", "REAL DEFAULT 0.0"),
            ("last_mirror_addon_ts",  "REAL DEFAULT 0.0"),
            ("close_reason",          "TEXT DEFAULT ''"),
        ]

        for col, col_def in migrations:
            if col not in existing:
                cur.execute(
                    f"ALTER TABLE futures_positions ADD COLUMN {col} {col_def}"
                )
                logger.info(f"[DB_MIGRATE] futures_positions.{col} column added")

    except Exception as e:
        logger.warning(f"[DB_MIGRATE] futures_columns fail | err={e}")


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
