-- execution/db/schema.sql
-- ============================================================
-- FIX C-2: schema.sql სინქრონიზებულია db.py init_db()-სთან.
--
-- ცვლილებები:
--   REMOVED:  orders        — კოდი არ იყენებს, db.py არ ქმნის
--   REMOVED:  risk_state    — კოდი არ იყენებს, db.py არ ქმნის
--   ADDED:    oco_links      — OCO order tracking (TP/SL მონიტორინგი)
--   ADDED:    executed_signals — idempotency (duplicate signal block)
--   ADDED:    trades         — realized P&L tracking
--   UPDATED:  system_state  — mode სვეტი ამოღებულია, DEFAULT-ები დამატებულია
--                              + consecutive_sl და sl_pause_until (SL cooldown)
--
-- ⚠️  ᲛᲜᲘᲨᲕᲜᲔᲚᲝᲕᲐᲜᲘ: ეს ფაილი დოკუმენტაციისთვისაა.
--     production-ში schema-ს db.py init_db() ქმნის ავტომატურად.
--     ამ ფაილს პირდაპირ გასვლა არ სჭირდება.
-- ============================================================


-- ─────────────────────────────────────────────────────────────
-- SYSTEM STATE — bot-ის გლობალური სტატუსი (ყოველთვის 1 row)
-- ─────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS system_state (
    id                INTEGER PRIMARY KEY CHECK (id = 1),
    status            TEXT    NOT NULL DEFAULT 'RUNNING',   -- RUNNING / PAUSED
    startup_sync_ok   INTEGER NOT NULL DEFAULT 0,           -- 0/1
    kill_switch       INTEGER NOT NULL DEFAULT 0,           -- 0/1
    updated_at        TEXT    NOT NULL,
    -- SL Cooldown (FIX: migration via _migrate_sl_cooldown_columns)
    consecutive_sl    INTEGER NOT NULL DEFAULT 0,
    sl_pause_until    TEXT    DEFAULT NULL
);

INSERT OR IGNORE INTO system_state
    (id, status, startup_sync_ok, kill_switch, updated_at)
VALUES
    (1, 'RUNNING', 0, 0, datetime('now'));


-- ─────────────────────────────────────────────────────────────
-- POSITIONS — legacy table (backward compat)
-- ─────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS positions (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol       TEXT    NOT NULL,
    side         TEXT    NOT NULL,
    size         REAL    NOT NULL,
    entry_price  REAL    NOT NULL,
    status       TEXT    NOT NULL,
    opened_at    TEXT    NOT NULL,
    closed_at    TEXT,
    pnl          REAL
);


-- ─────────────────────────────────────────────────────────────
-- AUDIT LOG — ყველა event-ის ჩანაწერი
-- ─────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS audit_log (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type   TEXT    NOT NULL,
    message      TEXT    NOT NULL,
    created_at   TEXT    NOT NULL
);


-- ─────────────────────────────────────────────────────────────
-- OCO LINKS — TP + SL ბრძანებების წყვილი თითო trade-ზე
-- reconcile_oco() ყოველ loop-ზე ამოწმებს შესრულდა თუ არა
-- ─────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS oco_links (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    signal_id       TEXT    NOT NULL,
    symbol          TEXT    NOT NULL,
    base_asset      TEXT    NOT NULL,
    tp_order_id     TEXT    NOT NULL,
    sl_order_id     TEXT    NOT NULL,
    tp_price        REAL    NOT NULL,
    sl_stop_price   REAL    NOT NULL,
    sl_limit_price  REAL    NOT NULL,
    amount          REAL    NOT NULL,
    status          TEXT    NOT NULL,   -- OPEN / CLOSED_TP / CLOSED_SL / DESYNC / CLOSED_SL_COOLDOWN
    created_at      TEXT    NOT NULL,
    updated_at      TEXT    NOT NULL
);


-- ─────────────────────────────────────────────────────────────
-- EXECUTED SIGNALS — idempotency: ერთი signal_id ერთხელ სრულდება
-- ─────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS executed_signals (
    signal_id    TEXT PRIMARY KEY,
    signal_hash  TEXT,
    action       TEXT,
    symbol       TEXT,
    executed_at  TEXT NOT NULL
);


-- ─────────────────────────────────────────────────────────────
-- TRADES — realized performance tracking
-- ─────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS trades (
    signal_id    TEXT PRIMARY KEY,
    symbol       TEXT NOT NULL,
    qty          REAL NOT NULL,
    quote_in     REAL NOT NULL,
    entry_price  REAL NOT NULL,
    opened_at    TEXT NOT NULL,

    exit_price   REAL,
    closed_at    TEXT,
    outcome      TEXT,    -- TP / SL / MANUAL_SELL
    pnl_quote    REAL,
    pnl_pct      REAL
);


-- ─────────────────────────────────────────────────────────────
-- INDEXES — სწრაფი lookup-ებისთვის
-- ─────────────────────────────────────────────────────────────
CREATE INDEX IF NOT EXISTS ix_audit_event_type    ON audit_log(event_type);
CREATE INDEX IF NOT EXISTS ix_positions_status    ON positions(status);
CREATE INDEX IF NOT EXISTS ix_oco_links_signal_id ON oco_links(signal_id);
CREATE INDEX IF NOT EXISTS ix_oco_links_status    ON oco_links(status);
CREATE INDEX IF NOT EXISTS ix_trades_symbol       ON trades(symbol);
CREATE INDEX IF NOT EXISTS ix_trades_closed_at    ON trades(closed_at);
