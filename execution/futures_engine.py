# execution/futures_engine.py
# ============================================================
# GENIUS DCA Bot — Futures Engine (SHORT positions)
# SMART LONG + SHORT სტრატეგია — Bear market hedge
#
# DEMO mode: ვირტუალური SHORT — Binance Futures API არ სჭირდება
# LIVE mode: FUTURES_ENABLED=true + ცალკე Futures API key
#
# ENV პარამეტრები:
#   FUTURES_ENABLED=false         ← default გათიშული (safe!)
#   FUTURES_LEVERAGE=2            ← x2 (უსაფრთხო)
#   FUTURES_QUOTE=50              ← $50 per SHORT position
#   FUTURES_TP_PCT=2.0            ← 2% TP (BTC -2% → SHORT მოიგო)
#   FUTURES_SL_PCT — გათიშულია (DCA mode, sl_price=0.0 hardcoded)
#   FUTURES_SYMBOLS=BTC/USDT,ETH/USDT,BNB/USDT
#   FUTURES_MODE=DEMO             ← DEMO/LIVE
#   FUTURES_MAX_OPEN=3            ← max ღია SHORT-ების რაოდენობა
#   FUTURES_COOLDOWN_SECONDS=300  ← 5 წუთი SHORT-ებს შორის
#
# DB: futures_positions ცხრილი (auto-created)
# ============================================================
from __future__ import annotations

import os
import logging
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("gbm")


# ─── ENV helpers ────────────────────────────────────────────
def _ef(name: str, default: float) -> float:
    try:
        v = os.getenv(name)
        return float(v) if v is not None else default
    except Exception:
        return default


def _ei(name: str, default: int) -> int:
    try:
        v = os.getenv(name)
        return int(v) if v is not None else default
    except Exception:
        return default


def _eb(name: str, default: bool = False) -> bool:
    v = os.getenv(name, "true" if default else "false").strip().lower()
    return v in ("1", "true", "yes", "y", "on")


# ─── DB init ────────────────────────────────────────────────
def _init_futures_table() -> None:
    """
    futures_positions ცხრილი auto-create.
    მთავარ DB-ში ემატება (გამოიყენება genius_bot_v2.db).
    """
    try:
        from execution.db.db import get_connection
        with get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS futures_positions (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    signal_id    TEXT UNIQUE,
                    symbol       TEXT NOT NULL,
                    direction    TEXT DEFAULT 'SHORT',
                    entry_price  REAL NOT NULL,
                    qty          REAL NOT NULL,
                    quote_in     REAL NOT NULL,
                    leverage     INTEGER DEFAULT 2,
                    tp_price     REAL NOT NULL,
                    sl_price     REAL NOT NULL,
                    status       TEXT DEFAULT 'OPEN',
                    pnl_quote    REAL DEFAULT 0.0,
                    pnl_pct      REAL DEFAULT 0.0,
                    opened_at    TEXT,
                    closed_at    TEXT,
                    outcome      TEXT,
                    mode         TEXT DEFAULT 'DEMO'
                )
            """)
            conn.commit()
        logger.info("[FUTURES] DB table futures_positions ready")
    except Exception as e:
        logger.error(f"[FUTURES] DB_INIT_FAIL | err={e}")


# ─── DB helpers ─────────────────────────────────────────────
def _get_open_shorts() -> List[Dict[str, Any]]:
    try:
        from execution.db.db import get_connection
        with get_connection() as conn:
            cur = conn.execute(
                "SELECT * FROM futures_positions WHERE status='OPEN' AND direction='SHORT'"
            )
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, r)) for r in cur.fetchall()]
    except Exception as e:
        logger.warning(f"[FUTURES] GET_OPEN_FAIL | err={e}")
        return []


def _get_open_short_for_symbol(symbol: str) -> Optional[Dict[str, Any]]:
    try:
        from execution.db.db import get_connection
        with get_connection() as conn:
            cur = conn.execute(
                "SELECT * FROM futures_positions WHERE status='OPEN' AND symbol=? AND direction='SHORT'",
                (symbol,)
            )
            row = cur.fetchone()
            if not row:
                return None
            cols = [d[0] for d in cur.description]
            return dict(zip(cols, row))
    except Exception as e:
        logger.warning(f"[FUTURES] GET_SHORT_SYM_FAIL | {symbol} err={e}")
        return None


def _open_short_db(
    signal_id: str,
    symbol: str,
    entry_price: float,
    qty: float,
    quote_in: float,
    leverage: int,
    tp_price: float,
    sl_price: float,
    mode: str,
    is_dca_hedge: int = 0,
    dca_pos_id: int = 0,
) -> int:
    from execution.db.db import get_connection
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).isoformat()
    with get_connection() as conn:
        cur = conn.execute(
            """
            INSERT INTO futures_positions
              (signal_id, symbol, direction, entry_price, qty, quote_in,
               leverage, tp_price, sl_price, status, opened_at, mode,
               is_dca_hedge, dca_pos_id, avg_entry_price)
            VALUES (?, ?, 'SHORT', ?, ?, ?, ?, ?, ?, 'OPEN', ?, ?, ?, ?, ?)
            """,
            (signal_id, symbol, entry_price, qty, quote_in,
             leverage, tp_price, sl_price, now, mode,
             is_dca_hedge, dca_pos_id, entry_price)   # avg_entry_price = entry_price initially
        )
        conn.commit()
        return cur.lastrowid


def _close_short_db(pos_id: int, exit_price: float, pnl_quote: float, pnl_pct: float, outcome: str) -> None:
    from execution.db.db import get_connection
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).isoformat()
    with get_connection() as conn:
        conn.execute(
            """
            UPDATE futures_positions
            SET status='CLOSED', exit_price=?, pnl_quote=?, pnl_pct=?,
                outcome=?, closed_at=?
            WHERE id=?
            """,
            (exit_price, pnl_quote, pnl_pct, outcome, now, pos_id)
        )
        conn.commit()


# exit_price column auto-add if missing
def _ensure_exit_price_col() -> None:
    try:
        from execution.db.db import get_connection
        with get_connection() as conn:
            cols = [r[1] for r in conn.execute("PRAGMA table_info(futures_positions)").fetchall()]
            if "exit_price" not in cols:
                conn.execute("ALTER TABLE futures_positions ADD COLUMN exit_price REAL DEFAULT 0.0")
                conn.commit()
    except Exception:
        pass


def _ensure_addon_cols() -> None:
    """
    ADD-ON + DCA HEDGE სვეტების auto-migration.
    იდემპოტენტური — მეორედ გაშვება უვნებელია.
    """
    try:
        from execution.db.db import get_connection
        with get_connection() as conn:
            cols = [r[1] for r in conn.execute(
                "PRAGMA table_info(futures_positions)"
            ).fetchall()]
            migrations = [
                ("add_on_count",      "INTEGER DEFAULT 0"),
                ("add_on_quote",      "REAL DEFAULT 0.0"),
                ("avg_entry_price",   "REAL DEFAULT 0.0"),
                ("sl_price_addon",    "REAL DEFAULT 0.0"),
                # DCA HEDGE — ახალი columns
                ("is_dca_hedge",      "INTEGER DEFAULT 0"),   # 1=DCA hedge SHORT
                ("dca_pos_id",        "INTEGER DEFAULT 0"),   # DCA position-ის ID
            ]
            for col, col_def in migrations:
                if col not in cols:
                    conn.execute(
                        f"ALTER TABLE futures_positions ADD COLUMN {col} {col_def}"
                    )
            conn.commit()
        logger.info("[FUTURES] ADD-ON + DCA HEDGE columns ready")
    except Exception as e:
        logger.warning(f"[FUTURES] ADDON_COLS_FAIL | err={e}")


# ─── FuturesEngine ──────────────────────────────────────────
class FuturesEngine:
    """
    Binance USDT-M Futures SHORT პოზიციების მართვა.

    DEMO mode:  ვირტუალური SHORT — ფასი public API-დან, PnL გათვლილი
    LIVE mode:  Binance Futures API (FUTURES_ENABLED=true)

    მეთოდები:
      check_and_open_short(market_regime)   → BEAR → SHORT გახსნა
      close_all_shorts(reason)              → BULL → ყველა SHORT-ი დახურვა
      check_tp_sl()                         → TP/SL hit შემოწმება loop-ში
      get_open_shorts()                     → ღია SHORT-ების სია
    """

    def __init__(self) -> None:
        self.enabled       = _eb("FUTURES_ENABLED", False)
        self.leverage      = _ei("FUTURES_LEVERAGE", 2)
        self.quote         = _ef("FUTURES_QUOTE", 50.0)
        self.tp_pct        = _ef("FUTURES_TP_PCT", 2.0)
        self.sl_pct        = 0.0   # DCA: SL გათიშულია
        self.max_open      = _ei("FUTURES_MAX_OPEN", 3)
        self.cooldown_s    = _ei("FUTURES_COOLDOWN_SECONDS", 300)
        self.mode          = os.getenv("FUTURES_MODE", "DEMO").upper()

        _symbols_raw       = os.getenv("FUTURES_SYMBOLS", "BTC/USDT,ETH/USDT,BNB/USDT")
        self.symbols       = [s.strip() for s in _symbols_raw.split(",") if s.strip()]

        self._last_open_ts: float = 0.0  # cooldown tracking

        # ADD-ON პარამეტრები
        self.addon_enabled      = _eb("FUTURES_ADDON_ENABLED", True)
        self.addon_trigger_pct  = _ef("FUTURES_ADDON_TRIGGER_PCT", 1.0)
        self.addon_quote        = _ef("FUTURES_ADDON_QUOTE", 12.0)
        self.addon_sl_pct       = 0.0
        self.max_addons         = _ei("FUTURES_MAX_ADDONS", 1)

        # EXCHANGE პარამეტრები
        self.exchange_enabled     = _eb("FUTURES_EXCHANGE_ENABLED", True)
        self.exchange_trigger_pct = _ef("FUTURES_EXCHANGE_TRIGGER_PCT", 2.5)

        # ── DCA HEDGE SHORT პარამეტრები ──────────────────────
        # trigger: DCA add_on_count == max_add_ons (-6.5% L1-დან)
        # TP: -3.5% entry-დან (DCA-ს TP-სგან დამოუკიდებელი)
        # ADD-ON: +1% bounce → avg_short ↑
        # L3 EXCHANGE: ADD-ONs exhausted + BTC ↑ → close+reopen
        self.dca_hedge_quote           = _ef("FUTURES_DCA_HEDGE_QUOTE",         20.0)
        self.dca_hedge_tp_pct          = _ef("FUTURES_DCA_HEDGE_TP_PCT",         3.5)
        self.dca_hedge_addon_trigger_pct = _ef("FUTURES_DCA_ADDON_TRIGGER_PCT",  1.0)
        self.dca_hedge_addon_quote     = _ef("FUTURES_DCA_ADDON_QUOTE",          12.0)
        self.dca_hedge_max_addons      = _ei("FUTURES_DCA_MAX_ADDONS",           5)
        self.dca_hedge_l3_trigger_pct  = _ef("FUTURES_DCA_L3_TRIGGER_PCT",       1.5)

        # in-memory cooldown timestamps (per-loop spam guard)
        self._last_hedge_addon_ts: float = 0.0   # ADD-ON cooldown
        self._last_hedge_l3_ts:    float = 0.0   # L3 exchange cooldown

        # DB init
        _init_futures_table()
        _ensure_exit_price_col()
        _ensure_addon_cols()  # ADD-ON სვეტების მიგრაცია

        logger.info(
            f"[FUTURES] Engine init | enabled={self.enabled} mode={self.mode} "
            f"leverage={self.leverage}x quote={self.quote} "
            f"tp={self.tp_pct}% addon={self.addon_trigger_pct}% "
            f"exchange={self.exchange_trigger_pct}% symbols={self.symbols}"
        )

    def _fetch_price(self, symbol: str) -> float:
        """ახლანდელი ფასი — public API (DEMO/LIVE ორივეზე მუშაობს)."""
        try:
            import ccxt
            exchange = ccxt.binance({"enableRateLimit": True})
            ticker = exchange.fetch_ticker(symbol)
            price = float(ticker.get("last") or 0.0)
            logger.debug(f"[FUTURES] PRICE | {symbol}={price:.4f}")
            return price
        except Exception as e:
            logger.warning(f"[FUTURES] PRICE_FAIL | {symbol} err={e}")
            return 0.0

    def _get_btc_24h_change(self) -> float:
        """BTC 24h price change % — BEAR/BULL detector-ისთვის."""
        try:
            import ccxt
            exchange = ccxt.binance({"enableRateLimit": True})
            ticker = exchange.fetch_ticker("BTC/USDT")
            last  = float(ticker.get("last") or 0.0)
            prev  = float(ticker.get("previousClose") or ticker.get("open") or 0.0)
            if prev <= 0:
                return 0.0
            change = (last - prev) / prev * 100.0
            return change
        except Exception as e:
            logger.warning(f"[FUTURES] BTC_CHANGE_FAIL | err={e}")
            return 0.0

    # ────────────────────────────────────────────────────────
    # PUBLIC: check_and_open_short
    # ────────────────────────────────────────────────────────
    def check_and_open_short(self, market_regime: str) -> None:
        """
        BEAR market → SHORT პოზიციების გახსნა.

        ლოგიკა:
          1. FUTURES_ENABLED=false → skip (safe default)
          2. market_regime != 'BEAR' → skip
          3. cooldown შემოწმება (FUTURES_COOLDOWN_SECONDS)
          4. max ღია SHORT-ები შემოწმება (FUTURES_MAX_OPEN)
          5. ყოველ symbol-ზე → SHORT გახსნა (თუ უკვე არ გახსნილია)
        """
        if not self.enabled:
            logger.debug("[FUTURES] DISABLED | FUTURES_ENABLED=false → skip")
            return

        if market_regime != "BEAR":
            logger.debug(f"[FUTURES] NOT_BEAR | regime={market_regime} → no short")
            return

        # cooldown
        elapsed = time.time() - self._last_open_ts
        if elapsed < self.cooldown_s:
            logger.debug(
                f"[FUTURES] COOLDOWN | remaining={int(self.cooldown_s - elapsed)}s → skip"
            )
            return

        # max open shorts check
        open_shorts = _get_open_shorts()
        if len(open_shorts) >= self.max_open:
            logger.info(
                f"[FUTURES] MAX_OPEN | {len(open_shorts)}/{self.max_open} shorts open → skip"
            )
            return

        for sym in self.symbols:
            # უკვე ღია SHORT ამ symbol-ზე?
            existing = _get_open_short_for_symbol(sym)
            if existing:
                logger.debug(f"[FUTURES] ALREADY_OPEN | {sym} → skip")
                continue

            self._open_short(sym)

    def _open_short(self, symbol: str) -> None:
        """SHORT position გახსნა (DEMO: ვირტუალური / LIVE: Binance Futures)."""
        try:
            current_price = self._fetch_price(symbol)
            if current_price <= 0:
                logger.warning(f"[FUTURES] OPEN_SHORT_NO_PRICE | {symbol}")
                return

            # SHORT: TP = ქვევით, SL = გათიშულია (0.0)
            tp_price = round(current_price * (1.0 - self.tp_pct / 100.0), 6)
            sl_price = 0.0   # DCA: SL გათიშულია
            qty      = round((self.quote * self.leverage) / current_price, 6)
            sig_id   = f"FUT-{symbol.replace('/', '')}-{uuid.uuid4().hex[:8]}"

            if self.mode == "DEMO" or not self.enabled:
                # DEMO: ვირტუალური — DB-ში ვწერთ, ბინანსს არ ვეხებით
                pos_id = _open_short_db(
                    signal_id=sig_id,
                    symbol=symbol,
                    entry_price=current_price,
                    qty=qty,
                    quote_in=self.quote,
                    leverage=self.leverage,
                    tp_price=tp_price,
                    sl_price=sl_price,
                    mode="DEMO",
                )
                logger.warning(
                    f"[FUTURES] SHORT_OPENED_DEMO | {symbol} "
                    f"entry={current_price:.4f} tp={tp_price:.4f} sl={sl_price:.4f} "
                    f"qty={qty:.6f} quote={self.quote} lev={self.leverage}x pos_id={pos_id}"
                )
            else:
                # LIVE: Binance Futures API
                # TODO: implement live futures API call
                logger.warning("[FUTURES] LIVE mode not yet implemented — use DEMO")
                return

            self._last_open_ts = time.time()

            # Telegram
            try:
                from execution.telegram_notifier import notify_short_opened
                notify_short_opened(
                    symbol=symbol,
                    entry_price=current_price,
                    tp_price=tp_price,
                    sl_price=sl_price,
                    quote=self.quote,
                    leverage=self.leverage,
                    mode=self.mode,
                )
            except Exception as _tg:
                logger.warning(f"[FUTURES] TG_OPEN_FAIL | err={_tg}")

            try:
                from execution.db.repository import log_event
                log_event(
                    "FUTURES_SHORT_OPENED",
                    f"sym={symbol} entry={current_price:.4f} "
                    f"tp={tp_price:.4f} sl={sl_price:.4f} "
                    f"quote={self.quote} lev={self.leverage}x mode={self.mode}"
                )
            except Exception:
                pass

        except Exception as e:
            logger.error(f"[FUTURES] OPEN_SHORT_FAIL | {symbol} err={e}")

    # ────────────────────────────────────────────────────────
    # PUBLIC: close_all_shorts
    # ────────────────────────────────────────────────────────
    def close_all_shorts(self, reason: str = "BULL_MARKET") -> None:
        """
        BULL market → BEAR hedge SHORT-ების დახურვა.
        DCA hedge SHORT-ები (is_dca_hedge=1) არ იხურება —
        ისინი DCA position-ის სიცოცხლეს მიყვება, BULL market-ს არა.
        """
        if not self.enabled:
            return

        open_shorts = _get_open_shorts()
        if not open_shorts:
            logger.debug("[FUTURES] CLOSE_ALL | no open shorts → skip")
            return

        # DCA hedge SHORT-ები გამოვრიცხოთ — ისინი ცალკე iხურება
        bear_shorts = [p for p in open_shorts if not int(p.get("is_dca_hedge", 0) or 0)]
        hedge_shorts = [p for p in open_shorts if int(p.get("is_dca_hedge", 0) or 0)]

        if hedge_shorts:
            logger.info(
                f"[FUTURES] CLOSE_ALL | skipping {len(hedge_shorts)} DCA hedge SHORT(s) "
                f"— they close independently via TP"
            )

        if not bear_shorts:
            logger.debug("[FUTURES] CLOSE_ALL | no BEAR shorts to close → skip")
            return

        logger.warning(
            f"[FUTURES] CLOSING_BEAR_SHORTS | count={len(bear_shorts)} reason={reason}"
        )

        for pos in bear_shorts:
            self._close_short(pos, reason=reason)

    def _close_short(self, pos: Dict[str, Any], reason: str = "MANUAL") -> None:
        """ერთი SHORT position-ის დახურვა."""
        try:
            symbol      = str(pos.get("symbol", ""))
            pos_id      = int(pos.get("id", 0))
            entry_price = float(pos.get("entry_price", 0.0))
            qty         = float(pos.get("qty", 0.0))
            quote_in    = float(pos.get("quote_in", 0.0))

            exit_price = self._fetch_price(symbol)
            if exit_price <= 0:
                exit_price = entry_price  # fallback

            # SHORT PnL: ბაზარი ეცა → SHORT მოიგო (entry > exit → profit)
            # PnL = (entry - exit) × qty (leverage-ის გარეშე gross)
            price_diff  = entry_price - exit_price
            pnl_quote   = round(price_diff * qty, 4)
            pnl_pct     = round((price_diff / entry_price) * 100.0 * self.leverage, 2)

            _close_short_db(pos_id, exit_price, pnl_quote, pnl_pct, reason)

            logger.warning(
                f"[FUTURES] SHORT_CLOSED | {symbol} "
                f"entry={entry_price:.4f} exit={exit_price:.4f} "
                f"pnl={pnl_quote:+.4f} pct={pnl_pct:+.2f}% reason={reason}"
            )

            # Telegram
            try:
                from execution.telegram_notifier import notify_short_closed
                notify_short_closed(
                    symbol=symbol,
                    entry_price=entry_price,
                    exit_price=exit_price,
                    pnl_quote=pnl_quote,
                    pnl_pct=pnl_pct,
                    reason=reason,
                )
            except Exception as _tg:
                logger.warning(f"[FUTURES] TG_CLOSE_FAIL | err={_tg}")

            try:
                from execution.db.repository import log_event
                log_event(
                    "FUTURES_SHORT_CLOSED",
                    f"sym={symbol} entry={entry_price:.4f} exit={exit_price:.4f} "
                    f"pnl={pnl_quote:+.4f} reason={reason}"
                )
            except Exception:
                pass

        except Exception as e:
            logger.error(f"[FUTURES] CLOSE_SHORT_FAIL | id={pos.get('id')} err={e}")

    # ────────────────────────────────────────────────────────
    # PUBLIC: check_tp_sl
    # ────────────────────────────────────────────────────────
    def check_tp_sl(self) -> None:
        """
        main loop-ში ყოველ iteration-ზე გამოიძახება.
        TP hit: current_price <= tp_price → SHORT win → close
        SL: გათიშულია — SHORT TP-ს ელოდება ან manual close.
        """
        if not self.enabled:
            return

        open_shorts = _get_open_shorts()
        if not open_shorts:
            return

        for pos in open_shorts:
            symbol    = str(pos.get("symbol", ""))
            tp_price  = float(pos.get("tp_price", 0.0))

            current_price = self._fetch_price(symbol)
            if current_price <= 0:
                continue

            entry_price = float(pos.get("entry_price", 0.0))
            logger.debug(
                f"[FUTURES] TP_SL_CHECK | {symbol} "
                f"current={current_price:.4f} tp={tp_price:.4f} sl=OFF"
            )

            # TP hit: ბაზარი ეცა → SHORT მოიგო!
            if tp_price > 0 and current_price <= tp_price:
                logger.warning(
                    f"[FUTURES] TP_HIT | {symbol} "
                    f"current={current_price:.4f} <= tp={tp_price:.4f}"
                )
                self._close_short(pos, reason="TP")
                continue

    # ────────────────────────────────────────────────────────
    # PUBLIC: get_open_shorts
    # ────────────────────────────────────────────────────────
    def get_open_shorts(self) -> List[Dict[str, Any]]:
        return _get_open_shorts()

    # ────────────────────────────────────────────────────────
    # PUBLIC: check_and_addon_short
    # ADD-ON ლოგიკა — ბაზარი +addon_trigger_pct% → ADD-ON
    # ────────────────────────────────────────────────────────
    def check_and_addon_short(self) -> None:
        """
        ყოველ loop-ზე — ღია SHORT-ებზე ADD-ON შემოწმება.

        ლოგიკა:
          1. addon_enabled=false → skip
          2. ღია SHORT-ების სია
          3. ყოველ SHORT-ზე:
             - add_on_count >= max_addons → skip
             - current_price >= entry × (1 + addon_trigger_pct%) → ADD-ON!
             - avg_entry გაახლება
             - SL: გათიშულია (DCA mode)
             - TP = avg × (1 - tp_pct%) გაახლება
        """
        if not self.enabled:
            return
        if not self.addon_enabled:
            return

        open_shorts = _get_open_shorts()
        if not open_shorts:
            return

        for pos in open_shorts:
            try:
                symbol      = str(pos.get("symbol", ""))
                pos_id      = int(pos.get("id", 0))
                entry_price = float(pos.get("entry_price", 0.0))
                add_on_count = int(pos.get("add_on_count", 0) or 0)
                quote_in    = float(pos.get("quote_in", 0.0))
                avg_entry   = float(pos.get("avg_entry_price", 0.0) or entry_price)

                # უკვე მაქსიმუმ ADD-ON?
                if add_on_count >= self.max_addons:
                    logger.debug(f"[FUTURES] ADDON_MAX | {symbol} add_ons={add_on_count}/{self.max_addons}")
                    continue

                current_price = self._fetch_price(symbol)
                if current_price <= 0:
                    continue

                # ბაზარი trigger-ზე ზევით?
                trigger_price = entry_price * (1.0 + self.addon_trigger_pct / 100.0)
                if current_price < trigger_price:
                    logger.debug(
                        f"[FUTURES] ADDON_WAIT | {symbol} "
                        f"price={current_price:.2f} trigger={trigger_price:.2f}"
                    )
                    continue

                # ADD-ON! — ახალი avg გათვლა
                total_quote  = quote_in + self.addon_quote
                new_avg      = (entry_price * quote_in + current_price * self.addon_quote) / total_quote
                new_tp       = round(new_avg * (1.0 - self.tp_pct / 100.0), 6)
                new_sl       = 0.0   # DCA: SL გათიშულია
                new_qty      = round((total_quote * self.leverage) / new_avg, 6)

                logger.warning(
                    f"[FUTURES] SHORT_ADDON | {symbol} "
                    f"entry={entry_price:.2f} addon_price={current_price:.2f} "
                    f"new_avg={new_avg:.2f} new_tp={new_tp:.2f} sl=OFF"
                )

                # DB განახლება
                try:
                    from execution.db.db import get_connection
                    with get_connection() as conn:
                        conn.execute("""
                            UPDATE futures_positions
                            SET add_on_count    = ?,
                                add_on_quote    = ?,
                                avg_entry_price = ?,
                                tp_price        = ?,
                                sl_price_addon  = ?,
                                qty             = ?,
                                quote_in        = ?
                            WHERE id = ?
                        """, (
                            add_on_count + 1,
                            self.addon_quote,
                            round(new_avg, 6),
                            new_tp,
                            0.0,   # sl_price_addon = 0 (გათიშულია)
                            new_qty,
                            total_quote,
                            pos_id,
                        ))
                        conn.commit()
                except Exception as db_err:
                    logger.error(f"[FUTURES] ADDON_DB_FAIL | {symbol} err={db_err}")
                    continue

                # Telegram
                try:
                    from execution.telegram_notifier import send_telegram_message
                    send_telegram_message(
                        f"➕ <b>SHORT ADD-ON</b>\n\n"
                        f"🪙 <b>Symbol:</b> <code>{symbol}</code>\n"
                        f"📈 <b>ბაზარი ზევით:</b> <code>+{self.addon_trigger_pct:.1f}%</code>\n"
                        f"💰 <b>ADD-ON ფასი:</b> <code>{current_price:.2f}</code>\n"
                        f"📊 <b>ახალი avg:</b> <code>{new_avg:.2f}</code>\n"
                        f"🎯 <b>ახალი TP:</b> <code>{new_tp:.2f}</code>\n"
                        f"🛡 <b>SL:</b> <code>გათიშულია</code>\n"
                        f"💼 <b>სულ:</b> <code>${total_quote:.2f}</code>\n"
                        f"🕒 <code>{__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</code>"
                    )
                except Exception as tg_err:
                    logger.warning(f"[FUTURES] ADDON_TG_FAIL | err={tg_err}")

            except Exception as e:
                logger.error(f"[FUTURES] ADDON_ERR | {pos.get('symbol')} err={e}")

    # ────────────────────────────────────────────────────────
    # PUBLIC: check_and_exchange_short
    # EXCHANGE ლოგიკა — ბაზარი +exchange_trigger_pct% → EXCHANGE
    # ძველი SHORT დაიხურება, ახალი ძვირ ფასზე გაიხსნება
    # ────────────────────────────────────────────────────────
    def check_and_exchange_short(self) -> None:
        """
        ყოველ loop-ზე — ღია SHORT-ებზე EXCHANGE შემოწმება.

        ლოგიკა:
          1. exchange_enabled=false → skip
          2. ADD-ON-ის შემდეგ (add_on_count > 0) EXCHANGE შეიძლება
          3. current_price >= entry × (1 + exchange_trigger_pct%) → EXCHANGE!
          4. ძველი SHORT დაიხურება მცირე ზარალით
          5. ახალი SHORT გაიხსნება მიმდინარე ფასზე (ახლა ძვირი = SHORT-ისთვის კარგი!)
        """
        if not self.enabled:
            return
        if not self.exchange_enabled:
            return

        open_shorts = _get_open_shorts()
        if not open_shorts:
            return

        for pos in open_shorts:
            try:
                symbol       = str(pos.get("symbol", ""))
                pos_id       = int(pos.get("id", 0))
                entry_price  = float(pos.get("entry_price", 0.0))
                add_on_count = int(pos.get("add_on_count", 0) or 0)
                quote_in     = float(pos.get("quote_in", 0.0))

                # EXCHANGE მხოლოდ ADD-ON-ის შემდეგ!
                # ADD-ON-ის გარეშე ჯერ "სუნთქვის" საშუალება მიეცეს
                if add_on_count < 1:
                    logger.debug(f"[FUTURES] EXCHANGE_WAIT_ADDON | {symbol} no add-on yet")
                    continue

                current_price = self._fetch_price(symbol)
                if current_price <= 0:
                    continue

                # EXCHANGE trigger
                exchange_trigger = entry_price * (1.0 + self.exchange_trigger_pct / 100.0)
                if current_price < exchange_trigger:
                    logger.debug(
                        f"[FUTURES] EXCHANGE_WAIT | {symbol} "
                        f"price={current_price:.2f} trigger={exchange_trigger:.2f}"
                    )
                    continue

                logger.warning(
                    f"[FUTURES] SHORT_EXCHANGE | {symbol} "
                    f"old_entry={entry_price:.2f} new_entry={current_price:.2f} "
                    f"trigger={exchange_trigger:.2f}"
                )

                # 1. ძველი SHORT-ის დახურვა
                old_qty      = float(pos.get("qty", 0.0))
                price_diff   = entry_price - current_price  # SHORT ზარალი (უარყოფითი)
                pnl_quote    = round(price_diff * old_qty, 4)
                pnl_pct      = round((price_diff / entry_price) * 100.0 * self.leverage, 2)

                _close_short_db(pos_id, current_price, pnl_quote, pnl_pct, "EXCHANGE")

                logger.warning(
                    f"[FUTURES] EXCHANGE_CLOSED | {symbol} "
                    f"entry={entry_price:.2f} exit={current_price:.2f} "
                    f"pnl={pnl_quote:+.4f}"
                )

                # 2. ახალი SHORT-ის გახსნა — ახლა ძვირ ფასზე (SHORT-ისთვის უკეთესი!)
                self._open_short(symbol)

                # Telegram
                try:
                    from execution.telegram_notifier import send_telegram_message
                    send_telegram_message(
                        f"🔄 <b>SHORT EXCHANGE</b>\n\n"
                        f"🪙 <b>Symbol:</b> <code>{symbol}</code>\n"
                        f"📤 <b>ძველი დაიხურა:</b> <code>{entry_price:.2f} → {current_price:.2f}</code>\n"
                        f"💰 <b>PnL:</b> <code>{pnl_quote:+.4f} USDT</code>\n"
                        f"📈 <b>ახალი SHORT გაიხსნა:</b> <code>{current_price:.2f}</code>\n"
                        f"💡 <b>ახლა TP უფრო ახლოს!</b>\n"
                        f"🕒 <code>{__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</code>"
                    )
                except Exception as tg_err:
                    logger.warning(f"[FUTURES] EXCHANGE_TG_FAIL | err={tg_err}")

                try:
                    from execution.db.repository import log_event
                    log_event(
                        "FUTURES_SHORT_EXCHANGE",
                        f"sym={symbol} old_entry={entry_price:.2f} "
                        f"new_entry={current_price:.2f} pnl={pnl_quote:+.4f}"
                    )
                except Exception:
                    pass

            except Exception as e:
                logger.error(f"[FUTURES] EXCHANGE_ERR | {pos.get('symbol')} err={e}")

    # ────────────────────────────────────────────────────────
    # PUBLIC: check_addon_sl
    # ADD-ON შემდეგ SL — გათიშულია (DCA: SL=0)
    # ────────────────────────────────────────────────────────
    def check_addon_sl(self) -> None:
        """DCA mode: SL გათიშულია. Stub — no-op."""
        pass

    # ────────────────────────────────────────────────────────
    # PUBLIC: open_dca_hedge_short
    # DCA add_on_count==max_add_ons → SHORT hedge გახსნა
    # ────────────────────────────────────────────────────────
    def open_dca_hedge_short(
        self,
        symbol: str,
        current_price: float,
        dca_pos_id: int,
    ) -> bool:
        """
        DCA L2/L3 boundary-ზე SHORT hedge გახსნა.

        trigger: add_on_count == max_add_ons (L2 exhausted, -6.5% L1-დან)
        ერთხელ გაიხსნება per DCA position — dca_pos_id unique guard.

        TP: FUTURES_DCA_TP_PCT (default 3.5%) — DCA TP-ზე გაცილებით ქვემოთ
            SHORT-ი DCA-სგან დამოუკიდებლად იხურება TP-ზე.

        Returns: True თუ გახსნა, False თუ skip.
        """
        if not self.enabled:
            return False

        # უკვე გახსნილია ამ DCA position-ისთვის?
        try:
            from execution.db.db import get_connection
            with get_connection() as conn:
                row = conn.execute(
                    "SELECT id FROM futures_positions "
                    "WHERE dca_pos_id=? AND is_dca_hedge=1 AND status='OPEN'",
                    (dca_pos_id,)
                ).fetchone()
                if row:
                    logger.debug(
                        f"[HEDGE] ALREADY_OPEN | dca_pos_id={dca_pos_id} → skip"
                    )
                    return False
        except Exception as e:
            logger.warning(f"[HEDGE] CHECK_FAIL | err={e}")
            return False

        # balance check
        hedge_quote = self.dca_hedge_quote
        try:
            from execution.dca_risk_manager import get_risk_manager as _rm
            bal_ok, bal_reason = _rm().can_l3_operation(hedge_quote)
            if not bal_ok:
                logger.warning(f"[HEDGE] BALANCE_BLOCK | {symbol} reason={bal_reason}")
                return False
        except Exception as e:
            logger.warning(f"[HEDGE] BALANCE_CHECK_FAIL | err={e}")

        # SHORT TP — DCA TP-ს ქვევით
        tp_price = round(current_price * (1.0 - self.dca_hedge_tp_pct / 100.0), 6)
        qty      = round((hedge_quote * self.leverage) / current_price, 6)
        sig_id   = f"HEDGE-{symbol.replace('/', '')}-{uuid.uuid4().hex[:8]}"

        pos_id = _open_short_db(
            signal_id=sig_id,
            symbol=symbol,
            entry_price=current_price,
            qty=qty,
            quote_in=hedge_quote,
            leverage=self.leverage,
            tp_price=tp_price,
            sl_price=0.0,
            mode=self.mode,
            is_dca_hedge=1,
            dca_pos_id=dca_pos_id,
        )

        logger.warning(
            f"[HEDGE] SHORT_OPENED | {symbol} "
            f"entry={current_price:.4f} tp={tp_price:.4f} "
            f"qty={qty:.6f} quote={hedge_quote} "
            f"dca_pos_id={dca_pos_id} pos_id={pos_id}"
        )

        self._last_open_ts = time.time()

        try:
            from execution.telegram_notifier import send_telegram_message
            send_telegram_message(
                f"🛡 <b>DCA HEDGE SHORT გახსნა</b>\n\n"
                f"🪙 <b>Symbol:</b> <code>{symbol}</code>\n"
                f"📉 <b>DCA L2 exhausted — hedge active</b>\n"
                f"💰 <b>Entry:</b> <code>{current_price:.2f}</code>\n"
                f"🎯 <b>SHORT TP:</b> <code>{tp_price:.2f}</code> "
                f"(<code>-{self.dca_hedge_tp_pct:.1f}%</code>)\n"
                f"💼 <b>Quote:</b> <code>${hedge_quote:.0f}</code> "
                f"<code>×{self.leverage}</code>\n"
                f"🕒 <code>{__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</code>"
            )
        except Exception as _tg:
            logger.warning(f"[HEDGE] TG_FAIL | err={_tg}")

        try:
            from execution.db.repository import log_event
            log_event(
                "DCA_HEDGE_OPENED",
                f"sym={symbol} entry={current_price:.4f} "
                f"tp={tp_price:.4f} quote={hedge_quote} "
                f"dca_pos_id={dca_pos_id} pos_id={pos_id}"
            )
        except Exception:
            pass

        return True

    # ────────────────────────────────────────────────────────
    # PUBLIC: check_dca_hedge_addons
    # SHORT ADD-ON ზევით bounce — avg_short ↑, TP ↑
    # ────────────────────────────────────────────────────────
    def check_dca_hedge_addons(self) -> None:
        """
        DCA hedge SHORT-ებზე ADD-ON შემოწმება.

        BTC bounce ↑ → avg_short ↑ → TP_short ↑ → bounce მოთხოვნა ↓

        trigger: current_price >= entry × (1 + FUTURES_DCA_ADDON_TRIGGER_PCT%)
        max: FUTURES_DCA_MAX_ADDONS

        ADD-ON მათემატიკა:
          entry=$69,190, +1% → $69,882
          add_on_quote=$12, leverage=×2
          new_avg = (69190×20 + 69882×12) / 32 = $69,449
          new_tp  = $69,449 × (1 - 3.5%) = $67,018
        """
        if not self.enabled:
            return

        # cooldown: 180s ADD-ON-ებს შორის (spam guard)
        _addon_cooldown = 180
        if (time.time() - self._last_hedge_addon_ts) < _addon_cooldown:
            return

        # მხოლოდ DCA hedge SHORT-ები
        try:
            from execution.db.db import get_connection
            with get_connection() as conn:
                cur = conn.execute(
                    "SELECT * FROM futures_positions "
                    "WHERE status='OPEN' AND is_dca_hedge=1"
                )
                cols = [d[0] for d in cur.description]
                hedge_shorts = [dict(zip(cols, r)) for r in cur.fetchall()]
        except Exception as e:
            logger.warning(f"[HEDGE] ADDON_FETCH_FAIL | err={e}")
            return

        for pos in hedge_shorts:
            try:
                symbol       = str(pos.get("symbol", ""))
                pos_id       = int(pos.get("id", 0))
                entry_price  = float(pos.get("entry_price", 0.0))
                add_on_count = int(pos.get("add_on_count", 0) or 0)
                quote_in     = float(pos.get("quote_in", 0.0))
                avg_entry    = float(pos.get("avg_entry_price", 0.0) or entry_price)

                if add_on_count >= self.dca_hedge_max_addons:
                    continue

                current_price = self._fetch_price(symbol)
                if current_price <= 0:
                    continue

                trigger = entry_price * (1.0 + self.dca_hedge_addon_trigger_pct / 100.0)
                if current_price < trigger:
                    continue

                # ADD-ON
                total_quote = quote_in + self.dca_hedge_addon_quote
                new_avg     = (avg_entry * quote_in + current_price * self.dca_hedge_addon_quote) / total_quote
                new_tp      = round(new_avg * (1.0 - self.dca_hedge_tp_pct / 100.0), 6)
                new_qty     = round((total_quote * self.leverage) / new_avg, 6)

                logger.warning(
                    f"[HEDGE] ADDON | {symbol} "
                    f"entry={entry_price:.2f} current={current_price:.2f} "
                    f"new_avg={new_avg:.2f} new_tp={new_tp:.2f}"
                )

                from execution.db.db import get_connection
                with get_connection() as conn:
                    conn.execute(
                        """
                        UPDATE futures_positions SET
                            add_on_count    = ?,
                            add_on_quote    = ?,
                            avg_entry_price = ?,
                            tp_price        = ?,
                            qty             = ?,
                            quote_in        = ?
                        WHERE id = ?
                        """,
                        (add_on_count + 1, self.dca_hedge_addon_quote,
                         round(new_avg, 6), new_tp, new_qty, total_quote, pos_id)
                    )
                    conn.commit()

                try:
                    from execution.telegram_notifier import send_telegram_message
                    send_telegram_message(
                        f"➕ <b>HEDGE SHORT ADD-ON</b>\n\n"
                        f"🪙 <b>Symbol:</b> <code>{symbol}</code>\n"
                        f"📈 <b>BTC bounce:</b> <code>+{self.dca_hedge_addon_trigger_pct:.1f}%</code>\n"
                        f"💰 <b>ADD-ON @ </b><code>{current_price:.2f}</code>\n"
                        f"📊 <b>avg_short:</b> <code>{entry_price:.2f} → {new_avg:.2f}</code> ↑\n"
                        f"🎯 <b>new TP:</b> <code>{new_tp:.2f}</code>\n"
                        f"🕒 <code>{__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</code>"
                    )
                except Exception as _tg:
                    logger.warning(f"[HEDGE] ADDON_TG_FAIL | err={_tg}")

                try:
                    from execution.db.repository import log_event
                    log_event(
                        "DCA_HEDGE_ADDON",
                        f"sym={symbol} addon_price={current_price:.4f} "
                        f"new_avg={new_avg:.4f} new_tp={new_tp:.4f} "
                        f"addon_count={add_on_count+1}"
                    )
                except Exception:
                    pass

                # cooldown განახლება — spam guard
                self._last_hedge_addon_ts = time.time()

            except Exception as e:
                logger.error(f"[HEDGE] ADDON_ERR | {pos.get('symbol')} err={e}")

    # ────────────────────────────────────────────────────────
    # PUBLIC: check_dca_hedge_l3
    # SHORT ADD-ONs exhausted + BTC კიდევ ↑ → EXCHANGE
    # ძველი SHORT დაიხურება, ახალი ძვირ ფასზე (SHORT-ისთვის უკეთესი)
    # ────────────────────────────────────────────────────────
    def check_dca_hedge_l3(self) -> None:
        """
        DCA hedge L3 — ADD-ONs exhausted + BTC კვლავ ↑.

        EXCHANGE ლოგიკა (SHORT mirror of DCA LIFO):
          ყველაზე იაფი SHORT unit (ყველაზე დიდი ზარალი) → close
          reinvest @ current (ახლა ძვირი SHORT entry → TP ახლოს)
          avg_short ↑, TP_short ↑ → bounce მოთხოვნა ↓

        trigger: add_on_count >= dca_hedge_max_addons
                 + current >= entry × (1 + L3_TRIGGER_PCT%)
        """
        if not self.enabled:
            return

        # cooldown: 300s L3 exchange-ებს შორის
        _l3_cooldown = 300
        if (time.time() - self._last_hedge_l3_ts) < _l3_cooldown:
            return

        try:
                cols = [d[0] for d in cur.description]
                hedge_shorts = [dict(zip(cols, r)) for r in cur.fetchall()]
        except Exception as e:
            logger.warning(f"[HEDGE] L3_FETCH_FAIL | err={e}")
            return

        for pos in hedge_shorts:
            try:
                symbol       = str(pos.get("symbol", ""))
                pos_id       = int(pos.get("id", 0))
                entry_price  = float(pos.get("entry_price", 0.0))
                add_on_count = int(pos.get("add_on_count", 0) or 0)
                dca_pos_id   = int(pos.get("dca_pos_id", 0) or 0)

                # L3 trigger: ADD-ONs exhausted
                if add_on_count < self.dca_hedge_max_addons:
                    continue

                current_price = self._fetch_price(symbol)
                if current_price <= 0:
                    continue

                l3_trigger = entry_price * (1.0 + self.dca_hedge_l3_trigger_pct / 100.0)
                if current_price < l3_trigger:
                    continue

                logger.warning(
                    f"[HEDGE] L3_EXCHANGE | {symbol} "
                    f"entry={entry_price:.2f} current={current_price:.2f} "
                    f"trigger={l3_trigger:.2f} → close+reopen"
                )

                # ძველი SHORT დახურვა
                qty         = float(pos.get("qty", 0.0))
                price_diff  = entry_price - current_price  # negative (SHORT ზარალი)
                pnl_quote   = round(price_diff * qty, 4)
                pnl_pct     = round((price_diff / entry_price) * 100.0 * self.leverage, 2)

                _close_short_db(pos_id, current_price, pnl_quote, pnl_pct, "HEDGE_L3_EXCHANGE")

                logger.warning(
                    f"[HEDGE] L3_CLOSED | {symbol} "
                    f"pnl={pnl_quote:+.4f} → reopening @ {current_price:.2f}"
                )

                # ახალი SHORT გახსნა ახლა ძვირ ფასზე
                # TP = current × (1 - dca_hedge_tp_pct%) — ახლა TP უფრო ახლოა!
                hedge_quote = self.dca_hedge_quote
                tp_new      = round(current_price * (1.0 - self.dca_hedge_tp_pct / 100.0), 6)
                qty_new     = round((hedge_quote * self.leverage) / current_price, 6)
                sig_new     = f"HEDGE-L3-{symbol.replace('/', '')}-{uuid.uuid4().hex[:8]}"

                new_pos_id = _open_short_db(
                    signal_id=sig_new,
                    symbol=symbol,
                    entry_price=current_price,
                    qty=qty_new,
                    quote_in=hedge_quote,
                    leverage=self.leverage,
                    tp_price=tp_new,
                    sl_price=0.0,
                    mode=self.mode,
                    is_dca_hedge=1,
                    dca_pos_id=dca_pos_id,
                )

                logger.warning(
                    f"[HEDGE] L3_REOPENED | {symbol} "
                    f"entry={current_price:.4f} tp={tp_new:.4f} "
                    f"pos_id={new_pos_id}"
                )

                try:
                    from execution.telegram_notifier import send_telegram_message
                    send_telegram_message(
                        f"🔄 <b>HEDGE L3 EXCHANGE</b>\n\n"
                        f"🪙 <b>Symbol:</b> <code>{symbol}</code>\n"
                        f"📤 <b>ძველი:</b> <code>{entry_price:.2f} → {current_price:.2f}</code> "
                        f"(<code>{pnl_quote:+.4f} USDT</code>)\n"
                        f"📥 <b>ახალი SHORT:</b> <code>{current_price:.2f}</code>\n"
                        f"🎯 <b>TP:</b> <code>{tp_new:.2f}</code> "
                        f"(<code>-{self.dca_hedge_tp_pct:.1f}%</code>)\n"
                        f"💡 <b>avg_short ↑ → TP ახლოს!</b>\n"
                        f"🕒 <code>{__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</code>"
                    )
                except Exception as _tg:
                    logger.warning(f"[HEDGE] L3_TG_FAIL | err={_tg}")

                try:
                    from execution.db.repository import log_event
                    log_event(
                        "DCA_HEDGE_L3_EXCHANGE",
                        f"sym={symbol} old_entry={entry_price:.4f} "
                        f"new_entry={current_price:.4f} pnl={pnl_quote:+.4f} "
                        f"new_tp={tp_new:.4f} new_pos_id={new_pos_id}"
                    )
                except Exception:
                    pass

                # cooldown განახლება — 300s სანამ შემდეგი L3 exchange
                self._last_hedge_l3_ts = time.time()

            except Exception as e:
                logger.error(f"[HEDGE] L3_ERR | {pos.get('symbol')} err={e}")

    # ────────────────────────────────────────────────────────
    # PUBLIC: get_summary
    # ────────────────────────────────────────────────────────
    def get_summary(self) -> Dict[str, Any]:
        """Heartbeat / snapshot-ისთვის."""
        open_shorts = _get_open_shorts()
        total_quote = sum(float(p.get("quote_in", 0.0)) for p in open_shorts)
        return {
            "enabled":     self.enabled,
            "mode":        self.mode,
            "open_count":  len(open_shorts),
            "total_quote": total_quote,
            "symbols":     [p.get("symbol") for p in open_shorts],
        }


# ─── module-level singleton ─────────────────────────────────
_futures_engine: Optional[FuturesEngine] = None


def get_futures_engine() -> FuturesEngine:
    global _futures_engine
    if _futures_engine is None:
        _futures_engine = FuturesEngine()
    return _futures_engine
