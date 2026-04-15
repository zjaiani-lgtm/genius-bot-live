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
#   FUTURES_SL_PCT=1.0            ← 1% SL (BTC +1% → close SHORT)
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
            rows = conn.execute(
                "SELECT * FROM futures_positions WHERE status='OPEN' AND direction='SHORT'"
            ).fetchall()
            if not rows:
                return []
            cols = [d[0] for d in conn.execute(
                "SELECT * FROM futures_positions WHERE status='OPEN' LIMIT 0"
            ).description or []]
            # description from actual query
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
) -> int:
    from execution.db.db import get_connection
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).isoformat()
    with get_connection() as conn:
        cur = conn.execute(
            """
            INSERT INTO futures_positions
              (signal_id, symbol, direction, entry_price, qty, quote_in,
               leverage, tp_price, sl_price, status, opened_at, mode)
            VALUES (?, ?, 'SHORT', ?, ?, ?, ?, ?, ?, 'OPEN', ?, ?)
            """,
            (signal_id, symbol, entry_price, qty, quote_in,
             leverage, tp_price, sl_price, now, mode)
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
        self.sl_pct        = _ef("FUTURES_SL_PCT", 1.0)
        self.max_open      = _ei("FUTURES_MAX_OPEN", 3)
        self.cooldown_s    = _ei("FUTURES_COOLDOWN_SECONDS", 300)
        self.mode          = os.getenv("FUTURES_MODE", "DEMO").upper()

        _symbols_raw       = os.getenv("FUTURES_SYMBOLS", "BTC/USDT,ETH/USDT,BNB/USDT")
        self.symbols       = [s.strip() for s in _symbols_raw.split(",") if s.strip()]

        self._last_open_ts: float = 0.0  # cooldown tracking

        # DB init
        _init_futures_table()
        _ensure_exit_price_col()

        logger.info(
            f"[FUTURES] Engine init | enabled={self.enabled} mode={self.mode} "
            f"leverage={self.leverage}x quote={self.quote} "
            f"tp={self.tp_pct}% sl={self.sl_pct}% symbols={self.symbols}"
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

            # SHORT: TP = ქვევით, SL = ზევით
            tp_price = round(current_price * (1.0 - self.tp_pct / 100.0), 6)
            sl_price = round(current_price * (1.0 + self.sl_pct / 100.0), 6)
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
        BULL market → ყველა ღია SHORT-ი დახურვა.
        DEMO: ვირტუალური close — ახლანდელი ფასით PnL გათვლა.
        """
        if not self.enabled:
            return

        open_shorts = _get_open_shorts()
        if not open_shorts:
            logger.debug("[FUTURES] CLOSE_ALL | no open shorts → skip")
            return

        logger.warning(
            f"[FUTURES] CLOSING_ALL_SHORTS | count={len(open_shorts)} reason={reason}"
        )

        for pos in open_shorts:
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
        TP hit:  current_price <= tp_price → SHORT win → close
        SL hit:  current_price >= sl_price → SHORT loss → close
        """
        if not self.enabled:
            return

        open_shorts = _get_open_shorts()
        if not open_shorts:
            return

        for pos in open_shorts:
            symbol    = str(pos.get("symbol", ""))
            tp_price  = float(pos.get("tp_price", 0.0))
            sl_price  = float(pos.get("sl_price", 0.0))

            current_price = self._fetch_price(symbol)
            if current_price <= 0:
                continue

            entry_price = float(pos.get("entry_price", 0.0))
            logger.debug(
                f"[FUTURES] TP_SL_CHECK | {symbol} "
                f"current={current_price:.4f} tp={tp_price:.4f} sl={sl_price:.4f}"
            )

            # TP hit: ბაზარი ეცა → SHORT მოიგო!
            if tp_price > 0 and current_price <= tp_price:
                logger.warning(
                    f"[FUTURES] TP_HIT | {symbol} "
                    f"current={current_price:.4f} <= tp={tp_price:.4f}"
                )
                self._close_short(pos, reason="TP")
                continue

            # SL hit: ბაზარი ავიდა → SHORT დაკარგა
            if sl_price > 0 and current_price >= sl_price:
                logger.warning(
                    f"[FUTURES] SL_HIT | {symbol} "
                    f"current={current_price:.4f} >= sl={sl_price:.4f}"
                )
                self._close_short(pos, reason="SL")
                continue

    # ────────────────────────────────────────────────────────
    # PUBLIC: get_open_shorts
    # ────────────────────────────────────────────────────────
    def get_open_shorts(self) -> List[Dict[str, Any]]:
        return _get_open_shorts()

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
