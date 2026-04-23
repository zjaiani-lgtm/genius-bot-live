# execution/futures_engine.py
# ============================================================
# GENIUS DCA Bot — Futures Engine (SHORT positions)
# SMART LONG + SHORT სტრატეგია — Bear market hedge
#
# DEMO mode: ვირტუალური SHORT — Binance Futures API არ სჭირდება
# LIVE mode: FUTURES_ENABLED=true + ცალკე Futures API key
#
# ENV პარამეტრები:
#   FUTURES_ENABLED=false              ← default გათიშული (safe!)
#   FUTURES_LEVERAGE=2                 ← x2 (უსაფრთხო)
#   FUTURES_QUOTE=85                   ← $85 per BEAR SHORT
#   FUTURES_TP_PCT=0.8                 ← 0.8% TP
#   FUTURES_SL_PCT — გათიშულია (DCA mode, sl_price=0.0 hardcoded)
#   FUTURES_SYMBOLS=BTC/USDT,ETH/USDT,BNB/USDT
#   FUTURES_MODE=DEMO                  ← DEMO/LIVE
#   FUTURES_MAX_OPEN=3                 ← max ღია SHORT-ების რაოდენობა
#   FUTURES_COOLDOWN_SECONDS=300       ← 5 წუთი SHORT-ებს შორის
#
#   DCA HEDGE SHORT (CASCADE):
#   FUTURES_DCA_HEDGE_QUOTE=20         ← $20 initial hedge
#   FUTURES_DCA_HEDGE_TP_PCT=3.5       ← 3.5% TP entry-დან
#   FUTURES_DCA_ADDON_TRIGGER_PCTS=1.0,2.2,3.5,5.0,6.5  ← multi-level triggers (ზევით)
#   FUTURES_DCA_ADDON_QUOTE=12         ← $12 per ADD-ON
#   FUTURES_DCA_MAX_ADDONS=5           ← max 5 ADD-ON
#   FUTURES_DCA_L3_TRIGGER_PCT=1.5     ← L3 EXCHANGE trigger
#   FUTURES_DCA_FC_MAX_DAYS=10         ← Force Close: max open days
#   FUTURES_DCA_FC_DRAWDOWN_PCT=22.0   ← Force Close: max +22% loss (SHORT)
#
# DB: futures_positions ცხრილი (auto-created)
#
# CHANGELOG:
#   FIX-S1: check_dca_hedge_addons() — multi-level triggers (1.0,2.2,3.5,5.0,6.5)
#            ENV: FUTURES_DCA_ADDON_TRIGGER_PCTS — indexed by add_on_count
#   FIX-S2: _close_short() — PnL იყენებს avg_entry_price-ს (არა entry_price)
#            ADD-ON-ის შემდეგ PnL სწორია
#   FIX-S3: check_tp_sl() — FC (Force Close) SHORT-ებისთვის
#            FUTURES_DCA_FC_MAX_DAYS + FUTURES_DCA_FC_DRAWDOWN_PCT
#   FIX-S4: check_dca_hedge_l3() — true LIFO: cheapest unit sell → reinvest proceeds
#            (არა full position close + fixed $20 reopen)
#   FIX-S5: per-position cooldown dict (არა global timestamp)
# ============================================================
from __future__ import annotations

import os
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

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


def _parse_list_float(name: str, default: List[float]) -> List[float]:
    """ENV-დან comma-separated float list წაკითხვა. DCA-ს იგივე pattern."""
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return [float(x.strip()) for x in raw.split(",") if x.strip()]
    except Exception:
        return default


# ─── DB init ────────────────────────────────────────────────
def _init_futures_table() -> None:
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
             is_dca_hedge, dca_pos_id, entry_price)
        )
        conn.commit()
        return cur.lastrowid


def _close_short_db(
    pos_id: int,
    exit_price: float,
    pnl_quote: float,
    pnl_pct: float,
    outcome: str,
) -> None:
    from execution.db.db import get_connection
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
                ("add_on_count",          "INTEGER DEFAULT 0"),
                ("add_on_quote",          "REAL DEFAULT 0.0"),
                ("avg_entry_price",       "REAL DEFAULT 0.0"),
                ("sl_price_addon",        "REAL DEFAULT 0.0"),
                ("is_dca_hedge",          "INTEGER DEFAULT 0"),
                ("dca_pos_id",            "INTEGER DEFAULT 0"),
                # INDEPENDENT SHORT DCA — ახალი სარკე სისტემა
                ("is_independent_short",  "INTEGER DEFAULT 0"),  # 1=independent SHORT DCA
                ("long_ref_price",        "REAL DEFAULT 0.0"),   # LONG L1 entry reference
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

    SHORT CASCADE (DCA hedge mirror):
      trigger: DCA add_on_count == max_add_ons
      ADD-ON #1-5: +1.0%, +2.2%, +3.5%, +5.0%, +6.5% ↑ (indexed)
      L3: LIFO — cheapest unit sell → reinvest proceeds
      FC: 10 days / +22% drawdown (mirror of DCA FC)
    """

    def __init__(self) -> None:
        self.enabled       = _eb("FUTURES_ENABLED", False)
        self.leverage      = _ei("FUTURES_LEVERAGE", 2)
        self.quote         = _ef("FUTURES_QUOTE", 85.0)
        self.tp_pct        = _ef("FUTURES_TP_PCT", 0.8)
        self.sl_pct        = 0.0
        self.max_open      = _ei("FUTURES_MAX_OPEN", 3)
        self.cooldown_s    = _ei("FUTURES_COOLDOWN_SECONDS", 300)
        self.mode          = os.getenv("FUTURES_MODE", "DEMO").upper()

        _symbols_raw       = os.getenv("FUTURES_SYMBOLS", "BTC/USDT,ETH/USDT,BNB/USDT")
        self.symbols       = [s.strip() for s in _symbols_raw.split(",") if s.strip()]

        self._last_open_ts: float = 0.0

        # BEAR SHORT ADD-ON
        self.addon_enabled      = _eb("FUTURES_ADDON_ENABLED", True)
        self.addon_trigger_pct  = _ef("FUTURES_ADDON_TRIGGER_PCT", 0.25)
        self.addon_quote        = _ef("FUTURES_ADDON_QUOTE", 50.0)
        self.addon_sl_pct       = 0.0
        self.max_addons         = _ei("FUTURES_MAX_ADDONS", 3)

        # BEAR SHORT EXCHANGE
        self.exchange_enabled     = _eb("FUTURES_EXCHANGE_ENABLED", True)
        self.exchange_trigger_pct = _ef("FUTURES_EXCHANGE_TRIGGER_PCT", 2.5)

        # ── DCA HEDGE SHORT CASCADE პარამეტრები ──────────────
        # ყველა პარამეტრი ENV-კონტროლირებადია
        self.dca_hedge_quote      = _ef("FUTURES_DCA_HEDGE_QUOTE",    20.0)
        self.dca_hedge_tp_pct     = _ef("FUTURES_DCA_HEDGE_TP_PCT",    3.5)
        self.dca_hedge_max_addons = _ei("FUTURES_DCA_MAX_ADDONS",        5)
        self.dca_hedge_l3_trigger_pct = _ef("FUTURES_DCA_L3_TRIGGER_PCT", 1.5)

        # FIX-S1: multi-level triggers — indexed by add_on_count
        # ENV: FUTURES_DCA_ADDON_TRIGGER_PCTS=1.0,2.2,3.5,5.0,6.5
        # mirror of DCA DCA_ADDON_TRIGGER_PCTS (ზევით bounce %-ები)
        self.dca_hedge_addon_trigger_pcts = _parse_list_float(
            "FUTURES_DCA_ADDON_TRIGGER_PCTS",
            [1.0, 2.2, 3.5, 5.0, 6.5],
        )
        self.dca_hedge_addon_quote = _ef("FUTURES_DCA_ADDON_QUOTE", 12.0)

        # FIX-S3: Force Close — ENV-კონტროლირებადი
        # SHORT FC: days ან +drawdown% (ზევით — SHORT-ისთვის ზარალი)
        self.dca_hedge_fc_max_days     = _ef("FUTURES_DCA_FC_MAX_DAYS",     10.0)
        self.dca_hedge_fc_drawdown_pct = _ef("FUTURES_DCA_FC_DRAWDOWN_PCT", 22.0)

        # FIX-S5: per-position cooldown (არა global timestamp)
        # key: pos_id → last_addon_ts
        self._hedge_addon_cooldown_map: Dict[int, float] = {}
        self._hedge_addon_cooldown_s = 180  # 3 წუთი ADD-ON-ებს შორის
        self._last_hedge_l3_ts: float = 0.0

        # ── INDEPENDENT SHORT DCA პარამეტრები ────────────────
        # სარკე სისტემა: LONG-ის L2-L3 midpoint-ზე SHORT იხსნება
        # ADD-ONs ვარდნაზე (LONG ADD-ON-ების სარკე)
        # TP + FC — სავალდებულო დახურვა, ღია ტრეიდი არ რჩება
        self.short_dca_enabled       = _eb("SHORT_DCA_ENABLED",        False)
        self.short_l1_trigger_pct    = _ef("SHORT_L1_TRIGGER_PCT",      1.6)
        self.short_addon_trigger_pcts = _parse_list_float(
            "SHORT_ADDON_TRIGGER_PCTS", [1.0, 2.2, 3.5]
        )
        self.short_addon_quote       = _ef("SHORT_ADDON_QUOTE",         25.0)
        self.short_max_addons        = _ei("SHORT_MAX_ADDONS",           3)
        self.short_tp_pct            = _ef("SHORT_TP_PCT",               0.55)
        self.short_fc_max_days       = _ef("SHORT_FC_MAX_DAYS",         10.0)
        self.short_fc_drawdown_pct   = _ef("SHORT_FC_DRAWDOWN_PCT",     15.0)

        # per-position cooldown for independent SHORT ADD-ONs
        self._short_addon_cooldown_map: Dict[int, float] = {}
        self._short_addon_cooldown_s = 300  # 5 წუთი ADD-ON-ებს შორის

        # DB init
        _init_futures_table()
        _ensure_exit_price_col()
        _ensure_addon_cols()

        logger.info(
            f"[FUTURES] Engine init | enabled={self.enabled} mode={self.mode} "
            f"leverage={self.leverage}x quote={self.quote} "
            f"tp={self.tp_pct}% symbols={self.symbols}"
        )
        logger.info(
            f"[FUTURES] DCA HEDGE CASCADE | "
            f"quote={self.dca_hedge_quote} tp={self.dca_hedge_tp_pct}% "
            f"addon_triggers={self.dca_hedge_addon_trigger_pcts} "
            f"addon_quote={self.dca_hedge_addon_quote} "
            f"max_addons={self.dca_hedge_max_addons} "
            f"l3_trigger={self.dca_hedge_l3_trigger_pct}% "
            f"fc_days={self.dca_hedge_fc_max_days}d "
            f"fc_drawdown={self.dca_hedge_fc_drawdown_pct}%"
        )
        logger.info(
            f"[SHORT_DCA] INDEPENDENT SHORT | enabled={self.short_dca_enabled} "
            f"l1_trigger={self.short_l1_trigger_pct}% "
            f"addon_triggers={self.short_addon_trigger_pcts} "
            f"addon_quote={self.short_addon_quote} "
            f"max_addons={self.short_max_addons} "
            f"tp={self.short_tp_pct}% "
            f"fc_days={self.short_fc_max_days}d "
            f"fc_drawdown={self.short_fc_drawdown_pct}%"
        )

    def _fetch_price(self, symbol: str) -> float:
        """ახლანდელი ფასი — public API (DEMO/LIVE ორივეზე)."""
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
        try:
            import ccxt
            exchange = ccxt.binance({"enableRateLimit": True})
            ticker = exchange.fetch_ticker("BTC/USDT")
            last  = float(ticker.get("last") or 0.0)
            prev  = float(ticker.get("previousClose") or ticker.get("open") or 0.0)
            if prev <= 0:
                return 0.0
            return (last - prev) / prev * 100.0
        except Exception as e:
            logger.warning(f"[FUTURES] BTC_CHANGE_FAIL | err={e}")
            return 0.0

    # ────────────────────────────────────────────────────────
    # INTERNAL: _close_short  [FIX-S2]
    # PnL = avg_entry_price × qty (არა original entry_price)
    # ADD-ON-ის შემდეგ avg_entry_price განახლდება DB-ში →
    # _close_short-ი ამ განახლებულ avg-ს კითხულობს → PnL სწორია
    # ────────────────────────────────────────────────────────
    def _close_short(self, pos: Dict[str, Any], reason: str = "MANUAL") -> None:
        """
        ერთი SHORT position-ის დახურვა.

        FIX-S2: PnL-ი avg_entry_price-ზეა (არა entry_price).
        ADD-ON-ის შემდეგ avg_entry_price > entry_price (SHORT-ი ძვირდება) →
        PnL = (avg_entry - exit) × qty — ზუსტი weighted average PnL.

        edge case: avg_entry_price=0 (ძველი DB row მიგრაციამდე) →
        fallback entry_price — არ ინახება ზარალი.
        """
        try:
            symbol      = str(pos.get("symbol", ""))
            pos_id      = int(pos.get("id", 0))
            qty         = float(pos.get("qty", 0.0))

            # FIX-S2: avg_entry_price გამოიყენება (entry_price-ის მაგივრად)
            avg_entry   = float(pos.get("avg_entry_price", 0.0) or 0.0)
            entry_price = float(pos.get("entry_price", 0.0))
            if avg_entry <= 0:
                avg_entry = entry_price  # fallback ძველი DB rows-ისთვის

            exit_price = self._fetch_price(symbol)
            if exit_price <= 0:
                exit_price = avg_entry  # safe fallback

            # SHORT PnL: avg_entry > exit → profit; avg_entry < exit → loss
            price_diff = avg_entry - exit_price
            pnl_quote  = round(price_diff * qty, 4)
            pnl_pct    = round((price_diff / avg_entry) * 100.0 * self.leverage, 2) if avg_entry > 0 else 0.0

            _close_short_db(pos_id, exit_price, pnl_quote, pnl_pct, reason)

            logger.warning(
                f"[FUTURES] SHORT_CLOSED | {symbol} "
                f"avg_entry={avg_entry:.4f} exit={exit_price:.4f} "
                f"pnl={pnl_quote:+.4f} pct={pnl_pct:+.2f}% reason={reason}"
            )

            try:
                from execution.telegram_notifier import notify_short_closed
                notify_short_closed(
                    symbol=symbol,
                    entry_price=avg_entry,
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
                    f"sym={symbol} avg_entry={avg_entry:.4f} exit={exit_price:.4f} "
                    f"pnl={pnl_quote:+.4f} reason={reason}"
                )
            except Exception:
                pass

            # FIX-S5: cleanup per-position cooldown (hedge + independent)
            self._hedge_addon_cooldown_map.pop(pos_id, None)
            self._short_addon_cooldown_map.pop(pos_id, None)

        except Exception as e:
            logger.error(f"[FUTURES] CLOSE_SHORT_FAIL | id={pos.get('id')} err={e}")

    # ────────────────────────────────────────────────────────
    # PUBLIC: check_and_open_short
    # ────────────────────────────────────────────────────────
    def check_and_open_short(self, market_regime: str) -> None:
        if not self.enabled:
            return
        if market_regime != "BEAR":
            return

        elapsed = time.time() - self._last_open_ts
        if elapsed < self.cooldown_s:
            return

        open_shorts = _get_open_shorts()
        if len(open_shorts) >= self.max_open:
            return

        for sym in self.symbols:
            existing = _get_open_short_for_symbol(sym)
            if existing:
                continue
            self._open_short(sym)

    def _open_short(self, symbol: str) -> None:
        try:
            current_price = self._fetch_price(symbol)
            if current_price <= 0:
                return

            tp_price = round(current_price * (1.0 - self.tp_pct / 100.0), 6)
            sl_price = 0.0
            qty      = round((self.quote * self.leverage) / current_price, 6)
            sig_id   = f"FUT-{symbol.replace('/', '')}-{uuid.uuid4().hex[:8]}"

            pos_id = _open_short_db(
                signal_id=sig_id,
                symbol=symbol,
                entry_price=current_price,
                qty=qty,
                quote_in=self.quote,
                leverage=self.leverage,
                tp_price=tp_price,
                sl_price=sl_price,
                mode=self.mode,
            )
            logger.warning(
                f"[FUTURES] SHORT_OPENED | {symbol} "
                f"entry={current_price:.4f} tp={tp_price:.4f} "
                f"qty={qty:.6f} quote={self.quote} lev={self.leverage}x"
            )
            self._last_open_ts = time.time()

            try:
                from execution.telegram_notifier import notify_short_opened
                notify_short_opened(
                    symbol=symbol, entry_price=current_price,
                    tp_price=tp_price, sl_price=sl_price,
                    quote=self.quote, leverage=self.leverage, mode=self.mode,
                )
            except Exception as _tg:
                logger.warning(f"[FUTURES] TG_OPEN_FAIL | err={_tg}")

            try:
                from execution.db.repository import log_event
                log_event("FUTURES_SHORT_OPENED",
                    f"sym={symbol} entry={current_price:.4f} tp={tp_price:.4f} "
                    f"quote={self.quote} lev={self.leverage}x")
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
        ისინი DCA position lifecycle-ს მიყვება.
        """
        if not self.enabled:
            return

        open_shorts = _get_open_shorts()
        if not open_shorts:
            return

        bear_shorts  = [p for p in open_shorts if not int(p.get("is_dca_hedge", 0) or 0)
                        and not int(p.get("is_independent_short", 0) or 0)]
        hedge_shorts = [p for p in open_shorts if int(p.get("is_dca_hedge", 0) or 0)]
        indep_shorts = [p for p in open_shorts if int(p.get("is_independent_short", 0) or 0)]

        if hedge_shorts:
            logger.info(
                f"[FUTURES] CLOSE_ALL | skipping {len(hedge_shorts)} DCA hedge SHORT(s) "
                f"— they close via TP or close_dca_hedge_for_position()"
            )
        if indep_shorts:
            logger.info(
                f"[FUTURES] CLOSE_ALL | skipping {len(indep_shorts)} independent SHORT(s) "
                f"— they close via own TP/FC lifecycle"
            )

        for pos in bear_shorts:
            self._close_short(pos, reason=reason)

    # ────────────────────────────────────────────────────────
    # PUBLIC: check_tp_sl  [FIX-S3 — FC დამატება]
    # ────────────────────────────────────────────────────────
    def check_tp_sl(self) -> None:
        """
        main loop-ში ყოველ iteration-ზე.

        FIX-S3: Force Close დამატება hedge SHORT-ებისთვის:
          - FC by time: opened_at + FUTURES_DCA_FC_MAX_DAYS
          - FC by drawdown: current > avg_entry × (1 + FUTURES_DCA_FC_DRAWDOWN_PCT%)
            (SHORT-ისთვის ზევით მოძრაობა = ზარალი)

        edge cases:
          - opened_at NULL → FC by time skip (safe)
          - avg_entry_price = 0 → FC by drawdown skip (safe)
          - non-hedge SHORTs: FC skip (მხოლოდ BEAR SHORTs — TP-ს ელოდება)
        """
        if not self.enabled:
            return

        open_shorts = _get_open_shorts()
        if not open_shorts:
            return

        for pos in open_shorts:
            symbol    = str(pos.get("symbol", ""))
            tp_price  = float(pos.get("tp_price", 0.0))
            is_hedge  = int(pos.get("is_dca_hedge", 0) or 0)
            is_indep  = int(pos.get("is_independent_short", 0) or 0)

            current_price = self._fetch_price(symbol)
            if current_price <= 0:
                continue

            avg_entry = float(pos.get("avg_entry_price", 0.0) or 0.0)
            if avg_entry <= 0:
                avg_entry = float(pos.get("entry_price", 0.0))

            logger.debug(
                f"[FUTURES] TP_SL_CHECK | {symbol} is_hedge={is_hedge} "
                f"current={current_price:.4f} tp={tp_price:.4f}"
            )

            # ── TP hit ──────────────────────────────────────
            if tp_price > 0 and current_price <= tp_price:
                logger.warning(
                    f"[FUTURES] TP_HIT | {symbol} "
                    f"current={current_price:.4f} <= tp={tp_price:.4f}"
                )
                self._close_short(pos, reason="TP")
                continue

            # ── FIX-S3: Force Close ──────────────────────────
            # hedge SHORTs: FUTURES_DCA_FC_* params
            # independent SHORTs: SHORT_FC_* params
            if not is_hedge and not is_indep:
                continue  # BEAR SHORT — FC არ ვრთავთ (TP-ს ელოდება)

            if is_indep:
                # INDEPENDENT SHORT — საკუთარი FC params
                fc_reason = self._check_independent_short_fc(pos, current_price, avg_entry)
            else:
                # DCA HEDGE SHORT — hedge FC params
                fc_reason = self._check_hedge_force_close(pos, current_price, avg_entry)

            if fc_reason:
                logger.warning(
                    f"[FUTURES] HEDGE_FC | {symbol} reason={fc_reason} "
                    f"current={current_price:.4f} avg_entry={avg_entry:.4f}"
                )
                self._close_short(pos, reason=f"HEDGE_FC_{fc_reason}")

                try:
                    from execution.telegram_notifier import send_telegram_message
                    pnl_approx = round((avg_entry - current_price) * float(pos.get("qty", 0.0)), 4)
                    fc_type = "INDEPENDENT SHORT" if is_indep else "HEDGE SHORT"
                    send_telegram_message(
                        f"⛔ <b>{fc_type} FORCE CLOSE</b>\n\n"
                        f"🪙 <b>Symbol:</b> <code>{symbol}</code>\n"
                        f"📋 <b>Reason:</b> <code>{fc_reason}</code>\n"
                        f"💰 <b>avg_entry:</b> <code>{avg_entry:.2f}</code>\n"
                        f"📈 <b>exit:</b> <code>{current_price:.2f}</code>\n"
                        f"💸 <b>PnL:</b> <code>{pnl_approx:+.4f} USDT</code>\n"
                        f"🕒 <code>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</code>"
                    )
                except Exception as _tg:
                    logger.warning(f"[FUTURES] FC_TG_FAIL | err={_tg}")
                continue  # FC დასრულდა — შემდეგ position-ზე გადასვლა

    def _check_hedge_force_close(
        self,
        pos: Dict[str, Any],
        current_price: float,
        avg_entry: float,
    ) -> Optional[str]:
        """
        Returns FC reason string ან None.

        FC by time: opened_at → days_open >= dca_hedge_fc_max_days
        FC by drawdown: current >= avg_entry × (1 + fc_drawdown_pct%)
          SHORT-ისთვის ზევით მოძრაობა = ზარალი
          FC drawdown % = (current - avg_entry) / avg_entry × 100
        """
        # ── FC by time ──────────────────────────────────────
        if self.dca_hedge_fc_max_days > 0:
            opened_at_str = str(pos.get("opened_at", "") or "")
            if opened_at_str:
                try:
                    opened_dt = datetime.fromisoformat(
                        opened_at_str.replace("Z", "+00:00")
                    )
                    if opened_dt.tzinfo is None:
                        opened_dt = opened_dt.replace(tzinfo=timezone.utc)
                    days_open = (datetime.now(timezone.utc) - opened_dt).total_seconds() / 86400.0
                    if days_open >= self.dca_hedge_fc_max_days:
                        return f"MAX_DAYS_{days_open:.1f}d>={self.dca_hedge_fc_max_days:.0f}d"
                except Exception as _e:
                    logger.warning(f"[FUTURES] FC_TIME_PARSE_FAIL | err={_e}")

        # ── FC by drawdown (SHORT: ზევით = ზარალი) ─────────
        if self.dca_hedge_fc_drawdown_pct > 0 and avg_entry > 0:
            upside_pct = (current_price - avg_entry) / avg_entry * 100.0
            if upside_pct >= self.dca_hedge_fc_drawdown_pct:
                return f"DRAWDOWN_{upside_pct:.2f}%>={self.dca_hedge_fc_drawdown_pct:.1f}%"

        return None

    # ────────────────────────────────────────────────────────
    # PUBLIC: get_open_shorts
    # ────────────────────────────────────────────────────────
    def get_open_shorts(self) -> List[Dict[str, Any]]:
        return _get_open_shorts()

    # ────────────────────────────────────────────────────────
    # PUBLIC: check_and_addon_short  (BEAR SHORT ADD-ON)
    # ────────────────────────────────────────────────────────
    def check_and_addon_short(self) -> None:
        """BEAR SHORT ADD-ON — single trigger (BEAR shorts, არა hedge)."""
        if not self.enabled or not self.addon_enabled:
            return

        open_shorts = _get_open_shorts()
        for pos in open_shorts:
            if int(pos.get("is_dca_hedge", 0) or 0):
                continue  # hedge shorts — ცალკე მეთოდი
            try:
                symbol       = str(pos.get("symbol", ""))
                pos_id       = int(pos.get("id", 0))
                entry_price  = float(pos.get("entry_price", 0.0))
                add_on_count = int(pos.get("add_on_count", 0) or 0)
                quote_in     = float(pos.get("quote_in", 0.0))
                avg_entry    = float(pos.get("avg_entry_price", 0.0) or entry_price)

                if add_on_count >= self.max_addons:
                    continue

                current_price = self._fetch_price(symbol)
                if current_price <= 0:
                    continue

                trigger_price = entry_price * (1.0 + self.addon_trigger_pct / 100.0)
                if current_price < trigger_price:
                    continue

                total_quote = quote_in + self.addon_quote
                new_avg     = (avg_entry * quote_in + current_price * self.addon_quote) / total_quote
                new_tp      = round(new_avg * (1.0 - self.tp_pct / 100.0), 6)
                new_qty     = round((total_quote * self.leverage) / new_avg, 6)

                logger.warning(
                    f"[FUTURES] BEAR_ADDON | {symbol} "
                    f"avg={avg_entry:.2f}→{new_avg:.2f} tp={new_tp:.2f}"
                )

                from execution.db.db import get_connection
                with get_connection() as conn:
                    conn.execute("""
                        UPDATE futures_positions SET
                            add_on_count=?, add_on_quote=?, avg_entry_price=?,
                            tp_price=?, sl_price_addon=?, qty=?, quote_in=?
                        WHERE id=?
                    """, (add_on_count + 1, self.addon_quote, round(new_avg, 6),
                          new_tp, 0.0, new_qty, total_quote, pos_id))
                    conn.commit()

            except Exception as e:
                logger.error(f"[FUTURES] BEAR_ADDON_ERR | {pos.get('symbol')} err={e}")

    # ────────────────────────────────────────────────────────
    # PUBLIC: check_and_exchange_short  (BEAR SHORT EXCHANGE)
    # ────────────────────────────────────────────────────────
    def check_and_exchange_short(self) -> None:
        if not self.enabled or not self.exchange_enabled:
            return

        open_shorts = _get_open_shorts()
        for pos in open_shorts:
            if int(pos.get("is_dca_hedge", 0) or 0):
                continue
            try:
                symbol       = str(pos.get("symbol", ""))
                pos_id       = int(pos.get("id", 0))
                entry_price  = float(pos.get("entry_price", 0.0))
                add_on_count = int(pos.get("add_on_count", 0) or 0)

                if add_on_count < 1:
                    continue

                current_price = self._fetch_price(symbol)
                if current_price <= 0:
                    continue

                exchange_trigger = entry_price * (1.0 + self.exchange_trigger_pct / 100.0)
                if current_price < exchange_trigger:
                    continue

                logger.warning(
                    f"[FUTURES] SHORT_EXCHANGE | {symbol} "
                    f"old={entry_price:.2f} new={current_price:.2f}"
                )

                old_qty    = float(pos.get("qty", 0.0))
                price_diff = entry_price - current_price
                pnl_quote  = round(price_diff * old_qty, 4)
                pnl_pct    = round((price_diff / entry_price) * 100.0 * self.leverage, 2)
                _close_short_db(pos_id, current_price, pnl_quote, pnl_pct, "EXCHANGE")
                self._open_short(symbol)

            except Exception as e:
                logger.error(f"[FUTURES] EXCHANGE_ERR | {pos.get('symbol')} err={e}")

    def check_addon_sl(self) -> None:
        """DCA mode: SL გათიშულია. Stub — no-op."""
        pass

    # ────────────────────────────────────────────────────────
    # PUBLIC: open_dca_hedge_short
    # ────────────────────────────────────────────────────────
    def open_dca_hedge_short(
        self,
        symbol: str,
        current_price: float,
        dca_pos_id: int,
    ) -> bool:
        """
        DCA L2/L3 boundary → SHORT hedge გახსნა.
        trigger: add_on_count == max_add_ons
        """
        if not self.enabled:
            return False

        try:
            from execution.db.db import get_connection
            with get_connection() as conn:
                row = conn.execute(
                    "SELECT id FROM futures_positions "
                    "WHERE dca_pos_id=? AND is_dca_hedge=1 AND status='OPEN'",
                    (dca_pos_id,)
                ).fetchone()
                if row:
                    logger.debug(f"[HEDGE] ALREADY_OPEN | dca_pos_id={dca_pos_id} → skip")
                    return False
        except Exception as e:
            logger.warning(f"[HEDGE] CHECK_FAIL | err={e}")
            return False

        hedge_quote = self.dca_hedge_quote
        try:
            from execution.dca_risk_manager import get_risk_manager as _rm
            bal_ok, bal_reason = _rm().can_l3_operation(hedge_quote)
            if not bal_ok:
                logger.warning(f"[HEDGE] BALANCE_BLOCK | {symbol} reason={bal_reason}")
                return False
        except Exception as e:
            logger.warning(f"[HEDGE] BALANCE_CHECK_FAIL | err={e} → blocking hedge (safe)")
            return False  # balance check-ის შეცდომა → safe-side: hedge არ გაიხსნება

        tp_price = round(current_price * (1.0 - self.dca_hedge_tp_pct / 100.0), 6)
        qty      = round((hedge_quote * self.leverage) / current_price, 6)
        sig_id   = f"HEDGE-{symbol.replace('/', '')}-{uuid.uuid4().hex[:8]}"

        pos_id = _open_short_db(
            signal_id=sig_id, symbol=symbol,
            entry_price=current_price, qty=qty, quote_in=hedge_quote,
            leverage=self.leverage, tp_price=tp_price, sl_price=0.0,
            mode=self.mode, is_dca_hedge=1, dca_pos_id=dca_pos_id,
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
                f"📉 <b>DCA L2 exhausted — CASCADE active</b>\n"
                f"💰 <b>Entry:</b> <code>{current_price:.2f}</code>\n"
                f"🎯 <b>TP:</b> <code>{tp_price:.2f}</code> "
                f"(<code>-{self.dca_hedge_tp_pct:.1f}%</code>)\n"
                f"📊 <b>ADD-ON triggers:</b> "
                f"<code>{self.dca_hedge_addon_trigger_pcts}</code>\n"
                f"💼 <b>Quote:</b> <code>${hedge_quote:.0f}</code> "
                f"<code>×{self.leverage}</code>\n"
                f"🕒 <code>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</code>"
            )
        except Exception as _tg:
            logger.warning(f"[HEDGE] TG_FAIL | err={_tg}")

        try:
            from execution.db.repository import log_event
            log_event("DCA_HEDGE_OPENED",
                f"sym={symbol} entry={current_price:.4f} "
                f"tp={tp_price:.4f} quote={hedge_quote} "
                f"dca_pos_id={dca_pos_id} pos_id={pos_id}"
            )
        except Exception:
            pass

        return True

    # ────────────────────────────────────────────────────────
    # PUBLIC: check_dca_hedge_addons  [FIX-S1 + FIX-S5]
    # ────────────────────────────────────────────────────────
    def check_dca_hedge_addons(self) -> None:
        """
        DCA hedge SHORT CASCADE — ADD-ON შემოწმება.

        FIX-S1: multi-level triggers indexed by add_on_count:
          ADD-ON #1: entry × (1 + triggers[0]) = entry × 1.010  (+1.0%)
          ADD-ON #2: entry × (1 + triggers[1]) = entry × 1.022  (+2.2%)
          ADD-ON #3: entry × (1 + triggers[2]) = entry × 1.035  (+3.5%)
          ADD-ON #4: entry × (1 + triggers[3]) = entry × 1.050  (+5.0%)
          ADD-ON #5: entry × (1 + triggers[4]) = entry × 1.065  (+6.5%)

        trigger reference: entry_price (ორიგინალი SHORT entry, ფიქსირებული)
        ეს სწორია — DCA-ს მსგავსად: avg_entry-დან კი არა,
        original entry-დან ვითვლით drawdown/bounce %-ს.

        FIX-S5: per-position cooldown — BTC ADD-ON არ ბლოკავს ETH-ს.

        edge cases:
          - add_on_count >= len(triggers) → no trigger defined → skip
          - add_on_count >= max_addons → exhausted → skip (L3 აიღებს)
          - entry_price = 0 → skip (invalid pos)
        """
        if not self.enabled:
            return

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

                if entry_price <= 0:
                    continue

                # ADD-ONs exhausted → L3 აიღებს
                if add_on_count >= self.dca_hedge_max_addons:
                    continue

                # trigger list bounds check
                if add_on_count >= len(self.dca_hedge_addon_trigger_pcts):
                    logger.warning(
                        f"[HEDGE] ADDON_NO_TRIGGER | {symbol} "
                        f"add_on={add_on_count} triggers_len={len(self.dca_hedge_addon_trigger_pcts)}"
                    )
                    continue

                # FIX-S5: per-position cooldown
                last_ts = self._hedge_addon_cooldown_map.get(pos_id, 0.0)
                if (time.time() - last_ts) < self._hedge_addon_cooldown_s:
                    remaining = int(self._hedge_addon_cooldown_s - (time.time() - last_ts))
                    logger.debug(
                        f"[HEDGE] ADDON_COOLDOWN | {symbol} pos_id={pos_id} "
                        f"remaining={remaining}s"
                    )
                    continue

                current_price = self._fetch_price(symbol)
                if current_price <= 0:
                    continue

                # FIX-S1: indexed trigger
                trigger_pct   = self.dca_hedge_addon_trigger_pcts[add_on_count]
                trigger_price = entry_price * (1.0 + trigger_pct / 100.0)

                if current_price < trigger_price:
                    logger.debug(
                        f"[HEDGE] ADDON_WAIT | {symbol} "
                        f"level={add_on_count+1} "
                        f"price={current_price:.2f} trigger={trigger_price:.2f} "
                        f"(+{trigger_pct:.1f}% from entry={entry_price:.2f})"
                    )
                    continue

                # ADD-ON — weighted avg გათვლა
                total_quote = quote_in + self.dca_hedge_addon_quote
                new_avg     = (avg_entry * quote_in + current_price * self.dca_hedge_addon_quote) / total_quote
                new_tp      = round(new_avg * (1.0 - self.dca_hedge_tp_pct / 100.0), 6)
                new_qty     = round((total_quote * self.leverage) / new_avg, 6)

                logger.warning(
                    f"[HEDGE] ADDON | {symbol} level={add_on_count+1} "
                    f"trigger=+{trigger_pct:.1f}% "
                    f"entry={entry_price:.2f} current={current_price:.2f} "
                    f"avg={avg_entry:.2f}→{new_avg:.2f} tp={new_tp:.2f}"
                )

                from execution.db.db import get_connection
                with get_connection() as conn:
                    conn.execute(
                        """
                        UPDATE futures_positions SET
                            add_on_count=?, add_on_quote=?, avg_entry_price=?,
                            tp_price=?, qty=?, quote_in=?
                        WHERE id=?
                        """,
                        (add_on_count + 1, self.dca_hedge_addon_quote,
                         round(new_avg, 6), new_tp, new_qty, total_quote, pos_id)
                    )
                    conn.commit()

                # FIX-S5: per-position cooldown განახლება
                self._hedge_addon_cooldown_map[pos_id] = time.time()

                try:
                    from execution.telegram_notifier import send_telegram_message
                    next_trigger = (
                        f"+{self.dca_hedge_addon_trigger_pcts[add_on_count+1]:.1f}%"
                        if (add_on_count + 1) < len(self.dca_hedge_addon_trigger_pcts)
                        else "L3"
                    )
                    send_telegram_message(
                        f"➕ <b>HEDGE CASCADE ADD-ON #{add_on_count+1}</b>\n\n"
                        f"🪙 <b>Symbol:</b> <code>{symbol}</code>\n"
                        f"📈 <b>BTC bounce:</b> <code>+{trigger_pct:.1f}%</code> "
                        f"from entry <code>{entry_price:.2f}</code>\n"
                        f"💰 <b>ADD-ON @ </b><code>{current_price:.2f}</code>\n"
                        f"📊 <b>avg_short:</b> <code>{avg_entry:.2f} → {new_avg:.2f}</code> ↑\n"
                        f"🎯 <b>new TP:</b> <code>{new_tp:.2f}</code>\n"
                        f"⏭ <b>Next trigger:</b> <code>{next_trigger}</code>\n"
                        f"🕒 <code>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</code>"
                    )
                except Exception as _tg:
                    logger.warning(f"[HEDGE] ADDON_TG_FAIL | err={_tg}")

                try:
                    from execution.db.repository import log_event
                    log_event("DCA_HEDGE_ADDON",
                        f"sym={symbol} level={add_on_count+1} "
                        f"trigger=+{trigger_pct:.1f}% "
                        f"addon_price={current_price:.4f} "
                        f"new_avg={new_avg:.4f} new_tp={new_tp:.4f}"
                    )
                except Exception:
                    pass

            except Exception as e:
                logger.error(f"[HEDGE] ADDON_ERR | {pos.get('symbol')} err={e}")

    # ────────────────────────────────────────────────────────
    # PUBLIC: check_dca_hedge_l3  [FIX-S4 — true LIFO]
    # ────────────────────────────────────────────────────────
    def check_dca_hedge_l3(self) -> None:
        """
        DCA hedge L3 — ADD-ONs exhausted + BTC კვლავ ↑.

        FIX-S4: TRUE LIFO (არა full position close):
          ყველაზე იაფი SHORT unit = lowest entry_price (ყველაზე დიდი ზარალი)
          unit sell proceeds → reinvest @ current (higher = better SHORT entry)
          avg_short ↑, TP_short ↑

        SHORT LIFO unit = cheapest buy (lowest entry_price ADD-ON):
          ლოგიკა: ყველაზე დაბალი entry_price ADD-ON იყო ყველაზე ადრე,
          BTC ეცა → ეს unit ყველაზე ძვირად გაიყიდება ახლა (ახლა BTC ამაღლდა).
          wait — SHORT-ისთვის: ეგ unit -ს ყველაზე მეტი ზარალი აქვს
          (entry LOW, exit HIGH = SHORT ზარალი).
          სწორი LIFO: ყველაზე LOW entry ADD-ON → sell (realize loss) → reinvest HIGH.
          new entry = HIGH → TP = HIGH × (1 - 3.5%) → TP ახლოვდება current-თან.

        implementation:
          1. DB-დან hedge unit-ების ADD-ON ჩანაწერები — ვერ ვინახავთ ცალ-ცალკე.
             გამოსავალი: avg_entry_price-დან ვთვლით unit-ს.
             LIFO unit: entry_price (ორიგინალი) — ყველაზე დაბალი ADD-ON ფასი.
          2. unit qty = dca_hedge_quote / entry_price (initial unit)
          3. proceeds = unit_qty × current_price × (1 - 0.001 fee)
          4. reinvest proceeds @ current → new partial qty
          5. remaining position = (total_qty - unit_qty) + new_partial_qty
          6. new_avg = weighted average

        edge case: თუ LIFO unit qty > total qty/2 →
          reinvest-ი მთელი position-ის სიდიდეს ცვლის → cap at 50% of total_qty.

        NOTE: ეს approximation-ია (exact per-unit DB tracking-ის გარეშე).
        Production-grade დამატება: dca_hedge_orders ცხრილი (like dca_orders).
        ახლა: initial unit LIFO — ყველაზე მარტივი და სწორი approximation.
        """
        if not self.enabled:
            return

        _l3_cooldown = 300
        if (time.time() - self._last_hedge_l3_ts) < _l3_cooldown:
            return

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
            logger.warning(f"[HEDGE] L3_FETCH_FAIL | err={e}")
            return

        for pos in hedge_shorts:
            try:
                symbol       = str(pos.get("symbol", ""))
                pos_id       = int(pos.get("id", 0))
                entry_price  = float(pos.get("entry_price", 0.0))
                add_on_count = int(pos.get("add_on_count", 0) or 0)
                dca_pos_id   = int(pos.get("dca_pos_id", 0) or 0)
                total_qty    = float(pos.get("qty", 0.0))
                total_quote  = float(pos.get("quote_in", 0.0))
                avg_entry    = float(pos.get("avg_entry_price", 0.0) or entry_price)

                if add_on_count < self.dca_hedge_max_addons:
                    continue

                if entry_price <= 0 or total_qty <= 0:
                    continue

                current_price = self._fetch_price(symbol)
                if current_price <= 0:
                    continue

                # L3 trigger: avg_entry-დან (entry_price-ის მაგივრად)
                # avg_entry = weighted average after ADD-ONs
                # trigger reference: avg_entry × (1 + L3_TRIGGER_PCT%)
                l3_trigger = avg_entry * (1.0 + self.dca_hedge_l3_trigger_pct / 100.0)
                if current_price < l3_trigger:
                    logger.debug(
                        f"[HEDGE] L3_WAIT | {symbol} "
                        f"price={current_price:.2f} trigger={l3_trigger:.2f} "
                        f"(avg={avg_entry:.2f} +{self.dca_hedge_l3_trigger_pct:.1f}%)"
                    )
                    continue

                logger.warning(
                    f"[HEDGE] L3_LIFO | {symbol} "
                    f"avg_entry={avg_entry:.2f} current={current_price:.2f} "
                    f"trigger={l3_trigger:.2f} → LIFO rotation"
                )

                # FIX-S4: TRUE LIFO
                # LIFO unit = initial hedge (entry_price = ყველაზე დაბალი = ყველაზე "ძვირი" ზარალი)
                # unit quote = dca_hedge_quote (initial quote)
                lifo_entry   = entry_price                              # ყველაზე დაბალი entry
                lifo_quote   = self.dca_hedge_quote                     # initial hedge quote
                lifo_qty     = round(lifo_quote * self.leverage / lifo_entry, 6)

                # safety cap: max 50% of total position
                if lifo_qty > total_qty * 0.5:
                    lifo_qty = round(total_qty * 0.5, 6)

                # SHORT LIFO sell (realized loss since lifo_entry < current)
                sell_proceeds  = lifo_qty * current_price
                fee            = sell_proceeds * 0.001   # 0.1% Binance fee
                net_proceeds   = sell_proceeds - fee
                realized_pnl   = (lifo_entry - current_price) * lifo_qty - fee  # negative

                logger.warning(
                    f"[HEDGE] L3_LIFO_SELL | {symbol} "
                    f"lifo_entry={lifo_entry:.4f} sell={current_price:.4f} "
                    f"qty={lifo_qty:.6f} pnl={realized_pnl:+.4f}"
                )

                # reinvest net_proceeds @ current_price (ახლა ძვირი = SHORT-ისთვის უკეთესი)
                reinvest_qty  = round(net_proceeds * self.leverage / current_price, 6)

                # new position avg:
                # remaining = total - lifo_qty (old units at various entries)
                # new avg = weighted:
                # remaining_value = total_qty × avg_entry - lifo_qty × lifo_entry
                remaining_qty   = total_qty - lifo_qty
                total_value     = total_qty * avg_entry
                remaining_value = total_value - lifo_qty * lifo_entry  # ზუსტი LIFO

                new_qty   = remaining_qty + reinvest_qty
                new_value = remaining_value + reinvest_qty * current_price
                new_avg   = round(new_value / new_qty, 6) if new_qty > 0 else avg_entry

                # TP = new_avg × (1 - tp_pct%)
                new_tp = round(new_avg * (1.0 - self.dca_hedge_tp_pct / 100.0), 6)

                # total_quote update
                new_total_quote = total_quote - lifo_quote + net_proceeds

                logger.warning(
                    f"[HEDGE] L3_LIFO_REINVEST | {symbol} "
                    f"@ {current_price:.4f} qty={reinvest_qty:.6f} "
                    f"old_avg={avg_entry:.4f} → new_avg={new_avg:.4f} "
                    f"new_tp={new_tp:.4f}"
                )

                # DB განახლება — position in-place (არ ვხურავთ, ვასწორებთ)
                from execution.db.db import get_connection
                with get_connection() as conn:
                    conn.execute(
                        """
                        UPDATE futures_positions SET
                            avg_entry_price = ?,
                            qty             = ?,
                            quote_in        = ?,
                            tp_price        = ?
                        WHERE id = ?
                        """,
                        (new_avg, new_qty, new_total_quote, new_tp, pos_id)
                    )
                    conn.commit()

                try:
                    from execution.telegram_notifier import send_telegram_message
                    send_telegram_message(
                        f"🔄 <b>HEDGE L3 LIFO ROTATION</b>\n\n"
                        f"🪙 <b>Symbol:</b> <code>{symbol}</code>\n"
                        f"💸 <b>LIFO sell:</b> <code>{lifo_entry:.2f}→{current_price:.2f}</code> "
                        f"(<code>{realized_pnl:+.4f} USDT</code>)\n"
                        f"♻️ <b>Reinvest:</b> <code>${net_proceeds:.2f}</code> "
                        f"@ <code>{current_price:.2f}</code>\n"
                        f"📊 <b>avg_short:</b> <code>{avg_entry:.2f}→{new_avg:.2f}</code> ↑\n"
                        f"🎯 <b>new TP:</b> <code>{new_tp:.2f}</code>\n"
                        f"💡 <b>TP ახლოვდება current-თან!</b>\n"
                        f"🕒 <code>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</code>"
                    )
                except Exception as _tg:
                    logger.warning(f"[HEDGE] L3_TG_FAIL | err={_tg}")

                try:
                    from execution.db.repository import log_event
                    log_event("DCA_HEDGE_L3_LIFO",
                        f"sym={symbol} lifo_entry={lifo_entry:.4f} "
                        f"current={current_price:.4f} pnl={realized_pnl:+.4f} "
                        f"old_avg={avg_entry:.4f} new_avg={new_avg:.4f} "
                        f"new_tp={new_tp:.4f}"
                    )
                except Exception:
                    pass

                self._last_hedge_l3_ts = time.time()

            except Exception as e:
                logger.error(f"[HEDGE] L3_ERR | {pos.get('symbol')} err={e}")

    # ────────────────────────────────────────────────────────
    # PUBLIC: close_dca_hedge_for_position  [BUG-1 FIX]
    # DCA TP / FORCE_CLOSE → hedge SHORT-ის დახურვა
    # ────────────────────────────────────────────────────────
    def close_dca_hedge_for_position(
        self,
        dca_pos_id: int,
        reason: str = "DCA_CLOSED",
    ) -> None:
        """
        DCA position დაიხურა → შესაბამისი hedge SHORT-ების დახურვა.

        edge cases:
          - hedge ჯერ არ გახსნილა → empty fetch → no-op ✓
          - hedge უკვე დაიხურა TP-ზე → status!='OPEN' → no-op ✓
          - L3 rotation-ის შემდეგ position in-place განახლდა (არ დაიხურა) → ✓
        """
        if not self.enabled:
            return

        try:
            from execution.db.db import get_connection
            with get_connection() as conn:
                cur = conn.execute(
                    "SELECT * FROM futures_positions "
                    "WHERE dca_pos_id=? AND is_dca_hedge=1 AND status='OPEN'",
                    (dca_pos_id,),
                )
                cols = [d[0] for d in cur.description]
                hedge_positions = [dict(zip(cols, r)) for r in cur.fetchall()]
        except Exception as e:
            logger.warning(f"[HEDGE] CLOSE_FOR_DCA_FETCH_FAIL | dca_pos_id={dca_pos_id} err={e}")
            return

        if not hedge_positions:
            logger.debug(
                f"[HEDGE] CLOSE_FOR_DCA | dca_pos_id={dca_pos_id} "
                f"no open hedge SHORT → skip"
            )
            return

        for pos in hedge_positions:
            logger.warning(
                f"[HEDGE] CLOSE_FOR_DCA | dca_pos_id={dca_pos_id} "
                f"sym={pos.get('symbol')} hedge_id={pos.get('id')} reason={reason}"
            )
            self._close_short(pos, reason=reason)

    # ────────────────────────────────────────────────────────
    # INTERNAL: _check_independent_short_fc
    # ────────────────────────────────────────────────────────
    def _check_independent_short_fc(
        self,
        pos: Dict[str, Any],
        current_price: float,
        avg_entry: float,
    ) -> Optional[str]:
        """
        FC check for independent SHORT DCA positions.
        SHORT-ისთვის ზევით მოძრაობა = ზარალი.
        FC by time: SHORT_FC_MAX_DAYS
        FC by drawdown: (current - avg) / avg >= SHORT_FC_DRAWDOWN_PCT%
        """
        if self.short_fc_max_days > 0:
            opened_at_str = str(pos.get("opened_at", "") or "")
            if opened_at_str:
                try:
                    opened_dt = datetime.fromisoformat(opened_at_str.replace("Z", "+00:00"))
                    if opened_dt.tzinfo is None:
                        opened_dt = opened_dt.replace(tzinfo=timezone.utc)
                    days_open = (datetime.now(timezone.utc) - opened_dt).total_seconds() / 86400.0
                    if days_open >= self.short_fc_max_days:
                        return f"MAX_DAYS_{days_open:.1f}d>={self.short_fc_max_days:.0f}d"
                except Exception as _e:
                    logger.warning(f"[SHORT_DCA] FC_TIME_PARSE_FAIL | err={_e}")

        if self.short_fc_drawdown_pct > 0 and avg_entry > 0:
            upside_pct = (current_price - avg_entry) / avg_entry * 100.0
            if upside_pct >= self.short_fc_drawdown_pct:
                return f"DRAWDOWN_{upside_pct:.2f}%>={self.short_fc_drawdown_pct:.1f}%"

        return None

    # ────────────────────────────────────────────────────────
    # PUBLIC: open_independent_short
    # LONG L1 გახსნის შემდეგ — price-level trigger-ზე SHORT
    # ────────────────────────────────────────────────────────
    def open_independent_short(
        self,
        symbol: str,
        long_entry_price: float,
    ) -> bool:
        """
        Independent SHORT DCA — LONG L1-ის გახსნის შემდეგ გამოიძახება.

        trigger: current_price <= long_entry_price * (1 - SHORT_L1_TRIGGER_PCT%)
        SHORT L1 opens at: long_entry_price * 0.984  (-1.6%)

        ეს არ არის hedge — LONG-ის სიცოცხლეს არ მიყვება.
        საკუთარი TP + FC lifecycle.

        edge cases:
          - SHORT_DCA_ENABLED=false → skip
          - already open for this symbol → skip (duplicate guard)
          - current_price > trigger → skip (not yet)
          - long_entry_price=0 → skip (invalid)
        """
        if not self.enabled or not self.short_dca_enabled:
            return False

        if long_entry_price <= 0:
            return False

        # duplicate guard — symbol-ზე უკვე ღია independent SHORT?
        try:
            from execution.db.db import get_connection
            with get_connection() as conn:
                row = conn.execute(
                    "SELECT id FROM futures_positions "
                    "WHERE symbol=? AND is_independent_short=1 AND status='OPEN'",
                    (symbol,)
                ).fetchone()
                if row:
                    logger.debug(f"[SHORT_DCA] ALREADY_OPEN | {symbol} → skip")
                    return False
        except Exception as e:
            logger.warning(f"[SHORT_DCA] CHECK_FAIL | {symbol} err={e}")
            return False

        # price check — trigger not yet reached?
        current_price = self._fetch_price(symbol)
        if current_price <= 0:
            return False

        trigger_price = round(long_entry_price * (1.0 - self.short_l1_trigger_pct / 100.0), 6)
        if current_price > trigger_price:
            logger.debug(
                f"[SHORT_DCA] TRIGGER_WAIT | {symbol} "
                f"price={current_price:.2f} trigger={trigger_price:.2f} "
                f"(LONG={long_entry_price:.2f} -{self.short_l1_trigger_pct:.1f}%)"
            )
            return False

        # balance check — same pattern as hedge
        short_quote = self.short_addon_quote  # L1 = same size as ADD-ON ($25)
        try:
            from execution.dca_risk_manager import get_risk_manager as _rm
            bal_ok, bal_reason = _rm().can_l3_operation(short_quote)
            if not bal_ok:
                logger.warning(f"[SHORT_DCA] BALANCE_BLOCK | {symbol} reason={bal_reason}")
                return False
        except Exception as e:
            logger.warning(f"[SHORT_DCA] BALANCE_CHECK_FAIL | {symbol} err={e} → blocking (safe)")
            return False

        # TP = entry * (1 - short_tp_pct%)
        tp_price = round(current_price * (1.0 - self.short_tp_pct / 100.0), 6)
        qty      = round((short_quote * self.leverage) / current_price, 6)
        sig_id   = f"SINDEP-{symbol.replace('/', '')}-{uuid.uuid4().hex[:8]}"

        try:
            from execution.db.db import get_connection
            now = datetime.now(timezone.utc).isoformat()
            with get_connection() as conn:
                cur = conn.execute(
                    """
                    INSERT INTO futures_positions
                      (signal_id, symbol, direction, entry_price, qty, quote_in,
                       leverage, tp_price, sl_price, status, opened_at, mode,
                       is_dca_hedge, dca_pos_id, avg_entry_price,
                       is_independent_short, long_ref_price)
                    VALUES (?, ?, 'SHORT', ?, ?, ?, ?, ?, ?, 'OPEN', ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (sig_id, symbol, current_price, qty, short_quote,
                     self.leverage, tp_price, 0.0, now, self.mode,
                     0, 0, current_price,
                     1, long_entry_price)
                )
                conn.commit()
                pos_id = cur.lastrowid
        except Exception as e:
            logger.error(f"[SHORT_DCA] OPEN_DB_FAIL | {symbol} err={e}")
            return False

        logger.warning(
            f"[SHORT_DCA] SHORT_OPENED | {symbol} "
            f"entry={current_price:.4f} tp={tp_price:.4f} "
            f"long_ref={long_entry_price:.4f} trigger={trigger_price:.4f} "
            f"qty={qty:.6f} quote={short_quote} pos_id={pos_id}"
        )

        try:
            from execution.telegram_notifier import send_telegram_message
            send_telegram_message(
                f"📉 <b>SHORT DCA გახსნა</b>\n\n"
                f"🪙 <b>Symbol:</b> <code>{symbol}</code>\n"
                f"💰 <b>Entry:</b> <code>{current_price:.2f}</code> "
                f"(<code>-{self.short_l1_trigger_pct:.1f}%</code> from LONG)\n"
                f"🎯 <b>TP:</b> <code>{tp_price:.2f}</code> "
                f"(<code>-{self.short_tp_pct:.1f}%</code>)\n"
                f"📊 <b>ADD-ON triggers:</b> <code>{self.short_addon_trigger_pcts}</code> (ქვევით)\n"
                f"💼 <b>Quote:</b> <code>${short_quote:.0f}</code> "
                f"<code>×{self.leverage}</code>\n"
                f"🔗 <b>LONG ref:</b> <code>{long_entry_price:.2f}</code>\n"
                f"🕒 <code>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</code>"
            )
        except Exception as _tg:
            logger.warning(f"[SHORT_DCA] TG_FAIL | err={_tg}")

        try:
            from execution.db.repository import log_event
            log_event("SHORT_DCA_OPENED",
                f"sym={symbol} entry={current_price:.4f} "
                f"tp={tp_price:.4f} long_ref={long_entry_price:.4f} "
                f"quote={short_quote} pos_id={pos_id}"
            )
        except Exception:
            pass

        return True

    # ────────────────────────────────────────────────────────
    # PUBLIC: check_independent_short_open
    # ყოველ loop-ზე — ყველა LONG position-ზე SHORT trigger check
    # ────────────────────────────────────────────────────────
    def check_independent_short_open(self) -> None:
        """
        ყოველ main loop iteration-ზე გამოიძახება.
        ყოველ ღია LONG DCA position-ზე შეამოწმებს:
          - SHORT უკვე ღიაა? → skip
          - current_price <= long_entry * (1 - 1.6%)? → open_independent_short()

        edge cases:
          - SHORT_DCA_ENABLED=false → skip
          - LONG position-ი არ არის → skip (nothing to mirror)
          - per-symbol: 1 SHORT max (duplicate guard in open_independent_short)
        """
        if not self.enabled or not self.short_dca_enabled:
            return

        try:
            from execution.db.repository import get_all_open_dca_positions
            long_positions = get_all_open_dca_positions()
        except Exception as e:
            logger.warning(f"[SHORT_DCA] LONG_FETCH_FAIL | err={e}")
            return

        if not long_positions:
            return

        import re as _re_sym
        for pos in long_positions:
            try:
                sym = str(pos.get("symbol", ""))
                # base symbol only (no _L2/_L3 suffix)
                exchange_sym = _re_sym.sub(r'_L\d+$', '', sym)
                long_entry = float(pos.get("initial_entry_price") or pos.get("avg_entry_price") or 0.0)
                if long_entry <= 0:
                    continue
                self.open_independent_short(exchange_sym, long_entry)
            except Exception as e:
                logger.warning(f"[SHORT_DCA] OPEN_CHECK_ERR | {pos.get('symbol')} err={e}")

    # ────────────────────────────────────────────────────────
    # PUBLIC: check_independent_short_addons
    # ვარდნაზე ADD-ONs — LONG ADD-ON-ების სარკე
    # ────────────────────────────────────────────────────────
    def check_independent_short_addons(self) -> None:
        """
        ყოველ loop-ზე — ღია independent SHORT-ებზე ADD-ON შემოწმება.

        trigger: current_price <= entry_price * (1 - addon_trigger_pcts[add_on_count])
        trigger reference: entry_price (SHORT L1 original entry — ფიქსირებული)

        ADD-ON direction: ვარდნაზე (ქვევით) — LONG-ის სარკე
          SHORT A1: entry * (1 - 1.0%) = -1.0% from SHORT L1
          SHORT A2: entry * (1 - 2.2%) = -2.2% from SHORT L1
          SHORT A3: entry * (1 - 3.5%) = -3.5% from SHORT L1

        avg update: weighted average (avg drops as we add lower)
        TP update: new_avg * (1 - short_tp_pct%) — TP ახლოვდება bounce-ზე

        edge cases:
          - add_on_count >= max_addons → exhausted, skip
          - add_on_count >= len(triggers) → no trigger defined, skip
          - current_price > trigger → not yet, skip
          - per-position cooldown 300s spam guard
        """
        if not self.enabled or not self.short_dca_enabled:
            return

        try:
            from execution.db.db import get_connection
            with get_connection() as conn:
                cur = conn.execute(
                    "SELECT * FROM futures_positions "
                    "WHERE status='OPEN' AND is_independent_short=1"
                )
                cols = [d[0] for d in cur.description]
                indep_shorts = [dict(zip(cols, r)) for r in cur.fetchall()]
        except Exception as e:
            logger.warning(f"[SHORT_DCA] ADDON_FETCH_FAIL | err={e}")
            return

        for pos in indep_shorts:
            try:
                symbol       = str(pos.get("symbol", ""))
                pos_id       = int(pos.get("id", 0))
                entry_price  = float(pos.get("entry_price", 0.0))
                add_on_count = int(pos.get("add_on_count", 0) or 0)
                quote_in     = float(pos.get("quote_in", 0.0))
                avg_entry    = float(pos.get("avg_entry_price", 0.0) or entry_price)

                if entry_price <= 0:
                    continue

                if add_on_count >= self.short_max_addons:
                    logger.debug(f"[SHORT_DCA] ADDON_EXHAUSTED | {symbol} {add_on_count}/{self.short_max_addons}")
                    continue

                if add_on_count >= len(self.short_addon_trigger_pcts):
                    logger.warning(
                        f"[SHORT_DCA] ADDON_NO_TRIGGER | {symbol} "
                        f"add_on={add_on_count} triggers={len(self.short_addon_trigger_pcts)}"
                    )
                    continue

                # per-position cooldown
                last_ts = self._short_addon_cooldown_map.get(pos_id, 0.0)
                if (time.time() - last_ts) < self._short_addon_cooldown_s:
                    continue

                current_price = self._fetch_price(symbol)
                if current_price <= 0:
                    continue

                # trigger: ქვევით (SHORT ADD-ON on DROP)
                trigger_pct   = self.short_addon_trigger_pcts[add_on_count]
                trigger_price = entry_price * (1.0 - trigger_pct / 100.0)

                if current_price > trigger_price:
                    logger.debug(
                        f"[SHORT_DCA] ADDON_WAIT | {symbol} level={add_on_count+1} "
                        f"price={current_price:.2f} trigger={trigger_price:.2f} "
                        f"(-{trigger_pct:.1f}% from entry={entry_price:.2f})"
                    )
                    continue

                # balance check
                try:
                    from execution.dca_risk_manager import get_risk_manager as _rm
                    bal_ok, bal_reason = _rm().can_l3_operation(self.short_addon_quote)
                    if not bal_ok:
                        logger.warning(f"[SHORT_DCA] ADDON_BALANCE_BLOCK | {symbol} reason={bal_reason}")
                        continue
                except Exception as _be:
                    logger.warning(f"[SHORT_DCA] ADDON_BALANCE_FAIL | {symbol} err={_be} → skip")
                    continue

                # ADD-ON — weighted avg (ვარდნაზე avg ეცემა)
                total_quote = quote_in + self.short_addon_quote
                new_avg     = (avg_entry * quote_in + current_price * self.short_addon_quote) / total_quote
                new_tp      = round(new_avg * (1.0 - self.short_tp_pct / 100.0), 6)
                new_qty     = round((total_quote * self.leverage) / new_avg, 6)

                logger.warning(
                    f"[SHORT_DCA] ADDON | {symbol} level={add_on_count+1} "
                    f"trigger=-{trigger_pct:.1f}% "
                    f"entry={entry_price:.2f} current={current_price:.2f} "
                    f"avg={avg_entry:.2f}→{new_avg:.2f} tp={new_tp:.2f}"
                )

                from execution.db.db import get_connection
                with get_connection() as conn:
                    conn.execute(
                        """
                        UPDATE futures_positions SET
                            add_on_count=?, add_on_quote=?, avg_entry_price=?,
                            tp_price=?, qty=?, quote_in=?
                        WHERE id=?
                        """,
                        (add_on_count + 1, self.short_addon_quote,
                         round(new_avg, 6), new_tp, new_qty, total_quote, pos_id)
                    )
                    conn.commit()

                self._short_addon_cooldown_map[pos_id] = time.time()

                try:
                    from execution.telegram_notifier import send_telegram_message
                    next_t = (
                        f"-{self.short_addon_trigger_pcts[add_on_count+1]:.1f}%"
                        if (add_on_count + 1) < len(self.short_addon_trigger_pcts)
                        else "MAX (FC/TP)"
                    )
                    send_telegram_message(
                        f"➕ <b>SHORT DCA ADD-ON #{add_on_count+1}</b>\n\n"
                        f"🪙 <b>Symbol:</b> <code>{symbol}</code>\n"
                        f"📉 <b>Drop:</b> <code>-{trigger_pct:.1f}%</code> "
                        f"from entry <code>{entry_price:.2f}</code>\n"
                        f"💰 <b>ADD-ON @ </b><code>{current_price:.2f}</code>\n"
                        f"📊 <b>avg_short:</b> <code>{avg_entry:.2f} → {new_avg:.2f}</code> ↓\n"
                        f"🎯 <b>new TP:</b> <code>{new_tp:.2f}</code>\n"
                        f"⏭ <b>Next trigger:</b> <code>{next_t}</code>\n"
                        f"🕒 <code>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</code>"
                    )
                except Exception as _tg:
                    logger.warning(f"[SHORT_DCA] ADDON_TG_FAIL | err={_tg}")

                try:
                    from execution.db.repository import log_event
                    log_event("SHORT_DCA_ADDON",
                        f"sym={symbol} level={add_on_count+1} "
                        f"trigger=-{trigger_pct:.1f}% "
                        f"addon_price={current_price:.4f} "
                        f"new_avg={new_avg:.4f} new_tp={new_tp:.4f}"
                    )
                except Exception:
                    pass

            except Exception as e:
                logger.error(f"[SHORT_DCA] ADDON_ERR | {pos.get('symbol')} err={e}")

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
