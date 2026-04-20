# execution/dca_risk_manager.py
# ============================================================
# DCA Risk Manager — Capital limits, exposure controls,
# position-level and portfolio-level risk checks.
#
# ENV პარამეტრები:
#   DCA_MAX_CAPITAL_USDT=40.0
#   DCA_MAX_TOTAL_USDT=60.0
#   DCA_MAX_DRAWDOWN_PCT=8.0
#   MAX_OPEN_TRADES=2 (გაზიარებული ძველ ბოტთან)
#
# FIX: Smart Add-on — Binance ბალანსის ავტომატური შემოწმება
#   ბალანსი >= addon_size + SMART_ADDON_BUFFER → add-on ✅
#   ბალანსი < addon_size + SMART_ADDON_BUFFER → add-on ❌
#   SMART_ADDON_BUFFER=5.0 (ENV-ით კონტროლი)
# ============================================================
from __future__ import annotations

import os
import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("gbm")


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


def _get_binance_usdt_balance() -> float:
    """
    USDT ბალანსის წამოღება.

    LIVE: Binance API-დან რეალური balance.
    DEMO: DEMO_INITIAL_BALANCE - ღია პოზიციების ჯამი.
          (API key არ არის → 0.0 → SMART_ADDON ყოველთვის blocked იყო!)

    შეცდომაზე 0.0 დაბრუნება (fail-safe: add-on ბლოკდება).
    """
    mode = os.getenv("MODE", "DEMO").upper()

    # ── DEMO MODE ────────────────────────────────────────────
    if mode == "DEMO":
        try:
            initial = float(os.getenv("DEMO_INITIAL_BALANCE", "120.0"))
            # ღია პოზიციების ჯამი DB-დან
            from execution.db.repository import get_all_open_dca_positions
            open_pos = get_all_open_dca_positions() or []
            invested = sum(float(p.get("total_quote_spent", 0.0)) for p in open_pos)
            free = round(initial - invested, 2)
            logger.debug(f"[SMART_ADDON] DEMO balance | initial={initial} invested={invested:.2f} free={free:.2f}")
            return max(free, 0.0)
        except Exception as e:
            logger.warning(f"[SMART_ADDON] DEMO balance_calc_fail | err={e} → initial fallback")
            return float(os.getenv("DEMO_INITIAL_BALANCE", "120.0"))

    # ── LIVE MODE ────────────────────────────────────────────
    try:
        from execution.exchange_client import BinanceSpotClient
        client = BinanceSpotClient()
        balance = float(client.fetch_balance_free("USDT") or 0.0)
        logger.debug(f"[SMART_ADDON] Binance USDT balance={balance:.2f}")
        return balance
    except Exception as e:
        logger.warning(f"[SMART_ADDON] balance_fetch_fail | err={e} → 0.0 (add-on blocked)")
        return 0.0


class DCARiskManager:
    """
    Portfolio-level risk controls for DCA positions.

    ამოწმებს:
      1. max open positions (MAX_OPEN_TRADES — გაზიარებული)
      2. total exposure limit (DCA_MAX_TOTAL_USDT)
      3. per-symbol capital limit (DCA_MAX_CAPITAL_USDT)
      4. symbol-level dedup (ერთ symbol-ზე ერთი DCA position)
      5. min notional per add-on (Binance $10 minimum)
      6. SMART: Binance real-time USDT balance check
    """

    def __init__(self) -> None:
        self.max_open_positions = _ei("MAX_OPEN_TRADES",       3)
        self.max_per_symbol     = _ef("DCA_MAX_CAPITAL_USDT",  20.0)
        self.max_total_exposure = _ef("DCA_MAX_TOTAL_USDT",    60.0)
        self.max_drawdown_pct   = _ef("DCA_MAX_DRAWDOWN_PCT",  999.0)
        self.min_notional       = _ef("DCA_MIN_NOTIONAL",      10.0)

        # Smart Add-on: minimum free USDT buffer after addon
        # addon_size + buffer უნდა იყოს ბალანსზე
        self.smart_addon_buffer = _ef("SMART_ADDON_BUFFER",    5.0)

        logger.info(
            f"[DCA] DCARiskManager init | max_positions={self.max_open_positions} "
            f"max_per_symbol={self.max_per_symbol} max_total={self.max_total_exposure} "
            f"smart_addon_buffer={self.smart_addon_buffer}"
        )

    def can_open_position(
        self,
        symbol: str,
        initial_size: float,
        open_positions: List[Dict[str, Any]],
    ) -> Tuple[bool, str]:
        """
        ახალი DCA position-ის გახსნა შეიძლება?
        """
        sym = str(symbol or "").upper().strip()

        # 1. max open positions
        if len(open_positions) >= self.max_open_positions:
            return False, f"MAX_OPEN_POSITIONS ({len(open_positions)}/{self.max_open_positions})"

        # 2. symbol already open
        sym_positions = [p for p in open_positions if str(p.get("symbol", "")).upper() == sym]
        if sym_positions:
            return False, f"SYMBOL_ALREADY_OPEN ({sym})"

        # 3. total exposure
        total = sum(float(p.get("total_quote_spent", 0.0)) for p in open_positions)
        if total + initial_size > self.max_total_exposure:
            return False, (
                f"MAX_TOTAL_EXPOSURE "
                f"({total:.1f}+{initial_size:.1f}>{self.max_total_exposure:.1f})"
            )

        # 4. min notional
        if initial_size < self.min_notional:
            return False, f"BELOW_MIN_NOTIONAL ({initial_size}<{self.min_notional})"

        return True, "OK"

    def can_add_on(
        self,
        position: Dict[str, Any],
        addon_size: float,
        open_positions: List[Dict[str, Any]],
    ) -> Tuple[bool, str]:
        """
        Add-on capital risk check.
        DCAPositionManager.should_add_on()-ს შემდეგ გამოიძახება.

        FIX: Smart Add-on — Binance real-time balance check.
        """
        total_spent = float(position.get("total_quote_spent", 0.0))

        # 1. per-symbol capital
        if total_spent + addon_size > self.max_per_symbol:
            return False, (
                f"PER_SYMBOL_CAPITAL "
                f"({total_spent:.1f}+{addon_size:.1f}>{self.max_per_symbol:.1f})"
            )

        # 2. total portfolio exposure
        total_all = sum(float(p.get("total_quote_spent", 0.0)) for p in open_positions)
        if total_all + addon_size > self.max_total_exposure:
            return False, (
                f"TOTAL_EXPOSURE "
                f"({total_all:.1f}+{addon_size:.1f}>{self.max_total_exposure:.1f})"
            )

        # 3. min notional
        if addon_size < self.min_notional:
            return False, f"ADDON_BELOW_MIN_NOTIONAL ({addon_size}<{self.min_notional})"

        # 4. SMART: Binance real-time USDT balance check
        # addon_size + buffer უნდა იყოს თავისუფლად
        required = addon_size + self.smart_addon_buffer
        free_usdt = _get_binance_usdt_balance()
        if free_usdt < required:
            logger.warning(
                f"[SMART_ADDON] BLOCKED | free={free_usdt:.2f} USDT "
                f"< required={required:.2f} (addon={addon_size:.1f} + buffer={self.smart_addon_buffer:.1f})"
            )
            return False, (
                f"INSUFFICIENT_BALANCE "
                f"(free={free_usdt:.2f}<required={required:.2f})"
            )

        logger.info(
            f"[SMART_ADDON] OK | free={free_usdt:.2f} USDT "
            f">= required={required:.2f} → add-on approved"
        )
        return True, "OK"

    def portfolio_summary(
        self,
        open_positions: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Portfolio risk snapshot — logging / Telegram-ისთვის.
        """
        total_spent = sum(float(p.get("total_quote_spent", 0.0)) for p in open_positions)
        symbols = [str(p.get("symbol", "?")) for p in open_positions]
        unrealized_pnl = sum(float(p.get("unrealized_pnl", 0.0)) for p in open_positions)

        return {
            "open_count":     len(open_positions),
            "max_open":       self.max_open_positions,
            "total_spent":    round(total_spent, 4),
            "max_total":      self.max_total_exposure,
            "exposure_pct":   round(total_spent / self.max_total_exposure * 100, 1) if self.max_total_exposure else 0,
            "symbols":        symbols,
            "unrealized_pnl": round(unrealized_pnl, 4),
        }


# module-level singleton
_risk_mgr: Optional[DCARiskManager] = None


def get_risk_manager() -> DCARiskManager:
    global _risk_mgr
    if _risk_mgr is None:
        _risk_mgr = DCARiskManager()
    return _risk_mgr
