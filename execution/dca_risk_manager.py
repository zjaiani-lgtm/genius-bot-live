# execution/dca_risk_manager.py
# ============================================================
# DCA Risk Manager — Capital limits, exposure controls,
# position-level and portfolio-level risk checks.
#
# ADDON CASCADE SYSTEM:
#   can_open_position() — ახალი L1 position-ის გახსნა
#   can_add_on()        — L2 zone ADD-ON (level 1-5)
#   can_l3_operation()  — L3 ADD-ON + LIFO rotation (balance only)
#
# ENV პარამეტრები:
#   DCA_MAX_CAPITAL_USDT=80.0   ← per-symbol max (L1 + ყველა ADD-ON)
#   DCA_MAX_TOTAL_USDT=80.0     ← portfolio total max
#   MAX_OPEN_TRADES=1           ← ერთი symbol (ADDON CASCADE)
#   SMART_ADDON_BUFFER=10.0     ← min free USDT after operation
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

    DEMO: DEMO_INITIAL_BALANCE - ღია პოზიციების ჯამი (DB-დან).
    LIVE: Binance API-დან რეალური free balance.

    შეცდომაზე 0.0 → fail-safe: operation ბლოკდება.
    """
    mode = os.getenv("MODE", "DEMO").upper()

    if mode == "DEMO":
        try:
            initial  = float(os.getenv("DEMO_INITIAL_BALANCE", "120.0"))
            from execution.db.repository import get_all_open_dca_positions
            open_pos = get_all_open_dca_positions() or []
            invested = sum(float(p.get("total_quote_spent", 0.0)) for p in open_pos)
            free     = round(initial - invested, 2)
            logger.debug(
                f"[SMART_ADDON] DEMO balance | "
                f"initial={initial} invested={invested:.2f} free={free:.2f}"
            )
            return max(free, 0.0)
        except Exception as e:
            logger.warning(f"[SMART_ADDON] DEMO balance_calc_fail | err={e} → initial fallback")
            return float(os.getenv("DEMO_INITIAL_BALANCE", "120.0"))

    try:
        from execution.exchange_client import BinanceSpotClient
        client  = BinanceSpotClient()
        balance = float(client.fetch_balance_free("USDT") or 0.0)
        logger.debug(f"[SMART_ADDON] Binance USDT balance={balance:.2f}")
        return balance
    except Exception as e:
        logger.warning(f"[SMART_ADDON] balance_fetch_fail | err={e} → 0.0 (blocked)")
        return 0.0


class DCARiskManager:
    """
    Portfolio-level risk controls — ADDON CASCADE SYSTEM.

    L2 zone (can_add_on):
      - per-symbol capital limit
      - total portfolio exposure limit
      - min notional ($10)
      - SMART balance check

    L3 zone (can_l3_operation):
      - მხოლოდ SMART balance check
      - per-symbol და total limits არ ვამოწმებთ:
        L3 operation-ი არ ამატებს ახალ capital-ს —
        LIFO: sell → reinvest (net zero), L3 ADD-ON: L2 resource-ი
    """

    def __init__(self) -> None:
        self.max_open_positions = _ei("MAX_OPEN_TRADES",      1)
        self.max_drawdown_pct   = _ef("DCA_MAX_DRAWDOWN_PCT", 999.0)
        self.min_notional       = _ef("DCA_MIN_NOTIONAL",     10.0)
        self.smart_addon_buffer = _ef("SMART_ADDON_BUFFER",   10.0)

        # AUTO-CALC: max_per_symbol = BOT_QUOTE_PER_TRADE + sum(DCA_ADDON_SIZES)
        # ENV DCA_MAX_CAPITAL_USDT-ი თუ დაყენებულია — ის იმარჯვებს (override)
        # თუ არ არის — ავტომატურად: BOT_QUOTE + sum(ADDON_SIZES)
        _quote     = _ef("BOT_QUOTE_PER_TRADE", 12.0)
        _sizes_raw = os.getenv("DCA_ADDON_SIZES", "12,15,18,15,10")
        try:
            _sizes = [float(x.strip()) for x in _sizes_raw.split(",") if x.strip()]
        except Exception:
            _sizes = [12.0, 15.0, 18.0, 15.0, 10.0]
        _auto_cap  = _quote + sum(_sizes)

        _env_cap = os.getenv("DCA_MAX_CAPITAL_USDT")
        self.max_per_symbol = float(_env_cap) if _env_cap is not None else _auto_cap

        # AUTO-CALC: max_total = MAX_OPEN_TRADES × max_per_symbol
        # ENV DCA_MAX_TOTAL_USDT-ი თუ დაყენებულია — ის იმარჯვებს (override)
        _auto_total = self.max_open_positions * self.max_per_symbol
        _env_total  = os.getenv("DCA_MAX_TOTAL_USDT")
        self.max_total_exposure = float(_env_total) if _env_total is not None else _auto_total

        logger.info(
            f"[DCA] DCARiskManager init | "
            f"max_positions={self.max_open_positions} "
            f"max_per_symbol={self.max_per_symbol:.1f} "
            f"({'ENV' if _env_cap else f'AUTO={_quote}+{sum(_sizes):.0f}'}) "
            f"max_total={self.max_total_exposure:.1f} "
            f"({'ENV' if _env_total else 'AUTO=positions×cap'}) "
            f"smart_buffer={self.smart_addon_buffer}"
        )

    def can_open_position(
        self,
        symbol: str,
        initial_size: float,
        open_positions: List[Dict[str, Any]],
    ) -> Tuple[bool, str]:
        """ახალი L1 position-ის გახსნა შეიძლება?"""
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
        L2 zone ADD-ON risk check.

        FIX: total_exposure check — position საკუთარი total_spent-ი
        open_positions-ში შედის, ამიტომ გამოვაკლოთ და ახლიდან ვამოწმოთ.
        სხვა შემთხვევაში current position double-count-დება:
          total_all = pos.total_spent (other positions) + pos.total_spent (current)
          → ADD-ON ბლოკდება ადრე ვიდრე per-symbol limit-ს მიაღწევს.
        """
        total_spent = float(position.get("total_quote_spent", 0.0))
        pos_id      = position.get("id")

        # 1. per-symbol capital
        if total_spent + addon_size > self.max_per_symbol:
            return False, (
                f"PER_SYMBOL_CAPITAL "
                f"({total_spent:.1f}+{addon_size:.1f}>{self.max_per_symbol:.1f})"
            )

        # 2. total portfolio exposure (current position გამოვრიცხოთ double-count-ის თავიდანაცილებისთვის)
        other_positions_total = sum(
            float(p.get("total_quote_spent", 0.0))
            for p in open_positions
            if p.get("id") != pos_id
        )
        if other_positions_total + total_spent + addon_size > self.max_total_exposure:
            return False, (
                f"TOTAL_EXPOSURE "
                f"(others={other_positions_total:.1f}+"
                f"current={total_spent:.1f}+"
                f"addon={addon_size:.1f}>"
                f"{self.max_total_exposure:.1f})"
            )

        # 3. min notional
        if addon_size < self.min_notional:
            return False, f"ADDON_BELOW_MIN_NOTIONAL ({addon_size}<{self.min_notional})"

        # 4. SMART balance check
        required  = addon_size + self.smart_addon_buffer
        free_usdt = _get_binance_usdt_balance()
        if free_usdt < required:
            logger.warning(
                f"[SMART_ADDON] BLOCKED | free={free_usdt:.2f} "
                f"< required={required:.2f} "
                f"(addon={addon_size:.1f} + buffer={self.smart_addon_buffer:.1f})"
            )
            return False, (
                f"INSUFFICIENT_BALANCE "
                f"(free={free_usdt:.2f}<required={required:.2f})"
            )

        logger.info(
            f"[SMART_ADDON] OK | free={free_usdt:.2f} "
            f">= required={required:.2f} → add-on approved"
        )
        return True, "OK"

    def can_l3_operation(
        self,
        operation_size: float,
    ) -> Tuple[bool, str]:
        """
        L3 zone operation risk check — L3 ADD-ON + LIFO rotation.

        მხოლოდ SMART balance check:
          LIFO rotation: sell + reinvest = net ~$0 capital change
          L3 ADD-ON: L2 resource-ი (BOT_QUOTE_PER_TRADE), არა ახალი capital

        per-symbol და total limits არ ვამოწმებთ — L3-ზე ეს limits
        უკვე გათვლილია ADD-ON #5-ის გახსნისას.

        operation_size: LIFO rotation-ზე = net_proceeds ($10-13)
                        L3 ADD-ON-ზე = BOT_QUOTE_PER_TRADE ($10)
        """
        # min notional — Binance minimum
        if operation_size < self.min_notional:
            return False, (
                f"L3_BELOW_MIN_NOTIONAL "
                f"({operation_size:.2f}<{self.min_notional:.1f})"
            )

        # SMART balance — საკმარისი free USDT?
        required  = operation_size + self.smart_addon_buffer
        free_usdt = _get_binance_usdt_balance()
        if free_usdt < required:
            logger.warning(
                f"[L3_RISK] BLOCKED | free={free_usdt:.2f} "
                f"< required={required:.2f} "
                f"(op={operation_size:.1f} + buffer={self.smart_addon_buffer:.1f})"
            )
            return False, (
                f"L3_INSUFFICIENT_BALANCE "
                f"(free={free_usdt:.2f}<required={required:.2f})"
            )

        logger.info(
            f"[L3_RISK] OK | free={free_usdt:.2f} "
            f">= required={required:.2f} → L3 operation approved"
        )
        return True, "OK"

    def portfolio_summary(
        self,
        open_positions: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Portfolio risk snapshot — heartbeat / Telegram."""
        total_spent    = sum(float(p.get("total_quote_spent", 0.0)) for p in open_positions)
        symbols        = [str(p.get("symbol", "?")) for p in open_positions]
        unrealized_pnl = sum(float(p.get("unrealized_pnl", 0.0)) for p in open_positions)

        return {
            "open_count":     len(open_positions),
            "max_open":       self.max_open_positions,
            "total_spent":    round(total_spent, 4),
            "max_total":      self.max_total_exposure,
            "exposure_pct":   round(total_spent / self.max_total_exposure * 100, 1)
                              if self.max_total_exposure else 0,
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
