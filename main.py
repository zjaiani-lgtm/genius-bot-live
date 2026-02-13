# main.py
import os
import time
import logging

from core.excel_reader import ExcelReader
from core.hardening_guard import validate_schema, readiness_score
from core.heartbeat_guard import check_system_alive
from core.risk_engine import check_kill_switch
from core.signal_validator import validate_signal

from adapters.binance_live import create_exchange
from adapters.virtual_wallet import VirtualWallet

from core.execution_engine import execute_order
from memory.trade_logger import TradeLogger


logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s | %(message)s"
)
log = logging.getLogger("gbm-lite")


def _to_bool(v: str, default=False) -> bool:
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "y", "on")


def _env(name: str, default: str = "") -> str:
    v = os.getenv(name)
    return default if v is None or v == "" else v


def _pick_symbols() -> list[str]:
    raw = _env("BOT_SYMBOLS", "BTC/USDT")
    return [s.strip() for s in raw.split(",") if s.strip()]


def _excel_boot_check(excel_path: str):
    exists = os.path.exists(excel_path)
    log.info(f"EXCEL_CHECK | exists={exists} path={excel_path}")

    if not exists:
        return

    try:
        size = os.path.getsize(excel_path)
        log.info(f"EXCEL_CHECK | size_bytes={size}")

        with open(excel_path, "rb") as f:
            head = f.read(32)

        log.info(f"EXCEL_CHECK | head={head!r}")

        # .xlsx/.xlsm should be a zip container, usually starts with PK
        if not head.startswith(b"PK"):
            # Common Git LFS pointer starts with text "version https://git-lfs.github.com/spec/v1"
            # or sometimes other text headers.
            log.warning("EXCEL_CHECK_WARN | file does not look like a real .xlsx/.xlsm (missing 'PK' zip header).")
    except Exception as e:
        log.warning(f"EXCEL_CHECK_WARN | could not inspect file: {e}")


def _calc_amount(exchange, symbol: str, quote_per_trade: float) -> float:
    """
    Simple sizing: amount = quote_per_trade / last_price
    """
    try:
        t = exchange.fetch_ticker(symbol)
        last = float(t.get("last") or t.get("close") or 0.0)
        if last > 0:
            amt = quote_per_trade / last
            return max(0.0, amt)
    except Exception as e:
        log.warning(f"SIZING_WARN | symbol={symbol} err={e}")

    # fallback conservative amount
    return 0.001


def main():
    run_mode = _env("RUN_MODE", "DEMO").upper()
    auto_trading = _to_bool(os.getenv("AUTO_TRADING", "false"))
    excel_path = _env("EXCEL_PATH", "DYZEN_CAPITAL_OS_AI_LIVE_CORE_READY_HARDENED.xlsx")
    loop_seconds = int(_env("LOOP_SECONDS", "30"))
    quote_per_trade = float(_env("BOT_QUOTE_PER_TRADE", "20"))

    symbols = _pick_symbols()

    log.info(f"BOOT | RUN_MODE={run_mode} AUTO_TRADING={auto_trading} LOOP_SECONDS={loop_seconds} EXCEL={excel_path}")
    log.info(f"BOOT | SYMBOLS={symbols} BOT_QUOTE_PER_TRADE={quote_per_trade}")

    _excel_boot_check(excel_path)

    reader = ExcelReader(excel_path)

    # This will now throw more accurate errors if engine missing or file invalid
    validate_schema(reader)

    # Exchange selection
    if run_mode == "LIVE":
        api_key = _env("BINANCE_API_KEY")
        api_secret = _env("BINANCE_API_SECRET")
        if not api_key or not api_secret:
            raise RuntimeError("LIVE mode requires BINANCE_API_KEY and BINANCE_API_SECRET")

        exchange = create_exchange(api_key, api_secret)
    else:
        exchange = VirtualWallet()

    trades = TradeLogger(db_path=_env("TRADES_DB_PATH", "trades.db"))

    while True:
        try:
            decision = reader.read_decision()
            sell_fw = reader.read_sell_firewall()
            heartbeat = reader.read_heartbeat()
            risk_lock = reader.read_risk_lock()

            ok_alive, alive_msg = check_system_alive(heartbeat)
            ok_kill, kill_msg = check_kill_switch(risk_lock)

            score = readiness_score(heartbeat, risk_lock, sell_fw)

            if not ok_alive:
                log.warning(f"PAUSE | {alive_msg} score={score}")
                time.sleep(loop_seconds)
                continue

            if not ok_kill:
                log.warning(f"KILL | {kill_msg} score={score}")
                time.sleep(loop_seconds)
                continue

            ok_sig, action = validate_signal(decision, sell_fw)
            if not ok_sig:
                log.info(f"HOLD | reason={action} score={score}")
                time.sleep(loop_seconds)
                continue

            # If Excel provides SYMBOL use it, else run across BOT_SYMBOLS
            target_symbols = [decision.get("SYMBOL")] if decision.get("SYMBOL") else symbols

            if action in ("BUY", "SELL") and auto_trading:
                for sym in target_symbols:
                    if not sym:
                        continue

                    amount = _calc_amount(exchange, sym, quote_per_trade)
                    if amount <= 0:
                        log.warning(f"SKIP | symbol={sym} amount_invalid={amount}")
                        continue

                    ok, result = execute_order(exchange, sym, action, amount)
                    if ok:
                        log.info(f"ORDER_OK | symbol={sym} side={action} amount={amount}")
                        try:
                            trades.log(result)
                        except Exception as e:
                            log.warning(f"TRADE_LOG_WARN | err={e}")
                    else:
                        log.error(f"ORDER_FAIL | symbol={sym} side={action} amount={amount} err={result}")
            else:
                log.info(f"HOLD | action={action} AUTO_TRADING={auto_trading} score={score}")

        except Exception as e:
            log.exception(f"LOOP_ERROR | {e}")

        time.sleep(loop_seconds)


if __name__ == "__main__":
    main()
