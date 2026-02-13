
import yaml
from core.excel_reader import ExcelReader
from core.heartbeat_guard import check_system_alive
from core.signal_validator import validate_signal
from core.risk_engine import check_kill_switch
from core.execution_engine import execute_order
from core.hardening_guard import validate_schema, fail_fast_action, readiness_score
from adapters.virtual_wallet import VirtualWallet
from adapters.binance_live import create_exchange
from memory.trade_logger import TradeLogger

def load_config():
    with open("config/settings.yaml", "r") as f:
        return yaml.safe_load(f)

def main():
    cfg = load_config()
    reader = ExcelReader(cfg["excel_path"])

    # --- Institutional Schema Validation ---
    validate_schema(reader)

    heartbeat = reader.read_heartbeat()
    risk = reader.read_risk_lock()
    sell_fw = reader.read_sell_firewall()
    decision = reader.read_decision()

    # --- Readiness score (informational) ---
    score = readiness_score(heartbeat, risk, sell_fw)
    print(f"READINESS_SCORE: {round(score*100,1)}%")

    alive, msg = check_system_alive(heartbeat)
    if not alive:
        print(msg)
        return

    ok, msg = check_kill_switch(risk)
    if not ok:
        print(msg)
        return

    # --- Fail-fast decision ---
    action = fail_fast_action(decision)

    valid, action = validate_signal({"FINAL_ACTION": action}, sell_fw)
    if not valid or action == "HOLD":
        print("NO TRADE:", action)
        return

    # Select execution mode
    if cfg["mode"] == "DEMO":
        exchange = VirtualWallet()
    else:
        ex_cfg = cfg["exchange"]
        exchange = create_exchange(ex_cfg["api_key"], ex_cfg["api_secret"])

    symbol = cfg["symbol"]
    amount = cfg["base_position_size"]

    success, result = execute_order(exchange, symbol, action, amount)
    print("EXECUTION:", success, result)

    if success:
        logger = TradeLogger()
        logger.log(result)

if __name__ == "__main__":
    main()
