
def validate_signal(decision: dict, sell_fw: dict):
    action = decision.get("FINAL_ACTION", "HOLD")

    if action == "SELL":
        if sell_fw.get("SELL_DECISION") != "SELL":
            return False, "SELL_BLOCKED"

    return True, action
