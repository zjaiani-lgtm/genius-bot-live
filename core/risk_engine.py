
def check_kill_switch(risk_lock: dict):
    if risk_lock.get("KILL_SWITCH") == "KILL":
        return False, "KILL_SWITCH_ACTIVE"
    return True, "OK"
