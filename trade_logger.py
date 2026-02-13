
def check_system_alive(heartbeat: dict):
    if heartbeat.get("GLOBAL_STATUS") != "RUN":
        return False, "SYSTEM_PAUSED"
    return True, "OK"
