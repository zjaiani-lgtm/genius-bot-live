
REQUIRED_SHEETS = [
    "AI_MASTER_LIVE_DECISION",
    "SELL_FIREWALL",
    "SYSTEM_HEARTBEAT",
    "RISK_ENVELOPE_LOCK",
]

REQUIRED_COLUMNS = {
    "AI_MASTER_LIVE_DECISION": ["FINAL_ACTION"],
    "SELL_FIREWALL": ["SELL_DECISION"],
    "SYSTEM_HEARTBEAT": ["GLOBAL_STATUS"],
    "RISK_ENVELOPE_LOCK": ["KILL_SWITCH"],
}

def validate_schema(excel_reader):
    for sheet in REQUIRED_SHEETS:
        try:
            excel_reader._read_first_row(sheet)
        except Exception as e:
            raise RuntimeError(f"SCHEMA_ERROR: Missing sheet {sheet}: {e}")

    # Column checks
    for sheet, cols in REQUIRED_COLUMNS.items():
        row = excel_reader._read_first_row(sheet)
        for col in cols:
            if col not in row:
                raise RuntimeError(f"SCHEMA_ERROR: Missing column {col} in {sheet}")

def fail_fast_action(decision: dict):
    if "FINAL_ACTION" not in decision:
        raise RuntimeError("FAIL_FAST: FINAL_ACTION missing")
    return decision["FINAL_ACTION"]

def readiness_score(heartbeat, risk_lock, sell_fw):
    score = 1.0
    if heartbeat.get("GLOBAL_STATUS") != "RUN":
        score -= 0.4
    if risk_lock.get("KILL_SWITCH") == "KILL":
        score -= 0.4
    if sell_fw.get("SELL_DECISION") not in ["SELL", "HOLD"]:
        score -= 0.2
    return max(0.0, min(1.0, score))
