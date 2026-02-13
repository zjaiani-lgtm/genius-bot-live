
import pandas as pd

class ExcelReader:
    def __init__(self, path):
        self.path = path

    def _read_first_row(self, sheet):
        df = pd.read_excel(self.path, sheet_name=sheet)
        return df.iloc[0].to_dict()

    def read_decision(self):
        return self._read_first_row("AI_MASTER_LIVE_DECISION")

    def read_sell_firewall(self):
        return self._read_first_row("SELL_FIREWALL")

    def read_heartbeat(self):
        return self._read_first_row("SYSTEM_HEARTBEAT")

    def read_risk_lock(self):
        return self._read_first_row("RISK_ENVELOPE_LOCK")
