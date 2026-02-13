# core/excel_reader.py
import os
import pandas as pd


class ExcelReader:
    def __init__(self, path: str):
        self.path = path
        # allow override from env if ever needed
        self.engine = os.getenv("PANDAS_EXCEL_ENGINE", "openpyxl")

    def _read_first_row(self, sheet: str) -> dict:
        """
        Reads first data row from a sheet and returns dict (column -> value).
        """
        try:
            df = pd.read_excel(self.path, sheet_name=sheet, engine=self.engine)
        except ImportError as e:
            # common when openpyxl is missing
            raise RuntimeError(
                f"EXCEL_ENGINE_ERROR: missing dependency for engine='{self.engine}'. "
                f"Install openpyxl (recommended). Original: {e}"
            )
        except ValueError as e:
            # "Excel file format cannot be determined" often lands here
            raise RuntimeError(
                f"EXCEL_READ_ERROR: cannot read Excel file '{self.path}' with engine='{self.engine}'. "
                f"Original: {e}"
            )
        except Exception as e:
            raise RuntimeError(
                f"EXCEL_READ_ERROR: failed reading sheet='{sheet}' from '{self.path}' with engine='{self.engine}'. "
                f"Original: {e}"
            )

        if df is None or df.empty:
            raise RuntimeError(f"EXCEL_SHEET_EMPTY: sheet='{sheet}' in '{self.path}' has no rows.")

        # ensure first row dict
        return df.iloc[0].to_dict()

    def read_decision(self) -> dict:
        return self._read_first_row("AI_MASTER_LIVE_DECISION")

    def read_sell_firewall(self) -> dict:
        return self._read_first_row("SELL_FIREWALL")

    def read_heartbeat(self) -> dict:
        return self._read_first_row("SYSTEM_HEARTBEAT")

    def read_risk_lock(self) -> dict:
        return self._read_first_row("RISK_ENVELOPE_LOCK")
