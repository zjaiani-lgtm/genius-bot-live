
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import os
import pandas as pd
import logging
import threading

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExcelOverrideConfig:
    enabled: bool = True
    kill_switch: bool = False
    risk_multiplier: float = 1.0
    min_confidence_override: Optional[float] = None
    disable_new_entries: bool = False


class ExcelOverrideBridge:

    def __init__(self, path: str) -> None:
        self.path = path
        self._last_mtime: Optional[float] = None
        self._cached: Optional[ExcelOverrideConfig] = None
        self._lock = threading.Lock()

    def read_override(self) -> Optional[ExcelOverrideConfig]:

        try:
            current_mtime = os.path.getmtime(self.path)
        except Exception:
            return self._cached

        with self._lock:

            if self._last_mtime == current_mtime:
                return self._cached

            override = self._reload_file()

            if override:
                self._cached = override
                self._last_mtime = current_mtime

            return self._cached

    def _reload_file(self) -> Optional[ExcelOverrideConfig]:
        try:
            df = pd.read_excel(self.path, sheet_name="CONTROL")
            row = df.iloc[0]

            enabled = bool(row.get("ENABLED", True))
            kill_switch = bool(row.get("KILL_SWITCH", False))

            risk_multiplier = float(row.get("RISK_MULTIPLIER", 1.0))
            risk_multiplier = max(0.0, min(risk_multiplier, 1.0))

            min_conf = row.get("MIN_CONFIDENCE_OVERRIDE", None)
            if pd.isna(min_conf):
                min_conf = None
            else:
                min_conf = float(min_conf)

            disable_entries = bool(row.get("DISABLE_NEW_ENTRIES", False))

            logger.info("Excel override reloaded")

            return ExcelOverrideConfig(
                enabled=enabled,
                kill_switch=kill_switch,
                risk_multiplier=risk_multiplier,
                min_confidence_override=min_conf,
                disable_new_entries=disable_entries,
            )

        except Exception as e:
            logger.warning(f"Excel reload failed, keeping cached config: {e}")
            return None
