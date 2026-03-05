
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import os
import logging

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EnvOverrideConfig:
    enabled: bool
    kill_switch: bool
    risk_multiplier: float
    min_confidence_override: Optional[float]
    disable_new_entries: bool


class EnvOverrideBridge:
    """
    Production-safe ENV-based override layer.

    - Zero disk I/O
    - Cloud-native
    - Instant toggle via Render dashboard
    - No Excel dependency
    - Never raises
    """

    @staticmethod
    def _get_bool(name: str, default: str = "false") -> bool:
        return os.getenv(name, default).strip().lower() == "true"

    @staticmethod
    def _get_float(name: str, default: str) -> float:
        try:
            return float(os.getenv(name, default))
        except Exception:
            return float(default)

    def read_override(self) -> EnvOverrideConfig:

        enabled = self._get_bool("OVERRIDE_ENABLED", "true")
        kill_switch = self._get_bool("KILL_SWITCH", "false")
        disable_new_entries = self._get_bool("DISABLE_NEW_ENTRIES", "false")

        risk_multiplier = self._get_float("RISK_MULTIPLIER", "1.0")
        risk_multiplier = max(0.0, min(risk_multiplier, 1.0))  # clamp safety

        min_conf_raw = os.getenv("MIN_CONFIDENCE_OVERRIDE", "")
        try:
            min_conf = float(min_conf_raw) if min_conf_raw else None
        except Exception:
            min_conf = None

        cfg = EnvOverrideConfig(
            enabled=enabled,
            kill_switch=kill_switch,
            risk_multiplier=risk_multiplier,
            min_confidence_override=min_conf,
            disable_new_entries=disable_new_entries,
        )

        logger.info(f"ENV override loaded: {cfg}")

        return cfg
