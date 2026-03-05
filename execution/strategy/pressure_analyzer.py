import logging
from collections import defaultdict

log = logging.getLogger("pressure")

# In-memory counters (safe for single worker process)
_pressure_counts = defaultdict(int)
_total_checks = defaultdict(int)


def log_pressure(
    *,
    symbol: str,
    up15: bool,
    up30: bool,
    up1h: bool,
    rsi_ok: bool,
    not_too_extended: bool,
    atr_ok: bool = True,
) -> None:
    """Non-invasive diagnostic logger for signal pressure analysis."""

    log.info(
        "pressure_check | %s | t15=%s t30=%s t1h=%s rsi=%s ext=%s atr=%s",
        symbol,
        up15,
        up30,
        up1h,
        rsi_ok,
        not_too_extended,
        atr_ok,
    )

    _total_checks[symbol] += 1

    if not up15:
        _pressure_counts[(symbol, "trend15")] += 1
    if not up30:
        _pressure_counts[(symbol, "trend30")] += 1
    if not up1h:
        _pressure_counts[(symbol, "trend1h")] += 1
    if not rsi_ok:
        _pressure_counts[(symbol, "rsi")] += 1
    if not not_too_extended:
        _pressure_counts[(symbol, "extension")] += 1
    if not atr_ok:
        _pressure_counts[(symbol, "atr")] += 1


def dump_pressure_summary() -> None:
    """Log aggregated failure statistics."""

    for symbol in sorted(_total_checks):
        total = _total_checks[symbol]
        if total == 0:
            continue

        log.info("pressure_summary | %s | samples=%d", symbol, total)

        for key in ("trend15", "trend30", "trend1h", "rsi", "extension", "atr"):
            fails = _pressure_counts.get((symbol, key), 0)
            pct = (fails / total) * 100.0
            log.info(
                "pressure_summary | %s | %s_fail=%.1f%% (%d/%d)",
                symbol,
                key,
                pct,
                fails,
                total,
            )
