import os
import logging
from datetime import datetime
from typing import Any, Dict, Optional
from zoneinfo import ZoneInfo

import requests

logger = logging.getLogger("gbm")


def _env_bool(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() in ("1", "true", "yes", "y", "on")


TELEGRAM_ENABLED = _env_bool("TELEGRAM_NOTIFICATIONS", "false")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_IDS = [
    x.strip()
    for x in os.getenv("TELEGRAM_CHAT_ID", "").split(",")
    if x.strip()
]
TELEGRAM_PARSE_MODE = os.getenv("TELEGRAM_PARSE_MODE", "HTML").strip().upper()

# Default timezone for all Telegram notifications
TELEGRAM_TIMEZONE = os.getenv("TELEGRAM_TIMEZONE", "Asia/Tbilisi").strip()


def _is_ready() -> bool:
    return TELEGRAM_ENABLED and bool(TELEGRAM_BOT_TOKEN) and len(TELEGRAM_CHAT_IDS) > 0


def _escape_html(text: Any) -> str:
    s = str(text if text is not None else "")
    return (
        s.replace("&", "&amp;")
         .replace("<", "&lt;")
         .replace(">", "&gt;")
    )


def send_telegram_message(text: str, disable_preview: bool = True) -> bool:
    if not _is_ready():
        logger.info("TG_SKIP | telegram not configured")
        return False

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    ok_any = False

    for chat_id in TELEGRAM_CHAT_IDS:
        payload = {
            "chat_id": chat_id,
            "text": text,
            "parse_mode": TELEGRAM_PARSE_MODE,
            "disable_web_page_preview": disable_preview,
        }

        try:
            r = requests.post(url, json=payload, timeout=15)
            if r.ok:
                logger.info("TG_SEND_OK | chat_id=%s", chat_id)
                ok_any = True
            else:
                logger.warning(
                    "TG_SEND_FAIL | chat_id=%s status=%s body=%s",
                    chat_id,
                    r.status_code,
                    r.text[:500],
                )
        except Exception as e:
            logger.warning("TG_SEND_EXC | chat_id=%s err=%s", chat_id, e)

    return ok_any


def _fmt_price(v: Any, digits: int = 6) -> str:
    try:
        return f"{float(v):.{digits}f}"
    except Exception:
        return str(v)


def _fmt_usdt(v: Any) -> str:
    try:
        x = float(v)
        sign = "+" if x > 0 else ""
        return f"{sign}{x:.4f} USDT"
    except Exception:
        return str(v)


def _fmt_pct(v: Any) -> str:
    try:
        x = float(v)
        sign = "+" if x > 0 else ""
        return f"{sign}{x:.2f}%"
    except Exception:
        return str(v)


def _fmt_plain(v: Any, digits: int = 2) -> str:
    try:
        return f"{float(v):.{digits}f}"
    except Exception:
        return str(v)


def _outcome_title(outcome: str) -> str:
    x = str(outcome or "").upper()
    if x == "TP":
        return "✅ <b>TRADE CLOSED — TP HIT</b>"
    if x == "SL":
        return "🛑 <b>TRADE CLOSED — SL HIT</b>"
    if x == "MANUAL_SELL":
        return "📤 <b>TRADE CLOSED — MANUAL SELL</b>"
    return f"📦 <b>TRADE CLOSED — {_escape_html(x)}</b>"


def _now_str() -> str:
    try:
        tz = ZoneInfo(TELEGRAM_TIMEZONE)
        return datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
    except Exception as e:
        logger.warning("TZ_FALLBACK | timezone=%s err=%s", TELEGRAM_TIMEZONE, e)
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def notify_signal_created(
    symbol: str,
    entry_price: float,
    quote_amount: float,
    tp_price: float,
    sl_price: float,
    verdict: str = "BUY",
    mode: str = "LIVE",
) -> None:
    tp_pct = ((float(tp_price) - float(entry_price)) / float(entry_price) * 100.0) if float(entry_price) else 0.0
    sl_pct = ((float(sl_price) - float(entry_price)) / float(entry_price) * 100.0) if float(entry_price) else 0.0

    msg = (
        f"🚀 <b>NEW SIGNAL OPENED</b>\n\n"
        f"🪙 <b>Symbol:</b> <code>{_escape_html(symbol)}</code>\n"
        f"💰 <b>Entry:</b> <code>{_fmt_price(entry_price)}</code> USDT\n"
        f"📦 <b>Size:</b> <code>{_fmt_plain(quote_amount, 2)}</code> USDT\n"
        f"🎯 <b>TP:</b> <code>{_fmt_price(tp_price)}</code> ({_fmt_pct(tp_pct)})\n"
        f"🛑 <b>SL:</b> <code>{_fmt_price(sl_price)}</code> ({_fmt_pct(sl_pct)})\n\n"
        f"🧠 <b>Verdict:</b> <code>{_escape_html(verdict)}</code>\n"
        f"📌 <b>Mode:</b> <code>{_escape_html(mode)}</code>\n"
        f"🕒 <b>Time:</b> <code>{_now_str()}</code>"
    )
    send_telegram_message(msg)


def notify_trade_closed(
    symbol: str,
    entry_price: float,
    exit_price: float,
    pnl_quote: float,
    pnl_pct: float,
    outcome: str,
    stats: Optional[Dict[str, Any]] = None,
) -> None:
    msg = (
        f"{_outcome_title(outcome)}\n\n"
        f"🪙 <b>Symbol:</b> <code>{_escape_html(symbol)}</code>\n"
        f"📥 <b>Entry:</b> <code>{_fmt_price(entry_price)}</code>\n"
        f"📤 <b>Exit:</b> <code>{_fmt_price(exit_price)}</code>\n"
        f"💵 <b>PnL:</b> <code>{_fmt_usdt(pnl_quote)}</code>\n"
        f"📈 <b>PnL %:</b> <code>{_fmt_pct(pnl_pct)}</code>\n"
        f"🎯 <b>Outcome:</b> <code>{_escape_html(str(outcome).upper())}</code>\n"
        f"🕒 <b>Time:</b> <code>{_now_str()}</code>"
    )

    if stats:
        msg += (
            f"\n\n📊 <b>Total closed:</b> <code>{int(stats.get('closed_trades', 0))}</code>\n"
            f"🏆 <b>Wins:</b> <code>{int(stats.get('wins', 0))}</code>\n"
            f"❌ <b>Losses:</b> <code>{int(stats.get('losses', 0))}</code>\n"
            f"🔥 <b>Winrate:</b> <code>{float(stats.get('winrate_pct', 0.0)):.2f}%</code>\n"
            f"💹 <b>ROI:</b> <code>{float(stats.get('roi_pct', 0.0)):.2f}%</code>\n"
            f"💰 <b>Total PnL:</b> <code>{float(stats.get('pnl_quote_sum', 0.0)):.4f} USDT</code>\n"
            f"📂 <b>Open trades:</b> <code>{int(stats.get('open_trades', 0))}</code>"
        )

    send_telegram_message(msg)


def notify_performance_snapshot(stats: Dict[str, Any]) -> None:
    msg = (
        f"📊 <b>BOT PERFORMANCE SNAPSHOT</b>\n\n"
        f"✅ <b>Closed trades:</b> <code>{int(stats.get('closed_trades', 0))}</code>\n"
        f"🏆 <b>Wins:</b> <code>{int(stats.get('wins', 0))}</code>\n"
        f"❌ <b>Losses:</b> <code>{int(stats.get('losses', 0))}</code>\n"
        f"🔥 <b>Winrate:</b> <code>{float(stats.get('winrate_pct', 0.0)):.2f}%</code>\n"
        f"💵 <b>Total PnL:</b> <code>{float(stats.get('pnl_quote_sum', 0.0)):.4f} USDT</code>\n"
        f"💹 <b>ROI:</b> <code>{float(stats.get('roi_pct', 0.0)):.2f}%</code>\n"
        f"⚔️ <b>Profit factor:</b> <code>{float(stats.get('profit_factor', 0.0)):.2f}</code>\n\n"
        f"📂 <b>Open trades:</b> <code>{int(stats.get('open_trades', 0))}</code>\n"
        f"💰 <b>Open capital:</b> <code>{float(stats.get('open_quote_in_sum', 0.0)):.2f} USDT</code>\n"
        f"🕒 <b>Time:</b> <code>{_now_str()}</code>"
    )
    send_telegram_message(msg)
