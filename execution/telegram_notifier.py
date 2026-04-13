import os
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, List
from zoneinfo import ZoneInfo

import requests

logger = logging.getLogger("gbm")


def _env_bool(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() in ("1", "true", "yes", "y", "on")


TELEGRAM_ENABLED = _env_bool("TELEGRAM_NOTIFICATIONS", "false")

def _clean_ascii(s: str) -> str:
    """
    Remove non-ASCII chars — prevent latin-1 UnicodeEncodeError in urllib3.
    Telegram bot token and chat_id must be pure ASCII.
    Copy-paste from Telegram can introduce invisible Unicode spaces.
    """
    return "".join(c for c in s if ord(c) < 128).strip()

TELEGRAM_BOT_TOKEN = _clean_ascii(os.getenv("TELEGRAM_BOT_TOKEN", ""))
TELEGRAM_CHAT_IDS = [
    _clean_ascii(x)
    for x in os.getenv("TELEGRAM_CHAT_ID", "").split(",")
    if _clean_ascii(x)
]
TELEGRAM_PARSE_MODE = _clean_ascii(os.getenv("TELEGRAM_PARSE_MODE", "HTML")).upper()
TELEGRAM_TIMEZONE = os.getenv("TELEGRAM_TIMEZONE", "Asia/Tbilisi").strip()


def _tz() -> ZoneInfo:
    try:
        return ZoneInfo(TELEGRAM_TIMEZONE)
    except Exception as e:
        logger.warning("TZ_FALLBACK | timezone=%s err=%s", TELEGRAM_TIMEZONE, e)
        return ZoneInfo("Asia/Tbilisi")


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
            # FIX: explicit utf-8 encoding — prevents latin-1 UnicodeEncodeError
            # in urllib3 when emoji/Georgian chars are in the message body.
            # json= uses utf-8 internally, but older urllib3 versions
            # can re-encode headers as latin-1. Force manual encoding.
            import json as _json
            encoded_body = _json.dumps(payload, ensure_ascii=False).encode("utf-8")
            headers = {
                "Content-Type": "application/json; charset=utf-8",
                "Accept": "application/json",
            }
            r = requests.post(url, data=encoded_body, headers=headers, timeout=15)
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
        return "✅ <b>!!SELL!! TRADE CLOSED — TP HIT</b>"
    if x == "SL":
        return "🛑 <b>!!SELL!! TRADE CLOSED — SL HIT</b>"
    if x == "MANUAL_SELL":
        return "📤 <b>TRADE CLOSED — MANUAL SELL</b>"
    return f"📦 <b>TRADE CLOSED — {_escape_html(x)}</b>"


def _now_dt() -> datetime:
    return datetime.now(_tz())


def _now_str() -> str:
    return _now_dt().strftime("%Y-%m-%d %H:%M:%S")


def _day_bounds_tbilisi(target_dt: Optional[datetime] = None) -> tuple[datetime, datetime]:
    now_local = target_dt.astimezone(_tz()) if target_dt else _now_dt()
    day_start = now_local.replace(hour=0, minute=0, second=0, microsecond=0)
    day_end = now_local.replace(hour=23, minute=59, second=59, microsecond=999999)
    return day_start, day_end


def _parse_trade_time(value: Any) -> Optional[datetime]:
    """
    ელის datetime-ს ან ISO string-ს.
    თუ timezone არ აქვს, ვთვლით რომ უკვე თბილისის დროა.
    """
    if value is None:
        return None

    if isinstance(value, datetime):
        dt = value
    else:
        s = str(value).strip()
        if not s:
            return None
        try:
            # მხარდაჭერა '2026-03-16T23:59:01', '2026-03-16 23:59:01', '...Z'
            s = s.replace("Z", "+00:00")
            dt = datetime.fromisoformat(s)
        except Exception:
            return None

    if dt.tzinfo is None:
        return dt.replace(tzinfo=_tz())

    return dt.astimezone(_tz())


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def build_daily_stats_from_closed_trades(
    closed_trades: List[Dict[str, Any]],
    target_dt: Optional[datetime] = None,
) -> Dict[str, Any]:
    """
    closed_trades ელემენტში სასურველია იყოს მინიმუმ:
      - pnl_quote
      - outcome
      - closed_at  (ან exit_time / closed_time)

    მაგალითი:
    {
        "symbol": "BTC/USDT",
        "pnl_quote": 1.25,
        "pnl_pct": 0.83,
        "outcome": "TP",
        "closed_at": "2026-03-16T21:15:00+04:00"
    }
    """
    day_start, day_end = _day_bounds_tbilisi(target_dt)

    day_trades: List[Dict[str, Any]] = []

    for trade in closed_trades:
        closed_at_raw = (
            trade.get("closed_at")
            or trade.get("exit_time")
            or trade.get("closed_time")
            or trade.get("updated_at")
        )
        closed_at = _parse_trade_time(closed_at_raw)
        if not closed_at:
            continue

        if day_start <= closed_at <= day_end:
            day_trades.append(trade)

    closed_count = len(day_trades)

    wins = 0
    losses = 0
    pnl_sum = 0.0
    roi_sum = 0.0
    gross_profit = 0.0
    gross_loss_abs = 0.0

    for t in day_trades:
        pnl_quote = _safe_float(t.get("pnl_quote"), 0.0)
        pnl_pct = _safe_float(t.get("pnl_pct"), 0.0)
        outcome = str(t.get("outcome", "")).upper()

        pnl_sum += pnl_quote
        roi_sum += pnl_pct

        if outcome == "TP" or pnl_quote > 0:
            wins += 1
            if pnl_quote > 0:
                gross_profit += pnl_quote
        else:
            losses += 1
            if pnl_quote < 0:
                gross_loss_abs += abs(pnl_quote)

    winrate_pct = (wins / closed_count * 100.0) if closed_count > 0 else 0.0
    profit_factor = (gross_profit / gross_loss_abs) if gross_loss_abs > 0 else (999.0 if gross_profit > 0 else 0.0)

    return {
        "date": day_start.strftime("%Y-%m-%d"),
        "day_start": day_start,
        "day_end": day_end,
        "closed_trades": closed_count,
        "wins": wins,
        "losses": losses,
        "winrate_pct": winrate_pct,
        "pnl_quote_sum": pnl_sum,
        "roi_pct": roi_sum,
        "profit_factor": profit_factor,
    }


def notify_signal_created(
    symbol: str,
    entry_price: float,
    quote_amount: float,
    tp_price: float,
    sl_price: float = 0.0,
    verdict: str = "BUY",
    mode: str = "LIVE",
) -> None:
    tp_pct = ((float(tp_price) - float(entry_price)) / float(entry_price) * 100.0) if float(entry_price) else 0.0

    msg = (
        f"🚀 <b>NEW SIGNAL OPENED</b>\n\n"
        f"🪙 <b>Symbol:</b> <code>{_escape_html(symbol)}</code>\n"
        f"💰 <b>Entry:</b> <code>{_fmt_price(entry_price)}</code> USDT\n"
        f"📦 <b>Size:</b> <code>{_fmt_plain(quote_amount, 2)}</code> USDT\n"
        f"🎯 <b>TP:</b> <code>{_fmt_price(tp_price)}</code> ({_fmt_pct(tp_pct)})\n\n"
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
    verdict = "SELL"

    msg = (
        f"{_outcome_title(outcome)}\n\n"
        f"🪙 <b>Symbol:</b> <code>{_escape_html(symbol)}</code>\n"
        f"📥 <b>Entry:</b> <code>{_fmt_price(entry_price)}</code>\n"
        f"📤 <b>Exit:</b> <code>{_fmt_price(exit_price)}</code>\n"
        f"💵 <b>PnL:</b> <code>{_fmt_usdt(pnl_quote)}</code>\n"
        f"📈 <b>PnL %:</b> <code>{_fmt_pct(pnl_pct)}</code>\n"
        f"🎯 <b>Outcome:</b> <code>{_escape_html(str(outcome).upper())}</code>\n"
        f"🧠 <b>Verdict:</b> <code>{_escape_html(verdict)}</code>\n"
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
        f"📊 <b>GENIUS-ALGO-BOT-DEMO PERFORMANCE</b>\n\n"
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


def notify_daily_close_summary(daily_stats: Dict[str, Any]) -> None:
    msg = (
        f"🌙 <b>DAILY DCA CLOSE SUMMARY</b>\n\n"
        f"📅 <b>Date:</b> <code>{_escape_html(daily_stats.get('date', ''))}</code>\n"
        f"🕛 <b>Window:</b> <code>00:00 - 23:59 Asia/Tbilisi</code>\n\n"
        f"✅ <b>Closed today:</b> <code>{int(daily_stats.get('closed_trades', 0))}</code>\n"
        f"🏆 <b>Wins:</b> <code>{int(daily_stats.get('wins', 0))}</code>\n"
        f"❌ <b>Losses:</b> <code>{int(daily_stats.get('losses', 0))}</code>\n"
        f"🔥 <b>Winrate:</b> <code>{float(daily_stats.get('winrate_pct', 0.0)):.2f}%</code>\n"
        f"💰 <b>Day PnL:</b> <code>{float(daily_stats.get('pnl_quote_sum', 0.0)):.4f} USDT</code>\n"
        f"📈 <b>Day ROI:</b> <code>{float(daily_stats.get('roi_pct', 0.0)):.2f}%</code>\n"
        f"⚔️ <b>Profit factor:</b> <code>{float(daily_stats.get('profit_factor', 0.0)):.2f}</code>\n\n"
        f"🕒 <b>Sent at:</b> <code>{_now_str()}</code>"
    )
    send_telegram_message(msg)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DCA NOTIFICATIONS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def notify_dca_position_opened(
    symbol: str,
    entry_price: float,
    qty: float,
    quote_spent: float,
    tp_price: float,
    sl_price: float = 0.0,
    tp_pct: float = 0.0,
    sl_pct: float = 0.0,
    max_add_ons: int = 3,
    max_capital: float = 40.0,
) -> None:
    """ახალი DCA position გახსნილია."""
    tp_pct_f = float(tp_pct)
    msg = (
        f"📈 <b>DCA POSITION OPENED</b>\n\n"
        f"🪙 <b>Symbol:</b> <code>{_escape_html(symbol)}</code>\n"
        f"💰 <b>Entry:</b> <code>{_fmt_price(entry_price)}</code> USDT\n"
        f"📦 <b>Size:</b> <code>{_fmt_plain(quote_spent, 2)}</code> USDT\n"
        f"🎯 <b>TP:</b> <code>{_fmt_price(tp_price)}</code> (<code>{_fmt_pct(tp_pct_f)}</code>)\n"
        f"🔄 <b>Max add-ons:</b> <code>{max_add_ons}</code>\n"
        f"💼 <b>Max capital:</b> <code>{_fmt_plain(max_capital, 1)}</code> USDT\n"
        f"🕒 <b>Time:</b> <code>{_now_str()}</code>"
    )
    send_telegram_message(msg)


def notify_dca_addon(
    symbol: str,
    addon_number: int,
    addon_price: float,
    addon_quote: float,
    new_avg_entry: float,
    total_quote_spent: float,
    new_tp_price: float,
    new_sl_price: float = 0.0,
    drawdown_pct: float = 0.0,
    recovery_score: int = 0,
) -> None:
    """DCA add-on order გახსნილია."""
    msg = (
        f"➕ <b>DCA ADD-ON #{addon_number}</b>\n\n"
        f"🪙 <b>Symbol:</b> <code>{_escape_html(symbol)}</code>\n"
        f"📉 <b>Drawdown:</b> <code>{_fmt_pct(-drawdown_pct)}</code> from avg\n"
        f"💰 <b>Add-on price:</b> <code>{_fmt_price(addon_price)}</code>\n"
        f"📦 <b>Add-on size:</b> <code>{_fmt_plain(addon_quote, 2)}</code> USDT\n"
        f"📊 <b>New avg entry:</b> <code>{_fmt_price(new_avg_entry)}</code>\n"
        f"💼 <b>Total spent:</b> <code>{_fmt_plain(total_quote_spent, 2)}</code> USDT\n"
        f"🎯 <b>New TP:</b> <code>{_fmt_price(new_tp_price)}</code>\n"
        f"🔍 <b>Recovery score:</b> <code>{recovery_score}/5</code>\n"
        f"🕒 <b>Time:</b> <code>{_now_str()}</code>"
    )
    send_telegram_message(msg)


def notify_dca_closed(
    symbol: str,
    avg_entry_price: float,
    exit_price: float,
    total_qty: float,
    total_quote_spent: float,
    pnl_quote: float,
    pnl_pct: float,
    outcome: str,
    add_on_count: int,
    stats: Optional[Dict[str, Any]] = None,
) -> None:
    """DCA position დახურულია."""
    outcome_upper = str(outcome or "").upper()

    if outcome_upper == "TP":
        title = "✅ <b>DCA POSITION CLOSED — TP HIT</b>"
    elif outcome_upper == "SL":
        title = "🛑 <b>DCA POSITION CLOSED — SL HIT</b>"
    elif outcome_upper == "FORCE_CLOSE":
        title = "⚠️ <b>DCA POSITION FORCE CLOSED</b>"
    else:
        title = f"📦 <b>DCA POSITION CLOSED — {_escape_html(outcome_upper)}</b>"

    msg = (
        f"{title}\n\n"
        f"🪙 <b>Symbol:</b> <code>{_escape_html(symbol)}</code>\n"
        f"📥 <b>Avg entry:</b> <code>{_fmt_price(avg_entry_price)}</code>\n"
        f"📤 <b>Exit:</b> <code>{_fmt_price(exit_price)}</code>\n"
        f"💵 <b>PnL:</b> <code>{_fmt_usdt(pnl_quote)}</code>\n"
        f"📈 <b>PnL %:</b> <code>{_fmt_pct(pnl_pct)}</code>\n"
        f"💼 <b>Total invested:</b> <code>{_fmt_plain(total_quote_spent, 2)}</code> USDT\n"
        f"🔄 <b>Add-ons used:</b> <code>{add_on_count}</code>\n"
        f"🎯 <b>Outcome:</b> <code>{_escape_html(outcome_upper)}</code>\n"
        f"🕒 <b>Time:</b> <code>{_now_str()}</code>"
    )

    if stats:
        msg += (
            f"\n\n📊 <b>Total closed:</b> <code>{int(stats.get('closed_trades', 0))}</code>\n"
            f"🏆 <b>Wins:</b> <code>{int(stats.get('wins', 0))}</code>\n"
            f"❌ <b>Losses:</b> <code>{int(stats.get('losses', 0))}</code>\n"
            f"🔥 <b>Winrate:</b> <code>{float(stats.get('winrate_pct', 0.0)):.2f}%</code>\n"
            f"💰 <b>Total PnL:</b> <code>{float(stats.get('pnl_quote_sum', 0.0)):.4f} USDT</code>"
        )

    send_telegram_message(msg)


def notify_dca_breakeven(
    symbol: str,
    avg_entry_price: float,
    old_sl: float = 0.0,
    new_sl: float = 0.0,
) -> None:
    """Breakeven protection ამოქმედდა."""
    msg = (
        f"🔒 <b>DCA BREAKEVEN ACTIVATED</b>\n\n"
        f"🪙 <b>Symbol:</b> <code>{_escape_html(symbol)}</code>\n"
        f"📊 <b>Avg entry:</b> <code>{_fmt_price(avg_entry_price)}</code>\n"
        f"✅ <b>Position protected at breakeven</b>\n"
        f"🕒 <b>Time:</b> <code>{_now_str()}</code>"
    )
    send_telegram_message(msg)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# D: CASCADE SELL — გაუმჯობესებული შეტყობინება
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def notify_cascade_exchange(
    symbol: str,
    old_avg: float,
    old_layer: str,
    new_avg: float,
    new_layer: str,
    sell_price: float,
    pnl_quote: float,
    drop_pct: float,
    new_tp: float,
) -> None:
    """CASCADE Rolling Exchange — გამარტივებული avg."""
    recovery_needed = ((new_tp / new_avg) - 1.0) * 100.0 if new_avg > 0 else 0.0
    pnl_icon = "🟢" if pnl_quote >= 0 else "🔴"
    msg = (
        f"🔄 <b>CASCADE EXCHANGE</b>\n\n"
        f"🪙 <b>Symbol:</b> <code>{_escape_html(symbol)}</code>\n"
        f"📤 <b>გაიყიდა:</b> <code>{_escape_html(old_layer)}</code> avg=<code>{_fmt_price(old_avg)}</code>\n"
        f"📥 <b>გაიხსნა:</b> <code>{_escape_html(new_layer)}</code> avg=<code>{_fmt_price(new_avg)}</code>\n"
        f"📉 <b>Avg ჩამოვიდა:</b> <code>{_fmt_price(old_avg)} → {_fmt_price(new_avg)}</code>\n"
        f"{pnl_icon} <b>Exchange PnL:</b> <code>{_fmt_usdt(pnl_quote)}</code>\n"
        f"📊 <b>Drop trigger:</b> <code>{drop_pct:.2f}%</code>\n"
        f"🎯 <b>TP:</b> <code>{_fmt_price(new_tp)}</code> (+{recovery_needed:.2f}%)\n"
        f"🕒 <b>Time:</b> <code>{_now_str()}</code>"
    )
    send_telegram_message(msg)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# E: HEARTBEAT — ბოტი ცოცხალია
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def notify_heartbeat(
    open_count: int,
    open_capital: float,
    prices: dict,
    memory_mb: float = 0.0,
    pnl_today: float = 0.0,
    positions: list = None,
) -> None:
    """ყოველ 10 წუთს — ბოტი ცოცხალია + პოზიციების სტატუსი."""
    btc = prices.get("BTC/USDT", 0.0)
    bnb = prices.get("BNB/USDT", 0.0)
    eth = prices.get("ETH/USDT", 0.0)
    mem_str = f" | 💾 <code>{memory_mb:.0f}MB</code>" if memory_mb > 0 else ""

    msg = (
        f"💚 <b>GENIUS ALGO BOT DEMO</b>\n\n"
        f"📂 <b>Open:</b> <code>{open_count}</code> | 💰 <code>{_fmt_plain(open_capital, 2)} USDT</code>\n"
        f"₿ BTC <code>{_fmt_price(btc, 0)}</code> | "
        f"Ξ ETH <code>{_fmt_price(eth, 0)}</code> | "
        f"BNB <code>{_fmt_price(bnb, 2)}</code>\n"
        f"📈 <b>PnL today:</b> <code>{_fmt_usdt(pnl_today)}</code>"
        f"{mem_str}\n"
    )

    # პოზიციების სტატუსი — TP-მდე დაშორება + ინდიკატორი
    if positions:
        msg += "\n<b>📊 პოზიციები:</b>\n"
        for pos in sorted(positions, key=lambda p: str(p.get("symbol", ""))):
            sym = str(pos.get("symbol", ""))
            avg = float(pos.get("avg_entry_price") or 0)
            tp  = float(pos.get("current_tp_price") or 0)
            if avg <= 0 or tp <= 0:
                continue

            # ახლანდელი ფასი cache-დან
            base_sym = sym.split("_")[0]  # BTC/USDT_L2 → BTC/USDT
            current = prices.get(base_sym, 0.0)
            if current <= 0:
                continue

            # TP-მდე რამდენი %
            tp_dist_pct = (tp - current) / current * 100.0
            # avg-დან რამდენი % ქვევით
            avg_dist_pct = (current - avg) / avg * 100.0

            # ინდიკატორი:
            # 🔴 ← TP-მდე > 1% (შორს)
            # 🟡 ← TP-მდე 0.3-1% (შუაში)
            # 🔵 ← TP-მდე < 0.3% (ახლოს!)
            # 🟢 ← TP hit (current >= tp)
            if current >= tp:
                indicator = "🟢"
            elif tp_dist_pct < 0.3:
                indicator = "🔵"
            elif tp_dist_pct < 1.0:
                indicator = "🟡"
            else:
                indicator = "🔴"

            msg += (
                f"{indicator} <code>{_escape_html(sym)}</code> "
                f"avg=<code>{_fmt_price(avg, 0)}</code> "
                f"tp=<code>{_fmt_price(tp, 0)}</code> "
                f"(<code>{tp_dist_pct:+.2f}%</code>)\n"
            )

    msg += f"🕒 <code>{_now_str()}</code>"
    send_telegram_message(msg)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# F: CASCADE DEPTH WARNING — სიღრმის გაფრთხილება
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def notify_cascade_depth(
    symbol: str,
    layer_num: int,
    max_layers: int,
    drop_from_high_pct: float,
    current_price: float,
    avg_entry: float,
    price_trend: str = "unknown",
) -> None:
    """CASCADE სიღრმის გაფრთხილება L7+."""
    remaining = max_layers - layer_num

    if layer_num >= max_layers:
        icon = "🚨"
        title = f"CASCADE MAX — ბოტი შეჩერდება!"
    elif layer_num >= max_layers - 1:
        icon = "🔴"
        title = f"CASCADE L{layer_num}/{max_layers} — კრიტიკული! {remaining} layer დარჩა"
    else:
        icon = "⚠️"
        title = f"CASCADE L{layer_num}/{max_layers} — ღრმა ვარდნა"

    drawdown = ((avg_entry - current_price) / avg_entry * 100.0) if avg_entry > 0 else 0.0

    if price_trend == "down":
        trend_str = "📉 ბაზარი კვლავ ეცემა"
    elif price_trend == "up":
        trend_str = "📈 ბაზარი იწყებს ზრდას"
    else:
        trend_str = "➡️ ბაზარი სტაბილურია"

    msg = (
        f"{icon} <b>{_escape_html(title)}</b>\n\n"
        f"🪙 <b>Symbol:</b> <code>{_escape_html(symbol)}</code>\n"
        f"📉 <b>HIGH-დან ვარდნა:</b> <code>{drop_from_high_pct:.2f}%</code>\n"
        f"📊 <b>Avg entry-დან:</b> <code>{drawdown:.2f}%</code>\n"
        f"💰 <b>ფასი:</b> <code>{_fmt_price(current_price)}</code>\n"
        f"🎯 <b>Avg entry:</b> <code>{_fmt_price(avg_entry)}</code>\n"
        f"{trend_str}\n"
        f"🕒 <code>{_now_str()}</code>"
    )
    send_telegram_message(msg)
