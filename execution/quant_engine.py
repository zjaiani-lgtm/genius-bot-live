import time


# =========================
# 🧩 UTILS
# =========================
def clamp(value, min_v, max_v):
    return max(min_v, min(value, max_v))


# =========================
# 📊 REGIME DETECTION
# =========================
def detect_regime(atr_pct):
    if atr_pct < 0.25:
        return "LOW"
    elif atr_pct < 0.50:
        return "NORMAL"
    else:
        return "HIGH"


# =========================
# 📈 ENTRY SCORING
# =========================
def calculate_entry_score(ai_conf, trend, sma_gap, vol_ratio):
    sma_norm = clamp(sma_gap / 0.05, 0.0, 1.0)
    vol_norm = clamp(vol_ratio, 0.0, 1.0)

    score = (
        ai_conf * 0.4 +
        trend * 0.3 +
        sma_norm * 0.2 +
        vol_norm * 0.1
    )

    return clamp(score, 0.0, 1.0)


# =========================
# 💰 POSITION SIZING
# =========================
def calculate_position_size(balance, risk_pct, atr_pct):
    risk_amount = balance * (risk_pct / 100.0)

    stop_distance = max(atr_pct * 1.5, 0.3)

    position_size = risk_amount / stop_distance

    return position_size


# =========================
# 🛑 STOP LOSS (ATR BASED)
# =========================
def compute_stop_loss(entry_price, atr_pct):
    stop_pct = max(atr_pct * 1.5, 0.4)
    return entry_price * (1 - stop_pct / 100.0)


# =========================
# 📉 TRAILING SYSTEM
# =========================
def get_base_trailing(score, regime):
    if regime == "LOW":
        return 0.18 if score < 0.6 else 0.22
    elif regime == "NORMAL":
        return 0.22 if score < 0.6 else 0.26
    else:  # HIGH
        return 0.26 if score < 0.6 else 0.32


def time_decay(minutes_open):
    return clamp(1.0 - (minutes_open / 300.0), 0.5, 1.0)


# =========================
# 🧠 TRADE OBJECT
# =========================
class Trade:
    def __init__(self, entry_price, score, position_size):
        self.entry_price = entry_price
        self.score = score
        self.position_size = position_size
        self.open_time = time.time()
        self.max_profit = 0.0

    def minutes_open(self):
        return (time.time() - self.open_time) / 60.0


# =========================
# 🚪 EXIT ENGINE
# =========================
def should_exit(trade, current_profit_pct, atr_pct):
    minutes = trade.minutes_open()

    regime = detect_regime(atr_pct)

    base_trailing = get_base_trailing(trade.score, regime)
    decay = time_decay(minutes)

    trailing = clamp(base_trailing * decay, 0.12, 0.35)

    # update peak profit
    trade.max_profit = max(trade.max_profit, current_profit_pct)

    drawdown = trade.max_profit - current_profit_pct

    # 🔴 TRAILING EXIT
    if drawdown >= trailing:
        return True, "TRAILING_EXIT"

    # 🔴 ATR-BASED HARD STOP
    if current_profit_pct <= -1.5 * atr_pct:
        return True, "HARD_STOP"

    # 🔴 SCORE-AWARE TIME EXIT
    max_hold = 120 + trade.score * 120
    if minutes > max_hold:
        return True, "TIME_EXIT"

    return False, None


# =========================
# 🧪 FULL EXAMPLE
# =========================
if __name__ == "__main__":
    balance = 1000

    # mock input (replace with real data)
    ai_conf = 0.65
    trend = 0.60
    sma_gap = 0.03
    vol_ratio = 0.9
    atr_pct = 0.23
    entry_price = 100

    # ENTRY
    score = calculate_entry_score(ai_conf, trend, sma_gap, vol_ratio)
    regime = detect_regime(atr_pct)

    position_size = calculate_position_size(balance, 0.5, atr_pct)
    stop_loss = compute_stop_loss(entry_price, atr_pct)

    trade = Trade(entry_price, score, position_size)

    print(f"Score: {score:.2f}, Regime: {regime}, Size: {position_size:.4f}, SL: {stop_loss:.2f}")

    # SIMULATION LOOP
    for i in range(1, 300):
        time.sleep(0.01)

        # fake profit curve
        profit = min(1.0, i * 0.01) - max(0, (i - 100)) * 0.01

        exit_signal, reason = should_exit(trade, profit, atr_pct)

        print(f"t={i} profit={profit:.2f} exit={exit_signal} reason={reason}")

        if exit_signal:
            break
