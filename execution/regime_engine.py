class MarketRegimeEngine:
    def __init__(self, config):
        self.config = config

    def detect_regime(self, trend, vol):
        if vol < 0.2:
            return "SIDEWAYS"
        elif trend > 0.5:
            return "TREND_UP"
        elif trend < -0.5:
            return "TREND_DOWN"
        return "UNCERTAIN"

    def apply(self, regime):
        if regime == "SIDEWAYS":
            return {
                "SKIP_TRADING": True,
                "QUOTE_SIZE": 0.5
            }
        return {
            "SKIP_TRADING": False
        }
