# my_adapter.py
# Adapter implementation to connect diagnostics_pro with your bot

from diagnostics_pro import Adapter

class MyAdapter(Adapter):
    def __init__(self, exchange, db):
        self.exchange = exchange
        self.db = db

    # ===== DB LAYER =====
    def get_trade(self, signal_id):
        return self.db.get_trade(signal_id)

    def get_oco_status(self, link_id):
        return self.db.get_oco_status(link_id)

    def get_close_events_count(self, signal_id):
        return self.db.count_close_events(signal_id)

    def get_trade_logs(self, signal_id):
        return self.db.get_logs(signal_id)

    def get_open_trades(self):
        return self.db.get_open_trades()

    # ===== EXCHANGE LAYER =====
    def get_order(self, order_id):
        if not order_id:
            return None
        try:
            return self.exchange.get_order(order_id)
        except Exception:
            return None

    def get_fills(self, order_id):
        try:
            return self.exchange.get_my_trades(order_id)
        except Exception:
            return []

    def get_position(self, symbol):
        try:
            return self.exchange.get_position(symbol)
        except Exception:
            return {"qty": 0}

    def get_balance(self):
        try:
            return self.exchange.get_balance()
        except Exception:
            return {}

    def get_fee_rate(self, symbol):
        # შეგიძლია აქ დააბრუნო dynamic fee exchange-დან
        return 0.001

    def get_latency_ms(self):
        try:
            return self.exchange.get_latency()
        except Exception:
            return 0
