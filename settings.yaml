
class VirtualWallet:
    def __init__(self):
        self.orders = []

    def create_market_order(self, symbol, side, amount):
        order = {
            "symbol": symbol,
            "side": side,
            "amount": amount,
            "status": "filled"
        }
        self.orders.append(order)
        return order
