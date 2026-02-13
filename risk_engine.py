
def execute_order(exchange, symbol, side, amount):
    try:
        order = exchange.create_market_order(
            symbol=symbol,
            side=side.lower(),
            amount=amount
        )
        return True, order
    except Exception as e:
        return False, str(e)
