def init_account(capital=100000):
    return {
        "cash": capital,
        "position": 0,
        "entry_price": None,
        "equity": capital,
        "trades": []
    }


def execute_trade(
    portfolio,
    ticker,
    signal,
    price,
    timestamp,
    size
):
    positions = portfolio["positions"]

    if signal == "BUY" and ticker not in positions and size > 0:
        cost = price * size
        if portfolio["cash"] >= cost:
            portfolio["cash"] -= cost
            positions[ticker] = {
                "size": size,
                "entry_price": price
            }
            portfolio["trades"].append({
                "time": timestamp,
                "ticker": ticker,
                "type": "BUY",
                "price": price,
                "size": size
            })

    elif signal == "SELL" and ticker in positions:
        pos = positions[ticker]
        portfolio["cash"] += price * pos["size"]
        portfolio["trades"].append({
            "time": timestamp,
            "ticker": ticker,
            "type": "SELL",
            "price": price,
            "size": pos["size"]
        })
        del positions[ticker]

    return portfolio

