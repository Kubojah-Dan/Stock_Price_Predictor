def init_portfolio(capital=100000):
    return {
        "cash": capital,
        "positions": {},   # {ticker: {size, entry_price}}
        "equity": capital,
        "trades": []
    }


def update_equity(portfolio, prices):
    equity = portfolio["cash"]
    for ticker, pos in portfolio["positions"].items():
        equity += prices[ticker] * pos["size"]
    portfolio["equity"] = equity
    return portfolio


