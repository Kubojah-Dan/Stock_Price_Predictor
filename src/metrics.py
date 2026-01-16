import numpy as np

def sharpe(returns, rf=0):
    return np.mean(returns - rf) / (np.std(returns) + 1e-9) * np.sqrt(252)

def sortino(returns, rf=0):
    downside = returns[returns < rf]
    return np.mean(returns - rf) / (np.std(downside) + 1e-9) * np.sqrt(252)
