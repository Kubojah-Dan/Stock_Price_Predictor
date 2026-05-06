import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime
import time

def init_mt5(account, password, server):
    """Initialize MT5 and login to the given account."""
    import os
    
    # Try to initialize
    if not mt5.initialize():
        error = mt5.last_error()
        print(f"MT5 initialize() failed, error code: {error}")
        # Common MT5 installation paths for XM
        common_paths = [
            "C:/Program Files/XM Global MT5/terminal64.exe",
            "C:/Program Files/MetaTrader 5/terminal64.exe"
        ]
        success = False
        for path in common_paths:
            if os.path.exists(path):
                print(f"Attempting to initialize with path: {path}")
                if mt5.initialize(path=path):
                    success = True
                    break
        
        if not success:
            return False
        
    print(f"Attempting login for account {account} on server {server}...")
    authorized = mt5.login(int(account), password=password, server=server)
    if not authorized:
        print(f"MT5 login failed for account #{account}, error code: {mt5.last_error()}")
        return False
        
    print("MT5 connection successful!")
    return True

def get_account_info():
    """Retrieve account summary like balance, equity, margin."""
    account_info = mt5.account_info()
    if account_info is None:
        return None
    return {
        "login": account_info.login,
        "balance": account_info.balance,
        "equity": account_info.equity,
        "margin": account_info.margin,
        "free_margin": account_info.margin_free,
        "currency": account_info.currency
    }

def get_positions():
    """Retrieve open positions."""
    positions = mt5.positions_get()
    if positions is None:
        return []
    
    res = []
    for p in positions:
        res.append({
            "ticket": p.ticket,
            "symbol": p.symbol,
            "type": "BUY" if p.type == mt5.POSITION_TYPE_BUY else "SELL",
            "volume": p.volume,
            "open_price": p.price_open,
            "current_price": p.price_current,
            "sl": p.sl,
            "tp": p.tp,
            "profit": p.profit
        })
    return res

def place_order(symbol: str, order_type: str, volume: float, price: float = None):
    """Place a market order."""
    if order_type == "BUY":
        mt5_type = mt5.ORDER_TYPE_BUY
        price = mt5.symbol_info_tick(symbol).ask if price is None else price
    elif order_type == "SELL":
        mt5_type = mt5.ORDER_TYPE_SELL
        price = mt5.symbol_info_tick(symbol).bid if price is None else price
    else:
        return {"error": "Invalid order type"}

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": float(volume),
        "type": mt5_type,
        "price": price,
        "deviation": 20,
        "magic": 234000,
        "comment": "AI Agent Trade",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    
    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        return {"error": f"Order failed: {result.comment}"}
        
    return {
        "success": True,
        "order": result.order,
        "volume": result.volume,
        "price": result.price
    }

def get_symbol_price(symbol):
    """Get the current live price for a symbol from MT5."""
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        # Try finding the symbol (sometimes brokers add suffixes like .m, .x)
        return None
    return (tick.bid + tick.ask) / 2

def shutdown_mt5():
    mt5.shutdown()
