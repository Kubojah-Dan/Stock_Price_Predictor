def generate_signal(prob, buy_th=0.6, sell_th=0.4):
    if prob >= buy_th:
        return "BUY"
    elif prob <= sell_th:
        return "SELL"
    else:
        return "HOLD"
