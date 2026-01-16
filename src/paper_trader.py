import time

def paper_trade_loop(
    get_latest_data,
    model,
    scaler,
    features,
    capital=100000
):
    position = 0

    while True:
        df = get_latest_data()
        x = df[features].iloc[-1].values.reshape(1, -1)
        x = scaler.transform(x)

        prob = model.predict_proba(x)[0, 1]
        signal = generate_signal(prob)

        print(f"Prob={prob:.2f} Signal={signal}")

        time.sleep(60)  # 1-minute frequency
