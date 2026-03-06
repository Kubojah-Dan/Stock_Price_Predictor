import numpy as np
import pandas as pd

def build_sequences_for_range(df_scaled, feature_cols, target_col, seq_len, start_idx, end_idx=None):
    arr = df_scaled[feature_cols].values
    targets = df_scaled[target_col].values
    X, y = [], []
    N = len(df_scaled)
    if end_idx is None:
        end_idx = N - 1
    for i in range(seq_len, N):
        if i < start_idx:
            continue
        if i > end_idx:
            break
        # X[i] uses data from [i-seq_len : i]
        X.append(arr[i-seq_len:i])
        y.append(targets[i-1])
    return np.array(X), np.array(y)

# Create dummy data
data = pd.DataFrame({
    "Close": [100, 110, 120, 130, 140],
    "feat": [1, 2, 3, 4, 5]
})
data["target_close_t1"] = data["Close"].shift(-1)

SEQ_LEN = 2
X, y = build_sequences_for_range(data, ["feat"], "target_close_t1", seq_len=SEQ_LEN, start_idx=SEQ_LEN)

print("Data:")
print(data)
print("\nX shape:", X.shape)
print("X[0]:\n", X[0])
print("y[0]:", y[0])

unscaled_close = data["Close"].values
unscaled_target = data["target_close_t1"].values
close_prev = unscaled_close[SEQ_LEN-1:-1]
target_actual = unscaled_target[SEQ_LEN-1:-1]

print("\nDirectional Alignment:")
print("Target Actuals:", target_actual)
print("Previous Closes:", close_prev)
print("Consistent?", all(target_actual == y))
