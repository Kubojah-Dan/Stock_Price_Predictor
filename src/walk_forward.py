import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

def walk_forward_validation(
    df,
    features,
    target_col,
    model_fn,
    train_window=1000,
    test_window=200
):
    accuracies = []
    for start in range(0, len(df) - train_window - test_window, test_window):
        train = df.iloc[start:start + train_window]
        test  = df.iloc[start + train_window:start + train_window + test_window]
        X_train = train[features].values
        y_train = train[target_col].values
        X_test  = test[features].values
        y_test  = test[target_col].values
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test  = scaler.transform(X_test)
        model = model_fn()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        accuracies.append(acc)
    return float(np.mean(accuracies)), accuracies

def walk_forward_train(df, train_fn, window=1000, step=50):
    models = []
    for start in range(0, len(df) - window, step):
        end = start + window
        df_slice = df.iloc[start:end]
        model = train_fn(df_slice)
        models.append(model)
    return models

