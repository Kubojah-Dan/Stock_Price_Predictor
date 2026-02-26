import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

def feature_ablation_test(df, feature_cols, target_col="target"):
    """
    Removes one feature at a time and measures accuracy drop.
    Higher drop = more important feature.
    """

    results = []

    # Base accuracy with all features
    X = df[feature_cols].values
    y = df[target_col].values

    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    base_model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        eval_metric="logloss"
    )
    base_model.fit(X_train, y_train)
    base_acc = accuracy_score(y_test, base_model.predict(X_test))

    print(f"Base accuracy: {base_acc:.4f}")

    for f in feature_cols:
        reduced_features = [x for x in feature_cols if x != f]

        X = df[reduced_features].values
        y = df[target_col].values

        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        model = XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            eval_metric="logloss"
        )
        model.fit(X_train, y_train)

        acc = accuracy_score(y_test, model.predict(X_test))
        drop = base_acc - acc

        results.append({
            "feature_removed": f,
            "accuracy": acc,
            "accuracy_drop": drop
        })

    df_res = pd.DataFrame(results).sort_values("accuracy_drop", ascending=False)
    return df_res
