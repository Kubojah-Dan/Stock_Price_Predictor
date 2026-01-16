import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

def feature_ablation_test(df, features, target):
    results = []

    for f in features:
        test_features = [x for x in features if x != f]

        X = df[test_features].values
        y = df[target].values

        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        model = XGBClassifier(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.05,
            eval_metric="logloss"
        )
        model.fit(X_train, y_train)

        acc = accuracy_score(y_test, model.predict(X_test) * 100)
        results.append((f, acc))

    return pd.DataFrame(results, columns=["Removed Feature", "Accuracy"])
