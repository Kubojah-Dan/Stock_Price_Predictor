import optuna
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import math
import warnings
warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

def tune_xgb(X_train, y_train, X_val, y_val, n_trials=20):
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 700),
            "max_depth": trial.suggest_int("max_depth", 2, 5),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.08),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 4, 20),
            "gamma": trial.suggest_float("gamma", 0.0, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 2.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 10.0),
        }
        model = XGBRegressor(**params, objective="reg:squarederror", random_state=42, verbosity=0)
        model.fit(X_train, y_train)
        pred = model.predict(X_val)
        rmse = math.sqrt(mean_squared_error(y_val, pred))
        return rmse

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params

def tune_lstm(X_train, y_train, X_val, y_val, n_trials=12):
    """
    Simple Optuna tuner for LSTM units/lr/batch; keep it lightweight.
    Returns dict with best 'units', 'lr', and 'batch'.
    """
    import tensorflow as tf
    def objective(trial):
        units = trial.suggest_categorical("units", [32, 48, 64, 96])
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        batch = trial.suggest_categorical("batch", [16, 32, 64])
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=X_train.shape[1:]),
            tf.keras.layers.LSTM(units),
            tf.keras.layers.Dense(max(16, units//2), activation="relu"),
            tf.keras.layers.Dense(1, activation="linear"),
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss="mse")
        es = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)
        # small number of epochs for trial speed
        history = model.fit(
            X_train,
            y_train,
            epochs=6,
            batch_size=batch,
            verbose=0,
            validation_data=(X_val, y_val),
            callbacks=[es],
            shuffle=False
        )
        val_loss = min(history.history.get("val_loss", [np.inf]))
        tf.keras.backend.clear_session()
        return float(val_loss)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = study.best_params
    return {
        "units": int(best.get("units", 64)),
        "lr": float(best.get("lr", 1e-3)),
        "batch": int(best.get("batch", 32)),
    }
