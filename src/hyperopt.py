import optuna
from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import mean_squared_error, f1_score, roc_auc_score
import numpy as np
import math
import warnings
warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

def tune_xgb(X_train, y_train, X_val, y_val, n_trials=50):
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 10.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 100.0, log=True),
        }
        model = XGBRegressor(**params, objective="reg:squarederror", random_state=42, verbosity=0, early_stopping_rounds=15)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        pred = model.predict(X_val)
        rmse = math.sqrt(mean_squared_error(y_val, pred))
        return rmse

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params

def tune_lstm(X_train, y_train, X_val, y_val, n_trials=30):
    """
    Enhanced Optuna tuner for LSTM with dropout and stacked layers.
    """
    import tensorflow as tf
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
    from tensorflow.keras.models import Sequential
    
    def objective(trial):
        units = trial.suggest_categorical("units", [32, 64, 128])
        lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
        batch = trial.suggest_categorical("batch", [16, 32, 64])
        dropout_rate = trial.suggest_float("dropout", 0.1, 0.4)
        # suggest_bool may not be available in older optuna versions
        use_second_layer = trial.suggest_categorical("use_second_layer", [True, False])

        model = Sequential()
        model.add(Input(shape=X_train.shape[1:]))
        
        if use_second_layer:
            model.add(LSTM(units, return_sequences=True))
            model.add(Dropout(dropout_rate))
            model.add(LSTM(units // 2, return_sequences=False))
        else:
            model.add(LSTM(units, return_sequences=False))
        
        model.add(Dropout(dropout_rate))
        model.add(Dense(max(16, units//4), activation="relu"))
        model.add(Dense(1, activation="linear"))
        
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss="mse")
        es = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
        
        history = model.fit(
            X_train,
            y_train,
            epochs=15, 
            batch_size=batch,
            verbose=0,
            validation_data=(X_val, y_val),
            callbacks=[es],
            shuffle=False
        )
        val_loss = min(history.history.get("val_loss", [np.inf]))
        tf.keras.backend.clear_session()
        return float(val_loss)

def tune_xgb_clf(X_train, y_train, X_val, y_val, n_trials=40):
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 800),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 0.8, 1.2), # Adjust for slight imbalance
        }
        model = XGBClassifier(**params, use_label_encoder=False, eval_metric="logloss", random_state=42, verbosity=0, early_stopping_rounds=15)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        pred = model.predict(X_val)
        return f1_score(y_val, pred, zero_division=0)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params

def tune_lstm_clf(X_train, y_train, X_val, y_val, n_trials=25):
    """
    Optuna tuner for LSTM Classification (Directional).
    """
    import tensorflow as tf
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
    from tensorflow.keras.models import Sequential
    
    def objective(trial):
        units = trial.suggest_categorical("units", [32, 64, 128])
        lr = trial.suggest_float("lr", 5e-5, 2e-3, log=True)
        batch = trial.suggest_categorical("batch", [16, 32, 64])
        dropout_rate = trial.suggest_float("dropout", 0.2, 0.5)
        use_second_layer = trial.suggest_categorical("use_second_layer", [True, False])

        model = Sequential()
        model.add(Input(shape=X_train.shape[1:]))
        if use_second_layer:
            model.add(LSTM(units, return_sequences=True))
            model.add(Dropout(dropout_rate))
            model.add(LSTM(units // 2, return_sequences=False))
        else:
            model.add(LSTM(units, return_sequences=False))
        
        model.add(Dropout(dropout_rate))
        model.add(Dense(1, activation="sigmoid")) # Probability of Up
        
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss="binary_crossentropy", metrics=["AUC"])
        es = tf.keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=5, restore_best_weights=True)
        
        history = model.fit(
            X_train, y_train,
            epochs=20,
            batch_size=batch,
            verbose=0,
            validation_data=(X_val, y_val),
            callbacks=[es],
            shuffle=False
        )
        # Maximize AUC or F1 is tricky in Keras during fit, so we use max val_auc from history
        val_auc = max(history.history.get("val_auc", [0.0]))
        tf.keras.backend.clear_session()
        return float(val_auc)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params
