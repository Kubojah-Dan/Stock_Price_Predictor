import os
import joblib
import numpy as np
import pandas as pd
import math
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef
)
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_selection import SelectFromModel, mutual_info_regression, mutual_info_classif, f_regression
from sklearn.linear_model import Ridge

from xgboost import XGBRegressor, XGBClassifier

from preprocessing import load_and_save_yahoo
from features import add_features, create_target
from hyperopt import tune_xgb, tune_lstm
from backtest import backtest_with_costs
from metrics import sharpe, sortino
from ablation import feature_ablation_test
from visualize import plot_equity_curve, plot_confidence_vs_return, plot_feature_importance

# ----------------- CONFIG -----------------
CLIP_RET = 0.10            # clip returns to +/- 10%
OUTPUT_DIR = "outputs"
MODEL_DIR = "models"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

TICKERS = ["AAPL", "MSFT", "NVDA", "GOOGL"]
START_DATE = "2015-01-01"
CLF_HORIZON = int(os.getenv("CLF_HORIZON", "1"))
REG_HORIZON = int(os.getenv("REG_HORIZON", "5"))

SEQ_LEN = 30              # LSTM lookback (days)
FUTURE_DAYS = 30          # naive multi-step horizon (approx)
CALIBRATION_FRAC = 0.2    # fraction of training to use for calibration
MAX_FEATURES = 25         # maximum number of features to keep after selection
MIN_FEATURES = 10         # enforce a minimum retained feature count
HIGH_CORR_THRESHOLD = 0.95
TUNE_VAL_FRAC = 0.2       # validation fraction for Optuna tuning (train-only)
LSTM_VAL_FRAC = 0.1       # temporal validation fraction for LSTM fit

_tickers_env = os.getenv("TRAIN_TICKERS", "").strip()
if _tickers_env:
    TICKERS = [t.strip().upper() for t in _tickers_env.split(",") if t.strip()]
SKIP_LSTM = os.getenv("SKIP_LSTM", "1") == "1"
SKIP_SEQ2SEQ = os.getenv("SKIP_SEQ2SEQ", "1") == "1"
WF_REG_TRAIN_WINDOW = int(os.getenv("WF_REG_TRAIN_WINDOW", "900"))
WF_REG_TEST_WINDOW = int(os.getenv("WF_REG_TEST_WINDOW", "160"))
WF_REG_STEP = int(os.getenv("WF_REG_STEP", "160"))
WF_GATE_MARGIN = float(os.getenv("WF_GATE_MARGIN", "0.0005"))
# ------------------------------------------

def build_lstm_model(n_features, seq_len=SEQ_LEN, units=64, lr=1e-3):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
    from tensorflow.keras.optimizers import Adam

    model = Sequential()
    model.add(Input(shape=(seq_len, n_features)))
    model.add(LSTM(units, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(max(16, units//2), activation="relu"))
    model.add(Dense(1, activation="linear"))
    model.compile(optimizer=Adam(learning_rate=lr), loss="mse")
    return model

def metrics_regression(y_true, y_pred):
    """Return r2, mae, rmse"""
    r2 = r2_score(y_true, y_pred) if len(y_true)>0 else float("nan")
    mae = mean_absolute_error(y_true, y_pred) if len(y_true)>0 else float("nan")
    rmse = math.sqrt(mean_squared_error(y_true, y_pred)) if len(y_true)>0 else float("nan")
    return r2, mae, rmse

def save_metrics_table(df_metrics, ticker):
    Path(OUTPUT_DIR).mkdir(exist_ok=True, parents=True)
    df_metrics.to_csv(os.path.join(OUTPUT_DIR, f"{ticker}_metrics.csv"), index=False)

def time_train_val_split(X, y, val_frac=0.2, min_val=64):
    """
    Time-ordered split: first block for fit, last block for validation.
    Keeps test set untouched for honest evaluation.
    """
    n = len(X)
    if n < 2:
        return X, y, X, y
    val_n = max(int(n * val_frac), min_val)
    val_n = min(max(1, val_n), n - 1)
    split = n - val_n
    return X[:split], y[:split], X[split:], y[split:]

def tune_threshold(y_true, proba):
    """
    Choose decision threshold on validation data.
    Optimizes a blend of balanced accuracy + F1 while avoiding degenerate predictions.
    """
    if len(y_true) == 0:
        return 0.5, 0.0
    best_t = 0.5
    best_score = -1.0
    for t in np.arange(0.25, 0.76, 0.01):
        pred = (proba >= t).astype(int)
        pos_rate = float(np.mean(pred))
        # avoid degenerate rules that predict one class almost always
        if pos_rate < 0.20 or pos_rate > 0.80:
            continue
        bal_acc = balanced_accuracy_score(y_true, pred)
        f1 = f1_score(y_true, pred, zero_division=0)
        mcc = matthews_corrcoef(y_true, pred)
        # penalize negative MCC but keep focus on class balance + hit rate
        score = 0.55 * bal_acc + 0.35 * f1 + 0.10 * max(0.0, mcc)
        if score > best_score:
            best_score = score
            best_t = float(t)
    if best_score < 0:
        pred = (proba >= 0.5).astype(int)
        best_t = 0.5
        best_score = float(
            0.55 * balanced_accuracy_score(y_true, pred) +
            0.35 * f1_score(y_true, pred, zero_division=0) +
            0.10 * max(0.0, matthews_corrcoef(y_true, pred))
        )
    return best_t, float(best_score)


def _normalize_scores(values):
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr
    vmin = np.nanmin(arr)
    vmax = np.nanmax(arr)
    if (not np.isfinite(vmin)) or (not np.isfinite(vmax)) or (vmax - vmin < 1e-12):
        return np.zeros_like(arr)
    return (arr - vmin) / (vmax - vmin)


def statistical_feature_filter(df_train, feature_cols, y_reg, y_cls, max_features=MAX_FEATURES, min_features=MIN_FEATURES):
    """
    Multi-stage filter:
    1) Remove near-constant features.
    2) Remove highly correlated features.
    3) Rank by univariate signal (MI + correlation + F-test).
    """
    X = df_train[feature_cols].copy()
    X = X.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)

    # 1) Variance filter
    variances = X.var(ddof=0)
    keep_var = variances[variances > 1e-8].index.tolist()
    if not keep_var:
        keep_var = list(X.columns)
    X = X[keep_var]

    # 2) Correlation filter
    if X.shape[1] > 1:
        corr = X.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        to_drop = [col for col in upper.columns if (upper[col] > HIGH_CORR_THRESHOLD).any()]
        if to_drop:
            X = X.drop(columns=to_drop)

    if X.shape[1] == 0:
        fallback = feature_cols[:min(max_features, len(feature_cols))]
        return fallback

    # 3) Score each feature with multiple filter metrics
    corr_scores = []
    for col in X.columns:
        v = X[col].values
        if np.std(v) < 1e-12:
            corr_scores.append(0.0)
            continue
        c = np.corrcoef(v, y_reg)[0, 1]
        corr_scores.append(abs(c) if np.isfinite(c) else 0.0)

    try:
        _, pvals = f_regression(X.values, y_reg)
        p_scores = 1.0 - np.clip(pvals, 0.0, 1.0)
    except Exception:
        p_scores = np.zeros(X.shape[1], dtype=float)

    try:
        mi_reg = mutual_info_regression(X.values, y_reg, random_state=42)
    except Exception:
        mi_reg = np.zeros(X.shape[1], dtype=float)

    try:
        mi_cls = mutual_info_classif(X.values, y_cls, random_state=42)
    except Exception:
        mi_cls = np.zeros(X.shape[1], dtype=float)

    score = (
        0.40 * _normalize_scores(mi_reg) +
        0.30 * _normalize_scores(mi_cls) +
        0.20 * _normalize_scores(corr_scores) +
        0.10 * _normalize_scores(p_scores)
    )
    ranking = pd.Series(score, index=X.columns).sort_values(ascending=False)

    k = min(max_features, len(ranking))
    k = max(min_features, k)
    k = min(k, len(ranking))
    selected = ranking.head(k).index.tolist()
    return selected


def tune_regression_shrinkage(y_true, pred_raw, baseline_value):
    """
    Blend model predictions with a baseline constant to reduce overfit variance.
    Returns best blend weight for model predictions.
    """
    if len(y_true) == 0:
        return 1.0, float("nan")
    best_w = 1.0
    best_r2 = -np.inf
    for w in np.arange(0.0, 1.01, 0.05):
        pred = w * pred_raw + (1.0 - w) * baseline_value
        score = r2_score(y_true, pred)
        if score > best_r2:
            best_r2 = score
            best_w = float(w)
    return best_w, float(best_r2)


def walk_forward_regression_score(X, y, model_fn, train_window=WF_REG_TRAIN_WINDOW, test_window=WF_REG_TEST_WINDOW, step=WF_REG_STEP):
    scores = []
    n = len(X)
    if n < (train_window + test_window + 1):
        return float("nan"), scores
    for start in range(0, n - train_window - test_window + 1, step):
        tr_slice = slice(start, start + train_window)
        te_slice = slice(start + train_window, start + train_window + test_window)
        m = model_fn()
        m.fit(X[tr_slice], y[tr_slice])
        pred = m.predict(X[te_slice])
        try:
            score = r2_score(y[te_slice], pred)
            if np.isfinite(score):
                scores.append(float(score))
        except Exception:
            continue
    if not scores:
        return float("nan"), scores
    return float(np.mean(scores)), scores


def horizon_return_to_daily(ret_h, horizon):
    horizon = max(1, int(horizon))
    ret_h = float(ret_h)
    ret_h = max(ret_h, -0.99)
    return (1.0 + ret_h) ** (1.0 / horizon) - 1.0

def build_sequences_for_range(df_all, feature_cols, seq_len, start_idx, end_idx=None):
    """
    Build sequences from df_all for indices i in [start_idx, end_idx] (inclusive).
    Each sequence X corresponds to features at indices [i-seq_len ... i-1] and target at i.
    df_all must contain a numeric target column named 'target_z'.
    """
    arr = df_all[feature_cols].values
    targets = df_all["target_z"].values
    X, y = [], []
    N = len(df_all)
    if end_idx is None:
        end_idx = N - 1
    for i in range(seq_len, N):
        if i < start_idx:
            continue
        if i > end_idx:
            break
        X.append(arr[i-seq_len:i])
        y.append(targets[i])
    if len(X) == 0:
        return np.empty((0, seq_len, len(feature_cols))), np.empty((0,))
    return np.array(X), np.array(y)

# ----------------- Main loop -----------------
for T in TICKERS:
    print("Processing", T)
    lstm_model = None
    # 1) Load and feature-engineer
    df_raw = load_and_save_yahoo(T, START_DATE)  # saves CSV under data/
    df = add_features(df_raw, ticker=T, start_date=START_DATE)

    # remove weekends (ensure only business days)
    if isinstance(df.index, pd.DatetimeIndex):
        df = df[df.index.dayofweek < 5]

    # classification target stays short horizon; regression can use longer horizon
    df = create_target(df, horizon=CLF_HORIZON)
    df["target_price_reg"] = df["Close"].shift(-REG_HORIZON)
    df["target_return_reg"] = (df["target_price_reg"] - df["Close"]) / df["Close"]

    # basic cleaning: drop rows with NA in essential columns
    df = df.dropna(subset=["target_return", "target", "target_return_reg"]).copy()

    # Choose numeric feature columns excluding leakage columns
    exclude = {"target_price", "target_return", "target", "target_price_reg", "target_return_reg", "ret_1d", "ret_5d"}
    feature_cols = [c for c in df.columns if c not in exclude and np.issubdtype(df[c].dtype, np.number)]
    print(f"Features ({len(feature_cols)}): {feature_cols[:12]}{'...' if len(feature_cols) > 12 else ''}")

    # Train/test split (time ordered)
    split_idx = int(len(df) * 0.8)
    df_train = df.iloc[:split_idx].copy()
    df_test  = df.iloc[split_idx:].copy()

    # ---- Clip returns to reduce extreme outliers ----
    df_train["target_return_clipped"] = df_train["target_return_reg"].clip(-CLIP_RET, CLIP_RET)
    df_test["target_return_clipped"]  = df_test["target_return_reg"].clip(-CLIP_RET, CLIP_RET)

    # ---- Standardize (z-score) train target for LSTM ----
    tr_mean = df_train["target_return_clipped"].mean()
    tr_std  = df_train["target_return_clipped"].std() + 1e-9

    df_train["target_z"] = (df_train["target_return_clipped"] - tr_mean) / tr_std
    df_test["target_z"]  = (df_test["target_return_clipped"]  - tr_mean) / tr_std

    # For regressor we use clipped returns (raw clipped)
    y_train_xgb_full = df_train["target_return_clipped"].values
    y_test_xgb_full  = df_test["target_return_clipped"].values

    # For direction classifier (binary)
    y_train_dir = df_train["target"].values
    y_test_dir  = df_test["target"].values
    majority_baseline_acc = max(np.mean(y_test_dir), 1 - np.mean(y_test_dir))
    print(f"Direction majority baseline Acc={majority_baseline_acc:.4f}")

    # ----------------- Feature selection (automatic) -----------------
    filtered = statistical_feature_filter(
        df_train=df_train,
        feature_cols=feature_cols,
        y_reg=y_train_xgb_full,
        y_cls=y_train_dir,
        max_features=max(MAX_FEATURES * 2, MIN_FEATURES),
        min_features=MIN_FEATURES
    )
    print(f"After statistical filter ({len(filtered)}): {filtered}")

    # Use a quick XGB regressor on filtered features for model-based ranking
    scaler_prefit = StandardScaler()
    X_train_prefit = scaler_prefit.fit_transform(df_train[filtered])
    temp = XGBRegressor(
        n_estimators=220,
        learning_rate=0.04,
        max_depth=3,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.2,
        reg_lambda=2.0,
        objective="reg:squarederror",
        random_state=42
    )
    temp.fit(X_train_prefit, y_train_xgb_full)
    importances = pd.Series(temp.feature_importances_, index=filtered).sort_values(ascending=False)
    imp_sum = float(importances.sum())
    if imp_sum <= 1e-12:
        importances_cum = pd.Series(np.linspace(0, 1, len(importances)), index=importances.index)
    else:
        importances_cum = importances.cumsum() / imp_sum
    # select features until cumulative importance >= 0.95 or up to MAX_FEATURES
    chosen = importances_cum[importances_cum <= 0.95].index.tolist()
    min_keep = min(MIN_FEATURES, len(importances))
    if len(chosen) < min_keep:
        chosen = importances.head(min_keep).index.tolist()
    chosen = chosen[:min(MAX_FEATURES, len(importances))]
    print(f"Selected top features ({len(chosen)}): {chosen}")

    # rebuild scaled datasets using chosen features
    scaler_xgb = StandardScaler()
    X_train_xgb = scaler_xgb.fit_transform(df_train[chosen])
    X_test_xgb  = scaler_xgb.transform(df_test[chosen])
    # keep feature_cols_updated for LSTM sequences
    feature_cols_updated = chosen

    # ----------------- XGBoost regressor (predict clipped return) -----------------
    X_xgb_fit, y_xgb_fit, X_xgb_val, y_xgb_val = time_train_val_split(
        X_train_xgb, y_train_xgb_full, val_frac=TUNE_VAL_FRAC, min_val=128
    )

    try:
        best_params = tune_xgb(X_xgb_fit, y_xgb_fit, X_xgb_val, y_xgb_val)
        print("Optuna found XGB params:", best_params)
        xgb_reg_candidate = XGBRegressor(
            **best_params,
            objective="reg:squarederror",
            random_state=42,
            verbosity=0
        )
    except Exception as e:
        print("Optuna tuning XGB failed/fallback:", e)
        xgb_reg_candidate = XGBRegressor(
            n_estimators=420,
            learning_rate=0.03,
            max_depth=3,
            min_child_weight=8,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.5,
            reg_lambda=4.0,
            objective="reg:squarederror",
            random_state=42,
            verbosity=0
        )

    safe_reg_candidate = Ridge(alpha=12.0)

    def _xgb_builder():
        params = xgb_reg_candidate.get_params()
        return XGBRegressor(**params)

    def _safe_builder():
        return Ridge(alpha=12.0)

    wf_xgb_r2, _ = walk_forward_regression_score(X_train_xgb, y_train_xgb_full, _xgb_builder)
    wf_safe_r2, _ = walk_forward_regression_score(X_train_xgb, y_train_xgb_full, _safe_builder)

    if (not np.isfinite(wf_xgb_r2)) or (np.isfinite(wf_safe_r2) and wf_xgb_r2 < wf_safe_r2 + WF_GATE_MARGIN):
        xgb_reg = safe_reg_candidate
        selected_reg_model = "ridge_safe"
    else:
        xgb_reg = xgb_reg_candidate
        selected_reg_model = "xgb_reg"

    print(f"WF gate -> selected {selected_reg_model} (wf_xgb_r2={wf_xgb_r2:.5f}, wf_safe_r2={wf_safe_r2:.5f})")

    # tune a shrinkage blend on validation to reduce overfit volatility
    xgb_reg.fit(X_xgb_fit, y_xgb_fit)
    pred_val_raw = xgb_reg.predict(X_xgb_val)
    baseline_val = float(np.mean(y_xgb_fit)) if len(y_xgb_fit) > 0 else 0.0
    reg_shrink_w, reg_val_r2 = tune_regression_shrinkage(y_xgb_val, pred_val_raw, baseline_val)
    print(f"Regression blend weight (model vs mean) = {reg_shrink_w:.2f}, val_r2={reg_val_r2:.4f}")

    # retrain on full train data for final test predictions
    xgb_reg.fit(X_train_xgb, y_train_xgb_full)
    pred_xgb_raw = xgb_reg.predict(X_test_xgb)
    baseline_full = float(np.mean(y_train_xgb_full)) if len(y_train_xgb_full) > 0 else 0.0
    pred_xgb_ret = reg_shrink_w * pred_xgb_raw + (1.0 - reg_shrink_w) * baseline_full

    r2_xgb, mae_xgb, rmse_xgb = metrics_regression(y_test_xgb_full, pred_xgb_ret)
    print(f"XGB reg R2={r2_xgb:.4f} MAE={mae_xgb:.6f} RMSE={rmse_xgb:.6f}")

    # ----------------- XGBoost classifier (direction) + calibration -----------------
    # Time-ordered split for calibration
    cal_size = max(1, int(len(X_train_xgb) * CALIBRATION_FRAC))
    train_end = len(X_train_xgb) - cal_size
    if train_end < 1:
        train_end = int(len(X_train_xgb) * 0.75)

    X_fit = X_train_xgb[:train_end]
    y_fit = y_train_dir[:train_end]
    X_cal = X_train_xgb[train_end:]
    y_cal = y_train_dir[train_end:]

    pos = max(1, int(np.sum(y_fit)))
    neg = max(1, int(len(y_fit) - np.sum(y_fit)))
    scale_pos_weight = neg / pos

    xgb_clf = XGBClassifier(
        n_estimators=420,
        learning_rate=0.03,
        max_depth=3,
        min_child_weight=6,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.3,
        reg_lambda=3.0,
        random_state=42,
        use_label_encoder=False,
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        verbosity=0
    )

    decision_threshold = 0.5
    calibrator = None
    invert_proba = False
    try:
        if len(X_cal) >= 20 and len(np.unique(y_fit)) > 1 and len(np.unique(y_cal)) > 1:
            xgb_clf.fit(X_fit, y_fit)
            calibrator = CalibratedClassifierCV(xgb_clf, method="sigmoid", cv="prefit")
            calibrator.fit(X_cal, y_cal)
            proba_cal = calibrator.predict_proba(X_cal)[:, 1]
            try:
                auc_cal = roc_auc_score(y_cal, proba_cal)
                if np.isfinite(auc_cal) and auc_cal < 0.5:
                    invert_proba = True
                    proba_cal = 1.0 - proba_cal
            except Exception:
                pass
            decision_threshold, _ = tune_threshold(y_cal, proba_cal)
            proba_test_cal = calibrator.predict_proba(X_test_xgb)[:, 1]
            if invert_proba:
                proba_test_cal = 1.0 - proba_test_cal
            clf_used = f"xgb_clf_calibrated@{decision_threshold:.2f}{'_inv' if invert_proba else ''}"
        else:
            xgb_clf.fit(X_train_xgb, y_train_dir)
            proba_test_cal = xgb_clf.predict_proba(X_test_xgb)[:, 1]
            clf_used = "xgb_clf_uncalibrated_small_cal"
    except Exception as e:
        print("Calibration failed:", e)
        xgb_clf.fit(X_train_xgb, y_train_dir)
        proba_test_cal = xgb_clf.predict_proba(X_test_xgb)[:, 1]
        clf_used = "xgb_clf_uncalibrated"

    proba_xgb_dir = proba_test_cal
    pred_xgb_dir = (proba_xgb_dir >= decision_threshold).astype(int)

    acc_xgb = accuracy_score(y_test_dir, pred_xgb_dir)
    bal_acc_xgb = balanced_accuracy_score(y_test_dir, pred_xgb_dir)
    prec_xgb = precision_score(y_test_dir, pred_xgb_dir, zero_division=0)
    rec_xgb = recall_score(y_test_dir, pred_xgb_dir, zero_division=0)
    f1_xgb = f1_score(y_test_dir, pred_xgb_dir, zero_division=0)
    mcc_xgb = matthews_corrcoef(y_test_dir, pred_xgb_dir)
    try:
        auc_xgb = roc_auc_score(y_test_dir, proba_xgb_dir)
    except Exception:
        auc_xgb = float("nan")
    print(
        f"XGB cls Acc={acc_xgb:.4f} BalAcc={bal_acc_xgb:.4f} "
        f"Prec={prec_xgb:.4f} Rec={rec_xgb:.4f} F1={f1_xgb:.4f} MCC={mcc_xgb:.4f} "
        f"AUC={auc_xgb:.4f} ({clf_used})"
    )

    # ----------------- LSTM (predict next-day z-scored return) -----------------
    # Fit scaler for LSTM features on TRAIN only (important)
    scaler_lstm = StandardScaler()
    df_train_lstm = df_train.copy()
    df_train_lstm[feature_cols_updated] = scaler_lstm.fit_transform(df_train_lstm[feature_cols_updated])

    df_all_lstm = df.copy()
    df_all_lstm["target_return_clipped"] = df_all_lstm["target_return"].clip(-CLIP_RET, CLIP_RET)
    df_all_lstm["target_z"] = (df_all_lstm["target_return_clipped"] - tr_mean) / tr_std
    df_all_lstm[feature_cols_updated] = scaler_lstm.transform(df_all_lstm[feature_cols_updated])

    # Build sequences:
    X_lstm_train, y_lstm_train = build_sequences_for_range(df_all_lstm, feature_cols_updated, SEQ_LEN, start_idx=SEQ_LEN, end_idx=split_idx-1)
    X_lstm_test,  y_lstm_test  = build_sequences_for_range(df_all_lstm, feature_cols_updated, SEQ_LEN, start_idx=split_idx, end_idx=len(df_all_lstm)-1)

    print("LSTM train samples:", X_lstm_train.shape[0], "test samples:", X_lstm_test.shape[0])

    if SKIP_LSTM:
        print("SKIP_LSTM=1 -> skipping LSTM branch.")
        pred_lstm_ret = np.array([])
        r2_lstm = mae_lstm = rmse_lstm = float("nan")
    elif X_lstm_train.shape[0] == 0 or X_lstm_test.shape[0] == 0:
        print("Not enough samples for LSTM training/testing. Skipping LSTM for this ticker.")
        pred_lstm_ret = np.array([])
        r2_lstm = mae_lstm = rmse_lstm = float("nan")
    else:
        X_lstm_fit, y_lstm_fit, X_lstm_val, y_lstm_val = time_train_val_split(
            X_lstm_train, y_lstm_train, val_frac=LSTM_VAL_FRAC, min_val=64
        )
        lstm_batch = 32
        # Optionally tune LSTM hyperparams with Optuna (fast search)
        try:
            lstm_params = tune_lstm(X_lstm_fit, y_lstm_fit, X_lstm_val, y_lstm_val, n_trials=12)
            print("LSTM tuning result:", lstm_params)
            lstm_units = lstm_params.get("units", 64)
            lstm_lr = lstm_params.get("lr", 1e-3)
            lstm_batch = int(lstm_params.get("batch", 32))
        except Exception as e:
            print("LSTM tuning failed, fallback:", e)
            lstm_units = 64
            lstm_lr = 1e-3

        from tensorflow.keras.callbacks import EarlyStopping
        lstm_model = build_lstm_model(n_features=len(feature_cols_updated), seq_len=SEQ_LEN, units=lstm_units, lr=lstm_lr)
        es = EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)
        lstm_model.fit(
            X_lstm_fit,
            y_lstm_fit,
            epochs=100,
            batch_size=lstm_batch,
            validation_data=(X_lstm_val, y_lstm_val),
            callbacks=[es],
            verbose=1,
            shuffle=False
        )

        # Predict z-scores then invert to clipped returns
        pred_lstm_z = lstm_model.predict(X_lstm_test).flatten()
        pred_lstm_ret = pred_lstm_z * tr_std + tr_mean

        # y_lstm_test are target_z; convert to returns
        y_lstm_test_ret = y_lstm_test * tr_std + tr_mean

        r2_lstm, mae_lstm, rmse_lstm = metrics_regression(y_lstm_test_ret, pred_lstm_ret)
        print(f"LSTM R2={r2_lstm:.4f} MAE={mae_lstm:.6f} RMSE={rmse_lstm:.6f}")

    # ----------------- Align & Ensemble -----------------
    if pred_lstm_ret.size > 0:
        n_common = min(len(pred_xgb_ret), len(pred_lstm_ret))
        xgb_reg_aligned = pred_xgb_ret[:n_common]
        lstm_aligned     = pred_lstm_ret[:n_common]
        y_common = df_test["target_return_clipped"].values[:n_common]
    else:
        n_common = 0
        xgb_reg_aligned = np.array([])
        lstm_aligned = np.array([])
        y_common = np.array([])

    # If LSTM does not beat naive mean baseline (R2 <= 0), do not let it hurt ensemble.
    use_lstm_in_ensemble = (pred_lstm_ret.size > 0) and (r2_lstm is not None) and (not math.isnan(r2_lstm)) and (r2_lstm > 0)
    if use_lstm_in_ensemble:
        # RMSE-based weighting is stable even when R2 is small.
        w_x = 1.0 / (rmse_xgb + 1e-9)
        w_l = 1.0 / (rmse_lstm + 1e-9)
        w_sum = w_x + w_l
        w_x /= w_sum
        w_l /= w_sum
    else:
        w_x = 1.0
        w_l = 0.0

    if n_common > 0 and w_l > 0:
        pred_ensemble_ret = w_x * xgb_reg_aligned + w_l * lstm_aligned
        r2_ens, mae_ens, rmse_ens = metrics_regression(y_common, pred_ensemble_ret)
    elif len(pred_xgb_ret) > 0:
        pred_ensemble_ret = pred_xgb_ret.copy()
        r2_ens, mae_ens, rmse_ens = metrics_regression(y_test_xgb_full, pred_ensemble_ret)
    else:
        r2_ens = mae_ens = rmse_ens = float("nan")

    # ----------------- Save metrics table -----------------
    metrics_table = pd.DataFrame([{
        "model":"xgb_reg",
        "r2": r2_xgb, "mae": mae_xgb, "rmse": rmse_xgb,
        "reg_horizon": REG_HORIZON,
        "reg_model": selected_reg_model,
        "wf_xgb_r2": wf_xgb_r2,
        "wf_safe_r2": wf_safe_r2,
        "acc": acc_xgb, "bal_acc": bal_acc_xgb, "mcc": mcc_xgb,
        "precision": prec_xgb, "recall": rec_xgb, "f1": f1_xgb, "auc": auc_xgb
    }, {
        "model":"lstm",
        "r2": r2_lstm, "mae": mae_lstm, "rmse": rmse_lstm, "reg_horizon": REG_HORIZON,
        "acc": None, "bal_acc": None, "mcc": None, "precision": None, "recall": None, "f1": None, "auc": None
    }, {
        "model":"ensemble",
        "r2": r2_ens, "mae": mae_ens, "rmse": rmse_ens, "reg_horizon": REG_HORIZON
    }])
    save_metrics_table(metrics_table, T)
    print(metrics_table)

    # ----------------- Backtest on TEST SET only (no in-sample leakage) -----------------
    signals_test = (proba_xgb_dir > decision_threshold).astype(int)
    # Provide df_test with Date if necessary
    bt_test = backtest_with_costs(df_test.copy().reset_index(drop=False), signals_test)

    try:
        sh = sharpe(bt_test["strategy"])
        so = sortino(bt_test["strategy"])
    except Exception:
        sh = so = float("nan")
    print(f"{T} TEST backtest Sharpe={sh:.2f} Sortino={so:.2f}")

    # ----------------- Next-day forecasts (ensemble) -----------------
    last_row = df.iloc[-1:]
    X_last = scaler_xgb.transform(last_row[feature_cols_updated])
    xgb_next_raw = xgb_reg.predict(X_last)[0] if hasattr(xgb_reg, "predict") else 0.0
    xgb_next_ret = reg_shrink_w * xgb_next_raw + (1.0 - reg_shrink_w) * baseline_full

    last_seq = df_all_lstm[feature_cols_updated].values[-SEQ_LEN:]
    last_seq = last_seq.reshape(1, SEQ_LEN, len(feature_cols_updated))
    try:
        lstm_next_z = lstm_model.predict(last_seq).flatten()[0] if lstm_model is not None else 0.0
        lstm_next_ret = lstm_next_z * tr_std + tr_mean
    except Exception:
        lstm_next_ret = 0.0

    reg_horizon_ret = w_x * xgb_next_ret + w_l * lstm_next_ret
    next_day_ensemble_ret = horizon_return_to_daily(reg_horizon_ret, REG_HORIZON)
    last_close = df["Close"].iloc[-1]
    next_day_price = last_close * (1 + next_day_ensemble_ret)

    print(f"{T} next-day ret (ensemble) = {next_day_ensemble_ret:.6f}, price ~ {next_day_price:.2f} (w_xgb={w_x:.2f}, w_lstm={w_l:.2f})")

    # ----------------- Save models/artifacts -----------------
    joblib.dump(xgb_reg, os.path.join(MODEL_DIR, f"xgb_reg_{T}.pkl"))
    joblib.dump(xgb_clf, os.path.join(MODEL_DIR, f"xgb_clf_{T}.pkl"))
    joblib.dump(scaler_xgb, os.path.join(MODEL_DIR, f"scaler_xgb_{T}.pkl"))
    joblib.dump(scaler_lstm, os.path.join(MODEL_DIR, f"scaler_lstm_{T}.pkl"))
    joblib.dump(feature_cols_updated, os.path.join(MODEL_DIR, f"features_ensemble_{T}.pkl"))
    meta = {
        "clf_horizon": int(CLF_HORIZON),
        "reg_horizon": int(REG_HORIZON),
        "reg_shrink_w": float(reg_shrink_w),
        "reg_baseline": float(baseline_full),
        "reg_model": selected_reg_model,
        "wf_xgb_r2": float(wf_xgb_r2) if np.isfinite(wf_xgb_r2) else None,
        "wf_safe_r2": float(wf_safe_r2) if np.isfinite(wf_safe_r2) else None,
    }
    joblib.dump(meta, os.path.join(MODEL_DIR, f"meta_{T}.pkl"))
    if lstm_model is not None:
        lstm_model.save(os.path.join(MODEL_DIR, f"lstm_{T}.keras"))
        
    print("\nRunning feature ablation...")
    ablation_df = feature_ablation_test(df_train, chosen)
    print(ablation_df.head(10))
    ablation_df.to_csv(f"outputs/{T}_ablation.csv", index=False)
    
    if SKIP_SEQ2SEQ:
        print("SKIP_SEQ2SEQ=1 -> skipping seq2seq training.")
    else:
        from seq2seq_train import train_seq2seq
        train_seq2seq(df, T, chosen)

    # ----------------- Visualizations (saved) -----------------
    try:
        plot_equity_curve(bt_test, save_path=os.path.join(OUTPUT_DIR, f"{T}_equity_test.html"))
        plot_confidence_vs_return(df_test, proba_xgb_dir, save_path=os.path.join(OUTPUT_DIR, f"{T}_conf_vs_return_test.html"))
        if hasattr(xgb_reg, "feature_importances_"):
            fi_values = xgb_reg.feature_importances_
        elif hasattr(xgb_reg, "coef_"):
            fi_values = np.abs(np.asarray(xgb_reg.coef_))
        else:
            fi_values = np.zeros(len(feature_cols_updated), dtype=float)
        plot_feature_importance(feature_cols_updated, fi_values, save_path=os.path.join(OUTPUT_DIR, f"{T}_fi.html"))
        print("Visualizations saved.")
    except Exception as e:
        print("Warning: visualization save failed:", e)

    print(f"{T} done. Artifacts saved to {MODEL_DIR} and {OUTPUT_DIR}.")




