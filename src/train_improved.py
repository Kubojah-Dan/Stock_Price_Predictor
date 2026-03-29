import os
import sys
import warnings
import joblib
import numpy as np
import pandas as pd
import optuna
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    roc_auc_score, matthews_corrcoef, log_loss, precision_score, recall_score
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import RobustScaler
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Force UTF-8 output on Windows
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

try:
    from src.preprocessing import load_and_save_yahoo
    from src.features import add_features, create_target
    from src.visualize import (
        plot_feature_importance, plot_roc_curve, plot_confusion_matrix,
        plot_calibration_curve, plot_walk_forward_accuracy,
        plot_equity_curve, plot_drawdown
    )
    from src.backtest import backtest_with_costs
    from src.metrics import sharpe, sortino
except ImportError:
    from preprocessing import load_and_save_yahoo
    from features import add_features, create_target
    from visualize import (
        plot_feature_importance, plot_roc_curve, plot_confusion_matrix,
        plot_calibration_curve, plot_walk_forward_accuracy,
        plot_equity_curve, plot_drawdown
    )
    from backtest import backtest_with_costs
    from metrics import sharpe, sortino

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TICKERS         = ["AAPL", "MSFT", "NVDA", "GOOGL"]
START_DATE      = "2015-01-01"
CLF_HORIZON     = 21         # 21-day horizon has more persistent signal
MAX_FEATURES    = 25
N_OPTUNA_TRIALS = 50
N_CV_SPLITS     = 5
MODEL_DIR       = "models"
OUTPUT_DIR      = "outputs"
# ---------------------------------------------------------------------------

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

DROP_COLS = {"target_return", "target", "Date", "Ticker",
             "Open", "High", "Low", "Close", "Adj Close", "Volume"}


def _clean(df):
    num = df.select_dtypes(include=[np.number]).columns
    df[num] = df[num].replace([np.inf, -np.inf], np.nan)
    return df.dropna()


def _feature_cols(df):
    return [c for c in df.columns if c not in DROP_COLS]


def _sample_weights(y):
    """Balanced sample weights so each class contributes equally."""
    return compute_sample_weight("balanced", y)


def _best_threshold(y_true, y_prob):
    """Pick threshold that maximises balanced accuracy on the given set."""
    best_t, best_ba = 0.5, 0.0
    for t in np.linspace(0.3, 0.7, 41):
        ba = balanced_accuracy_score(y_true, (y_prob >= t).astype(int))
        if ba > best_ba:
            best_ba, best_t = ba, t
    return best_t


def _optuna_tune(X_tr, y_tr, n_trials=N_OPTUNA_TRIALS):
    tscv = TimeSeriesSplit(n_splits=N_CV_SPLITS)
    sw = _sample_weights(y_tr)

    def objective(trial):
        params = dict(
            n_estimators      = trial.suggest_int("n_estimators", 200, 800),
            max_depth         = trial.suggest_int("max_depth", 3, 7),
            learning_rate     = trial.suggest_float("learning_rate", 0.005, 0.15, log=True),
            subsample         = trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree  = trial.suggest_float("colsample_bytree", 0.5, 1.0),
            min_child_weight  = trial.suggest_int("min_child_weight", 1, 10),
            gamma             = trial.suggest_float("gamma", 0.0, 1.0),
            reg_alpha         = trial.suggest_float("reg_alpha", 0.0, 2.0),
            reg_lambda        = trial.suggest_float("reg_lambda", 0.5, 3.0),
            eval_metric       = "auc",
            use_label_encoder = False,
            verbosity         = 0,
            random_state      = 42,
            n_jobs            = -1,
        )
        aucs = []
        for tr_idx, val_idx in tscv.split(X_tr):
            clf = XGBClassifier(**params)
            clf.fit(X_tr[tr_idx], y_tr[tr_idx],
                    sample_weight=sw[tr_idx],
                    eval_set=[(X_tr[val_idx], y_tr[val_idx])],
                    verbose=False)
            prob = clf.predict_proba(X_tr[val_idx])[:, 1]
            if len(np.unique(y_tr[val_idx])) > 1:
                aucs.append(roc_auc_score(y_tr[val_idx], prob))
        return float(np.mean(aucs)) if aucs else 0.5

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def _walk_forward_eval(df, feature_cols, train_frac=0.6, n_folds=5):
    """Expanding-window walk-forward returning OOS predictions and per-fold AUCs."""
    n = len(df)
    min_train = int(n * train_frac)
    fold_size = (n - min_train) // n_folds

    all_y, all_prob, fold_aucs = [], [], []

    for i in range(n_folds):
        train_end = min_train + i * fold_size
        test_end  = train_end + fold_size
        if test_end > n:
            break

        df_tr = df.iloc[:train_end]
        df_te = df.iloc[train_end:test_end]

        X_tr = df_tr[feature_cols].values
        y_tr = df_tr["target"].values
        X_te = df_te[feature_cols].values
        y_te = df_te["target"].values

        sc = RobustScaler()
        X_tr_s = sc.fit_transform(X_tr)
        X_te_s = sc.transform(X_te)

        clf = XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="auc", use_label_encoder=False,
            verbosity=0, random_state=42, n_jobs=-1
        )
        clf.fit(X_tr_s, y_tr, sample_weight=_sample_weights(y_tr))
        prob = clf.predict_proba(X_te_s)[:, 1]

        all_y.extend(y_te)
        all_prob.extend(prob)
        if len(np.unique(y_te)) > 1:
            fold_aucs.append(roc_auc_score(y_te, prob))

    return np.array(all_y), np.array(all_prob), fold_aucs


def train_ticker(ticker):
    print(f"\n{'='*50}")
    print(f"  Training: {ticker}")
    print(f"{'='*50}")

    # 1. Load & engineer features
    df_raw = load_and_save_yahoo(ticker, START_DATE)
    df = add_features(df_raw, ticker=ticker, start_date=START_DATE)
    df = create_target(df, horizon=CLF_HORIZON)
    df = _clean(df)

    if "Date" in df.columns:
        df = df.set_index("Date")

    feature_cols = _feature_cols(df)
    print(f"  Raw features : {len(feature_cols)}")
    print(f"  Total rows   : {len(df)}")
    print(f"  Class balance: {df['target'].mean():.2%} Up")

    if len(df) < 300:
        print(f"  Skipping {ticker}: insufficient data")
        return None

    # 2. Train / holdout split (80/20, time-ordered)
    split = int(len(df) * 0.80)
    df_train, df_test = df.iloc[:split], df.iloc[split:]

    X_tr_raw = df_train[feature_cols].values
    y_tr     = df_train["target"].values
    X_te_raw = df_test[feature_cols].values
    y_te     = df_test["target"].values

    # 3. Scale
    scaler = RobustScaler()
    X_tr = scaler.fit_transform(X_tr_raw)
    X_te = scaler.transform(X_te_raw)

    # 4. Feature selection
    print("  Selecting features...")
    selector = SelectFromModel(
        RandomForestClassifier(n_estimators=150, max_depth=6,
                               class_weight="balanced", random_state=42, n_jobs=-1),
        max_features=MAX_FEATURES, threshold="median"
    )
    X_tr_sel = selector.fit_transform(X_tr, y_tr)
    X_te_sel = selector.transform(X_te)
    sel_features = [feature_cols[i] for i in selector.get_support(indices=True)]
    print(f"  Selected {len(sel_features)} features: {sel_features}")

    # 5. Optuna tuning (optimises AUC via TimeSeriesSplit + balanced sample weights)
    print(f"  Running Optuna ({N_OPTUNA_TRIALS} trials)...")
    best_params = _optuna_tune(X_tr_sel, y_tr)
    best_params.update({"eval_metric": "auc", "use_label_encoder": False,
                        "verbosity": 0, "random_state": 42, "n_jobs": -1})
    print(f"  Best params: {best_params}")

    # 6. Train final XGBoost with balanced weights
    xgb = XGBClassifier(**best_params)
    xgb.fit(X_tr_sel, y_tr, sample_weight=_sample_weights(y_tr))

    # 7. Calibrate probabilities (isotonic on a held-out validation slice)
    val_split = int(len(X_tr_sel) * 0.8)
    X_cal_tr, X_cal_val = X_tr_sel[:val_split], X_tr_sel[val_split:]
    y_cal_tr, y_cal_val = y_tr[:val_split], y_tr[val_split:]

    xgb_cal = XGBClassifier(**best_params)
    xgb_cal.fit(X_cal_tr, y_cal_tr, sample_weight=_sample_weights(y_cal_tr))
    calibrated = CalibratedClassifierCV(xgb_cal, method="isotonic", cv="prefit")
    calibrated.fit(X_cal_val, y_cal_val)

    # 8. Threshold tuning on validation set (avoid leaking test set)
    val_prob = calibrated.predict_proba(X_cal_val)[:, 1]
    threshold = _best_threshold(y_cal_val, val_prob)
    print(f"  Optimal threshold (val): {threshold:.2f}")

    # 9. Evaluate on holdout test set
    y_prob = calibrated.predict_proba(X_te_sel)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "ticker"      : ticker,
        "threshold"   : threshold,
        "accuracy"    : accuracy_score(y_te, y_pred),
        "bal_accuracy": balanced_accuracy_score(y_te, y_pred),
        "precision"   : precision_score(y_te, y_pred, zero_division=0),
        "recall"      : recall_score(y_te, y_pred, zero_division=0),
        "f1"          : f1_score(y_te, y_pred, zero_division=0),
        "auc"         : roc_auc_score(y_te, y_prob) if len(np.unique(y_te)) > 1 else float("nan"),
        "mcc"         : matthews_corrcoef(y_te, y_pred),
        "log_loss"    : log_loss(y_te, y_prob),
    }

    print("\n  -- Holdout Metrics --")
    for k, v in metrics.items():
        if k not in ("ticker",):
            print(f"  {k:15s}: {v:.4f}")

    # 10. Walk-forward OOS evaluation
    print("\n  Running walk-forward evaluation...")
    wf_y, wf_prob, fold_aucs = _walk_forward_eval(df, sel_features)
    if len(wf_y) > 0 and len(np.unique(wf_y)) > 1:
        wf_auc = roc_auc_score(wf_y, wf_prob)
        wf_acc = accuracy_score(wf_y, (wf_prob >= threshold).astype(int))
        wf_ba  = balanced_accuracy_score(wf_y, (wf_prob >= threshold).astype(int))
        print(f"  Walk-forward AUC: {wf_auc:.4f} | Acc: {wf_acc:.4f} | BalAcc: {wf_ba:.4f}")
        print(f"  Per-fold AUCs   : {[f'{a:.3f}' for a in fold_aucs]}")
        metrics["wf_auc"] = wf_auc
        metrics["wf_acc"] = wf_acc
        metrics["wf_bal_acc"] = wf_ba

    # 11. Save artifacts
    joblib.dump(calibrated,   os.path.join(MODEL_DIR, f"model_{ticker}.pkl"))
    joblib.dump(scaler,       os.path.join(MODEL_DIR, f"scaler_{ticker}.pkl"))
    joblib.dump(selector,     os.path.join(MODEL_DIR, f"selector_{ticker}.pkl"))
    joblib.dump(sel_features, os.path.join(MODEL_DIR, f"features_{ticker}.pkl"))
    joblib.dump(threshold,    os.path.join(MODEL_DIR, f"threshold_{ticker}.pkl"))
    pd.DataFrame([metrics]).to_csv(
        os.path.join(OUTPUT_DIR, f"{ticker}_metrics.csv"), index=False
    )
    print(f"  Saved artifacts -> {MODEL_DIR}/")

    # 12. Visualisations
    _make_visuals(ticker, df_test, sel_features, y_te, y_pred, y_prob,
                  xgb, sel_features, fold_aucs)

    return metrics


def _make_visuals(ticker, df_test, sel_features, y_te, y_pred, y_prob,
                  xgb_model, feature_names, fold_aucs):
    prefix = os.path.join(OUTPUT_DIR, ticker)

    try:
        plot_feature_importance(
            feature_names, xgb_model.feature_importances_,
            save_path=f"{prefix}_feature_importance.html",
            title=f"{ticker} Feature Importance"
        )
    except Exception as e:
        print(f"  Feature importance plot failed: {e}")

    try:
        plot_roc_curve(y_te, y_prob, save_path=f"{prefix}_roc.html")
    except Exception as e:
        print(f"  ROC plot failed: {e}")

    try:
        plot_confusion_matrix(y_te, y_pred, save_path=f"{prefix}_confusion.html")
    except Exception as e:
        print(f"  Confusion matrix plot failed: {e}")

    try:
        plot_calibration_curve(y_te, y_prob, save_path=f"{prefix}_calibration.html")
    except Exception as e:
        print(f"  Calibration plot failed: {e}")

    if fold_aucs:
        try:
            plot_walk_forward_accuracy(
                fold_aucs, ticker, save_path=f"{prefix}_walk_forward.html"
            )
        except Exception as e:
            print(f"  Walk-forward plot failed: {e}")

    try:
        signals = (y_prob >= 0.55).astype(int)
        bt = backtest_with_costs(df_test.reset_index(), signals)
        bt = bt.set_index("Date") if "Date" in bt.columns else bt
        sh = sharpe(bt["strategy"])
        so = sortino(bt["strategy"])
        ret = bt["cum_strategy"].iloc[-1] - 1
        print(f"  Backtest -> Sharpe: {sh:.2f}  Sortino: {so:.2f}  Return: {ret:.2%}")
        plot_equity_curve(bt, save_path=f"{prefix}_equity.html")
        plot_drawdown(bt, save_path=f"{prefix}_drawdown.html")
    except Exception as e:
        print(f"  Equity/drawdown plot failed: {e}")


def run():
    all_metrics = []
    for ticker in TICKERS:
        try:
            m = train_ticker(ticker)
            if m:
                all_metrics.append(m)
        except Exception as e:
            print(f"ERROR training {ticker}: {e}")
            import traceback; traceback.print_exc()

    if all_metrics:
        summary = pd.DataFrame(all_metrics)
        summary.to_csv(os.path.join(OUTPUT_DIR, "summary_metrics.csv"), index=False)
        print("\n" + "="*50)
        print("SUMMARY")
        print("="*50)
        cols = ["ticker", "accuracy", "bal_accuracy", "f1", "auc", "mcc", "wf_auc"]
        cols = [c for c in cols if c in summary.columns]
        print(summary[cols].to_string(index=False))
        print(f"\nOutputs -> {OUTPUT_DIR}/")
        print(f"Models  -> {MODEL_DIR}/")


if __name__ == "__main__":
    run()
