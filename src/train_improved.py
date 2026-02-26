import os
import joblib
import numpy as np
import pandas as pd
import math
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    log_loss
)
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.feature_selection import SelectFromModel
from xgboost import XGBClassifier

from preprocessing import load_and_save_yahoo
from features import add_features, create_target
from visualize import plot_feature_importance

# ----------------- CONFIG -----------------
OUTPUT_DIR = "outputs"
MODEL_DIR = "models_improved"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

TICKERS = ["AAPL", "MSFT", "NVDA", "GOOGL"]
START_DATE = "2015-01-01"
CLF_HORIZON = 5
MAX_FEATURES = 30  # Increased since we have more features now

# ----------------- HELPERS -----------------
def metrics_classification(y_true, y_pred, y_proba):
    """Return dict of classification metrics"""
    return {
        "acc": accuracy_score(y_true, y_pred),
        "bal_acc": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "auc": roc_auc_score(y_true, y_proba) if len(np.unique(y_true)) > 1 else float("nan"),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "log_loss": log_loss(y_true, y_proba) if len(np.unique(y_true)) > 1 else float("nan")
    }

def save_metrics_table(df_metrics, ticker):
    Path(OUTPUT_DIR).mkdir(exist_ok=True, parents=True)
    df_metrics.to_csv(os.path.join(OUTPUT_DIR, f"{ticker}_improved_metrics.csv"), index=False)

def get_base_models():
    xgb = XGBClassifier(
        n_estimators=200,
        learning_rate=0.03,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss',
        use_label_encoder=False,
        verbosity=0
    )
    
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        min_samples_leaf=4,
        random_state=42,
        n_jobs=-1
    )
    
    et = ExtraTreesClassifier(
        n_estimators=200,
        max_depth=6,
        min_samples_leaf=4,
        random_state=42,
        n_jobs=-1
    )
    return {'xgb': xgb, 'rf': rf, 'et': et}

def train_manual_stacking(X_train, y_train, X_test):
    """
    Manual stacking for Time Series.
    1. Generate meta-features using TimeSeriesSplit on Train.
    2. Train Meta-Learner (LogReg).
    3. Retrain Base Models on all Train.
    4. Generate Test meta-features.
    5. Predict Test.
    """
    tscv = TimeSeriesSplit(n_splits=5)
    base_models = get_base_models()
    
    # Meta features storage
    # We only get predictions for the validation parts of the splits
    meta_predictions_train = {name: [] for name in base_models}
    y_meta_train = []
    
    print("Generating meta-features via TimeSeriesSplit...")
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train)):
        X_tr, y_tr = X_train[train_idx], y_train[train_idx]
        X_val, y_val = X_train[val_idx], y_train[val_idx]
        y_meta_train.append(y_val)
        
        for name, model in base_models.items():
            # Clone model to reset it (or just create new)
            from sklearn.base import clone
            clf = clone(model)
            clf.fit(X_tr, y_tr)
            # Predict proba for class 1
            pred = clf.predict_proba(X_val)[:, 1]
            meta_predictions_train[name].append(pred)
            
            # Debug: Check AUC of this fold
            try:
                fold_auc = roc_auc_score(y_val, pred)
                print(f"  Fold {fold} {name} AUC: {fold_auc:.4f}")
            except:
                pass
            
    # Concatenate all validation predictions
    X_meta_train = []
    for name in base_models:
        meta_predictions_train[name] = np.concatenate(meta_predictions_train[name])
        
    # Stack features: shape (N_samples_meta, N_models)
    X_meta_train = np.column_stack([meta_predictions_train[name] for name in base_models])
    y_meta_train = np.concatenate(y_meta_train)
    
    print(f"Meta-training set size: {X_meta_train.shape}")
    
    # Train Meta Learner
    # Relax regularization to allow it to learn from base models
    meta_learner = LogisticRegression(C=1.0, random_state=42)
    meta_learner.fit(X_meta_train, y_meta_train)
    
    # Retrain Base Models on FULL Train and predict Test
    print("Retraining base models on full training set...")
    meta_predictions_test = {}
    fitted_base_models = {}
    
    for name, model in base_models.items():
        model.fit(X_train, y_train)
        fitted_base_models[name] = model
        pred = model.predict_proba(X_test)[:, 1]
        meta_predictions_test[name] = pred
        
    X_meta_test = np.column_stack([meta_predictions_test[name] for name in base_models])
    
    # Final prediction
    y_pred = meta_learner.predict(X_meta_test)
    y_proba = meta_learner.predict_proba(X_meta_test)[:, 1]
    
    return y_pred, y_proba, fitted_base_models, meta_learner, list(base_models.keys())

# ----------------- MAIN LOOP -----------------
def run_training():
    for T in TICKERS:
        print(f"Processing {T} with Improved Stacking Pipeline...")
        
        # 1. Load Data
        df_raw = load_and_save_yahoo(T, START_DATE)
        print(f"Loaded raw data: {len(df_raw)} rows")
        
        df = add_features(df_raw, ticker=T, start_date=START_DATE)
        print(f"After features: {len(df)} rows")
        
        # 2. Target Creation
        df = create_target(df, horizon=CLF_HORIZON)
        print(f"After target: {len(df)} rows")
        
        # 3. Cleaning
        # Ensure Date is not a column when cleaning if it exists
        if "Date" in df.columns:
            df = df.set_index("Date")
            
        # Select numeric columns for cleaning reference
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        # Replace inf with nan in numeric columns only
        df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
        
        # Identify columns with excessive NaNs
        nan_counts = df[numeric_cols].isna().sum()
        high_nan_cols = nan_counts[nan_counts > len(df) * 0.1]
        if not high_nan_cols.empty:
            print(f"Columns with >10% NaNs: {high_nan_cols.to_dict()}")
        
        # Drop rows with any NaN
        before_drop = len(df)
        df = df.dropna()
        after_drop = len(df)
        print(f"Dropped {before_drop - after_drop} rows with NaNs/Infs.")
        
        if df.empty:
            print(f"Skipping {T}: DataFrame empty after cleaning.")
            continue
            
        # 4. Feature Selection / Preprocessing
        drop_cols = ["target_price", "target_return", "target", "Date", "Ticker", 
                     "Open", "High", "Low", "Close", "Adj Close", "Volume"]
        feature_cols = [c for c in df.columns if c not in drop_cols]
        
        # Double check for ANY remaining infinite/nan
        X_check = df[feature_cols].values
        if not np.all(np.isfinite(X_check)):
            print(f"CRITICAL: Infinite values still present for {T}!")
            # Last resort fill
            df[feature_cols] = df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
        
        print(f"Initial feature count: {len(feature_cols)}")
        
        # Split Train/Test (Time Ordered)
        if len(df) < 100:
            print(f"Skipping {T}: Not enough data ({len(df)} rows).")
            continue
            
        split_idx = int(len(df) * 0.85) # 15% Holdout Test
        df_train = df.iloc[:split_idx].copy()
        df_test = df.iloc[split_idx:].copy()
        
        X_train = df_train[feature_cols].values
        y_train = df_train["target"].values
        X_test = df_test[feature_cols].values
        y_test = df_test["target"].values
        
        # 5. Robust Scaling (better for financial outliers)
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # 6. Feature Selection (Embeded RF)
        print("Running feature selection...")
        selector = SelectFromModel(
            RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1),
            max_features=MAX_FEATURES,
            threshold="median"
        )
        X_train_sel = selector.fit_transform(X_train_scaled, y_train)
        X_test_sel = selector.transform(X_test_scaled)
        
        selected_indices = selector.get_support(indices=True)
        selected_features = [feature_cols[i] for i in selected_indices]
        print(f"Selected {len(selected_features)} features: {selected_features}")
        
        # 7. Train Stacking Classifier
        print("Training Stacking Ensemble (Manual TimeSeriesSplit)...")
        y_pred, y_proba, fitted_base_models, meta_learner, base_names = train_manual_stacking(X_train_sel, y_train, X_test_sel)
        
        # 8. Evaluate
        metrics = metrics_classification(y_test, y_pred, y_proba)
        
        print("-" * 30)
        print(f"Results for {T}:")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")
        print("-" * 30)
        
        # Save metrics
        metrics["model"] = "StackingEnsemble_Improved"
        metrics["ticker"] = T
        df_metrics = pd.DataFrame([metrics])
        save_metrics_table(df_metrics, T)
        
        # Save Artifacts
        # Save Meta Learner
        joblib.dump(meta_learner, os.path.join(MODEL_DIR, f"stacking_meta_lr_{T}.pkl"))
        # Save Base Models
        for name, model in fitted_base_models.items():
            joblib.dump(model, os.path.join(MODEL_DIR, f"stacking_base_{name}_{T}.pkl"))
            
        joblib.dump(scaler, os.path.join(MODEL_DIR, f"scaler_robust_{T}.pkl"))
        joblib.dump(selected_features, os.path.join(MODEL_DIR, f"features_{T}.pkl"))
        
        # Visualization (Feature Importance - derived from RF base learner)
        try:
            if 'rf' in fitted_base_models:
                rf_base = fitted_base_models['rf']
                fi = rf_base.feature_importances_
                plot_feature_importance(selected_features, fi, save_path=os.path.join(OUTPUT_DIR, f"{T}_improved_fi.html"))
        except Exception as e:
            print(f"Could not plot feature importance: {e}")

    print(f"\nTraining complete. Models saved to {MODEL_DIR}, metrics to {OUTPUT_DIR}")

if __name__ == "__main__":
    run_training()
