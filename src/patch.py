import pathlib
p = pathlib.Path(r'd:\Stock_Price_Predictor\src\hyperopt.py')
text = p.read_text(encoding='utf-8')
old_text = """    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params"""
new_text = """    # Use Tree-structured Parzen Estimator (TPE) sampler for Bayesian optimization of residuals
    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    print(f"[Optuna] Starting Bayesian Optimization (TPE) for Bo-XGBoost with {n_trials} trials...")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    print(f"[Optuna] Best params: {study.best_params}")
    return study.best_params"""
text = text.replace(old_text, new_text)
p.write_text(text, encoding='utf-8')
print("Patch successful!")
