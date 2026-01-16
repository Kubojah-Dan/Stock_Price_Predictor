import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = BASE_DIR / "models"

def load_xgb_final(ticker):
    model = joblib.load(MODELS_DIR / f"xgb_{ticker}_final.pkl")
    scaler = joblib.load(MODELS_DIR / f"scaler_{ticker}_final.pkl")
    features = joblib.load(MODELS_DIR / f"features_{ticker}_final.pkl")
    return model, scaler, features
