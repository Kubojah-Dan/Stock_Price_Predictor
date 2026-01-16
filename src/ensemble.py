import numpy as np

def ensemble_predict(xgb_proba, lstm_proba, w_xgb=0.6, w_lstm=0.4):
    return w_xgb * xgb_proba + w_lstm * lstm_proba

def ensemble_proba(xgb_p, lstm_p, w_xgb=0.6, w_lstm=0.4):
    return w_xgb * xgb_p + w_lstm * lstm_p

