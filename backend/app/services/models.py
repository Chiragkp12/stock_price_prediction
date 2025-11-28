from typing import Optional, Literal, Dict, Any
from pydantic import BaseModel
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

try:
    import tensorflow as tf  # optional
    from tensorflow.keras import layers, models
    TENSORFLOW_AVAILABLE = True
except Exception:
    TENSORFLOW_AVAILABLE = False

def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / np.clip(y_true, 1e-8, None)))

class ModelRequest(BaseModel):
    ticker: str
    horizon: Literal[7, 15, 30] = 7
    model: Literal["linear","rf","gbm","lstm","gru"] = "linear"
    years: Optional[int] = 2

def _split(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    return X_train, X_test, y_train, y_test

def _metrics(y_true, y_pred):
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    mape_v = float(mape(y_true, y_pred))
    return {"rmse": rmse, "mae": mae, "r2": r2, "mape": mape_v}

def _train_model(X_train, y_train, model_type: str):
    if model_type == "linear":
        model = LinearRegression()
    elif model_type == "rf":
        model = RandomForestRegressor(n_estimators=200, random_state=42)
    elif model_type == "gbm":
        model = GradientBoostingRegressor(random_state=42)
    elif model_type in ("lstm","gru") and TENSORFLOW_AVAILABLE:
        # Simple sequence model over features; note: this is illustrative
        # Convert tabular to sequences of length 1 for demo simplicity
        input_shape = (1, X_train.shape[1])
        model = models.Sequential()
        if model_type == "lstm":
            model.add(layers.LSTM(64, input_shape=input_shape, return_sequences=False))
        else:
            model.add(layers.GRU(64, input_shape=input_shape, return_sequences=False))
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dense(1))
        model.compile(optimizer='adam', loss='mse')
        # Fit with sequences of len 1
        X_seq = X_train.values.reshape((-1, 1, X_train.shape[1]))
        model.fit(X_seq, y_train.values, epochs=10, batch_size=32, verbose=0)
    else:
        raise ValueError("Model type not supported or optional dependency missing")

    if model_type in ("lstm","gru") and TENSORFLOW_AVAILABLE:
        return model
    else:
        model.fit(X_train, y_train)
        return model

def _predict(model, X) -> np.ndarray:
    if TENSORFLOW_AVAILABLE and isinstance(model, models.Model):
        X_seq = X.values.reshape((-1, 1, X.shape[1]))
        return model.predict(X_seq, verbose=0).ravel()
    return model.predict(X)

def run_prediction(X: pd.DataFrame, y: pd.Series, meta: Dict[str,Any], req: ModelRequest):
    X_train, X_test, y_train, y_test = _split(X, y)
    try:
        model = _train_model(X_train, y_train, req.model)
    except ValueError as e:
        return {"error": str(e), "hint": "Install tensorflow or choose linear/rf/gbm"}

    y_pred_test = _predict(model, X_test)
    metrics = _metrics(y_test, y_pred_test)

    # Roll-forward multi-step forecast using last available features (naive)
    horizon = req.horizon
    last_X = X.iloc[-1:].copy()
    preds = []
    residual_std = float(np.std(y_test.values - y_pred_test))
    for _ in range(horizon):
        next_pred = float(_predict(model, last_X)[0])
        preds.append(next_pred)
        # Simple update: shift lag features
        for l in [3,2,1]:
            lag_col = f"lag_close_{l}"
            if lag_col in last_X.columns:
                if l == 1:
                    last_X[lag_col] = next_pred
                else:
                    prev = f"lag_close_{l-1}"
                    last_X[lag_col] = float(last_X[prev])
        # leave other features unchanged for simplicity

    # Confidence intervals via residual std
    ci = [{"lower": float(p - 1.96 * residual_std), "upper": float(p + 1.96 * residual_std)} for p in preds]

    return {
        "ticker": req.ticker,
        "horizon": horizon,
        "predictions": preds,
        "confidence": ci,
        "metrics": metrics,
    }

def compare_models(X: pd.DataFrame, y: pd.Series, meta: Dict[str,Any], req: ModelRequest):
    X_train, X_test, y_train, y_test = _split(X, y)
    results = []
    for m in ["linear","rf","gbm"] + (["lstm","gru"] if TENSORFLOW_AVAILABLE else []):
        try:
            model = _train_model(X_train, y_train, m)
            y_pred = _predict(model, X_test)
            results.append({"model": m, **_metrics(y_test, y_pred)})
        except Exception as e:
            results.append({"model": m, "error": str(e)})
    return {"ticker": req.ticker, "results": results}