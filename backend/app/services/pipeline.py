import pandas as pd
from typing import Tuple, Dict
from sklearn.preprocessing import StandardScaler
from .indicators import add_features

def prepare_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, Dict]:
    xdf = add_features(df)
    xdf = xdf.dropna()
    features = [c for c in xdf.columns if c not in ["Adj Close","Close"]]
    X = xdf[features]
    y = xdf["Close"]
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), index=X.index, columns=features)
    meta = {
        "scaler": scaler,
        "features": features,
        "last_date": X.index.max(),
    }
    return X_scaled, y, meta