import os
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

from .services.data_fetch import search_tickers, get_historical_prices
from .services.indicators import compute_indicators_bundle
from .services.pipeline import prepare_dataset
from .services.models import ModelRequest, run_prediction, compare_models
from .db.session import init_db

app = FastAPI(title="Stock Forecast API", version="0.1.0")

origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in origins],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def on_startup():
    init_db()


class SearchResponse(BaseModel):
    symbols: list[str]


@app.get("/api/stocks/search", response_model=SearchResponse)
def api_search(q: str = Query(..., min_length=1, description="Search query")):
    return SearchResponse(symbols=search_tickers(q))


@app.get("/api/stocks/{ticker}/historical")
def api_historical(ticker: str, years: int = 2):
    df = get_historical_prices(ticker, years=years)
    if df is None or df.empty:
        raise HTTPException(status_code=404, detail="No historical data")
    return {
        "ticker": ticker,
        "rows": df.reset_index().to_dict(orient="records"),
    }


@app.get("/api/indicators/{ticker}")
def api_indicators(ticker: str, years: int = 2):
    df = get_historical_prices(ticker, years=years)
    if df is None or df.empty:
        raise HTTPException(status_code=404, detail="No historical data")
    indicators = compute_indicators_bundle(df)
    return {"ticker": ticker, "indicators": indicators}


@app.post("/api/predict")
def api_predict(req: ModelRequest):
    df = get_historical_prices(req.ticker, years=req.years or 2)
    if df is None or df.empty:
        raise HTTPException(status_code=404, detail="No historical data")
    X, y, meta = prepare_dataset(df)
    result = run_prediction(X, y, meta, req)
    return result


@app.post("/api/models/compare")
def api_models_compare(req: ModelRequest):
    df = get_historical_prices(req.ticker, years=req.years or 2)
    if df is None or df.empty:
        raise HTTPException(status_code=404, detail="No historical data")
    X, y, meta = prepare_dataset(df)
    return compare_models(X, y, meta, req)