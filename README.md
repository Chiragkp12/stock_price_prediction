# Stock Price Prediction – Full-Stack ML Web App

End-to-end system to forecast stock prices from historical OHLCV data with technical indicators, multiple models (Linear Regression, Random Forest, Gradient Boosting, optional LSTM/GRU), FastAPI backend, Next.js + Tailwind frontend, PostgreSQL + Redis, and deployment guides.

## Quick Start

1. Backend
   - Create and populate `.env` in `backend/` (see `.env.example`).
   - Create a virtual environment and install dependencies:
     - PowerShell:
       - `python -m venv backend/.venv`
       - `backend/.venv/Scripts/Activate.ps1`
       - `pip install -r backend/requirements.txt`
   - Start API (dev): `uvicorn app.main:app --reload --port 8000 --app-dir backend`

2. Frontend
   - See `frontend/README.md` after scaffolding. Start dev server: `npm run dev`.
   - Set `NEXT_PUBLIC_API_BASE_URL` to your backend URL.

3. Services (optional for local dev)
   - `docker-compose up -d` to start PostgreSQL and Redis.

## Features

- Data pipeline: fetch (Alpha Vantage, Yahoo, yfinance), clean, scale, engineer features (SMA 20/50/200, EMA, RSI, MACD, Bollinger Bands), lag (t-1..t-3), temporal (day, month, quarter)
- Models: Linear Regression baseline; RandomForest, GradientBoosting; optional LSTM/GRU if `tensorflow` available
- Horizons: 7, 15, 30 days; metrics: RMSE, MAE, MAPE, R²
- FastAPI endpoints:
  - `/api/stocks/search`
  - `/api/stocks/{ticker}/historical`
  - `/api/indicators/{ticker}`
  - `/api/predict`
  - `/api/models/compare`
- Frontend (Next.js + Tailwind): search, historical charts, predictions + confidence, model dashboard, indicators panel, buy/sell signals (optional)
- Storage: PostgreSQL (prices, predictions, metrics), Redis cache (recent results)

## Deployment
- Frontend: Vercel
- Backend: Render/Railway/AWS

## Disclaimer
This application is for educational purposes only and not financial advice.# stock_price_prediction
