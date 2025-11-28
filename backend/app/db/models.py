from sqlalchemy import Column, Integer, String, Float, Date
from .session import Base

class StockPrice(Base):
    __tablename__ = "stock_prices"
    id = Column(Integer, primary_key=True, index=True)
    ticker = Column(String, index=True)
    date = Column(Date, index=True)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    adj_close = Column(Float)
    volume = Column(Float)

class Prediction(Base):
    __tablename__ = "predictions"
    id = Column(Integer, primary_key=True, index=True)
    ticker = Column(String, index=True)
    horizon = Column(Integer)
    model = Column(String)
    date = Column(Date, index=True)
    predicted = Column(Float)
    lower = Column(Float)
    upper = Column(Float)

class ModelMetric(Base):
    __tablename__ = "model_metrics"
    id = Column(Integer, primary_key=True, index=True)
    ticker = Column(String, index=True)
    model = Column(String)
    rmse = Column(Float)
    mae = Column(Float)
    r2 = Column(Float)
    mape = Column(Float)