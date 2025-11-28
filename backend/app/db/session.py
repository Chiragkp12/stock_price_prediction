import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./stocks.db")

class Base(DeclarativeBase):
    pass

engine = create_engine(DATABASE_URL, echo=False, future=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    # Lazy import models to avoid circulars
    from . import models  # noqa: F401
    Base.metadata.create_all(bind=engine)