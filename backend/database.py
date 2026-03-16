"""
database.py — SQLAlchemy engine + session factory
Reads DATABASE_URL from .env.
Defaults to SQLite (chargesync.db) if env var not set.
Switch to PostgreSQL for hosting by setting:
  DATABASE_URL=postgresql://user:pass@host:5432/chargesync
"""
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase

load_dotenv()  # reads .env file

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./chargesync.db")

# SQLite needs check_same_thread=False for FastAPI's threading model
connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(DATABASE_URL, connect_args=connect_args)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


def get_db():
    """FastAPI dependency — yields a DB session and closes it after the request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
