from sqlmodel import SQLModel, create_engine, Session, Relationship, select
import os
from contextlib import contextmanager
from dotenv import load_dotenv

load_dotenv()

DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST", "mysql-senu-jhonnybarrios968.b.aivencloud.com")
DB_PORT = os.getenv("DB_PORT", "15797")
DB_NAME = os.getenv("DB_NAME")

DATABASE_URL = (
    f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    "?charset=utf8mb4"
)

engine = create_engine(
    DATABASE_URL,
    echo=True,
    pool_pre_ping=True,
    pool_recycle=3600,
    future=True,
)

def get_session():
    s = Session(engine)
    try:
        yield s
        s.commit()
    except:
        s.rollback()
        raise
    finally:
        s.close()