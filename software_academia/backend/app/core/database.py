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
    # sin connect_args["ssl"]
)

print("DATABASE_URL:", DATABASE_URL)
print("engine:", engine)

from sqlalchemy import text
from sqlmodel import Session

with Session(engine) as s:
    s.exec(text("SELECT 1"))
    print("✅ Conexión OK")

from sqlalchemy import text
from sqlmodel import Session

with Session(engine) as s:
    # CAST a entero por si la columna es texto
    r = s.exec(text("""
        SELECT COUNT(*) AS cantidad_1
        FROM `churn_training_unified`
        WHERE CAST(`abandono` AS UNSIGNED) = 1
    """))
    print("abandono = 1 ->", r.scalar())
with Session(engine) as s:
    cols_students = s.exec(text("""
        SELECT COLUMN_NAME, DATA_TYPE
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_SCHEMA='Ingles_academia' AND TABLE_NAME='students'
        ORDER BY ORDINAL_POSITION
    """)).all()
    cols_churn = s.exec(text("""
        SELECT COLUMN_NAME, DATA_TYPE
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_SCHEMA='Ingles_academia' AND TABLE_NAME='churn_training_unified'
        ORDER BY ORDINAL_POSITION
    """)).all()

print("students:", cols_students[:5], "…")
print("churn_training_unified:", cols_churn[:5], "…")
