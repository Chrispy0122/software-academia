from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sqlmodel import Session

app = FastAPI(title="Churn Scoring API", version="1.0.0")

app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/")
def root():
    return {"ok": True, "endpoints": ["/health", "/docs", "/students/{id}/score"]}


from software_academia.backend.app.core.database import engine

def get_session():
    with Session(engine) as session:
        yield session

import joblib

# ajusta a tu ruta real del modelo
MODEL_PATH = r"C:/Users/Windows/software-academia/software_academia/ml/models/model_rf.pkl"

# carga única
_bundle = joblib.load(MODEL_PATH)
_pipe     = _bundle["model"]
_expected = _bundle["feature_order"]
_cat_cols = _bundle.get("categorical_cols", [])
_num_cols = _bundle.get("numeric_cols", [])



from sqlalchemy import text
import pandas as pd

def df_from_sql(session: Session, sql: str, params: dict | None = None) -> pd.DataFrame:
    stmt = text(sql)
    if params: stmt = stmt.bindparams(**params)
    rows = session.exec(stmt).mappings().all()
    return pd.DataFrame(rows)

def infer_asof_date(session: Session) -> str:
    att = df_from_sql(session, "SELECT MAX(Class_Date) AS max_att FROM attendance")
    pay = df_from_sql(session, "SELECT MAX(Payment_Date) AS max_pay FROM payments")
    max_att = pd.to_datetime(att["max_att"].iloc[0], errors="coerce") if not att.empty else pd.NaT
    max_pay = pd.to_datetime(pay["max_pay"].iloc[0], errors="coerce") if not pay.empty else pd.NaT
    asof = max(max_att, max_pay)
    if pd.isna(asof):
        raise HTTPException(status_code=500, detail="No hay fechas en attendance/payments")
    return asof.strftime("%Y-%m-%d")

def build_features_from_db(session: Session, asof_date: str) -> pd.DataFrame:
    asof = pd.to_datetime(asof_date); cutoff = asof - pd.Timedelta(days=1); cutoff_s = cutoff.strftime("%Y-%m-%d")
    students = df_from_sql(session, "SELECT Student_ID, Signup_Date, Plan, Price_USD FROM students")
    if students.empty: raise HTTPException(status_code=500, detail="Tabla students vacía")
    students["Signup_Date"] = pd.to_datetime(students["Signup_Date"], errors="coerce")
    students["tenure_days"] = (cutoff - students["Signup_Date"]).dt.days.clip(lower=0)
    students["plan"] = students["Plan"].astype(str)
    students["price_usd"] = pd.to_numeric(students["Price_USD"], errors="coerce").fillna(0)

    att_sql = """
    WITH hist AS (
      SELECT Student_ID, Class_Date, COALESCE(Present, 1) AS Present
      FROM attendance
      WHERE Class_Date <= :cutoff
    ),
    last_att AS (SELECT Student_ID, MAX(Class_Date) AS last_class FROM hist GROUP BY Student_ID),
    win30 AS (SELECT Student_ID, SUM(Present) AS total_classes_30d FROM hist
              WHERE Class_Date BETWEEN DATE_SUB(:cutoff, INTERVAL 29 DAY) AND :cutoff GROUP BY Student_ID),
    win60 AS (SELECT Student_ID, SUM(Present) AS total_classes_60d FROM hist
              WHERE Class_Date BETWEEN DATE_SUB(:cutoff, INTERVAL 59 DAY) AND :cutoff GROUP BY Student_ID),
    win90 AS (SELECT Student_ID, SUM(Present) AS total_classes_90d FROM hist
              WHERE Class_Date BETWEEN DATE_SUB(:cutoff, INTERVAL 89 DAY) AND :cutoff GROUP BY Student_ID)
    SELECT s.Student_ID, last_att.last_class,
           COALESCE(win30.total_classes_30d,0) AS total_classes_30d,
           COALESCE(win60.total_classes_60d,0) AS total_classes_60d,
           COALESCE(win90.total_classes_90d,0) AS total_classes_90d
    FROM (SELECT DISTINCT Student_ID FROM students) s
    LEFT JOIN last_att ON s.Student_ID = last_att.Student_ID
    LEFT JOIN win30    ON s.Student_ID = win30.Student_ID
    LEFT JOIN win60    ON s.Student_ID = win60.Student_ID
    LEFT JOIN win90    ON s.Student_ID = win90.Student_ID
    """
    attendance_agg = df_from_sql(session, att_sql, {"cutoff": cutoff_s})

    pay_sql = """
    WITH pay AS (
      SELECT Student_ID, Payment_Date, Amount, Status, Payment_ID
      FROM payments
      WHERE Payment_Date <= :cutoff
    ),
    ok AS (
      SELECT * FROM pay
      WHERE COALESCE(LOWER(Status), 'paid') IN ('completed','paid','success','approved','ok')
    ),
    last_pay AS (SELECT Student_ID, MAX(Payment_Date) AS last_payment FROM ok GROUP BY Student_ID),
    win90 AS (
      SELECT Student_ID, SUM(Amount) AS payments_90d_usd, COUNT(Payment_ID) AS n_payments_90d
      FROM ok
      WHERE Payment_Date BETWEEN DATE_SUB(:cutoff, INTERVAL 89 DAY) AND :cutoff
      GROUP BY Student_ID
    )
    SELECT s.Student_ID, last_pay.last_payment,
           COALESCE(win90.payments_90d_usd,0) AS payments_90d_usd,
           COALESCE(win90.n_payments_90d,0)  AS n_payments_90d
    FROM (SELECT DISTINCT Student_ID FROM students) s
    LEFT JOIN last_pay ON s.Student_ID = last_pay.Student_ID
    LEFT JOIN win90    ON s.Student_ID = win90.Student_ID
    """
    payments_agg = df_from_sql(session, pay_sql, {"cutoff": cutoff_s})

    out = students.merge(attendance_agg, on="Student_ID", how="left").merge(payments_agg, on="Student_ID", how="left")
    out["last_class"] = pd.to_datetime(out["last_class"], errors="coerce")
    out["days_since_last_attendance"] = (cutoff - out["last_class"]).dt.days
    out["days_since_last_attendance"] = out["days_since_last_attendance"].fillna(9999).astype(int)
    out["last_payment"] = pd.to_datetime(out["last_payment"], errors="coerce")
    out["months_since_last_payment"] = ((cutoff - out["last_payment"]).dt.days / 30.0).fillna(9999.0)

    keep = ["Student_ID","plan","price_usd","tenure_days",
            "days_since_last_attendance","total_classes_30d","total_classes_60d","total_classes_90d",
            "months_since_last_payment","payments_90d_usd","n_payments_90d"]
    out = out[keep].fillna(0)
    out["plan"] = out["plan"].astype(str)
    return out

def score_one_student(session: Session, student_id: int, asof_date: str | None = None) -> float:
    # 1) fecha de corte
    if asof_date is None:
        asof_date = infer_asof_date(session)
    # 2) features
    feat = build_features_from_db(session, asof_date)
    row = feat.loc[feat["Student_ID"] == student_id].copy()
    if row.empty:
        raise HTTPException(status_code=404, detail=f"Student_ID {student_id} no encontrado")

    # 3) alinear al contrato del modelo
    X = row.copy()
    if "Student_ID" not in _expected:
        X = X.drop(columns=["Student_ID"], errors="ignore")
    for c in _expected:
        if c not in X.columns:
            X[c] = 0
    X = X[_expected].copy()
    for c in _cat_cols:
        if c in X.columns:
            X[c] = X[c].astype(str).fillna("<UNK>")
    for c in _num_cols:
        if c in X.columns:
            X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0)

    # 4) predecir
    return float(_pipe.predict_proba(X)[:, 1][0])




class ScoreResponse(BaseModel):
    student_id: int
    asof_date: str
    churn_risk: float

@app.get("/students/{student_id}/score", response_model=ScoreResponse)
def get_student_score(student_id: int, asof: str | None = None):
    # abre sesión a BD
    with Session(engine) as session:
        asof_date = asof or infer_asof_date(session)
        prob = score_one_student(session, student_id=student_id, asof_date=asof_date)
        return ScoreResponse(student_id=student_id, asof_date=asof_date, churn_risk=prob)
