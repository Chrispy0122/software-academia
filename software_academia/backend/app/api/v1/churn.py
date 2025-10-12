from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sqlmodel import Session

app = FastAPI(title="Churn Scoring API", version="1.0.0")

@app.get("/health")
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
from software_academia.ml.src.scoring import pipe as _pipe, expected as _expected, cat_cols as _cat_cols, num_cols as _num_cols
from software_academia.ml.src.feature_builder import build_features_from_db_sqlmodel_session, infer_asof_date, df_from_sql

def score_one_student(session: Session, student_id: int, asof_date: str | None = None) -> float:
    # 1) fecha de corte
    if asof_date is None:
        asof_date = infer_asof_date(session)
    # 2) features
    feat = build_features_from_db_sqlmodel_session(session, asof_date)
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
