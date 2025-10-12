import pandas as pd
import numpy as np
import joblib
import os

from sqlalchemy import text
from sqlmodel import Session


MODEL_PATH = r"C:/Users/Windows/software-academia/software_academia/ml/models/model_rf.pkl"
bundle = joblib.load(MODEL_PATH)
pipe = bundle["model"]
expected = bundle["feature_order"]
cat_cols = bundle["categororical_cols"] if "categororical_cols" in bundle else bundle["categorical_cols"]
num_cols = bundle["numeric_cols"]

print("modelo cargado. Columnas esperadas:", len(expected))

from software_academia.ml.src.feature_builder import build_features_from_db_sqlmodel_session, infer_asof_date
from software_academia.backend.app.core.database import engine

def score_batch_from_db(session: Session, model_path: str | None = None, asof_date: str | None = None) -> pd.DataFrame:
    # 1) fecha de corte
    if asof_date is None:
        asof_date = infer_asof_date(session)

    # 2) features desde BD
    feat = build_features_from_db_sqlmodel_session(session, asof_date)

    # 3) usar bundle cargado arriba (si pasas otro path, lo recargo)
    global pipe, expected, cat_cols, num_cols
    if model_path is not None:
        _b = joblib.load(model_path)
        pipe = _b["model"]; expected = _b["feature_order"]
        cat_cols = _b.get("categorical_cols", []); num_cols = _b.get("numeric_cols", [])

    # 4) alinear columnas y tipos
    ids = feat["Student_ID"].values
    X = feat.copy()
    if "Student_ID" not in expected:
        X = X.drop(columns=["Student_ID"], errors="ignore")

    for c in expected:
        if c not in X.columns:
            X[c] = 0
    X = X[expected].copy()

    for c in cat_cols:
        if c in X.columns:
            X[c] = X[c].astype(str).fillna("<UNK>")
    for c in num_cols:
        if c in X.columns:
            X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0)

    # 5) predecir
    proba = pipe.predict_proba(X)[:, 1]
    scores = pd.DataFrame({
        "Student_ID": ids,
        "asof_date": asof_date,
        "churn_risk": proba
    }).sort_values("churn_risk", ascending=False)
    return scores

def score_one_student_from_db(session: Session, student_id: int, model_path: str | None = None, asof_date: str | None = None) -> float:
    if asof_date is None:
        asof_date = infer_asof_date(session)

    # Reusar builder y filtrar al alumno
    feat = build_features_from_db_sqlmodel_session(session, asof_date)
    row = feat.loc[feat["Student_ID"] == student_id].copy()
    if row.empty:
        raise ValueError(f"Student_ID {student_id} no encontrado.")

    # Bundle (si pasas otro path, recarga)
    global pipe, expected, cat_cols, num_cols
    if model_path is not None:
        _b = joblib.load(model_path)
        pipe = _b["model"]; expected = _b["feature_order"]
        cat_cols = _b.get("categorical_cols", []); num_cols = _b.get("numeric_cols", [])

    X = row.copy()
    if "Student_ID" not in expected:
        X = X.drop(columns=["Student_ID"], errors="ignore")

    for c in expected:
        if c not in X.columns:
            X[c] = 0
    X = X[expected].copy()

    for c in cat_cols:
        if c in X.columns:
            X[c] = X[c].astype(str).fillna("<UNK>")
    for c in num_cols:
        if c in X.columns:
            X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0)

    prob = float(pipe.predict_proba(X)[:, 1][0])
    return prob

# 1) Crea tu Session(engine) donde tú ya la usas
# from sqlmodel import Session
# from somewhere import engine
from software_academia.backend.app.core.database import engine

# asumiendo que ya tienes 'engine' y abriste la sesión:
with Session(engine) as session:
    prob = score_one_student_from_db(session, student_id=1001)
    print("Prob alumno 1001:", prob)



