import pandas as pd
import shap
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sqlmodel import Session

import software_academia.ml.src.feature_builder as fb
print(fb.__file__)
import inspect
print(inspect.getsource(fb.df_from_sql))

# shap_utils.py
from software_academia.ml.src.scoring import (
    pipe as _pipe,
    expected as _expected,
    cat_cols as _cat_cols,
    num_cols as _num_cols,
)
from software_academia.ml.src.feature_builder import (
    build_features_from_db_sqlmodel_session,
    infer_asof_date,
    df_from_sql,
)
from software_academia.backend.app.core.database import engine


def get_preprocessor_and_estimator(pipe: Pipeline):
    preproc = None
    est = None
    for name, step in pipe.named_steps.items():
        if isinstance(step, ColumnTransformer) or name.lower() == "preprocessor":
            preproc = step
        est = step
    if preproc is None and isinstance(pipe, Pipeline):
        for _, step in pipe.named_steps.items():
            if isinstance(step, ColumnTransformer):
                preproc = step
                break
    return preproc, est


def get_transformed_feature_names(
    preproc: ColumnTransformer,
    cat_cols: list[str],
    num_cols: list[str],
) -> list[str]:
    """Nombres robustos para TODAS las columnas resultantes del ColumnTransformer."""
    # 1) Intento directo
    try:
        names = preproc.get_feature_names_out().tolist()
        if names:
            return names
    except Exception:
        pass

    feature_names: list[str] = []

    # Columnas originales de entrada (si están)
    try:
        original_in = list(preproc.feature_names_in_)
    except Exception:
        original_in = []

    assigned: list[str] = []

    def last_step(transformer):
        if isinstance(transformer, Pipeline) and transformer.steps:
            return transformer.steps[-1][1]
        return transformer

    def names_from_transformer(trans, cols):
        t = last_step(trans)
        cols = list(cols) if not isinstance(cols, str) else [cols]
        if hasattr(t, "get_feature_names_out"):
            try:
                return t.get_feature_names_out(cols).tolist()
            except TypeError:
                return t.get_feature_names_out().tolist()
        elif t == "passthrough":
            return cols
        else:
            return cols

    # Explícitos
    for name, trans, cols in preproc.transformers_:
        if name == "remainder":
            continue
        cols = list(cols) if not isinstance(cols, str) else [cols]
        assigned.extend(cols)
        feature_names.extend(names_from_transformer(trans, cols))

    # Remainder
    for name, trans, cols in preproc.transformers_:
        if name != "remainder":
            continue
        if trans == "passthrough":
            if original_in:
                remainder_cols = [c for c in original_in if c not in set(assigned)]
            else:
                remainder_cols = [c for c in (num_cols or []) if c not in set(assigned)]
            feature_names.extend(remainder_cols)

    return feature_names


def build_background_matrix(session, asof_date: str, sample_size: int = 300):
    preproc, est = get_preprocessor_and_estimator(_pipe)
    if preproc is None:
        raise RuntimeError("No se encontró preprocesador en el pipeline")

    feat = build_features_from_db_sqlmodel_session(session, asof_date)
    X = feat.copy()

    # Drop Student_ID solo si NO está en el contrato
    if "Student_ID" in X.columns and "Student_ID" not in _expected:
        X = X.drop(columns=["Student_ID"], errors="ignore")

    # Alinear con el contrato
    for c in _expected:
        if c not in X.columns:
            X[c] = 0
    X = X[_expected].copy()

    # Tipos
    for c in _cat_cols:
        if c in X.columns:
            X[c] = X[c].astype(str).fillna("<UNK>")
    for c in _num_cols:
        if c in X.columns:
            X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0)

    # Muestra de fondo
    X_bg = X.sample(sample_size, random_state=42) if len(X) > sample_size else X

    # Transformación
    X_bg_trans = preproc.transform(X_bg)
    if hasattr(X_bg_trans, "toarray"):
        X_bg_trans = X_bg_trans.toarray()

    transformed_names = get_transformed_feature_names(preproc, _cat_cols, _num_cols)

    # Asegurar longitud exacta
    if len(transformed_names) != X_bg_trans.shape[1]:
        try:
            names2 = preproc.get_feature_names_out().tolist()
        except Exception:
            names2 = None
        if names2 is not None and len(names2) == X_bg_trans.shape[1]:
            transformed_names = names2
        else:
            transformed_names = [f"feat_{i}" for i in range(X_bg_trans.shape[1])]

    return X_bg, X_bg_trans, transformed_names


def explain_student(session, student_id: int, asof_date: str | None = None, top_k: int = None):
    if asof_date is None:
        asof_date = infer_asof_date(session)

    preproc, est = get_preprocessor_and_estimator(_pipe)
    if preproc is None:
        raise RuntimeError("No se encontró preprocesador en el pipeline")

    # Background (informa sobre categorías OHE) y nombres "buenos"
    X_bg, X_bg_trans, transformed_names_bg = build_background_matrix(session, asof_date, sample_size=300)

    # Features del alumno
    feat = build_features_from_db_sqlmodel_session(session, asof_date)
    row = feat.loc[feat["Student_ID"] == student_id].copy()
    if row.empty:
        raise ValueError(f"Student_ID {student_id} no encontrado")

    X = row.copy()
    if "Student_ID" in X.columns and "Student_ID" not in _expected:
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

    # Transformar fila objetivo
    X_trans = preproc.transform(X)
    if hasattr(X_trans, "toarray"):
        X_trans = X_trans.toarray()

    # Nombres transformados
    transformed_names = get_transformed_feature_names(preproc, _cat_cols, _num_cols)
    if len(transformed_names) != X_trans.shape[1]:
        try:
            names2 = preproc.get_feature_names_out().tolist()
        except Exception:
            names2 = None
        if names2 is not None and len(names2) == X_trans.shape[1]:
            transformed_names = names2
        else:
            transformed_names = [f"feat_{i}" for i in range(X_trans.shape[1])]

    # TreeExplainer SIN feature_names (evita validación interna de longitudes)
    explainer = shap.TreeExplainer(est)
    shap_values = explainer.shap_values(X_trans)

    # Vector de la clase positiva (binario) o único vector (regresión)
    if isinstance(shap_values, list) and len(shap_values) == 2:
        sv = shap_values[1][0]
    else:
        sv = shap_values[0] if np.ndim(shap_values) == 2 else shap_values

    sv = np.asarray(sv).ravel()

    # Validación final
    assert len(sv) == len(transformed_names), f"Len mismatch: sv={len(sv)} vs names={len(transformed_names)}"

    # Mapear dummies OHE a su feature original
    def original_feature_name(trans_name: str) -> str:
        for cat in _cat_cols:
            prefix = f"{cat}_"
            if trans_name.startswith(prefix):
                return cat
        return trans_name

    contrib = pd.DataFrame({
        "transformed_feature": transformed_names,
        "shap_value": sv
    })
    contrib["orig_feature"] = contrib["transformed_feature"].apply(original_feature_name)
    contrib["abs_val"] = contrib["shap_value"].abs()

    agg = (contrib.groupby("orig_feature", as_index=False)
                  .agg(total_abs=("abs_val", "sum"))
                  .sort_values("total_abs", ascending=False))

    top_features = agg.head(top_k)["orig_feature"].tolist()

    explanations = []
    row_values = X.iloc[0].to_dict()
    for feat_name in top_features:
        mask = contrib["orig_feature"] == feat_name
        shap_net = contrib.loc[mask, "shap_value"].sum()
        direction = "sube_riesgo" if shap_net > 0 else "baja_riesgo"
        explanations.append({
            "feature": feat_name,
            "impact": float(shap_net),
            "effect": direction,
            "value": row_values.get(feat_name, None)
        })

    return {
        "student_id": student_id,
        "asof_date": asof_date,
        "top_k": top_k,
        "reasons": explanations
    }


# Ejecución de prueba
with Session(engine) as session:
    exp = explain_student(session, student_id=1001, top_k=1)
    print(exp)
