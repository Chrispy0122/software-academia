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


def robust_feature_names_from_preproc(
    preproc: ColumnTransformer,
    input_cols: list[str]
) -> list[str]:
    """
    Nombres legibles y completos para TODAS las columnas transformadas,
    usando las columnas de entrada ACTUALES (input_cols) para calcular el remainder.
    - OneHotEncoder -> "col=cat"
    - passthrough   -> nombre original
    - otros         -> nombre de columna (1:1)
    """
    # 1) Intento directo (si está fitted y soportado)
    try:
        names = list(preproc.get_feature_names_out())
        if names:
            return names
    except Exception:
        pass

    names: list[str] = []

    def last_step(transformer):
        if isinstance(transformer, Pipeline) and transformer.steps:
            return transformer.steps[-1][1]
        return transformer

    assigned_cols = set()

    # 2) Construcción manual para transformadores explícitos
    for name, trans, cols in preproc.transformers_:
        if name == "remainder":
            continue
        cols = list(cols) if isinstance(cols, (list, tuple)) else [cols]
        t = last_step(trans)
        assigned_cols.update(cols)

        if isinstance(t, OneHotEncoder):
            cats = t.categories_
            for col, cat_values in zip(cols, cats):
                for cat in cat_values:
                    names.append(f"{col}={cat}")
        elif t == "passthrough":
            names.extend(cols)
        elif t == "drop":
            # no agrega columnas
            continue
        else:
            # StandardScaler y similares (1:1 por columna)
            names.extend(cols)

    # 3) Remainder basado en las columnas ACTUALES (input_cols)
    #    No uses feature_names_in_ aquí: usa input_cols para evitar desajustes.
    try:
        # Si el remainder es passthrough, añadimos todas las columnas
        # de input_cols que no hayan sido asignadas arriba.
        for name, trans, cols in preproc.transformers_:
            if name == "remainder" and trans == "passthrough":
                remainder_cols = [c for c in input_cols if c not in assigned_cols]
                names.extend(remainder_cols)
                break
    except Exception:
        pass

    # Último recurso si no obtuvimos nada
    if not names:
        names = list(input_cols)

    return names


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

    # input_cols son las columnas REALES que entraron al preproc hoy
    input_cols = list(X_bg.columns)
    transformed_names = robust_feature_names_from_preproc(preproc, input_cols)
    if len(transformed_names) != X_bg_trans.shape[1]:
    # Como diagnóstico temporal, imprime para ver la discrepancia:
    # print("len(transformed_names)=", len(transformed_names), " X_bg_trans.shape[1]=", X_bg_trans.shape[1])
        transformed_names = [f"feat_{i}" for i in range(X_bg_trans.shape[1])]



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

def explain_student(session, student_id: int, asof_date: str | None = None, top_k: int = 5):
    import numpy as np
    import pandas as pd
    import shap
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer

    if asof_date is None:
        asof_date = infer_asof_date(session)

    preproc, est = get_preprocessor_and_estimator(_pipe)
    if preproc is None:
        raise RuntimeError("No se encontró preprocesador en el pipeline")

    # ---------- Helpers internos ----------
    def _last_step(transformer):
        if isinstance(transformer, Pipeline) and transformer.steps:
            return transformer.steps[-1][1]
        return transformer

    def _orig_index_map(preproc: ColumnTransformer, input_cols: list[str]) -> list[str]:
        """
        Devuelve una lista de largo = n_cols_transformadas donde cada posición i
        indica el NOMBRE DE LA VARIABLE ORIGINAL que generó esa columna transformada.
        No depende de nombres del OHE; calcula longitudes por transformador.
        Soporta:
          - OneHotEncoder (respeta drop_idx_)
          - Passthrough / Drop
          - Otros (Scaler/Imputer...) 1:1
          - remainder='passthrough' (usa input_cols reales)
        """
        orig_names_by_pos: list[str] = []
        assigned = set()

        # Transformadores explícitos
        for name, trans, cols in preproc.transformers_:
            if name == "remainder":
                continue
            cols = list(cols) if isinstance(cols, (list, tuple)) else [cols]
            t = _last_step(trans)
            assigned.update(cols)

            if isinstance(t, OneHotEncoder):
                # Para cada columna categórica, el OHE produce (#categorías - drops) columnas
                cats_list = t.categories_
                drop_idx = getattr(t, "drop_idx_", None)  # None o lista/array por columna
                for i, col in enumerate(cols):
                    n_cats = len(cats_list[i])
                    dropped = set()
                    if drop_idx is not None:
                        di = drop_idx[i] if i < len(np.atleast_1d(drop_idx)) else None
                        if di is not None:
                            arr = np.atleast_1d(di)
                            dropped = set(int(x) for x in arr.tolist())
                    out_dim = n_cats - len(dropped)
                    orig_names_by_pos.extend([col] * out_dim)
            else:
                # passthrough: 1:1 por columna
                if t == "passthrough":
                    orig_names_by_pos.extend(cols)
                elif t == "drop":
                    continue
                else:
                    # Scaler/Imputer/etc. normalmente 1:1
                    orig_names_by_pos.extend(cols)

        # Remainder contra input_cols REALES
        for name, trans, cols in preproc.transformers_:
            if name == "remainder" and trans == "passthrough":
                remainder_cols = [c for c in input_cols if c not in assigned]
                orig_names_by_pos.extend(remainder_cols)
                break

        return orig_names_by_pos

    # ---------- Background (opcional para consistencia con tu flujo) ----------
    _ = build_background_matrix(session, asof_date, sample_size=300)

    # ---------- Fila del alumno (espacio original) ----------
    feat = build_features_from_db_sqlmodel_session(session, asof_date)
    row = feat.loc[feat["Student_ID"] == student_id].copy()
    if row.empty:
        raise ValueError(f"Student_ID {student_id} no encontrado")

    X = row.copy()
    if "Student_ID" in X.columns and "Student_ID" not in _expected:
        X = X.drop(columns=["Student_ID"], errors="ignore")

    # Alinear contrato
    for c in _expected:
        if c not in X.columns:
            X[c] = 0
    X = X[_expected].copy()
    proba = float(_pipe.predict_proba(X)[:, 1][0])  # prob. de la clase “abandona”

    # Tipos correctos
    for c in _cat_cols:
        if c in X.columns:
            X[c] = X[c].astype(str).fillna("<UNK>")
    for c in _num_cols:
        if c in X.columns:
            X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0)

    # ---------- Transformación (espacio transformado) ----------
    X_trans = preproc.transform(X)
    if hasattr(X_trans, "toarray"):
        X_trans = X_trans.toarray()

    # ---------- Mapeo índice→variable original ----------
    input_cols = list(X.columns)
    orig_by_idx = _orig_index_map(preproc, input_cols)

    # Alinear longitudes del mapeo con X_trans
    n_trans = X_trans.shape[1]
    if len(orig_by_idx) != n_trans:
        if len(orig_by_idx) > n_trans:
            orig_by_idx = orig_by_idx[:n_trans]
        else:
            orig_by_idx = orig_by_idx + (["<unk>"] * (n_trans - len(orig_by_idx)))

    # ---------- SHAP sobre el ESTIMADOR ----------
    explainer = shap.TreeExplainer(est)
    shap_values = explainer.shap_values(X_trans)

    # Binario vs regresión
    if isinstance(shap_values, list) and len(shap_values) >= 2:
        sv = shap_values[1][0]  # clase positiva
    else:
        sv = shap_values[0] if np.ndim(shap_values) == 2 else shap_values
    sv = np.asarray(sv).ravel()

    # Seguridad: alinear longitudes
    if len(sv) != len(orig_by_idx):
        if len(sv) > len(orig_by_idx):
            orig_by_idx = orig_by_idx + (["<unk>"] * (len(sv) - len(orig_by_idx)))
        else:
            sv = sv[:len(orig_by_idx)]

    # ---------- Agregar, LIMPIAR y agrupar por variable original ----------
    contrib = pd.DataFrame({
        "orig_feature": orig_by_idx,
        "shap_value": sv
    })
    contrib["abs_val"] = contrib["shap_value"].abs()

    # Limpieza: quita '<unk>' y ruido numérico
    EPS = 1e-10
    contrib = contrib[contrib["orig_feature"] != "<unk>"].copy()
    contrib = contrib[contrib["abs_val"] > EPS].copy()

    # Si quedó vacío (raro), reconstituir con nombres originales de input y sv (trunc/extend)
    if contrib.empty:
        # Empareja sv con input_cols (contrato original)
        m = min(len(input_cols), len(sv))
        contrib = pd.DataFrame({
            "orig_feature": input_cols[:m],
            "shap_value": sv[:m]
        })
        contrib["abs_val"] = contrib["shap_value"].abs()

    agg = (contrib.groupby("orig_feature", as_index=False)
                  .agg(total_abs=("abs_val", "sum"),
                       net=("shap_value", "sum"))
                  .sort_values("total_abs", ascending=False))

    top_rows = agg.head(top_k)

    explanations = []
    row_values = X.iloc[0].to_dict()
    for _, r in top_rows.iterrows():
        feat_name = r["orig_feature"]
        shap_net = float(r["net"])
        direction = "sube_riesgo" if shap_net > 0 else "baja_riesgo"
        explanations.append({
            "feature": feat_name,
            "impact": shap_net,
            "effect": direction,
            "value": row_values.get(feat_name, None)
        })

    return {
    "student_id": student_id,
    "asof_date": asof_date,
    "top_k": top_k,
    "prob_churn": proba,     # <-- aquí
    "reasons": explanations
   }


# Ejecución de prueba
with Session(engine) as session:
    exp = explain_student(session, student_id=1001, top_k=5)
    print(exp)





from __future__ import annotations
from typing import List, Dict, Any, Iterable
import json
from datetime import datetime
from pathlib import Path

from software_academia.ml.automatizacion.config import SHAP_RESULTS_JSON, TOP_RISK_REASONS
from software_academia.ml.src.shap_catalog import translate_feature_to_reason

def _normalize_customer_record(raw: Dict[str, Any], top_k: int) -> Dict[str, Any]:
    customer_id = raw.get("customer_id") or raw.get("student_id")
    asof_date = raw.get("asof_date")
    prob = float(raw.get("prob_churn", 0.0))

    # Nos quedamos solo con razones que SUBEN el riesgo, ordenadas por impacto desc
    reasons_raw: List[Dict[str, Any]] = raw.get("reasons", [])
    risk_raisers = [r for r in reasons_raw if str(r.get("effect")).lower() == "sube_riesgo"]
    risk_raisers.sort(key=lambda r: abs(float(r.get("impact", 0.0))), reverse=True)

    # Tomamos top_k y construimos estructura uniforme
    top_reasons = []
    for r in risk_raisers[:top_k]:
        feat = r.get("feature")
        top_reasons.append({
            "feature": feat,
            "reason_human": translate_feature_to_reason(feat),
            "impact": float(r.get("impact", 0.0)),
            "value": r.get("value"),
        })

    return {
        "customer_id": customer_id,
        "asof_date": asof_date,
        "prob_churn": prob,
        "top_reasons": top_reasons,
    }

def build_shap_json_payload(records: Iterable[Dict[str, Any]], top_k: int = TOP_RISK_REASONS) -> List[Dict[str, Any]]:
    """
    Toma una lista/iterable de registros (como los que ya imprime tu shap_utils.py)
    y devuelve una lista JSON normalizada lista para guardar.
    """
    output: List[Dict[str, Any]] = []
    for raw in records:
        try:
            norm = _normalize_customer_record(raw, top_k=top_k)
            # Validaciones básicas
            if norm["customer_id"] is None:
                continue
            if not (0.0 <= norm["prob_churn"] <= 1.0):
                continue
            output.append(norm)
        except Exception:
            # En producción: loggear el error con el customer id si existe
            continue
    return output

def save_shap_json(payload: List[Dict[str, Any]], path: Path = SHAP_RESULTS_JSON) -> Path:
    """
    Guarda el payload en disco como JSON, con indentado legible.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path



# -------------------------------------------------------------------------
# PUNTO DE ENTRADA DE EXPORTACIÓN (llámalo desde tu flujo actual)
# -------------------------------------------------------------------------
def export_shap_results_to_json(shap_records: Iterable[Dict[str, Any]], *, top_k: int = TOP_RISK_REASONS) -> Path:
    """
    API pública para tu pipeline:
    1) Le pasas los registros ya calculados por tu lógica SHAP (prob + reasons)
    2) Te devuelve la ruta del JSON estandarizado (/data/output/shap_results.json)
    """
    payload = build_shap_json_payload(shap_records, top_k=top_k)
    return save_shap_json(payload, SHAP_RESULTS_JSON)

# Ejemplo de uso (puedes quitarlo si ya tienes un caller):
if __name__ == "__main__":
    # Simulación mínima de un registro como el tuyo
    sample = [{
        'student_id': 1001,
        'asof_date': '2025-10-31',
        'top_k': 5,
        'prob_churn': 0.5366243621,
        'reasons': [
            {'feature': 'plan', 'impact': -0.0174, 'effect': 'baja_riesgo', 'value': 'Standard'},
            {'feature': 'payments_90d_usd', 'impact': 0.0174, 'effect': 'sube_riesgo', 'value': 147.0},
            {'feature': 'days_since_last_attendance', 'impact': 0.0164, 'effect': 'sube_riesgo', 'value': 1},
            {'feature': 'price_usd', 'impact': -0.0164, 'effect': 'baja_riesgo', 'value': 49.0},
            {'feature': 'months_since_last_payment', 'impact': -0.0111, 'effect': 'baja_riesgo', 'value': 0.76}
        ]
    }]
    out_path = export_shap_results_to_json(sample, top_k=3)
    print(f"JSON exportado en: {out_path.resolve()}")