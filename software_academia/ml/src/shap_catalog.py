# software_academia/ml/src/shap_stage1.py
from __future__ import annotations
from typing import List, Optional
import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sqlmodel import Session

# Importa TUS utilidades existentes (tal cual las pasaste)
from software_academia.ml.src.scoring import (
    pipe as _pipe,
)
from software_academia.ml.src.feature_builder import (
    infer_asof_date,
)
from software_academia.ml.src.shap_utils import (
    get_preprocessor_and_estimator,
    build_background_matrix,
)

# ---- 1) Versión pública del mapeo índice->feature original (idéntico a tu helper interno) ----
def _last_step(transformer):
    if isinstance(transformer, Pipeline) and transformer.steps:
        return transformer.steps[-1][1]
    return transformer

def map_orig_feature_by_index(preproc: ColumnTransformer, input_cols: List[str]) -> List[str]:
    """
    Devuelve una lista de largo = n_cols_transformadas indicando la COLUMNA ORIGINAL
    que generó cada columna transformada. Respeta OneHotEncoder (drop_idx_), passthrough, drop,
    y remainder='passthrough'.
    """
    orig_names_by_pos: List[str] = []
    assigned = set()

    # Transformadores explícitos
    for name, trans, cols in preproc.transformers_:
        if name == "remainder":
            continue
        cols = list(cols) if isinstance(cols, (list, tuple)) else [cols]
        t = _last_step(trans)
        assigned.update(cols)

        if isinstance(t, OneHotEncoder):
            cats_list = t.categories_
            drop_idx = getattr(t, "drop_idx_", None)
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
            if t == "passthrough":
                orig_names_by_pos.extend(cols)
            elif t == "drop":
                continue
            else:
                # Scaler/Imputer/etc.: 1:1
                orig_names_by_pos.extend(cols)

    # Remainder contra input_cols REALES
    for name, trans, cols in preproc.transformers_:
        if name == "remainder" and trans == "passthrough":
            remainder_cols = [c for c in input_cols if c not in assigned]
            orig_names_by_pos.extend(remainder_cols)
            break

    return orig_names_by_pos


# ---- 2) Normalizador robusto de SHAP para dejarlo SIEMPRE en 2D (n_muestras, n_features) ----
def _normalize_shap_to_2d(sv_raw) -> np.ndarray:
    """
    Acepta:
      - lista por clase (formato clásico de shap para clasificación) -> toma clase positiva si existe
      - array 3D (n, f, c) -> toma última dimensión como clase positiva
      - array 2D (n, f)
    Devuelve: np.ndarray de shape (n_muestras, n_features)
    """
    if isinstance(sv_raw, list):
        if len(sv_raw) >= 2:
            return np.asarray(sv_raw[1])  # clase positiva
        return np.asarray(sv_raw[0])
    arr = np.asarray(sv_raw)
    if arr.ndim == 3:
        # (n, f, c) -> toma clase positiva (último índice)
        return arr[:, :, -1]
    if arr.ndim == 2:
        return arr
    raise ValueError(f"Forma inesperada de shap_values: {arr.shape}")


# ---- 3) FUNCIÓN PRINCIPAL: Catálogo GLOBAL de razones (features ORIGINALES) ----
def compute_global_shap_catalog(
    session: Session,
    asof_date: Optional[str] = None,
    sample_size: int = 300,
) -> pd.DataFrame:
    """
    Calcula importancia GLOBAL de SHAP agregada por columna ORIGINAL.

    Retorna DataFrame con columnas:
      - orig_feature
      - global_sum_abs_shap
      - global_mean_abs_shap
      - global_net
      - rank
    """
    if asof_date is None:
        asof_date = infer_asof_date(session)

    preproc, est = get_preprocessor_and_estimator(_pipe)
    if preproc is None:
        raise RuntimeError("No se encontró preprocesador en el pipeline")

    # Matriz de fondo + nombres de entrada reales (tus utilidades)
    X_bg, X_bg_trans, _ = build_background_matrix(session, asof_date, sample_size=sample_size)
    input_cols = list(X_bg.columns)

    # Mapeo índice->feature original
    orig_by_idx = map_orig_feature_by_index(preproc, input_cols)

    # Alineación longitudes
    n_trans = X_bg_trans.shape[1]
    if len(orig_by_idx) != n_trans:
        if len(orig_by_idx) > n_trans:
            orig_by_idx = orig_by_idx[:n_trans]
        else:
            orig_by_idx = orig_by_idx + (["<unk>"] * (n_trans - len(orig_by_idx)))

    # SHAP global sobre el estimador
    explainer = shap.TreeExplainer(est)
    sv_raw = explainer.shap_values(X_bg_trans)
    sv = _normalize_shap_to_2d(sv_raw)  # <- robusto 2D

    # Sanidad de dimensiones
    if sv.shape[1] != len(orig_by_idx):
        m = min(sv.shape[1], len(orig_by_idx))
        sv = sv[:, :m]
        orig_by_idx = orig_by_idx[:m]

    # Construcción de stats por columna transformada
    df = pd.DataFrame(sv, columns=[f"t_{i}" for i in range(sv.shape[1])])
    abs_df = df.abs()
    mapping = pd.DataFrame({"t_col": abs_df.columns, "orig_feature": orig_by_idx})

    sum_abs_by_t = abs_df.sum(axis=0).rename("sum_abs")
    mean_abs_by_t = abs_df.mean(axis=0).rename("mean_abs")
    net_mean_by_t = df.mean(axis=0).rename("net_mean")

    stats = pd.concat([sum_abs_by_t, mean_abs_by_t, net_mean_by_t], axis=1).reset_index().rename(columns={"index": "t_col"})
    stats = stats.merge(mapping, on="t_col", how="left")

    agg = (stats.groupby("orig_feature", as_index=False)
                 .agg(global_sum_abs_shap=("sum_abs", "sum"),
                      global_mean_abs_shap=("mean_abs", "mean"),
                      global_net=("net_mean", "mean"))
                 .sort_values("global_sum_abs_shap", ascending=False))

    # Limpieza opcional
    agg = agg[agg["orig_feature"] != "<unk>"].copy()
    agg["rank"] = np.arange(1, len(agg) + 1)
    return agg



from sqlmodel import Session
from software_academia.backend.app.core.database import engine
from software_academia.ml.src.feature_builder import infer_asof_date

with Session(engine) as session:
    asof = infer_asof_date(session)
    df_catalog = compute_global_shap_catalog(session, asof_date=asof, sample_size=300)
    print(df_catalog.head(15)[["orig_feature","global_sum_abs_shap","global_mean_abs_shap","rank"]])

from typing import Dict

SHAP_REASON_CATALOG: Dict[str, str] = {
    "months_since_last_payment": "atraso_pago",
    "payments_90d_usd": "bajo_pago_reciente",
    "plan": "plan_inadecuado",
    "price_usd": "precio_alto",
    "total_classes_90d": "bajo_uso_3m",
    "days_since_last_attendance": "inactividad_reciente",
    "total_classes_60d": "bajo_uso_2m",
    "total_classes_30d": "bajo_uso_1m",
}

def translate_feature_to_reason(feature_name: str) -> str:
    return SHAP_REASON_CATALOG.get(feature_name, feature_name)