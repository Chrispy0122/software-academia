import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score
import joblib  # opcional si quieres guardar el modelo

# Ajusta la ruta a tu archivo
CHURN_TRAIN = "C:/Users/Windows/Downloads/churn_training_unified.csv"

df = pd.read_csv(CHURN_TRAIN)

# Tu columna objetivo:
TARGET = "abandono"   # debe ser 0/1
assert TARGET in df.columns, "No encuentro la columna 'abandono' (target)"
df[TARGET] = df[TARGET].astype(int)

drop_cols = [
    TARGET,
    "asof_date", "customer_id", "first_name", "last_name", "email", "phone",
    "signup_date", "last_attendance_date",
    # muy propensas a fuga si existen:
    "is_active", "paid_current_month", "last_payment_date"
]

cols_to_drop = [c for c in drop_cols if c in df.columns]
X = df.drop(columns=cols_to_drop, errors="ignore").copy()
y = df[TARGET].copy()

# Tip: si quedaron fechas en X, conviene quitarlas o convertirlas a "recencias" antes de entrenar.
date_like = [c for c in X.columns if "date" in c.lower() or "fecha" in c.lower()]
if date_like:
    X = X.drop(columns=date_like, errors="ignore")

numeric_cols = [c for c in X.columns if np.issubdtype(X[c].dtype, np.number)]
categorical_cols = [c for c in X.columns if c not in numeric_cols]

pre = ColumnTransformer(
    transformers=[
        ("num", "passthrough", numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ]
)

rf = RandomForestClassifier(
    n_estimators=600,
    max_depth=None,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

pipe = Pipeline([("pre", pre), ("rf", rf)])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
auc_scores = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc")
print("AUC por fold:", np.round(auc_scores, 4))
print("AUC promedio:", np.round(auc_scores.mean(), 4))

pipe.fit(X, y)

# Guardar para usar después (opcional)
joblib.dump({
    "model": pipe,
    "feature_order": list(X.columns),
    "categorical_cols": categorical_cols,
    "numeric_cols": numeric_cols
}, "model_rf.pkl")

print("Modelo entrenado y guardado como model_rf.pkl")
