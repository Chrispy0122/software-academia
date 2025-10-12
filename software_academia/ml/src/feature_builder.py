# ============================================================
# Churn features + scoring usando SQLModel Session + SQL text
# ============================================================
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from sqlalchemy import text
from sqlmodel import Session

# ---------------------------
# 1) Utilidad: leer a DataFrame
# ---------------------------

def df_from_sql(session: Session, sql: str, params: dict | None = None) -> pd.DataFrame:
    stmt = text(sql)
    if params:
        stmt = stmt.bindparams(**params)   # ✅ kwargs
    res = session.exec(stmt)               # ✅ un solo argumento
    rows = res.mappings().all()
    return pd.DataFrame(rows)



# ---------------------------
# 2) Inferir asof_date desde la BD
# ---------------------------
def infer_asof_date(session: Session, attendance_table="attendance", payments_table="payments", schema: str | None = None) -> str:
    def T(name): return f"{schema}.{name}" if schema else name

    q_att = f"SELECT MAX(Class_Date) AS max_att FROM {T(attendance_table)}"
    q_pay = f"SELECT MAX(Payment_Date) AS max_pay FROM {T(payments_table)}"

    att = df_from_sql(session, q_att)
    pay = df_from_sql(session, q_pay)

    max_att = pd.to_datetime(att["max_att"].iloc[0]) if not att.empty else pd.NaT
    max_pay = pd.to_datetime(pay["max_pay"].iloc[0]) if not pay.empty else pd.NaT

    asof = max(max_att, max_pay)
    if pd.isna(asof):
        raise RuntimeError("No pude inferir asof_date (no hay fechas en attendance/payments).")
    return asof.strftime("%Y-%m-%d")

# ---------------------------
# 3) Builder de features (SQL + Session)
# ---------------------------
def build_features_from_db_sqlmodel_session(
    session: Session,
    asof_date: str,
    students_table: str = "students",
    attendance_table: str = "attendance",
    payments_table: str = "payments",
    schema: str | None = None,
) -> pd.DataFrame:
    """
    Construye features por Student_ID usando SOLO historia <= asof_date - 1 día.
    Calcula agregados en SQL y deriva recencias en Python (sin fuga).
    """
    asof = pd.to_datetime(asof_date)
    cutoff = asof - pd.Timedelta(days=1)
    cutoff_s = cutoff.strftime("%Y-%m-%d")
    def T(name): return f"{schema}.{name}" if schema else name

    # --- BASE: students ---
    students = df_from_sql(session, f"""
        SELECT Student_ID, Signup_Date, Plan, Price_USD
        FROM {T(students_table)}
    """)
    if students.empty:
        raise RuntimeError("Tabla students vacía.")
    students["Signup_Date"] = pd.to_datetime(students["Signup_Date"], errors="coerce")
    students["tenure_days"] = (cutoff - students["Signup_Date"]).dt.days.clip(lower=0)
    students["plan"] = students["Plan"].astype(str)
    students["price_usd"] = pd.to_numeric(students["Price_USD"], errors="coerce").fillna(0)

    # --- ATTENDANCE agregada en SQL (última clase y ventanas 30/60/90) ---
    attendance_agg = df_from_sql(session, f"""
        WITH hist AS (
         SELECT Student_ID, Class_Date, COALESCE(Present, 1) AS Present
         FROM attendance
         WHERE Class_Date <= :cutoff
        ),
        last_att AS (
            SELECT Student_ID, MAX(Class_Date) AS last_class
            FROM hist GROUP BY Student_ID
        ),
        win30 AS (
            SELECT Student_ID, SUM(Present) AS classes_30d
            FROM hist
            WHERE Class_Date BETWEEN DATE_SUB(:cutoff, INTERVAL 29 DAY) AND :cutoff
            GROUP BY Student_ID
         ),
        win60 AS (
            SELECT Student_ID, SUM(Present) AS classes_60d
            FROM hist
            WHERE Class_Date BETWEEN DATE_SUB(:cutoff, INTERVAL 59 DAY) AND :cutoff
            GROUP BY Student_ID
        ),
        win90 AS (
            SELECT Student_ID, SUM(Present) AS classes_90d
            FROM hist
            WHERE Class_Date BETWEEN DATE_SUB(:cutoff, INTERVAL 89 DAY) AND :cutoff
            GROUP BY Student_ID
        )
        SELECT s.Student_ID,
               last_att.last_class,
               COALESCE(win30.classes_30d,0) AS total_classes_30d,
               COALESCE(win60.classes_60d,0) AS total_classes_60d,
               COALESCE(win90.classes_90d,0) AS total_classes_90d
        FROM (SELECT DISTINCT Student_ID FROM students) s
        LEFT JOIN last_att ON s.Student_ID = last_att.Student_ID
        LEFT JOIN win30    ON s.Student_ID = win30.Student_ID
        LEFT JOIN win60    ON s.Student_ID = win60.Student_ID
        LEFT JOIN win90    ON s.Student_ID = win90.Student_ID
         """, {"cutoff": cutoff_s})

    # --- PAYMENTS agregada en SQL (último pago y 90d) ---
    payments_agg = df_from_sql(session, f"""
        WITH pay AS (
          SELECT Student_ID, Payment_Date, Amount, Status, Payment_ID
          FROM {T(payments_table)}
          WHERE Payment_Date <= :cutoff
        ),
        ok AS (
          SELECT * FROM pay
          WHERE COALESCE(LOWER(Status), 'paid') IN ('completed','paid','success','approved','ok')
        ),
        last_pay AS (
          SELECT Student_ID, MAX(Payment_Date) AS last_payment
          FROM ok GROUP BY Student_ID
        ),
        win90 AS (
          SELECT Student_ID,
                 SUM(Amount) AS payments_90d_usd,
                 COUNT(Payment_ID) AS n_payments_90d
          FROM ok
          WHERE Payment_Date BETWEEN DATE_SUB(:cutoff, INTERVAL 89 DAY) AND :cutoff
          GROUP BY Student_ID
        )
        SELECT s.Student_ID,
               last_pay.last_payment,
               COALESCE(win90.payments_90d_usd,0) AS payments_90d_usd,
               COALESCE(win90.n_payments_90d,0)  AS n_payments_90d
        FROM (SELECT DISTINCT Student_ID FROM {T(students_table)}) s
        LEFT JOIN last_pay ON s.Student_ID = last_pay.Student_ID
        LEFT JOIN win90    ON s.Student_ID = win90.Student_ID
    """, {"cutoff": cutoff_s})

    # --- UNIR + DERIVADOS ---
    out = students.merge(attendance_agg, on="Student_ID", how="left") \
                  .merge(payments_agg,   on="Student_ID", how="left")

    out["last_class"] = pd.to_datetime(out["last_class"], errors="coerce")
    out["days_since_last_attendance"] = (cutoff - out["last_class"]).dt.days
    out["days_since_last_attendance"] = out["days_since_last_attendance"].fillna(9999).astype(int)

    out["last_payment"] = pd.to_datetime(out["last_payment"], errors="coerce")
    out["months_since_last_payment"] = ((cutoff - out["last_payment"]).dt.days / 30.0).fillna(9999.0)

    # Selección final (mismo contrato de features que tu modelo espera)
    keep = [
    "Student_ID","plan","price_usd","tenure_days",
    "days_since_last_attendance","total_classes_30d","total_classes_60d","total_classes_90d",
    "months_since_last_payment","payments_90d_usd","n_payments_90d"
     ]

    out = out[keep].fillna(0)
    out["plan"] = out["plan"].astype(str)
    return out

# ---------------------------
# 4) Scoring batch (para UI o jobs)
# ---------------------------
def score_batch_from_db(
    session: Session,
    model_path: str,
    asof_date: str | None = None,
    students_table="students",
    attendance_table="attendance",
    payments_table="payments",
    schema: str | None = None,
) -> pd.DataFrame:
    """
    Recalcula features desde la BD y devuelve un DataFrame con Student_ID, asof_date y churn_risk.
    No escribe en DB: pensado para mostrar en la interfaz (probabilidad en vivo).
    """
    if asof_date is None:
        asof_date = infer_asof_date(session, attendance_table, payments_table, schema)

    # 1) Features desde BD
    feat = build_features_from_db_sqlmodel_session(
        session, asof_date,
        students_table=students_table,
        attendance_table=attendance_table,
        payments_table=payments_table,
        schema=schema
    )

    # 2) Cargar modelo + contrato
    bundle = joblib.load(model_path)
    pipe = bundle["model"]
    expected = bundle["feature_order"]
    cat_cols = bundle["categorical_cols"]
    num_cols = bundle["numeric_cols"]

    # 3) Alinear columnas y tipos
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

    # 4) Predecir
    proba = pipe.predict_proba(X)[:, 1]
    scores = pd.DataFrame({
        "Student_ID": ids,
        "asof_date": asof_date,
        "churn_risk": proba
    }).sort_values("churn_risk", ascending=False)
    return scores

# ---------------------------
# 5) Scoring de un alumno (para endpoint /students/{id}/score)
# ---------------------------
def score_one_student_from_db(
    session: Session,
    model_path: str,
    student_id: int,
    asof_date: str | None = None,
    students_table="students",
    attendance_table="attendance",
    payments_table="payments",
    schema: str | None = None,
) -> float:
    if asof_date is None:
        asof_date = infer_asof_date(session, attendance_table, payments_table, schema)

    feat = build_features_from_db_sqlmodel_session(
        session, asof_date,
        students_table=students_table,
        attendance_table=attendance_table,
        payments_table=payments_table,
        schema=schema
    )
    row = feat.loc[feat["Student_ID"] == student_id].copy()
    if row.empty:
        raise ValueError("Student_ID no encontrado en BD.")

    bundle = joblib.load(model_path)
    pipe = bundle["model"]
    expected = bundle["feature_order"]
    cat_cols = bundle["categorical_cols"]
    num_cols = bundle["numeric_cols"]

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


