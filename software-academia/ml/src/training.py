# train_rf.py
import os
import pandas as pd
from sqlalchemy import create_engine, text
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from joblib import dump
from backend.core.database import engine


T0 = "2025-09-21"

# 3) SQL del Paso 3 (tal cual, usando tus nombres de columnas/tablas)
SQL = f"""
WITH params AS ( SELECT DATE('{T0}') AS t0 ),
activos_t0 AS (
  SELECT s.Student_ID AS student_id, 1 AS was_active_t0
  FROM students s
  JOIN params p ON 1=1
  LEFT JOIN payments pay
         ON pay.Student_ID = s.Student_ID
        AND LOWER(pay.Status) = 'paid'
        AND pay.Payment_Date BETWEEN DATE_SUB(p.t0, INTERVAL 30 DAY) AND p.t0
  LEFT JOIN attendance att
         ON att.Student_ID = s.Student_ID
        AND att.Class_Date BETWEEN DATE_SUB(p.t0, INTERVAL 30 DAY) AND p.t0
  WHERE pay.Student_ID IS NOT NULL OR att.Student_ID IS NOT NULL
  GROUP BY s.Student_ID
),
mantiene_next30 AS (
  SELECT s.Student_ID AS student_id, 1 AS kept_signal
  FROM students s
  JOIN params p ON 1=1
  LEFT JOIN payments pay
         ON pay.Student_ID = s.Student_ID
        AND LOWER(pay.Status) = 'paid'
        AND pay.Payment_Date > p.t0
        AND pay.Payment_Date <= DATE_ADD(p.t0, INTERVAL 30 DAY)
  LEFT JOIN attendance att
         ON att.Student_ID = s.Student_ID
        AND att.Class_Date > p.t0
        AND att.Class_Date <= DATE_ADD(p.t0, INTERVAL 30 DAY)
  WHERE pay.Student_ID IS NOT NULL OR att.Student_ID IS NOT NULL
  GROUP BY s.Student_ID
),
att_30 AS (
  SELECT a.Student_ID AS student_id, COUNT(*) AS classes_30d
  FROM attendance a
  JOIN params p ON 1=1
  WHERE a.Class_Date BETWEEN DATE_SUB(p.t0, INTERVAL 30 DAY) AND p.t0
  GROUP BY a.Student_ID
),
att_60 AS (
  SELECT a.Student_ID AS student_id, COUNT(*) AS classes_60d
  FROM attendance a
  JOIN params p ON 1=1
  WHERE a.Class_Date BETWEEN DATE_SUB(p.t0, INTERVAL 60 DAY) AND p.t0
  GROUP BY a.Student_ID
),
last_class AS (
  SELECT a.Student_ID AS student_id, MIN(DATEDIFF(p.t0, a.Class_Date)) AS last_class_days
  FROM attendance a
  JOIN params p ON 1=1
  WHERE a.Class_Date <= p.t0
  GROUP BY a.Student_ID
),
pay_paid_60 AS (
  SELECT pay.Student_ID AS student_id, SUM(pay.Amount) AS paid_sum_60d, COUNT(*) AS paid_cnt_60d
  FROM payments pay
  JOIN params p ON 1=1
  WHERE LOWER(pay.Status)='paid'
    AND pay.Payment_Date BETWEEN DATE_SUB(p.t0, INTERVAL 60 DAY) AND p.t0
  GROUP BY pay.Student_ID
),
pay_failed_60 AS (
  SELECT pay.Student_ID AS student_id, COUNT(*) AS failed_cnt_60d
  FROM payments pay
  JOIN params p ON 1=1
  WHERE LOWER(pay.Status) <> 'paid'
    AND pay.Payment_Date BETWEEN DATE_SUB(p.t0, INTERVAL 60 DAY) AND p.t0
  GROUP BY pay.Student_ID
),
last_paid AS (
  SELECT pay.Student_ID AS student_id, MIN(DATEDIFF(p.t0, pay.Payment_Date)) AS last_payment_days
  FROM payments pay
  JOIN params p ON 1=1
  WHERE LOWER(pay.Status)='paid'
    AND pay.Payment_Date <= p.t0
  GROUP BY pay.Student_ID
),
months_paid_6m AS (
  SELECT pay.Student_ID AS student_id, COUNT(DISTINCT pay.Period_Month) AS months_paid_last_6m
  FROM payments pay
  JOIN params p ON 1=1
  WHERE LOWER(pay.Status)='paid'
    AND pay.Payment_Date BETWEEN DATE_SUB(p.t0, INTERVAL 180 DAY) AND p.t0
  GROUP BY pay.Student_ID
),
first_activity AS (
  SELECT s.Student_ID AS student_id,
         LEAST(
           COALESCE( (SELECT MIN(pay2.Payment_Date) FROM payments pay2 WHERE pay2.Student_ID = s.Student_ID), '2999-12-31'),
           COALESCE( (SELECT MIN(att2.Class_Date)   FROM attendance att2 WHERE att2.Student_ID = s.Student_ID), '2999-12-31')
         ) AS first_seen
  FROM students s
),
tenure AS (
  SELECT f.student_id,
         CASE WHEN f.first_seen = '2999-12-31' THEN NULL
              ELSE DATEDIFF(p.t0, f.first_seen)
         END AS tenure_days
  FROM first_activity f
  JOIN params p ON 1=1
)
SELECT
  a.student_id,
  (SELECT t0 FROM params) AS t0_date,
  CASE WHEN m.kept_signal IS NULL THEN 1 ELSE 0 END AS churn_30d,
  COALESCE(att30.classes_30d, 0)       AS classes_30d,
  COALESCE(att60.classes_60d, 0)       AS classes_60d,
  COALESCE(lc.last_class_days, 9999)   AS last_class_days,
  COALESCE(pp.paid_sum_60d, 0.0)       AS paid_sum_60d,
  COALESCE(pp.paid_cnt_60d, 0)         AS paid_cnt_60d,
  COALESCE(pf.failed_cnt_60d, 0)       AS failed_cnt_60d,
  COALESCE(lp.last_payment_days, 9999) AS last_payment_days,
  COALESCE(mp.months_paid_last_6m, 0)  AS months_paid_last_6m,
  COALESCE(tn.tenure_days, 0)          AS tenure_days
FROM activos_t0 a
LEFT JOIN mantiene_next30 m ON m.student_id = a.student_id
LEFT JOIN att_30 att30      ON att30.student_id = a.student_id
LEFT JOIN att_60 att60      ON att60.student_id = a.student_id
LEFT JOIN last_class lc     ON lc.student_id = a.student_id
LEFT JOIN pay_paid_60 pp    ON pp.student_id = a.student_id
LEFT JOIN pay_failed_60 pf  ON pf.student_id = a.student_id
LEFT JOIN last_paid lp      ON lp.student_id = a.student_id
LEFT JOIN months_paid_6m mp ON mp.student_id = a.student_id
LEFT JOIN tenure tn         ON tn.student_id = a.student_id
;
"""

print("Ejecutando SQL y cargando dataset...")
df = pd.read_sql(text(SQL), engine)