# -*- coding: utf-8 -*-
"""
Carga múltiples CSV a MySQL usando mysql.connector.
- Infiero esquema (INT / DECIMAL / DATETIME / DATE / TINYINT / VARCHAR/TEXT).
- Creo tabla si no existe (opcional: DROP & CREATE).
- Inserto en lotes con transacciones.

Requisitos:
  pip install mysql-connector-python

Ejecución:
  python load_csvs_to_mysql.py
"""

import os
import csv
import math
import decimal
from datetime import datetime
from typing import List, Dict, Tuple, Any
import mysql.connector

# ========= CONFIG =========
MYSQL_CFG = {
    "host": "mysql-senu-jhonnybarrios968.b.aivencloud.com",              # p.ej. "mysql-senu-jhonnybarrios968.b.aivencloud.com"
    "port": 15797,                     # p.ej. 15797
    "user": "avnadmin",              # p.ej. "avnadmin"
    "password": "***REDACTED***",
    "database": "Ingles_academia",      # p.ej. "Ingles_academia"
    "autocommit": False,
}

# Si quieres forzar reconstrucción de tablas (DROP & CREATE)
DROP_AND_RECREATE = False

# Tamaño de lote de inserción
BATCH_SIZE = 1000

# Archivos -> tablas
FILES_AND_TABLES: List[Tuple[str, str]] = [
    (r"C:/Users/Windows/Downloads/synthetic_students.csv",     "students"),
    (r"C:/Users/Windows/Downloads/synthetic_teachers.csv",     "teachers"),
    (r"C:/Users/Windows/Downloads/synthetic_payments.csv",     "payments"),
    (r"C:/Users/Windows/Downloads/synthetic_attendance.csv",   "attendance"),
    (r"C:/Users/Windows/Downloads/synthetic_emails.csv",       "emails"),
    (r"C:/Users/Windows/Downloads/churn_training_unified.csv", "churn_training_unified"),
]

# ========= UTIL: detección tipos =========

def try_parse_int(s: str) -> bool:
    try:
        int(s)
        return True
    except:
        return False

def try_parse_decimal(s: str) -> bool:
    try:
        decimal.Decimal(s)
        # Evitar tratar como decimal cosas tipo "00123" si ya es int
        return not try_parse_int(s)
    except:
        return False

# Fechas: probamos algunos formatos comunes; si ninguna, no es fecha
DATE_FORMATS = [
    "%Y-%m-%d",
    "%d/%m/%Y",
    "%m/%d/%Y",
    "%Y/%m/%d",
    "%d-%m-%Y",
    "%m-%d-%Y",
]

DATETIME_FORMATS = [
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S",
    "%d/%m/%Y %H:%M:%S",
    "%m/%d/%Y %H:%M:%S",
    "%Y/%m/%d %H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%Y-%m-%dT%H:%M",
]

def looks_date(s: str) -> bool:
    for fmt in DATE_FORMATS:
        try:
            datetime.strptime(s, fmt)
            return True
        except:
            pass
    return False

def looks_datetime(s: str) -> bool:
    # ISO-like rápido:
    try:
        # Esto cubre 2025-10-14T12:34:56, 2025-10-14 12:34:56, etc.
        datetime.fromisoformat(s.replace("Z", "+00:00"))
        return True
    except:
        pass
    for fmt in DATETIME_FORMATS:
        try:
            datetime.strptime(s, fmt)
            return True
        except:
            pass
    return False

def looks_bool(s: str) -> bool:
    return s.lower() in {"true", "false", "0", "1", "yes", "no", "y", "n"}

def normalize_bool(s: str) -> int:
    return 1 if s.lower() in {"true", "1", "yes", "y"} else 0

def infer_mysql_type(samples: List[str]) -> str:
    """
    Devuelve un tipo MySQL apropiado para una columna dado un muestreo de valores.
    """
    non_nulls = [x for x in samples if x is not None and x != ""]
    if not non_nulls:
        return "VARCHAR(255)"  # por defecto

    # Si todos son boolean-like
    if all(looks_bool(x) for x in non_nulls):
        return "TINYINT(1)"

    # Si todos son INT
    if all(try_parse_int(x) for x in non_nulls):
        # Elegimos BIGINT si hay números grandes
        max_abs = max(abs(int(x)) for x in non_nulls)
        if max_abs <= 2_147_483_647:
            return "INT"
        return "BIGINT"

    # Si todos son DECIMAL/float
    if all(try_parse_decimal(x) or try_parse_int(x) for x in non_nulls):
        # Decimals: tamaño razonable
        return "DECIMAL(18,6)"

    # Si todos parecen DATETIME
    if all(looks_datetime(x) for x in non_nulls):
        return "DATETIME"

    # Si todos parecen DATE
    if all(looks_date(x) for x in non_nulls):
        return "DATE"

    # Si hay textos largos
    maxlen = max(len(x) for x in non_nulls)
    if maxlen > 1000:
        return "LONGTEXT"
    if maxlen > 255:
        return "TEXT"

    return "VARCHAR(255)"

# ========= LECTURA CSV & ESQUEMA =========

def read_csv_rows(path: str, limit_for_infer: int = 1000) -> Tuple[List[str], List[List[Any]]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        headers = next(reader)
        rows = []
        for i, r in enumerate(reader):
            rows.append(r)
            if i + 1 >= limit_for_infer:
                break
    return headers, rows

def full_csv_iterator(path: str):
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        headers = next(reader)
        for r in reader:
            yield r

def build_schema_from_samples(headers: List[str], sample_rows: List[List[str]]) -> Dict[str, str]:
    cols = {h: [] for h in headers}
    for row in sample_rows:
        for h, v in zip(headers, row):
            v = v.strip()
            cols[h].append(v if v != "" else None)

    schema = {}
    for h, samples in cols.items():
        schema[h] = infer_mysql_type(samples)
    return schema

# ========= SQL helpers =========

def quote_ident(ident: str) -> str:
    return f"`{ident.replace('`', '``')}`"

def make_create_table_sql(table: str, schema: Dict[str, str]) -> str:
    cols_sql = []
    for col, coltype in schema.items():
        cols_sql.append(f"{quote_ident(col)} {coltype} NULL")
    # Clave primaria artificial para no depender del CSV
    cols_sql.append("`_row_id` BIGINT NOT NULL AUTO_INCREMENT")
    cols_sql.append("PRIMARY KEY (`_row_id`)")
    cols_clause = ",\n  ".join(cols_sql)
    return f"CREATE TABLE {quote_ident(table)} (\n  {cols_clause}\n) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;"

def make_insert_sql(table: str, headers: List[str]) -> str:
    cols = ", ".join(quote_ident(h) for h in headers)
    placeholders = ", ".join(["%s"] * len(headers))
    return f"INSERT INTO {quote_ident(table)} ({cols}) VALUES ({placeholders})"

def coerce_row(values: List[str], headers: List[str], schema: Dict[str, str]) -> List[Any]:
    out = []
    for h, v in zip(headers, values):
        v = None if v is None else v.strip()
        if v == "":
            v = None
        coltype = schema[h].upper()

        if v is None:
            out.append(None)
            continue

        try:
            if coltype.startswith("TINYINT(1)") and looks_bool(v):
                out.append(normalize_bool(v))
            elif coltype == "INT":
                out.append(int(v))
            elif coltype == "BIGINT":
                out.append(int(v))
            elif coltype.startswith("DECIMAL"):
                out.append(str(decimal.Decimal(v)))
            elif coltype == "DATE":
                # Normalizamos a YYYY-MM-DD si podemos
                if looks_date(v):
                    # intentamos varios formatos
                    dt = None
                    for fmt in DATE_FORMATS:
                        try:
                            dt = datetime.strptime(v, fmt).date()
                            break
                        except:
                            pass
                    out.append(dt.isoformat() if dt else v)
                else:
                    out.append(None)
            elif coltype == "DATETIME":
                # Normalizamos a 'YYYY-MM-DD HH:MM:SS'
                if looks_datetime(v):
                    try:
                        dt = datetime.fromisoformat(v.replace("Z", "+00:00"))
                    except:
                        dt = None
                        for fmt in DATETIME_FORMATS:
                            try:
                                dt = datetime.strptime(v, fmt)
                                break
                            except:
                                pass
                    out.append(dt.strftime("%Y-%m-%d %H:%M:%S") if dt else v)
                else:
                    out.append(None)
            else:
                out.append(v)
        except:
            # Si algo falla en la coerción, lo guardamos como texto
            out.append(v)
    return out

# ========= MAIN PIPELINE =========

def ensure_table(cursor, table: str, headers: List[str], schema: Dict[str, str]):
    cursor.execute(f"SHOW TABLES LIKE %s", (table,))
    exists = cursor.fetchone() is not None

    if exists and DROP_AND_RECREATE:
        print(f"[INFO] Dropping table {table} ...")
        cursor.execute(f"DROP TABLE {quote_ident(table)}")
        exists = False

    if not exists:
        # Crear tabla nueva
        print(f"[INFO] Creating table {table} ...")
        create_sql = make_create_table_sql(table, schema)
        cursor.execute(create_sql)
    else:
        # Verificar columnas faltantes y agregarlas si no existen
        cursor.execute(f"SHOW COLUMNS FROM {quote_ident(table)}")
        existing_cols = {row[0] for row in cursor.fetchall()}  # set de nombres
        to_add = [h for h in headers if h not in existing_cols]
        for col in to_add:
            coltype = schema[col]
            alter = f"ALTER TABLE {quote_ident(table)} ADD COLUMN {quote_ident(col)} {coltype} NULL"
            print(f"[INFO] Altering {table}: ADD COLUMN {col} {coltype}")
            cursor.execute(alter)

def insert_csv(conn, path: str, table: str):
    print(f"\n[JOB] {os.path.basename(path)} -> {table}")

    # 1) Muestra para inferir esquema
    headers, sample_rows = read_csv_rows(path, limit_for_infer=1000)
    schema = build_schema_from_samples(headers, sample_rows)

    with conn.cursor() as cur:
        ensure_table(cur, table, headers, schema)

        insert_sql = make_insert_sql(table, headers)
        batch = []
        count = 0

        for row in full_csv_iterator(path):
            coerced = coerce_row(row, headers, schema)
            batch.append(coerced)

            if len(batch) >= BATCH_SIZE:
                cur.executemany(insert_sql, batch)
                count += len(batch)
                batch.clear()
                print(f"[INFO] Inserted {count} rows into {table} ...")

        if batch:
            cur.executemany(insert_sql, batch)
            count += len(batch)

        print(f"[OK] Total inserted into {table}: {count}")
    conn.commit()

def main():
    # Conexión
    print(f"[CFG] host={MYSQL_CFG['host']} user={MYSQL_CFG['user']} db={MYSQL_CFG['database']} port={MYSQL_CFG['port']}")
    conn = mysql.connector.connect(
        host=MYSQL_CFG["host"],
        port=MYSQL_CFG["port"],
        user=MYSQL_CFG["user"],
        password=MYSQL_CFG["password"],
        database=MYSQL_CFG["database"],
        autocommit=MYSQL_CFG["autocommit"],
    )

    try:
        for path, table in FILES_AND_TABLES:
            if not os.path.exists(path):
                print(f"[WARN] No existe el archivo: {path} (saltando)")
                continue
            insert_csv(conn, path, table)
    finally:
        conn.close()
        print("\n[DONE] Ingesta finalizada.")

if __name__ == "__main__":
    main()
