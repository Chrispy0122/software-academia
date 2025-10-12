import csv
import re
import os
import sys
import datetime as dt
import mysql.connector

# ===================== CONFIG (con fallback) =====================
DEFAULT_DB_NAME = "Ingles_academia"
BATCH_SIZE = 2000

# Rutas ABSOLUTAS (6 archivos)
CSV_STUDENTS   = r"C:/Users/Windows/Downloads/synthetic_students.csv"
CSV_TEACHERS   = r"C:/Users/Windows/Downloads/synthetic_teachers.csv"
CSV_PAYMENTS   = r"C:/Users/Windows/Downloads/synthetic_payments.csv"
CSV_ATTENDANCE = r"C:/Users/Windows/Downloads/synthetic_attendance.csv"
CSV_EMAILS     = r"C:/Users/Windows/Downloads/synthetic_emails.csv"
CSV_CHURN      = r"C:/Users/Windows/Downloads/training_panel_template.csv"  # tu panel/plantilla

# Variables de entorno (con valores por defecto si no están seteadas)
MYSQL_HOST = os.getenv("DB_HOST")
MYSQL_USER = os.getenv("DB_USER")
MYSQL_PASS = os.getenv("DB_PASSWORD")
MYSQL_PORT = os.getenv("DB_PORT")
DB_NAME    = os.getenv("DB_NAME")
# ================================================================

def qi(name: str) -> str:
    return "`" + str(name).replace("`", "``") + "`"

# ------------------- Parsers & type guess -------------------
NULLS = {"", "na", "n/a", "null", "none", "nan", "nil"}

DATE_FMTS = ("%Y-%m-%d","%d/%m/%Y","%m/%d/%Y","%Y/%m/%d","%d-%m-%Y")
DT_FMTS   = ("%Y-%m-%d %H:%M:%S","%Y-%m-%d %H:%M","%d/%m/%Y %H:%M:%S","%m/%d/%Y %H:%M:%S")

def parse_date(s):
    if s is None: return None
    t = str(s).strip()
    if t.lower() in NULLS: return None
    try:
        return dt.date.fromisoformat(t).isoformat()
    except Exception:
        for f in DATE_FMTS:
            try: return dt.datetime.strptime(t, f).date().isoformat()
            except Exception: pass
    return None

def parse_datetime(s):
    if s is None: return None
    t = str(s).strip()
    if t.lower() in NULLS: return None
    for f in DT_FMTS:
        try: return dt.datetime.strptime(t, f)
        except Exception: pass
    try:
        return dt.datetime.fromisoformat(t.replace("Z","+00:00")).replace(tzinfo=None)
    except Exception:
        return None

def looks_int(s):   return re.fullmatch(r"[+-]?\d+", s) is not None
def looks_float(s): return re.fullmatch(r"[+-]?\d+\.\d+", s) is not None

def clean_money_like(s):
    t = re.sub(r"[^\d,.\-]", "", s)
    if t.count(",") > 0 and t.count(".") == 0:
        t = t.replace(",", ".")
    if t.count(".") > 1:
        parts = t.split("."); t = "".join(parts[:-1]) + "." + parts[-1]
    return t

def guess_type(colname, sample_vals):
    lower = colname.lower()
    # Heurísticas por nombre
    if lower.endswith("_id") or lower == "id" or lower in {"student_id","teacher_id","payment_id","attendance_id","email_id"}:
        return ("BIGINT", "int")
    if any(k in lower for k in ("amount","price","usd","monto","importe","sum_","avg_")):
        return ("DECIMAL(10,2)", "float")
    if "date" in lower or "fecha" in lower or lower.endswith("_at"):
        # prefer DATETIME si vemos horas
        for v in sample_vals:
            if v is None: continue
            if parse_datetime(v): return ("DATETIME", "datetime")
        return ("DATE", "date")
    if lower.startswith("is_") or lower in {"present","active","isactive","is_active"}:
        return ("TINYINT(1)", "bool")

    has_text=False; has_float=False; has_int=True
    maxlen = 0
    for v in sample_vals:
        if v is None: continue
        s = str(v).strip()
        if s.lower() in NULLS: continue
        maxlen = max(maxlen, len(s))
        if looks_int(s):
            continue
        elif looks_float(clean_money_like(s)):
            has_float = True
            has_int = False
        elif parse_date(s) or parse_datetime(s):
            has_text = True
            has_int = False
        else:
            has_text = True
            has_int = False
    if not has_text and not has_float and has_int:
        return ("BIGINT", "int")
    if has_float and not has_text:
        return ("DECIMAL(10,2)", "float")
    if maxlen <= 255:   return ("VARCHAR(255)", "text")
    if maxlen <= 1024:  return ("VARCHAR(1024)", "text")
    return ("TEXT", "text")

def convert_value(py_label, raw):
    if raw is None: return None
    s = str(raw).strip()
    if s.lower() in NULLS: return None
    if py_label == "int":
        try: return int(s)
        except:
            try: return int(float(clean_money_like(s)))
            except: return None
    if py_label == "float":
        t = clean_money_like(s)
        try: return float(t)
        except: return None
    if py_label == "bool":
        return 1 if s.lower() in {"1","true","yes","y","si","sí"} else 0 if s.lower() in {"0","false","no","n"} else None
    if py_label == "date":
        return parse_date(s)
    if py_label == "datetime":
        d = parse_datetime(s)
        return d.strftime("%Y-%m-%d %H:%M:%S") if d else None
    return s  # texto

# ------------------- MySQL helpers -------------------
def connect_mysql():
    return mysql.connector.connect(
        host=MYSQL_HOST, user=MYSQL_USER, password=MYSQL_PASS, port=MYSQL_PORT,
        charset="utf8mb4", use_unicode=True
    )

def ensure_database(cur):
    cur.execute(f"CREATE DATABASE IF NOT EXISTS {qi(DB_NAME)} DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;")
    cur.execute(f"USE {qi(DB_NAME)};")

def drop_if_exists(cur, tname):
    cur.execute(f"DROP TABLE IF EXISTS {qi(tname)};")

# === versión con PK obligatoria (si no hay, se crea _row_id) ===
def create_table_from_csv(cur, table_name, header, sample_rows, all_rows_for_pk):
    # Inferir tipos
    col_types, py_labels = {}, {}
    for col in header:
        sample_vals = [r.get(col) for r in sample_rows]
        sqlt, pyl = guess_type(col, sample_vals)
        col_types[col] = sqlt
        py_labels[col] = pyl

    # Detectar candidata a PK (flexible, case-insensitive)
    pk_col = None
    candidates_by_name = ["student_id","teacher_id","payment_id","attendance_id","email_id","id"]
    header_lower_map = {c.lower(): c for c in header}
    for cand in candidates_by_name:
        if cand in header_lower_map:
            pk_col = header_lower_map[cand]; break
    if pk_col is None:
        # última opción: cualquier columna que termine en _id
        for c in header:
            if c.lower().endswith("_id"):
                pk_col = c; break

    # Verificar unicidad para usar como PK
    use_primary = False
    add_unique  = False
    if pk_col:
        seen, has_null, has_dup = set(), False, False
        for r in all_rows_for_pk:
            v = r.get(pk_col)
            if v is None or str(v).strip()=="" or str(v).strip().lower() in NULLS:
                has_null = True
                continue
            k = str(v).strip()
            if k in seen: has_dup = True
            else: seen.add(k)
        if not has_null and not has_dup:
            use_primary = True
        else:
            add_unique = True

    cols_sql = []
    add_surrogate_pk = not use_primary
    if add_surrogate_pk:
        cols_sql.append("`_row_id` BIGINT NOT NULL AUTO_INCREMENT")

    for c in header:
        nullability = "NOT NULL" if (use_primary and c == pk_col) else "NULL"
        cols_sql.append(f"{qi(c)} {col_types[c]} {nullability}")

    ddl = f"CREATE TABLE {qi(table_name)} (\n  " + ",\n  ".join(cols_sql)
    ddl += f",\n  PRIMARY KEY ({qi(pk_col)})" if use_primary else f",\n  PRIMARY KEY (`_row_id`)"
    ddl += f"\n) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;"
    cur.execute(ddl)

    if add_unique and pk_col:
        try:
            cur.execute(f"CREATE UNIQUE INDEX {qi('uniq_'+table_name+'_'+pk_col)} ON {qi(table_name)}({qi(pk_col)});")
        except Exception:
            pass

    return py_labels

def bulk_insert(cur, table_name, header, py_labels, rows_iter):
    placeholders = ", ".join(["%s"] * len(header))
    cols_sql = ", ".join(qi(c) for c in header)
    sql = f"INSERT INTO {qi(table_name)} ({cols_sql}) VALUES ({placeholders}) " \
          f"ON DUPLICATE KEY UPDATE " + ", ".join(f"{qi(c)}=VALUES({qi(c)})" for c in header)
    batch, total = [], 0
    pyl = [py_labels[c] for c in header]
    for r in rows_iter:
        conv = [convert_value(pyl[i], r.get(header[i])) for i in range(len(header))]
        batch.append(conv)
        if len(batch) >= BATCH_SIZE:
            cur.executemany(sql, batch); total += len(batch); batch.clear()
    if batch:
        cur.executemany(sql, batch); total += len(batch)
    return total

def read_csv(path, sample_limit=500):
    # intentamos utf-8-sig y luego latin-1
    for enc in ("utf-8-sig","utf-8","latin-1"):
        try:
            with open(path, "r", newline="", encoding=enc) as fh:
                rdr = csv.DictReader(fh)
                header = rdr.fieldnames or []
                sample_rows, rows = [], []
                for i, row in enumerate(rdr):
                    if i < sample_limit: sample_rows.append(row)
                    rows.append(row)
            return header, sample_rows, rows
        except Exception:
            continue
    raise RuntimeError(f"No pude leer el CSV: {path}")

def add_fk_if_possible(cur, child_table, child_col, parent_table, parent_col, on_delete="RESTRICT"):
    cur.execute(f"SHOW COLUMNS FROM {qi(child_table)} LIKE %s;", (child_col,))
    if not cur.fetchall(): return
    cur.execute(f"SHOW COLUMNS FROM {qi(parent_table)} LIKE %s;", (parent_col,))
    if not cur.fetchall(): return
    try:
        cur.execute(f"CREATE INDEX {qi('idx_'+child_table+'_'+child_col)} ON {qi(child_table)}({qi(child_col)});")
    except Exception:
        pass
    fk_name = f"fk_{child_table}_{child_col}_{parent_table}_{parent_col}"
    try:
        cur.execute(f"ALTER TABLE {qi(child_table)} ADD CONSTRAINT {qi(fk_name)} "
                    f"FOREIGN KEY ({qi(child_col)}) REFERENCES {qi(parent_table)}({qi(parent_col)}) "
                    f"ON DELETE {on_delete} ON UPDATE CASCADE;")
    except Exception:
        pass

# ------------------- Pipeline -------------------
def build_table_from_file(cur, path, logical_name):
    print(f"[INFO] {logical_name}: creando tabla igual al CSV…")
    header, sample_rows, rows = read_csv(path)
    if not header:
        print(f"[WARN] {logical_name}: CSV sin encabezado. Saltado.")
        return 0
    drop_if_exists(cur, logical_name)
    py_labels = create_table_from_csv(cur, logical_name, header, sample_rows, rows)
    inserted = bulk_insert(cur, logical_name, header, py_labels, rows)
    print(f"[OK] {logical_name}: {inserted} filas.")
    return inserted

def main():
    print(f"[CFG] Host={MYSQL_HOST} User={MYSQL_USER} Port={MYSQL_PORT} DB={DB_NAME}")
    cnx = connect_mysql()
    cur = cnx.cursor()
    ensure_database(cur)

    # Tablas base
    build_table_from_file(cur, CSV_STUDENTS,   "students")
    build_table_from_file(cur, CSV_TEACHERS,   "teachers")
    cnx.commit()

    # Dependientes + churn/panel
    build_table_from_file(cur, CSV_PAYMENTS,   "payments")
    build_table_from_file(cur, CSV_ATTENDANCE, "attendance")
    build_table_from_file(cur, CSV_EMAILS,     "emails")
    build_table_from_file(cur, CSV_CHURN,      "training_panel")  # nombre lógico más claro
    cnx.commit()

    # FKs (si existen columnas compatibles)
    # Nota: intenta con minísculas por defecto; ajusta si tus CSV usan otra convención
    add_fk_if_possible(cur, "payments",   "student_id", "students", "student_id", on_delete="RESTRICT")
    add_fk_if_possible(cur, "attendance", "student_id", "students", "student_id", on_delete="SET NULL")
    add_fk_if_possible(cur, "emails",     "student_id", "students", "student_id", on_delete="SET NULL")

    cnx.commit()

    for t in ["students","teachers","payments","attendance","emails","training_panel"]:
        try:
            cur.execute(f"SELECT COUNT(*) FROM {qi(t)};")
            print(f" - {t}: {cur.fetchone()[0]} filas")
        except Exception as e:
            print(f" - {t}: (no se pudo contar) {e}")

    cur.close()
    cnx.close()
    print("✅ Listo: tablas creadas 1:1 con sus CSV, PK garantizada y upsert activo.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("[FATAL]", e)
        sys.exit(1)
