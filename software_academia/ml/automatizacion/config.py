# config.py (versión simple, sin labels ni threading)

from __future__ import annotations
from pathlib import Path
import os

# =========
# Archivos / Datos
# =========
# Tu JSON normalizado desde shap_utils.py
SHAP_RESULTS_JSON = Path(
    os.getenv(
        "SHAP_RESULTS_JSON",
        r"C:/Users/Windows/software-academia-2/software_academia/ml/automatizacion/output/shap_results.json"
    )
)

# Cuántas razones máximo por cliente (las que SUBEN riesgo)
TOP_RISK_REASONS = int(os.getenv("TOP_RISK_REASONS", "3"))

# Umbrales de riesgo
TIER_HIGH = float(os.getenv("TIER_HIGH", "0.70"))
TIER_MED  = float(os.getenv("TIER_MED", "0.40"))

# =========
# Runtime
# =========
# ÚNICO flag global: si True NO enviamos correos reales ni llamamos a OpenAI real
DRY_RUN   = os.getenv("DRY_RUN", "true").strip().lower() in ("1", "true", "yes", "y")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# =========
# OpenAI (para writer)
# =========
OPENAI_API_KEY      = os.getenv("OPENAI_API_KEY")  # pon tu key en variable de entorno
OPENAI_MODEL        = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_TIMEOUT_SECS = int(os.getenv("OPENAI_TIMEOUT_SECS", "30"))
OPENAI_MAX_RETRIES  = int(os.getenv("OPENAI_MAX_RETRIES", "2"))

# =========
# Gmail (solo ENVIAR correos)
# =========
# Cuenta remitente (la misma con la que harás OAuth)
GMAIL_SENDER = os.getenv("GMAIL_SENDER", "tu_cuenta@tu_dominio.com")

# Rutas a credenciales OAuth de Google:
# - credentials.json: secreto de cliente (descargado de Google Cloud Console)
# - token.json: se genera en el primer login OAuth (y se reusa luego)
GOOGLE_CREDENTIALS_PATH = Path(os.getenv("GOOGLE_CREDENTIALS_PATH", "./credentials.json"))
GOOGLE_TOKEN_PATH       = Path(os.getenv("GOOGLE_TOKEN_PATH", "./token.json"))

# Scope mínimo para ENVIAR correos
GMAIL_SCOPES = ["https://www.googleapis.com/auth/gmail.send"]

# =========
# Google Drive (OPCIONAL — para guardar archivos/reportes)
# =========
ENABLE_DRIVE = os.getenv("ENABLE_DRIVE", "false").strip().lower() in ("1", "true", "yes", "y")
DRIVE_SCOPES = ["https://www.googleapis.com/auth/drive.file"]
DRIVE_FOLDER_ID = os.getenv("DRIVE_FOLDER_ID", "")  # déjalo vacío si no lo usas
