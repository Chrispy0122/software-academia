# reasons.py
from typing import List

# 1) Diccionario: feature -> razón legible
REASON_MAP = {
    # Pagos
    "failed_payments_60d": "Problemas de pago",
    "days_since_last_payment": "Problemas de pago",

    # Uso / asistencia
    "classes_30d": "Bajo uso / no asiste",
    "last_class_days": "Bajo uso / no asiste",
    "logins_30d": "Bajo uso / no asiste",

    # Emails / comunicación
    "emails_open_30d": "No responde a emails",
    "emails_click_30d": "No responde a emails",

    # Precio / plan
    "price_usd": "Precio/plan inadecuado",
    "plan": "Precio/plan inadecuado",

    # Antigüedad
    "tenure_days": "Etapa del ciclo de vida",
}

# 2) Si una feature no está en el diccionario:
DEFAULT_REASON = "Otros factores"

def translate_reasons(top_features: List[str], max_reasons: int = 3) -> List[str]:
    """
    Recibe una lista de features top (por ejemplo, salidas de SHAP ordenadas)
    y devuelve hasta 'max_reasons' razones únicas y legibles.
    """
    reasons = []
    for f in top_features:
        reason = REASON_MAP.get(f, DEFAULT_REASON)
        if reason not in reasons:
            reasons.append(reason)
        if len(reasons) >= max_reasons:
            break
    return reasons



