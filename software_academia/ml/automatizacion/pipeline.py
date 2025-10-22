import json
from typing import Dict, Any, List  
from software_academia.ml.automatizacion.config import SHAP_RESULTS_JSON
from software_academia.ml.automatizacion.config import TOP_RISK_REASONS, TIER_HIGH, TIER_MED


def load_shap_results() -> List[Dict[str, Any]]:
    """
    Lee el archivo shap_results.json y devuelve una lista de registros.
    Cada registro debe tener: customer_id, asof_date, prob_churn, top_reasons.
    """
    with open(SHAP_RESULTS_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("El JSON esperado debe ser una lista de objetos.")
    return data

def tier_from_prob(p: float) -> str:
    if p >= TIER_HIGH:
        return "high"
    if p >= TIER_MED:
        return "medium"
    return "low"

def pick_candidates(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Filtra candidatos válidos: prob_churn dentro de [0,1] y al menos 1 razón.
    Añade el 'tier' calculado.
    """
    out = []
    for r in records:
        p = float(r.get("prob_churn", -1))
        reasons = r.get("top_reasons", [])
        if 0.0 <= p <= 1.0 and isinstance(reasons, list) and len(reasons) > 0:
            r["tier"] = tier_from_prob(p)
            out.append(r)
    return out

if __name__ == "__main__":
    data = load_shap_results()
    candidates = pick_candidates(data)
    print(f"Total registros: {len(data)} | Candidatos válidos: {len(candidates)}")
    # Vista previa del primero
    if candidates:
        from pprint import pprint
        pprint(candidates[0])


# opcional_schema_check.py
from typing import Dict, Any

REQUIRED_FIELDS = {"customer_id", "asof_date", "prob_churn", "top_reasons"}

def validate_record_shape(r: Dict[str, Any]) -> bool:
    if not REQUIRED_FIELDS.issubset(r.keys()):
        return False
    if not isinstance(r["top_reasons"], list) or len(r["top_reasons"]) == 0:
        return False
    return True



from typing import Dict, Any, List
from software_academia.ml.automatizacion.writer import generate_message


def build_writer_payload(candidate: Dict[str, Any]) -> Dict[str, Any]:
    # Asegúrate de traer nombre/teléfono si están en otra fuente/BD.
    customer = {
        "id": candidate["customer_id"],
        "customer_id": candidate["customer_id"],
        "name": candidate.get("name") or "Cliente",
        "phone": candidate.get("phone") or "+11111111111",
        "product": candidate.get("product") or "SaaS-X",
        "plan": candidate.get("plan") or "Mensual",
        "segment": candidate.get("segment") or "General",
        "language": candidate.get("language") or "es",
    }
    risk = {
        "churn_prob": candidate["prob_churn"],
        "tier": candidate["tier"],
        "top_reasons": candidate["top_reasons"],  # ya con reason_human
    }
    context = {
        "days_inactive": candidate.get("days_inactive"),
        "last_interaction_at": candidate.get("last_interaction_at"),
    }
    offer_policy = {
        "eligible": True if candidate["tier"] == "high" else False,
        "code": "SAVE10" if candidate["tier"] in ("high","medium") else None,
        "discount": 10 if candidate["tier"] in ("high","medium") else 0,
    }
    cta = {"label": "Reactivar ahora", "url": "https://tusitio.com/reactiva"}
    return {"customer": customer, "risk": risk, "context": context, "offer_policy": offer_policy, "cta": cta}

def generate_preview_for_one():
    data = load_shap_results()
    candidates = pick_candidates(data)
    assert candidates, "No hay candidatos válidos"
    cand = candidates[0]
    payload = build_writer_payload(cand)
    msg = generate_message(payload)
    from pprint import pprint
    print("[WRITER_OUTPUT]")
    pprint(msg)

if __name__ == "__main__":
    # ...
    generate_preview_for_one()