# writer.py
import json
import os
import time
from typing import Dict, Any, List

from software_academia.ml.automatizacion.writer_schema import MsgSchema
from software_academia.ml.automatizacion.config import (
    DRY_RUN, OPENAI_API_KEY, OPENAI_MODEL,
    OPENAI_TIMEOUT_SECS, OPENAI_MAX_RETRIES
)

# --- Si usas el SDK OpenAI v1 ---
try:
    from openai import OpenAI
    _openai_client = OpenAI(api_key=OPENAI_API_KEY or os.getenv("OPENAI_API_KEY"))
except Exception:
    _openai_client = None


SYSTEM_PROMPT = (
    "Eres un redactor experto en retención para WhatsApp. "
    "Escribes mensajes breves, empáticos y accionables. "
    "Siempre devuelves SOLO JSON válido con las claves: "
    "headline, body, cta, compliance_note. "
    "Respeta el idioma solicitado."
)

def _build_user_prompt(payload: Dict[str, Any]) -> str:
    c = payload["customer"]
    r = payload["risk"]
    ctx = payload.get("context", {}) or {}
    offer = payload.get("offer_policy", {}) or {}
    cta = payload["cta"]

    # Razones humanas (ya vienen traducidas en top_reasons.reason_human)
    reasons_human: List[str] = [tr.get("reason_human", tr.get("feature","")) for tr in r["top_reasons"]]
    reasons_txt = ", ".join(reasons_human) if reasons_human else "sin_razones"

    language = c.get("language") or "es"  # por defecto español

    instr = (
        "Instrucciones:\n"
        "- Tono: cercano, respetuoso y directo (WhatsApp). 2–3 oraciones en 'body'.\n"
        "- Personaliza según razones:\n"
        "  * 'precio_alto' → resalta valor/ahorro; usa oferta si elegible.\n"
        "  * 'inactividad_reciente' o 'bajo_uso_*' → da 2 pasos prácticos.\n"
        "  * 'plan_inadecuado' → sugiere ajuste/downgrade o plan alterno.\n"
        "  * 'atraso_pago' o 'bajo_pago_reciente' → facilita pago, empatía.\n"
        "- Incluye UNA sola CTA clara (usa la provista).\n"
        "- Agrega compliance_note con 'Reply STOP to opt-out'.\n"
        "- Devuelve SOLO JSON válido, sin texto adicional."
    )

    customer_block = (
        f"Cliente:\n"
        f"- id: {c.get('id') or c.get('customer_id')}\n"
        f"- nombre: {c.get('name','Cliente')}\n"
        f"- phone: {c.get('phone','N/A')}\n"
        f"- producto/plan: {c.get('product','N/A')} / {c.get('plan','N/A')}\n"
        f"- segmento: {c.get('segment','General')}\n"
        f"- idioma: {language}\n"
    )
    risk_block = (
        f"Riesgo:\n"
        f"- prob_churn: {r['churn_prob']:.4f} (tier={r['tier']})\n"
        f"- razones: {reasons_txt}\n"
    )
    context_block = (
        "Contexto:\n"
        f"- dias_sin_uso: {ctx.get('days_inactive','')}\n"
        f"- ultima_interaccion: {ctx.get('last_interaction_at','')}\n"
    )
    offer_block = (
        "Oferta:\n"
        f"- elegible: {offer.get('eligible', False)}\n"
        f"- codigo: {offer.get('code')}\n"
        f"- descuento: {offer.get('discount', 0)}\n"
    )
    cta_block = (
        "CTA:\n"
        f"- texto: {cta.get('label','')}\n"
        f"- url: {cta.get('url','')}\n"
    )

    # Reglas de idioma: pedimos respuesta en el idioma del cliente
    lang_rule = "Responde en español neutro." if language.startswith("es") else "Reply in natural U.S. English."

    return (
        f"{lang_rule}\n\n{instr}\n\n{customer_block}\n{risk_block}\n{context_block}\n{offer_block}\n{cta_block}"
    )


def _fake_message_for_dry_run(payload: Dict[str, Any]) -> Dict[str, Any]:
    name = payload["customer"].get("name") or "Cliente"
    reasons = ", ".join([r.get("reason_human", r.get("feature","")) for r in payload["risk"]["top_reasons"]]) or "señales de baja"
    tier = payload["risk"]["tier"]
    cta_label = payload["cta"]["label"]
    cta_url = payload["cta"]["url"]
    msg = {
        "headline": f"Hola {name}, te ayudo con tu plan",
        "body": f"Vimos {reasons}. Nivel de riesgo: {tier}. Puedo guiarte en 2 pasos para retomar valor hoy.",
        "cta": f"{cta_label}: {cta_url}",
        "compliance_note": "Reply STOP to opt-out."
    }
    # valida igual que OpenAI real
    return MsgSchema(**msg).model_dump()


def _call_openai_and_parse(user_prompt: str) -> Dict[str, Any]:
    # Usamos Chat Completions con response_format JSON para robustez
    # Si tu SDK no soporta response_format, parsearemos manual.
    for attempt in range(1, OPENAI_MAX_RETRIES + 1):
        try:
            resp = _openai_client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,
                response_format={"type": "json_object"},
                timeout=OPENAI_TIMEOUT_SECS,
            )
            raw_txt = resp.choices[0].message.content.strip()
            data = json.loads(raw_txt)  # debe ser JSON puro
            # Validación estricta
            msg = MsgSchema(**data).model_dump()
            return msg
        except Exception as e:
            if attempt >= OPENAI_MAX_RETRIES:
                raise
            time.sleep(0.8 * attempt)  # backoff simple y rápido
    # No debería llegar aquí
    raise RuntimeError("OpenAI: agotados los reintentos")


def generate_message(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Entrada: payload armado por el pipeline (customer, risk, context, offer_policy, cta).
    Salida: dict validado con {headline, body, cta, compliance_note}.
    """
    if DRY_RUN:
        return _fake_message_for_dry_run(payload)

    if _openai_client is None:
        raise RuntimeError("OpenAI client no inicializado. Configura OPENAI_API_KEY.")

    prompt = _build_user_prompt(payload)
    msg = _call_openai_and_parse(prompt)
    return msg
