import os
import requests
from typing import Dict, Any
from config import WHATSAPP_API_BASE, WHATSAPP_PHONE_NUMBER_ID, DRY_RUN

def _auth_headers():
    token = os.getenv("WHATSAPP_TOKEN")  # o léelo desde config si lo pones ahí
    if not token:
        raise RuntimeError("Falta WHATSAPP_TOKEN (variable de entorno).")
    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }

def send_template(to_phone_e164: str, template_name: str, lang_code: str = "es", components: list | None = None) -> str:
    """
    Envía una plantilla (HSM) para abrir la ventana 24h.
    'components' permite pasar variables a la plantilla.
    """
    if DRY_RUN:
        print(f"[DRY_RUN][TEMPLATE] to={to_phone_e164} template={template_name} components={components}")
        return "dryrun-template-id"

    url = f"{WHATSAPP_API_BASE}/{WHATSAPP_PHONE_NUMBER_ID}/messages"
    payload: Dict[str, Any] = {
        "messaging_product": "whatsapp",
        "to": to_phone_e164,
        "type": "template",
        "template": {
            "name": template_name,
            "language": {"code": lang_code},
        },
    }
    if components:
        payload["template"]["components"] = components

    resp = requests.post(url, headers=_auth_headers(), json=payload, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return data.get("messages", [{}])[0].get("id", "unknown-id")

def send_freeform(to_phone_e164: str, text: str) -> str:
    """
    Envía un mensaje de texto libre (requiere ventana activa de 24h).
    """
    if DRY_RUN:
        print(f"[DRY_RUN][FREEFORM] to={to_phone_e164}\n{text}\n")
        return "dryrun-freeform-id"

    url = f"{WHATSAPP_API_BASE}/{WHATSAPP_PHONE_NUMBER_ID}/messages"
    payload = {
        "messaging_product": "whatsapp",
        "to": to_phone_e164,
        "type": "text",
        "text": {"body": text}
    }
    resp = requests.post(url, headers=_auth_headers(), json=payload, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return data.get("messages", [{}])[0].get("id", "unknown-id")
