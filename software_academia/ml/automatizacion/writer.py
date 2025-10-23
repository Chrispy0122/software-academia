# writer.py
from __future__ import annotations

import json
import os
import time
from typing import Dict, Any, List

from software_academia.ml.automatizacion.writer_schema import EmailMsgSchema
from software_academia.ml.automatizacion.config import (
    DRY_RUN, OPENAI_API_KEY, OPENAI_MODEL,
    OPENAI_TIMEOUT_SECS, OPENAI_MAX_RETRIES,
)

# --- OpenAI client (SDK v1) ---
try:
    from openai import OpenAI
    _openai_client = OpenAI(api_key=OPENAI_API_KEY or os.getenv("OPENAI_API_KEY"))
except Exception:
    _openai_client = None


# =========================
#  Prompts & Helpers
# =========================
SYSTEM_PROMPT = (
    "You are a retention-focused email copywriter. "
    "Write short, empathetic, and actionable emails optimized for deliverability. "
    "ALWAYS return ONLY a valid JSON object with the exact keys: "
    "subject, html_body, text_body, cta_label, compliance_note, language, idempotency_key, metadata. "
    "No extra text, no markdown, no explanations."
)

INSTRUCTIONS_EMAIL = (
    "Instructions:\n"
    "- Language: natural U.S. English.\n"
    "- Audience: existing customer at risk of churn.\n"
    "- Style: respectful, concise, value-driven. Avoid spammy words.\n"
    "- Structure:\n"
    "  1) subject: clear value + reason-aware (<= 80 chars).\n"
    "  2) html_body: 2–4 short paragraphs. No links. Include ONE clear reply-based CTA.\n"
    "  3) text_body: plain-text equivalent (no HTML). No links.\n"
    "- Personalize using the top churn reasons (humanized) if available.\n"
    "- If offer is eligible, mention it once (code/discount). Do not overpromise.\n"
    "- CTA must be reply-based, e.g., 'Reply YES to continue' or 'Reply HELP for assistance'.\n"
    "- compliance_note: include a short sentence about opting out by reply.\n"
    "- Output ONLY a valid JSON object with the required keys."
)

def _extract_reasons_human(top_reasons: List[Dict[str, Any]]) -> str:
    if not top_reasons:
        return "no specific reasons available"
    items = [r.get("reason_human") or r.get("feature") or "reason" for r in top_reasons]
    return ", ".join([str(x) for x in items if x])

def _build_user_prompt(payload: Dict[str, Any]) -> str:
    """
    Build the user prompt for OpenAI based on the unified payload from pipeline:
    {
      'customer': {...}, 'risk': {...}, 'context': {...},
      'offer_policy': {...},
      'cta': {'label'}  # NO url
      'idempotency_key' (optional), 'campaign_id' (optional)
    }
    """
    c: Dict[str, Any] = payload.get("customer", {}) or {}
    r: Dict[str, Any] = payload.get("risk", {}) or {}
    ctx: Dict[str, Any] = payload.get("context", {}) or {}
    offer: Dict[str, Any] = payload.get("offer_policy", {}) or {}
    cta: Dict[str, Any] = payload.get("cta", {}) or {}

    name = c.get("name") or "Customer"
    product = c.get("product") or "your plan"
    plan = c.get("plan") or ""
    segment = c.get("segment") or "General"

    churn_prob = r.get("churn_prob", 0.0)
    tier = r.get("tier", "medium")
    reasons_txt = _extract_reasons_human(r.get("top_reasons", []))

    # Optional context
    days_inactive = ctx.get("days_inactive")
    last_interaction_at = ctx.get("last_interaction_at")

    # Offer
    eligible = bool(offer.get("eligible", False))
    code = offer.get("code")
    discount = offer.get("discount", 0)

    # CTA (reply-based)
    cta_label = cta.get("label") or "Reply YES to continue"

    lines = [
        "Language: en",
        INSTRUCTIONS_EMAIL,
        "",
        "Customer profile:",
        f"- id: {c.get('id') or c.get('customer_id')}",
        f"- name: {name}",
        f"- product/plan: {product} / {plan}",
        f"- segment: {segment}",
        "",
        "Risk summary:",
        f"- churn_probability: {churn_prob:.4f} (tier={tier})",
        f"- top_reasons_human: {reasons_txt}",
        "",
        "Context:",
        f"- days_inactive: {days_inactive}",
        f"- last_interaction_at: {last_interaction_at}",
        "",
        "Offer policy:",
        f"- eligible: {eligible}",
        f"- code: {code}",
        f"- discount_percent: {discount}",
        "",
        "CTA provided (reply-based):",
        f"- cta_label: {cta_label}",
        "",
        "Output JSON schema (exact keys):",
        "{",
        '  "subject": "...",',
        '  "html_body": "<p>...</p>",',
        '  "text_body": "...",',
        '  "cta_label": "Reply YES to continue",',
        '  "compliance_note": "You can reply STOP to opt out.",',
        '  "language": "en",',
        '  "idempotency_key": "user|campaign|date",',
        '  "metadata": {"campaign_id": "campaign-xyz", "tier": "high"}',
        "}",
    ]
    return "\n".join(lines)

def _fake_email_for_dry_run(payload: Dict[str, Any]) -> Dict[str, Any]:
    c = payload.get("customer", {}) or {}
    r = payload.get("risk", {}) or {}
    offer = payload.get("offer_policy", {}) or {}
    cta = payload.get("cta", {}) or {}

    name = c.get("name") or "Customer"
    product = c.get("product") or "your plan"
    tier = r.get("tier", "medium")
    reasons = _extract_reasons_human(r.get("top_reasons", []))
    cta_label = cta.get("label") or "Reply YES to continue"

    offer_line = ""
    if offer.get("eligible"):
        disc = offer.get("discount", 0)
        code = offer.get("code") or ""
        offer_line = f" As a courtesy, you can use code {code} for {disc}% off."

    subject = f"{name}, quick help to get more value — based on recent usage"
    html_body = (
        f"<p>Hi {name},</p>"
        f"<p>We noticed some signals: <strong>{reasons}</strong> (risk tier: {tier}). "
        f"Let me guide you to get immediate value from {product}.{offer_line}</p>"
        f"<p><strong>{cta_label}</strong></p>"
        f"<p>If you prefer not to receive these emails, just reply STOP.</p>"
    )
    text_body = (
        f"Hi {name},\n\n"
        f"We noticed some signals: {reasons} (risk tier: {tier}). "
        f"Let me guide you to get immediate value from {product}.{offer_line}\n\n"
        f"{cta_label}\n\n"
        f"If you prefer not to receive these emails, just reply STOP."
    )

    msg = {
        "subject": subject[:120],
        "html_body": html_body,
        "text_body": text_body,
        "cta_label": cta_label,
        "compliance_note": "You can reply STOP to opt out.",
        "language": "en",
        "idempotency_key": payload.get("idempotency_key") or None,
        "metadata": {
            "campaign_id": payload.get("campaign_id") or "",
            "tier": r.get("tier", "medium"),
        },
    }
    return EmailMsgSchema(**msg).model_dump()

def _call_openai_and_parse(user_prompt: str) -> Dict[str, Any]:
    if _openai_client is None:
        raise RuntimeError("OpenAI client not initialized. Set OPENAI_API_KEY.")

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
            data = json.loads(raw_txt)  # must be pure JSON
            msg = EmailMsgSchema(**data).model_dump()  # strict validation
            return msg
        except Exception:
            if attempt >= OPENAI_MAX_RETRIES:
                raise
            time.sleep(0.8 * attempt)

    raise RuntimeError("OpenAI: retries exhausted")

# =========================
#  Public API
# =========================
def generate_email_message(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Input (from pipeline):
      - customer: {id/customer_id, name, product, plan, segment, language? ...}
      - risk: {churn_prob, tier, top_reasons[{reason_human, feature, ...}]}
      - context: {...}
      - offer_policy: {eligible, code, discount}
      - cta: {label}   # NO url
      - (optional) idempotency_key, campaign_id
    Output: dict validated against EmailMsgSchema (no URL).
    """
    # Forzar inglés como solicitaste
    payload = dict(payload or {})
    cust = dict(payload.get("customer") or {})
    cust["language"] = "en"
    payload["customer"] = cust

    if DRY_RUN:
        return _fake_email_for_dry_run(payload)

    user_prompt = _build_user_prompt(payload)
    return _call_openai_and_parse(user_prompt)
