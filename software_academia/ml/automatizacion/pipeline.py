from __future__ import annotations

import json
from typing import Dict, Any, List, Optional
from datetime import datetime

from software_academia.ml.automatizacion.config import (
    SHAP_RESULTS_JSON,
    TOP_RISK_REASONS,
    TIER_HIGH,
    TIER_MED,
    DRY_RUN,
)
from software_academia.ml.automatizacion.writer import generate_email_message


# =========================
#  Load & basic filters
# =========================
def load_shap_results() -> List[Dict[str, Any]]:
    """
    Reads the normalized SHAP output JSON (list of records).
    Each record must include: customer_id, asof_date, prob_churn, top_reasons (list).
    May include extra fields such as: email, name, days_inactive, last_interaction_at, etc.
    """
    with open(SHAP_RESULTS_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Expected a list of records in SHAP_RESULTS_JSON.")
    return data


def tier_from_prob(p: float) -> str:
    if p >= TIER_HIGH:
        return "high"
    if p >= TIER_MED:
        return "medium"
    return "low"


def pick_candidates(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Keep candidates with valid prob and at least one reason.
    Also attach 'tier' computed from prob.
    """
    out: List[Dict[str, Any]] = []
    for r in records:
        try:
            p = float(r.get("prob_churn", -1))
        except Exception:
            continue
        reasons = r.get("top_reasons", [])
        if 0.0 <= p <= 1.0 and isinstance(reasons, list) and len(reasons) > 0:
            r["tier"] = tier_from_prob(p)
            out.append(r)
    return out


# =========================
#  Payload builder for writer
# =========================
def _idempotency_key(customer_id: Any, asof_date: Optional[str]) -> str:
    """
    Simple idempotency key. If you later add a campaign id, append it here.
    """
    parts = [str(customer_id)]
    if asof_date:
        parts.append(str(asof_date))
    return "|".join(parts)


def _safe_name(rec: Dict[str, Any]) -> str:
    return rec.get("name") or rec.get("customer_name") or "Customer"


def _safe_email(rec: Dict[str, Any]) -> Optional[str]:
    # We expect the email to be present either here or from an enrichment step.
    email = rec.get("email") or rec.get("customer_email")
    return email.strip() if isinstance(email, str) and email.strip() else None


def build_writer_payload(candidate: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Build the payload for writer.generate_email_message (reply-based CTA; no URLs).
    Returns None if the candidate has no usable email (we skip it).
    """
    email = _safe_email(candidate)
    if not email:
        # No destination email — skip this candidate
        return None

    customer = {
        "id": candidate["customer_id"],
        "customer_id": candidate["customer_id"],
        "name": _safe_name(candidate),
        "email": email,
        # Optional fields if you have them in your data source
        "product": candidate.get("product") or "Your plan",
        "plan": candidate.get("plan") or "Monthly",
        "segment": candidate.get("segment") or "General",
        # We force English in writer, but keep language here for completeness
        "language": candidate.get("language") or "en",
    }

    risk = {
        "churn_prob": float(candidate["prob_churn"]),
        "tier": candidate["tier"],
        # expects each item to already contain a human-readable reason if available
        "top_reasons": candidate.get("top_reasons", [])[:TOP_RISK_REASONS],
    }

    context = {
        "days_inactive": candidate.get("days_inactive"),
        "last_interaction_at": candidate.get("last_interaction_at"),
        "asof_date": candidate.get("asof_date"),
    }

    # Simple pilot offer policy by tier (adjust later if needed)
    offer_policy = {
        "eligible": True if candidate["tier"] in ("high", "medium") else False,
        "code": "SAVE10" if candidate["tier"] in ("high", "medium") else None,
        "discount": 10 if candidate["tier"] in ("high", "medium") else 0,
    }

    # Reply-based CTA (no links)
    cta = {"label": "Reply YES to continue"}

    payload = {
        "customer": customer,
        "risk": risk,
        "context": context,
        "offer_policy": offer_policy,
        "cta": cta,
        "idempotency_key": _idempotency_key(candidate["customer_id"], candidate.get("asof_date")),
        # "campaign_id": "your-campaign-id",  # optional — add later if you want
    }
    return payload


# =========================
#  Batch preview runner (no send)
# =========================
def preview_emails(limit: Optional[int] = 5) -> None:
    """
    Generates and prints email previews for the first N valid candidates.
    This does not send anything (sending will be wired later).
    """
    records = load_shap_results()
    candidates = pick_candidates(records)

    if not candidates:
        print("No valid candidates found.")
        return

    printed = 0
    for cand in candidates:
        if limit is not None and printed >= limit:
            break

        payload = build_writer_payload(cand)
        if not payload:
            # missing email — skip
            continue

        try:
            email_msg = generate_email_message(payload)
        except Exception as e:
            print(f"[ERROR] writer failed for customer_id={cand.get('customer_id')}: {e}")
            continue

        # Pretty preview
        print("=" * 70)
        print(f"To:   {payload['customer']['email']}")
        print(f"Name: {payload['customer']['name']}")
        print(f"Tier: {payload['risk']['tier']} | Prob: {payload['risk']['churn_prob']:.4f}")
        print(f"Idempotency: {email_msg.get('idempotency_key')}")
        print("-" * 70)
        print(f"Subject: {email_msg['subject']}")
        print("-" * 70)
        print("TEXT BODY:")
        print(email_msg["text_body"])
        print("-" * 70)
        print("HTML BODY:")
        print(email_msg["html_body"])
        print("=" * 70)
        printed += 1

    if printed == 0:
        print("No candidates with email found to preview.")





import json
from typing import Dict, Any, List, Optional
from datetime import datetime

from software_academia.ml.automatizacion.config import (
    SHAP_RESULTS_JSON,
    TOP_RISK_REASONS,
    TIER_HIGH,
    TIER_MED,
    DRY_RUN,
)
from software_academia.ml.automatizacion.writer import generate_email_message
from software_academia.ml.automatizacion.gmail_sender import send_email_message


# =========================
#  Load & basic filters
# =========================
def load_shap_results() -> List[Dict[str, Any]]:
    with open(SHAP_RESULTS_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Expected a list of records in SHAP_RESULTS_JSON.")
    return data


def tier_from_prob(p: float) -> str:
    if p >= TIER_HIGH:
        return "high"
    if p >= TIER_MED:
        return "medium"
    return "low"


def pick_candidates(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for r in records:
        try:
            p = float(r.get("prob_churn", -1))
        except Exception:
            continue
        reasons = r.get("top_reasons", [])
        if 0.0 <= p <= 1.0 and isinstance(reasons, list) and len(reasons) > 0:
            r["tier"] = tier_from_prob(p)
            out.append(r)
    return out


# =========================
#  Payload builder for writer
# =========================
def _idempotency_key(customer_id: Any, asof_date: Optional[str]) -> str:
    parts = [str(customer_id)]
    if asof_date:
        parts.append(str(asof_date))
    return "|".join(parts)


def _safe_name(rec: Dict[str, Any]) -> str:
    return rec.get("name") or rec.get("customer_name") or "Customer"


def _safe_email(rec: Dict[str, Any]) -> Optional[str]:
    email = rec.get("email") or rec.get("customer_email")
    return email.strip() if isinstance(email, str) and email.strip() else None


def build_writer_payload(candidate: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    email = _safe_email(candidate)
    if not email:
        return None

    customer = {
        "id": candidate["customer_id"],
        "customer_id": candidate["customer_id"],
        "name": _safe_name(candidate),
        "email": email,
        "product": candidate.get("product") or "Your plan",
        "plan": candidate.get("plan") or "Monthly",
        "segment": candidate.get("segment") or "General",
        "language": candidate.get("language") or "en",
    }

    risk = {
        "churn_prob": float(candidate["prob_churn"]),
        "tier": candidate["tier"],
        "top_reasons": candidate.get("top_reasons", [])[:TOP_RISK_REASONS],
    }

    context = {
        "days_inactive": candidate.get("days_inactive"),
        "last_interaction_at": candidate.get("last_interaction_at"),
        "asof_date": candidate.get("asof_date"),
    }

    offer_policy = {
        "eligible": True if candidate["tier"] in ("high", "medium") else False,
        "code": "SAVE10" if candidate["tier"] in ("high", "medium") else None,
        "discount": 10 if candidate["tier"] in ("high", "medium") else 0,
    }

    cta = {"label": "Reply YES to continue"}  # reply-based CTA, no links

    payload = {
        "customer": customer,
        "risk": risk,
        "context": context,
        "offer_policy": offer_policy,
        "cta": cta,
        "idempotency_key": _idempotency_key(candidate["customer_id"], candidate.get("asof_date")),
    }
    return payload


# =========================
#  Preview (no send)
# =========================
def preview_emails(limit: Optional[int] = 5) -> None:
    records = load_shap_results()
    candidates = pick_candidates(records)

    if not candidates:
        print("No valid candidates found.")
        return

    printed = 0
    for cand in candidates:
        if limit is not None and printed >= limit:
            break

        payload = build_writer_payload(cand)
        if not payload:
            continue

        try:
            email_msg = generate_email_message(payload)
        except Exception as e:
            print(f"[ERROR] writer failed for customer_id={cand.get('customer_id')}: {e}")
            continue

        print("=" * 70)
        print(f"To:   {payload['customer']['email']}")
        print(f"Name: {payload['customer']['name']}")
        print(f"Tier: {payload['risk']['tier']} | Prob: {payload['risk']['churn_prob']:.4f}")
        print(f"Idempotency: {email_msg.get('idempotency_key')}")
        print("-" * 70)
        print(f"Subject: {email_msg['subject']}")
        print("-" * 70)
        print("TEXT BODY:")
        print(email_msg["text_body"])
        print("-" * 70)
        print("HTML BODY:")
        print(email_msg["html_body"])
        print("=" * 70)
        printed += 1

    if printed == 0:
        print("No candidates with email found to preview.")


# =========================
#  Send batch (respects DRY_RUN)
# =========================
def send_batch(limit: Optional[int] = 10) -> None:
    """
    Generates email content and sends (or prints) up to N messages.
    - If DRY_RUN=True: prints previews using gmail_sender's DRY_RUN path.
    - If DRY_RUN=False: sends via Gmail API and prints Gmail message_id.
    """
    records = load_shap_results()
    candidates = pick_candidates(records)

    sent = 0
    for cand in candidates:
        if limit is not None and sent >= limit:
            break

        payload = build_writer_payload(cand)
        if not payload:
            continue

        try:
            email_msg = generate_email_message(payload)
            to_email = payload["customer"]["email"]
            gmail_id = send_email_message(to_email, email_msg)
            if gmail_id:
                print(f"[SENT] customer_id={cand.get('customer_id')} gmail_id={gmail_id}")
            else:
                print(f"[DRY_RUN PREVIEW] customer_id={cand.get('customer_id')} to={to_email}")
            sent += 1
        except Exception as e:
            print(f"[ERROR] send failed for customer_id={cand.get('customer_id')}: {e}")
            continue

    print(f"Done. Processed: {sent} message(s).")


if __name__ == "__main__":
    # First, you can preview:
    # preview_emails(limit=3)

    # Then run the batch (respects DRY_RUN from config):
    send_batch(limit=3)
