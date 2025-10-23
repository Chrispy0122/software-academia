# writer_schema.py
from __future__ import annotations
from pydantic import BaseModel, Field, validator
from typing import Optional, Dict

class EmailMsgSchema(BaseModel):
    """
    Esquema de MENSAJE PARA EMAIL (sin URL).
    La CTA se ejecuta por respuesta (ej.: Reply YES).
    """
    subject: str = Field(..., min_length=3, max_length=120)
    html_body: str = Field(..., min_length=20)
    text_body: str = Field(..., min_length=10, max_length=2000)
    cta_label: str = Field(..., min_length=2, max_length=100)  # ej.: "Reply YES"
    compliance_note: Optional[str] = Field(default=None, max_length=200)
    language: str = Field(default="en", min_length=2, max_length=5)
    idempotency_key: Optional[str] = Field(default=None, max_length=200)
    metadata: Optional[Dict[str, str]] = Field(default=None)

    @validator("subject")
    def no_crlf_in_subject(cls, v: str) -> str:
        if "\r" in v or "\n" in v:
            raise ValueError("Subject cannot contain newlines.")
        return v.strip()

    @validator("html_body")
    def minimal_html_sanity(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("html_body cannot be empty.")
        return v

    @validator("text_body")
    def no_html_in_text_body(cls, v: str) -> str:
        if "<" in v or ">" in v:
            raise ValueError("text_body must not contain HTML.")
        return v.strip()

    @validator("language")
    def normalize_language(cls, v: str) -> str:
        return v.strip().lower()
