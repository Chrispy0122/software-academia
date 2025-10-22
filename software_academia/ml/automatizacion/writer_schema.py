# writer_schema.py
from pydantic import BaseModel, Field

class MsgSchema(BaseModel):
    headline: str = Field(..., min_length=3, max_length=120)
    body: str = Field(..., min_length=10, max_length=500)
    cta: str = Field(..., min_length=5, max_length=200)
    compliance_note: str = Field(..., min_length=5, max_length=200)
