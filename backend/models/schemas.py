from pydantic import BaseModel
from typing import Optional

class TransactionRequest(BaseModel):
    name: str
    amount: float
    description: str

class ClassificationResponse(BaseModel):
    category: str
    confidence: Optional[float] = None
    transaction_id: Optional[str] = None 