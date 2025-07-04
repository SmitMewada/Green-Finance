from fastapi import APIRouter
from models.schemas import TransactionRequest, ClassificationResponse
from services.classifier import classify_with_model

router = APIRouter()

@router.get("/")
def home():
    return {
        "message": "Green Finance Transaction Classifier API",
        "status": "active",
        "version": "1.0.0"
    }

@router.get("/health")
def health_check():
    return {"status": "healthy"}

@router.post("/classify", response_model=ClassificationResponse)
def classify_transaction(transaction: TransactionRequest):
    predicted_category = classify_with_model(
        transaction.name,
        transaction.amount,
        transaction.description
    )
    
    return ClassificationResponse(
        category=predicted_category,
        confidence=0.95,
        transaction_id=f"txn_{hash(transaction.name + transaction.description)}"
    ) 