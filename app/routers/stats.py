from fastapi import APIRouter, Depends
from sqlalchemy import func
from sqlalchemy.orm import Session
from ..services.auth import get_db, get_current_user
from ..models.prediction import PredictionEvent
from ..schemas.prediction import StatsResponse

router = APIRouter(prefix="/v1", tags=["stats"])


@router.get("/stats", response_model=StatsResponse)
def stats(db: Session = Depends(get_db), user=Depends(get_current_user)):
    total = db.query(func.count(PredictionEvent.id)).scalar() or 0
    reviewed = (
        db.query(func.count(PredictionEvent.id))
        .filter(PredictionEvent.confirmed.is_(True))
        .scalar()
        or 0
    )
    corrected = (
        db.query(func.count(PredictionEvent.id))
        .filter(
            PredictionEvent.confirmed.is_(True),
            PredictionEvent.corrected_label.is_not(None),
        )
        .scalar()
        or 0
    )
    # Accuracy = confirmed and not corrected
    correct = reviewed - corrected
    accuracy_ratio = (correct / reviewed) if reviewed else 0.0
    return StatsResponse(
        total=total,
        reviewed=reviewed,
        corrected=corrected,
        accuracy_ratio=accuracy_ratio,
    )
