from fastapi import APIRouter, Depends, Query
from sqlalchemy import func
from sqlalchemy.orm import Session
from datetime import datetime
from ..services.auth import get_db, get_current_user
from ..models.prediction import PredictionEvent
from ..schemas.prediction import StatsResponse

router = APIRouter(prefix="/v1", tags=["stats"])


@router.get("/stats", response_model=StatsResponse)
def stats(
    start_date: datetime | None = Query(None),
    end_date: datetime | None = Query(None),
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
):
    q = db.query(PredictionEvent)
    if start_date:
        q = q.filter(PredictionEvent.created_at >= start_date)
    if end_date:
        q = q.filter(PredictionEvent.created_at <= end_date)

    total = q.with_entities(func.count(PredictionEvent.id)).scalar() or 0

    reviewed_q = q.filter(PredictionEvent.confirmed.is_(True))
    reviewed = reviewed_q.with_entities(func.count(PredictionEvent.id)).scalar() or 0

    corrected = (
        reviewed_q.filter(PredictionEvent.corrected_label.isnot(None))
        .with_entities(func.count(PredictionEvent.id))
        .scalar()
        or 0
    )
    correct = reviewed - corrected
    accuracy_ratio = (correct / reviewed) if reviewed else 0.0
    return StatsResponse(
        total=total,
        reviewed=reviewed,
        corrected=corrected,
        accuracy_ratio=accuracy_ratio,
    )
