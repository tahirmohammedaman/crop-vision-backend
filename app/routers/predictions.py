from datetime import datetime
from typing import List
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from PIL import Image
import os

from ..services.auth import get_db, get_current_user
from ..services.storage import save_upload
from ..services import inference
from ..core.config import get_settings
from ..models.prediction import PredictionEvent
from ..schemas.prediction import (
    PredictionResponse,
    PredictionHistoryItem,
    FeedbackRequest,
)

router = APIRouter(prefix="/v1", tags=["predictions"])


@router.post("/predict", response_model=PredictionResponse)
def predict(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
):
    settings = get_settings()
    # Save upload
    rel_path = save_upload(settings.media_root, file)
    abs_path = os.path.join(settings.media_root, rel_path)

    # Load model and infer
    model_path = os.path.join(settings.models_path, settings.model_file)
    inference.ensure_loaded(model_path)
    img = Image.open(abs_path).convert("RGB")
    predicted_class, probs, top_conf, _ = inference.predict_pil(img)

    # Persist
    prob_map = {cls: float(p) for cls, p in zip(inference.class_names, probs)}
    ev = PredictionEvent(
        user_id=user.id,
        image_path=rel_path,
        predicted_label=predicted_class,
        predicted_confidence=float(top_conf),
        probabilities=prob_map,
    )
    db.add(ev)
    db.commit()
    db.refresh(ev)

    return PredictionResponse(
        id=ev.id,
        predicted_class=predicted_class,
        confidence=float(top_conf),
        classes=inference.class_names,
        probabilities=[float(p) for p in probs],
        image_url=f"/media/{rel_path}",
    )


@router.post("/predictions/{prediction_id}/feedback")
def feedback(
    prediction_id: int,
    data: FeedbackRequest,
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
):
    ev = db.query(PredictionEvent).filter(PredictionEvent.id == prediction_id).first()
    if not ev:
        raise HTTPException(status_code=404, detail="Prediction not found")
    if data.is_correct:
        ev.confirmed = True
        ev.corrected_label = None
    else:
        if not data.corrected_label:
            raise HTTPException(
                status_code=400, detail="corrected_label required when is_correct=false"
            )
        if data.corrected_label not in inference.class_names:
            raise HTTPException(
                status_code=400, detail="corrected_label not in class list"
            )
        ev.confirmed = True
        ev.corrected_label = data.corrected_label
    ev.confirmed_at = datetime.utcnow()
    db.add(ev)
    db.commit()
    return JSONResponse({"status": "ok"})


@router.get("/history", response_model=List[PredictionHistoryItem])
def history(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
):
    q = (
        db.query(PredictionEvent)
        .order_by(PredictionEvent.id.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )
    return [
        PredictionHistoryItem(
            id=ev.id,
            created_at=ev.created_at,
            predicted_label=ev.predicted_label,
            predicted_confidence=ev.predicted_confidence,
            corrected_label=ev.corrected_label,
            confirmed=ev.confirmed,
            image_url=f"/media/{ev.image_path}",
        )
        for ev in q
    ]
