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