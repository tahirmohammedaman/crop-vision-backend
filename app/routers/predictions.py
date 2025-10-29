from datetime import datetime
from typing import List, Optional
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, Query, Form
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from sqlalchemy import or_, and_
from PIL import Image
import os
import re

from ..services.auth import get_db, get_current_user
from ..services.storage import save_upload
from ..services import inference
from ..core.config import get_settings
from ..models.prediction import PredictionEvent
from ..schemas.prediction import (
    PredictionResponse,
    PredictionHistoryItem,
    FeedbackRequest,
    PaginatedResponse,
)

router = APIRouter(prefix="/v1", tags=["predictions"])

TAG_CLEAN_RE = re.compile(r"[^a-z0-9\-]+")

def normalize_tag(s: str) -> str:
    if s is None:
        return ""
    t = s.strip().lower()
    t = re.sub(r"[\s_]+", "-", t)
    t = re.sub(r"-{2,}", "-", t).strip("-")
    t = TAG_CLEAN_RE.sub("", t)
    return t

def parse_crop_disease(label: str) -> tuple[Optional[str], Optional[str]]:
    # Expected formats like "Tomato___Late_blight", "Apple___healthy", etc.
    if not label:
        return None, None
    parts = label.split("___", 1)
    if len(parts) == 2:
        crop, disease = parts[0], parts[1]
    else:
        # fallback if no delimiter
        crop, disease = None, label
    return crop, disease

@router.post("/predict", response_model=PredictionResponse)
def predict(
    file: UploadFile = File(...),
    tags: Optional[List[str]] = Form(None, description="Optional list of tags"),
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

    # Derive crop/disease (must not be null)
    crop, _disease = parse_crop_disease(predicted_class)
    if crop is None:
        # This should never happen if class names use "Crop___Disease" format
        raise HTTPException(status_code=500, detail="Unable to parse crop from predicted class")

    # Normalize tags
    norm_tags = []
    if tags:
        for t in tags:
            nt = normalize_tag(t)
            if nt:
                norm_tags.append(nt)
        # de-duplicate
        norm_tags = sorted(set(norm_tags))

    # Persist
    prob_map = {cls: float(p) for cls, p in zip(inference.class_names, probs)}
    ev = PredictionEvent(
        user_id=user.id,
        image_path=rel_path,
        predicted_label=predicted_class,
        predicted_confidence=float(top_conf),
        probabilities=prob_map,
        crop=crop,  # enforce non-null
        tags=norm_tags,
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
        # Ensure crop is set (should already be from predict)
        if not ev.crop:
            pred_crop, _ = parse_crop_disease(ev.predicted_label)
            if pred_crop is None:
                raise HTTPException(status_code=500, detail="Unable to parse crop from predicted label")
            ev.crop = pred_crop
    else:
        if not data.corrected_label:
            raise HTTPException(
                status_code=400, detail="corrected_label required when is_correct=false"
            )
        if data.corrected_label not in inference.class_names:
            raise HTTPException(
                status_code=400, detail="corrected_label not in class list"
            )
        # Always set crop from corrected label; must not be null
        crop, _ = parse_crop_disease(data.corrected_label)
        if crop is None:
            raise HTTPException(status_code=400, detail="corrected_label must include crop (format 'Crop___Disease')")
        ev.confirmed = True
        ev.corrected_label = data.corrected_label
        ev.crop = crop
    ev.confirmed_at = datetime.utcnow()
    db.add(ev)
    db.commit()
    return JSONResponse({"status": "ok"})

@router.get("/history", response_model=PaginatedResponse[PredictionHistoryItem])
def history(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    crop: Optional[str] = Query(None),
    disease: Optional[str] = Query(None, description="Predicted or corrected label"),
    tags: Optional[List[str]] = Query(None, description="Filter by tags"),
    min_confidence: float = Query(0.0, ge=0.0, le=1.0),
    max_confidence: float = Query(1.0, ge=0.0, le=1.0),
    start_date: Optional[datetime] = Query(None),
    end_date: Optional[datetime] = Query(None),
    confirmed: Optional[bool] = Query(None, description="True=only confirmed, False=only unconfirmed, None=all"),
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
):
    q = db.query(PredictionEvent)

    if crop:
        q = q.filter(PredictionEvent.crop == crop)

    if disease:
        q = q.filter(
            or_(
                PredictionEvent.predicted_label == disease,
                PredictionEvent.corrected_label == disease,
            )
        )

    if start_date:
        q = q.filter(PredictionEvent.created_at >= start_date)
    if end_date:
        q = q.filter(PredictionEvent.created_at <= end_date)

    if min_confidence is not None:
        q = q.filter(PredictionEvent.predicted_confidence >= min_confidence)
    if max_confidence is not None:
        q = q.filter(PredictionEvent.predicted_confidence <= max_confidence)

    if confirmed is True:
        q = q.filter(PredictionEvent.confirmed.is_(True))
    elif confirmed is False:
        q = q.filter(or_(PredictionEvent.confirmed.is_(False), PredictionEvent.confirmed.is_(None)))

    if tags:
        norm = [normalize_tag(t) for t in tags if normalize_tag(t)]
        if norm:
            q = q.filter(PredictionEvent.tags.contains(norm))

    total = q.count()

    rows = (
        q.order_by(PredictionEvent.id.desc())
         .offset(skip)
         .limit(limit)
         .all()
    )

    return {
        "total": total,
        "items": [
            PredictionHistoryItem(
                id=ev.id,
                created_at=ev.created_at,
                predicted_label=ev.predicted_label,
                predicted_confidence=ev.predicted_confidence,
                corrected_label=ev.corrected_label,
                confirmed=ev.confirmed,
                image_url=f"/media/{ev.image_path}",
                crop=ev.crop,
                tags=ev.tags or [],
            )
            for ev in rows
        ],
    }

@router.get("/review/queue", response_model=PaginatedResponse[PredictionHistoryItem])
def review_queue(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
):
    base = db.query(PredictionEvent).filter(
        PredictionEvent.predicted_confidence < 0.90,
        or_(PredictionEvent.confirmed.is_(False), PredictionEvent.confirmed.is_(None)),
    )

    total = base.count()

    rows = (
        base.order_by(PredictionEvent.predicted_confidence.asc(), PredictionEvent.id.desc())
            .offset(skip)
            .limit(limit)
            .all()
    )

    return {
        "total": total,
        "items": [
            PredictionHistoryItem(
                id=ev.id,
                created_at=ev.created_at,
                predicted_label=ev.predicted_label,
                predicted_confidence=ev.predicted_confidence,
                corrected_label=ev.corrected_label,
                confirmed=ev.confirmed,
                image_url=f"/media/{ev.image_path}",
                crop=ev.crop,
                tags=ev.tags or [],
            )
            for ev in rows
        ],
    }

@router.get("/tags", response_model=List[str])
def all_tags(
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
):
    # Aggregate distinct tags in Python (portable across DBs)
    rows = db.query(PredictionEvent.tags).all()
    seen = set()
    for (tlist,) in rows:
        if not tlist:
            continue
        for t in tlist:
            if t:
                seen.add(normalize_tag(t))
    return sorted(seen)

@router.get("/crops", response_model=List[str])
def all_crops(
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
):
    # Prefer model classes; fallback to DB
    settings = get_settings()
    model_path = os.path.join(settings.models_path, settings.model_file)
    inference.ensure_loaded(model_path)
    crops = set()
    for label in getattr(inference, "class_names", []) or []:
        crop, _ = parse_crop_disease(label)
        if crop:
            crops.add(crop)
    if not crops:
        # fallback to DB crops
        for (c,) in db.query(PredictionEvent.crop).filter(PredictionEvent.crop.isnot(None)).distinct():
            crops.add(c)
    return sorted(crops)

@router.get("/diseases", response_model=List[str])
def all_diseases(
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
):
    settings = get_settings()
    model_path = os.path.join(settings.models_path, settings.model_file)
    inference.ensure_loaded(model_path)
    diseases = set()
    for label in getattr(inference, "class_names", []) or []:
        _crop, disease = parse_crop_disease(label)
        if disease:
            diseases.add(disease)
    if not diseases:
        # fallback to DB labels
        for (pl,) in db.query(PredictionEvent.predicted_label).distinct():
            if not pl:
                continue
            _c, d = parse_crop_disease(pl)
            if d:
                diseases.add(d)
        for (cl,) in db.query(PredictionEvent.corrected_label).filter(PredictionEvent.corrected_label.isnot(None)).distinct():
            _c, d = parse_crop_disease(cl)
            if d:
                diseases.add(d)
    return sorted(diseases)
