from datetime import datetime, timezone
from typing import List, Optional, cast
import json
import mimetypes
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, Query, Form
from fastapi.responses import JSONResponse, FileResponse
from sqlalchemy.orm import Session, selectinload
from sqlalchemy import or_, func
from PIL import Image
import os
import re

from ..services.auth import (
    get_db,
    get_current_user,
    get_current_user_optional,
)
from ..services.device_auth import get_current_device_optional, require_device
from ..services.storage import save_upload
from ..services import inference
from ..core.config import get_settings
from ..models.prediction import PredictionEvent
from ..models.device import Device
from ..schemas.prediction import (
    PredictionResponse,
    PredictionHistoryItem,
    PredictionHistoryUser,
    FeedbackRequest,
    PaginatedResponse,
    OriginLiteral,
)

router = APIRouter(prefix="/v1", tags=["predictions"])

TAG_CLEAN_RE = re.compile(r"[^a-z0-9\-:]+")
ALLOWED_ORIGINS = {"server_web", "server_edge", "device_offline"}


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


def parse_device_timestamp(
    value: Optional[str], field_name: str = "device_local_timestamp"
) -> Optional[datetime]:
    if not value:
        return None
    ts_value = value.strip()
    if not ts_value:
        return None
    if ts_value.endswith("Z"):
        ts_value = ts_value.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(ts_value)
    except ValueError as exc:
        raise HTTPException(
            status_code=400, detail=f"{field_name} must be ISO-8601"
        ) from exc
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
    return parsed


def parse_probabilities_json(value: Optional[str]) -> dict[str, float]:
    if not value:
        return {}
    try:
        raw = json.loads(value)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=400, detail="probabilities_json must be valid JSON object"
        ) from exc
    if not isinstance(raw, dict):
        raise HTTPException(
            status_code=400, detail="probabilities_json must be an object"
        )
    prob_map: dict[str, float] = {}
    for key, val in raw.items():
        if not isinstance(key, str):
            raise HTTPException(
                status_code=400, detail="probabilities_json keys must be strings"
            )
        try:
            prob_map[key] = float(val)
        except (TypeError, ValueError) as exc:
            raise HTTPException(
                status_code=400, detail=f"Invalid probability for class '{key}'"
            ) from exc
    return prob_map


def ensure_model_classes(settings) -> List[str]:
    model_path = os.path.join(settings.models_path, settings.model_file)
    inference.ensure_loaded(model_path)
    return getattr(inference, "class_names", []) or []


def _serialize_prediction_events(
    events: List[PredictionEvent],
) -> List[PredictionHistoryItem]:
    items: List[PredictionHistoryItem] = []
    for ev in events:
        prob_map = {
            str(label): float(value)
            for label, value in (ev.probabilities or {}).items()
        }
        user_summary = None
        if ev.user:
            user_summary = PredictionHistoryUser(
                id=ev.user.id, username=ev.user.username
            )
        items.append(
            PredictionHistoryItem(
                id=ev.id,
                created_at=ev.created_at,
                predicted_label=ev.predicted_label,
                predicted_confidence=ev.predicted_confidence,
                corrected_label=ev.corrected_label,
                confirmed=ev.confirmed,
                confirmed_at=ev.confirmed_at,
                image_url=f"/media/{ev.image_path}",
                crop=ev.crop,
                tags=ev.tags or [],
                origin=cast(OriginLiteral, ev.origin or "server_web"),
                device_id=ev.device_id,
                device_local_timestamp=ev.device_local_timestamp,
                probabilities=prob_map,
                user_id=ev.user_id,
                user=user_summary,
            )
        )
    return items


@router.post("/predict", response_model=PredictionResponse)
def predict(
    file: UploadFile = File(...),
    tags: Optional[List[str]] = Form(None, description="Optional list of tags"),
    origin: Optional[str] = Form(
        None,
        description="Origin of the prediction (server_web, server_edge, device_offline)",
    ),
    device_id: Optional[str] = Form(None, description="Registered device identifier"),
    device_local_timestamp: Optional[str] = Form(
        None, description="ISO-8601 timestamp from device when prediction was made"
    ),
    db: Session = Depends(get_db),
    user=Depends(get_current_user_optional),
    device=Depends(get_current_device_optional),
):
    settings = get_settings()
    classes = ensure_model_classes(settings)
    # Save upload
    rel_path = save_upload(settings.media_root, file)
    abs_path = os.path.join(settings.media_root, rel_path)

    # Load model and infer
    img = Image.open(abs_path).convert("RGB")
    predicted_class, probs, top_conf, _ = inference.predict_pil(img)

    # Derive crop/disease (must not be null)
    crop, _disease = parse_crop_disease(predicted_class)
    if crop is None:
        # This should never happen if class names use "Crop___Disease" format
        raise HTTPException(
            status_code=500, detail="Unable to parse crop from predicted class"
        )

    # Determine auth context
    if not user and not device:
        raise HTTPException(status_code=401, detail="Authentication required")

    device_obj: Optional[Device] = None
    if device:
        if device_id and device.device_id != device_id:
            raise HTTPException(status_code=403, detail="Device ID mismatch")
        device_id = device.device_id

    if device_id:
        device_obj = (
            db.query(Device)
            .filter(Device.device_id == device_id, Device.is_active.is_(True))
            .first()
        )
        if not device_obj:
            raise HTTPException(status_code=400, detail="Unknown or inactive device_id")

    resolved_origin = origin or ("server_edge" if device_obj else "server_web")
    if resolved_origin not in ALLOWED_ORIGINS:
        raise HTTPException(status_code=400, detail="Invalid origin value")

    if resolved_origin in {"server_edge", "device_offline"}:
        if not device_obj:
            raise HTTPException(
                status_code=400,
                detail="device_id is required when origin is server_edge or device_offline",
            )

    parsed_device_ts = parse_device_timestamp(device_local_timestamp)

    # Normalize tags
    norm_tags: list[str] = []
    if tags:
        for t in tags:
            nt = normalize_tag(t)
            if nt:
                norm_tags.append(nt)

    # Auto-tags based on origin/device
    norm_tags.append(normalize_tag(f"origin:{resolved_origin}"))
    if resolved_origin == "server_edge":
        norm_tags.append(normalize_tag("edge"))
    elif resolved_origin == "device_offline":
        norm_tags.append(normalize_tag("local"))
    if device_obj:
        norm_tags.append(normalize_tag(f"device:{device_obj.device_id}"))

    # Remove empties and de-duplicate
    norm_tags = sorted({t for t in norm_tags if t})

    # Persist
    prob_map = {cls: float(p) for cls, p in zip(classes, probs)}
    ev = PredictionEvent(
        user_id=user.id if user else None,
        image_path=rel_path,
        predicted_label=predicted_class,
        predicted_confidence=float(top_conf),
        probabilities=prob_map,
        crop=crop,  # enforce non-null
        tags=norm_tags,
        origin=resolved_origin,
        device_id=device_obj.device_id if device_obj else None,
        device_local_timestamp=parsed_device_ts,
    )
    db.add(ev)
    if device_obj:
        device_obj.last_seen = datetime.utcnow()
        db.add(device_obj)
    db.commit()
    db.refresh(ev)

    return PredictionResponse(
        id=ev.id,
        predicted_class=predicted_class,
        confidence=float(top_conf),
        classes=classes,
        probabilities=[float(p) for p in probs],
        image_url=f"/media/{rel_path}",
        origin=cast(OriginLiteral, resolved_origin),
        device_id=device_obj.device_id if device_obj else None,
        device_local_timestamp=ev.device_local_timestamp,
        tags=ev.tags or [],
    )


@router.post("/devices/{device_id}/predictions", response_model=PredictionResponse)
def sync_device_prediction(
    device_id: str,
    file: UploadFile = File(...),
    predicted_label: str = Form(
        ..., description="Predicted class label (e.g., Crop___Disease)"
    ),
    predicted_confidence: float = Form(
        ..., description="Confidence score for the predicted label"
    ),
    probabilities_json: Optional[str] = Form(
        None,
        description="Optional JSON object mapping class labels to probabilities",
    ),
    tags: Optional[List[str]] = Form(None, description="Optional list of tags"),
    device_local_timestamp: Optional[str] = Form(
        None, description="ISO-8601 timestamp when prediction was computed on device"
    ),
    db: Session = Depends(get_db),
    device=Depends(require_device),
):
    if device.device_id != device_id:
        raise HTTPException(status_code=403, detail="Device credential mismatch")

    settings = get_settings()
    classes = ensure_model_classes(settings)

    # Persist uploaded image
    rel_path = save_upload(settings.media_root, file)

    parsed_device_ts = parse_device_timestamp(device_local_timestamp)

    crop, _ = parse_crop_disease(predicted_label)
    if crop is None:
        raise HTTPException(
            status_code=400,
            detail="predicted_label must include crop (format 'Crop___Disease')",
        )

    prob_map = parse_probabilities_json(probabilities_json)
    if not prob_map:
        prob_map = {predicted_label: float(predicted_confidence)}
    else:
        prob_map.setdefault(predicted_label, float(predicted_confidence))

    probabilities_dict = {k: float(v) for k, v in prob_map.items()}

    norm_tags: list[str] = []
    if tags:
        for t in tags:
            nt = normalize_tag(t)
            if nt:
                norm_tags.append(nt)

    norm_tags.append(normalize_tag("origin:device_offline"))
    norm_tags.append(normalize_tag("local"))
    norm_tags.append(normalize_tag(f"device:{device.device_id}"))
    norm_tags = sorted({t for t in norm_tags if t})

    event_kwargs = dict(
        image_path=rel_path,
        predicted_label=predicted_label,
        predicted_confidence=float(predicted_confidence),
        probabilities=probabilities_dict,
        crop=crop,
        tags=norm_tags,
        origin="device_offline",
        device_id=device.device_id,
        device_local_timestamp=parsed_device_ts,
    )
    if parsed_device_ts is not None:
        event_kwargs["created_at"] = parsed_device_ts

    ev = PredictionEvent(**event_kwargs)
    db.add(ev)

    device.last_seen = datetime.utcnow()
    db.add(device)

    db.commit()
    db.refresh(ev)

    probabilities_list = [float(probabilities_dict.get(cls, 0.0)) for cls in classes]
    if not classes:
        probabilities_list = [float(v) for v in probabilities_dict.values()]

    return PredictionResponse(
        id=ev.id,
        predicted_class=predicted_label,
        confidence=float(predicted_confidence),
        classes=classes,
        probabilities=probabilities_list,
        image_url=f"/media/{rel_path}",
        origin=cast(OriginLiteral, "device_offline"),
        device_id=device.device_id,
        device_local_timestamp=ev.device_local_timestamp,
        tags=ev.tags or [],
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
                raise HTTPException(
                    status_code=500, detail="Unable to parse crop from predicted label"
                )
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
            raise HTTPException(
                status_code=400,
                detail="corrected_label must include crop (format 'Crop___Disease')",
            )
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
    confirmed: Optional[bool] = Query(
        None, description="True=only confirmed, False=only unconfirmed, None=all"
    ),
    origin: Optional[str] = Query(None, description="Filter by origin"),
    device_id_filter: Optional[str] = Query(None, alias="device_id"),
    search: Optional[str] = Query(
        None,
        description="Wildcard search across crop, disease, tags, or device id",
    ),
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
):
    q = db.query(PredictionEvent).options(selectinload(PredictionEvent.user))

    if crop:
        q = q.filter(PredictionEvent.crop == crop)

    if disease:
        term = disease.strip().lower()
        if term:
            pattern = f"%{term}%"
            # match when predicted or corrected label ends with the disease term (case-insensitive)
            pattern = f"%{term}"
            q = q.filter(
                or_(
                    func.lower(PredictionEvent.predicted_label).like(pattern),
                    func.lower(PredictionEvent.corrected_label).like(pattern),
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
        q = q.filter(
            or_(
                PredictionEvent.confirmed.is_(False),
                PredictionEvent.confirmed.is_(None),
            )
        )

    if origin:
        if origin not in ALLOWED_ORIGINS:
            raise HTTPException(status_code=400, detail="Invalid origin filter")
        q = q.filter(PredictionEvent.origin == origin)

    if device_id_filter:
        q = q.filter(PredictionEvent.device_id == device_id_filter)

    if tags:
        norm = [normalize_tag(t) for t in tags if normalize_tag(t)]
        if norm:
            q = q.filter(PredictionEvent.tags.contains(norm))

    if search:
        term = search.strip()
        if term:
            pattern = f"%{term.lower()}%"
            tag_expr = func.lower(
                func.coalesce(func.array_to_string(PredictionEvent.tags, ","), "")
            )
            q = q.filter(
                or_(
                    func.lower(PredictionEvent.crop).like(pattern),
                    func.lower(PredictionEvent.predicted_label).like(pattern),
                    func.lower(PredictionEvent.corrected_label).like(pattern),
                    func.lower(PredictionEvent.device_id).like(pattern),
                    tag_expr.like(pattern),
                )
            )

    total = q.count()

    rows = q.order_by(PredictionEvent.id.desc()).offset(skip).limit(limit).all()

    return {
        "total": total,
        "items": _serialize_prediction_events(rows),
    }


@router.get("/review/queue", response_model=PaginatedResponse[PredictionHistoryItem])
def review_queue(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
):
    base = (
        db.query(PredictionEvent)
        .options(selectinload(PredictionEvent.user))
        .filter(
            PredictionEvent.predicted_confidence < 0.90,
            or_(
                PredictionEvent.confirmed.is_(False),
                PredictionEvent.confirmed.is_(None),
            ),
        )
    )

    total = base.count()

    rows = (
        base.order_by(
            PredictionEvent.predicted_confidence.asc(), PredictionEvent.id.desc()
        )
        .offset(skip)
        .limit(limit)
        .all()
    )

    return {
        "total": total,
        "items": _serialize_prediction_events(rows),
    }


@router.get("/media/{image_path:path}")
def get_uploaded_image(
    image_path: str,
    _user=Depends(get_current_user),
):
    settings = get_settings()
    media_root = os.path.abspath(settings.media_root)
    candidate = os.path.abspath(os.path.join(media_root, image_path))

    try:
        common = os.path.commonpath([media_root, candidate])
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid media path")

    if common != media_root:
        raise HTTPException(status_code=400, detail="Invalid media path")

    if not os.path.isfile(candidate):
        raise HTTPException(status_code=404, detail="Image not found")

    media_type, _ = mimetypes.guess_type(candidate)
    return FileResponse(candidate, media_type=media_type or "application/octet-stream")


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
        for (c,) in (
            db.query(PredictionEvent.crop)
            .filter(PredictionEvent.crop.isnot(None))
            .distinct()
        ):
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
        for (cl,) in (
            db.query(PredictionEvent.corrected_label)
            .filter(PredictionEvent.corrected_label.isnot(None))
            .distinct()
        ):
            _c, d = parse_crop_disease(cl)
            if d:
                diseases.add(d)
    return sorted(diseases)
