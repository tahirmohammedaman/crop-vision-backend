import json
import os
from fastapi import APIRouter, Depends, HTTPException, Query
from typing import List, Optional
from ..core.config import get_settings
from ..schemas.disease import DiseaseOut
from ..services.auth import get_current_user

router = APIRouter(prefix="/v1/catalog", tags=["catalog"])

def _load_catalog():
    settings = get_settings()
    path = os.path.join(settings.models_path, "disease_catalog.json")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="disease_catalog.json not found")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

@router.get("/diseases", response_model=List[DiseaseOut])
def list_diseases(
    crop: Optional[str] = Query(None),
    user=Depends(get_current_user),
):
    data = _load_catalog()
    items = data if isinstance(data, list) else data.get("diseases", [])
    if crop:
        items = [d for d in items if d.get("crop") == crop]
    return items

@router.get("/diseases/{slug}", response_model=DiseaseOut)
def get_disease(slug: str, user=Depends(get_current_user)):
    data = _load_catalog()
    items = data if isinstance(data, list) else data.get("diseases", [])
    for d in items:
        if d.get("slug") == slug:
            return d
    raise HTTPException(status_code=404, detail="Disease not found")