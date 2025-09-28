import json
import os
from fastapi import APIRouter, HTTPException
from ..core.config import get_settings

router = APIRouter(prefix="/v1", tags=["model"])


@router.get("/model/info")
def model_info():
    settings = get_settings()
    path = os.path.join(settings.models_path, "model_info.json")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="model_info.json not found")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
