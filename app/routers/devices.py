from __future__ import annotations

import secrets
import uuid
from datetime import datetime
from typing import List

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from ..models.device import Device
from ..schemas.device import (
    DeviceCreate,
    DeviceOut,
    DeviceOutWithKey,
    DeviceHeartbeatRequest,
    DeviceListResponse,
    DeviceUpdate,
)
from ..services.auth import get_db, get_current_user, get_password_hash
from ..services.device_auth import require_device

router = APIRouter(prefix="/v1/devices", tags=["devices"])


def _normalize_tags(tags: List[str] | None) -> List[str]:
    if not tags:
        return []
    cleaned = set()
    for tag in tags:
        value = (tag or "").strip().lower()
        if not value:
            continue
        value = value.replace(" ", "-")
        cleaned.add(value)
    return sorted(cleaned)


def _generate_device_id() -> str:
    return uuid.uuid4().hex[:12]


@router.post("", response_model=DeviceOutWithKey, status_code=status.HTTP_201_CREATED)
def register_device(
    payload: DeviceCreate,
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
):
    device_id = payload.device_id or _generate_device_id()
    existing = db.query(Device).filter(Device.device_id == device_id).first()
    if existing:
        raise HTTPException(status_code=409, detail="device_id already exists")

    api_key = secrets.token_urlsafe(32)
    device = Device(
        device_id=device_id,
        name=payload.name,
        location=payload.location,
        tags=_normalize_tags(payload.tags),
        api_key_hash=get_password_hash(api_key),
    )
    db.add(device)
    db.commit()
    db.refresh(device)

    base = DeviceOut.from_orm(device)
    return DeviceOutWithKey(**base.model_dump(), api_key=api_key)


@router.get("", response_model=DeviceListResponse)
def list_devices(
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
):
    devices = db.query(Device).order_by(Device.created_at.desc()).all()
    return DeviceListResponse(
        total=len(devices),
        items=[DeviceOut.from_orm(device) for device in devices],
    )


@router.get("/{device_id}", response_model=DeviceOut)
def get_device(
    device_id: str,
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
):
    device = db.query(Device).filter(Device.device_id == device_id).first()
    if not device:
        raise HTTPException(status_code=404, detail="Device not found")
    return DeviceOut.from_orm(device)


@router.patch("/{device_id}", response_model=DeviceOut)
def update_device(
    device_id: str,
    payload: DeviceUpdate,
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
):
    device = db.query(Device).filter(Device.device_id == device_id).first()
    if not device:
        raise HTTPException(status_code=404, detail="Device not found")

    if payload.name is not None:
        device.name = payload.name
    if payload.location is not None:
        device.location = payload.location
    if payload.tags is not None:
        device.tags = _normalize_tags(payload.tags)
    if payload.is_active is not None:
        device.is_active = payload.is_active

    db.add(device)
    db.commit()
    db.refresh(device)
    return DeviceOut.from_orm(device)


@router.post("/{device_id}/heartbeat", response_model=DeviceOut)
def device_heartbeat(
    device_id: str,
    payload: DeviceHeartbeatRequest,
    db: Session = Depends(get_db),
    device=Depends(require_device),
):
    if device.device_id != device_id:
        raise HTTPException(status_code=403, detail="Device credential mismatch")

    device.last_seen = datetime.utcnow()
    device.last_metrics = payload.metrics.model_dump()
    db.add(device)
    db.commit()
    db.refresh(device)

    return DeviceOut.from_orm(device)
