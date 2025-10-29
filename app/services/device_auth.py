from __future__ import annotations

from fastapi import Depends, Header, HTTPException, status
from sqlalchemy.orm import Session

from ..models.device import Device
from .auth import get_db, verify_password


DEVICE_ID_HEADER = "X-Device-ID"
DEVICE_KEY_HEADER = "X-Device-Key"


def get_current_device_optional(
    device_identifier: str | None = Header(None, alias=DEVICE_ID_HEADER),
    device_key: str | None = Header(None, alias=DEVICE_KEY_HEADER),
    db: Session = Depends(get_db),
) -> Device | None:
    if not device_identifier or not device_key:
        return None
    device = (
        db.query(Device)
        .filter(Device.device_id == device_identifier, Device.is_active.is_(True))
        .first()
    )
    if not device or not verify_password(device_key, device.api_key_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid device credentials",
        )
    return device


def require_device(device: Device | None = Depends(get_current_device_optional)) -> Device:
    if device is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Device credentials required",
        )
    return device
