from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field, ConfigDict


class DeviceMetrics(BaseModel):
    cpu_percent: float = Field(ge=0.0)
    mem_percent: float = Field(ge=0.0)
    disk_percent: float = Field(ge=0.0)
    temp_c: Optional[float] = Field(default=None)
    uptime_seconds: Optional[int] = Field(default=None, ge=0)
    camera_status: Optional[str] = Field(default=None)
    extra: Optional[dict[str, Any]] = Field(default=None)


class DeviceBase(BaseModel):
    id: int
    device_id: str
    name: str
    location: Optional[str] = None
    tags: list[str] = Field(default_factory=list)
    is_active: bool = True
    last_seen: Optional[datetime] = None
    last_metrics: Optional[DeviceMetrics] = None
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class DeviceCreate(BaseModel):
    name: str
    location: Optional[str] = None
    tags: list[str] = Field(default_factory=list)
    device_id: Optional[str] = None


class DeviceUpdate(BaseModel):
    name: Optional[str] = None
    location: Optional[str] = None
    tags: Optional[list[str]] = None
    is_active: Optional[bool] = None


class DeviceOut(DeviceBase):
    pass


class DeviceOutWithKey(DeviceBase):
    api_key: str = Field(description="Device API key (return only once)")


class DeviceHeartbeatRequest(BaseModel):
    metrics: DeviceMetrics


class DeviceListResponse(BaseModel):
    total: int
    items: list[DeviceOut]


class DeviceCredentials(BaseModel):
    device_id: str
    api_key: str
