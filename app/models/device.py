from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from sqlalchemy import String, DateTime, Boolean
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.types import JSON
from sqlalchemy.orm import Mapped, mapped_column

from ..db.session import Base

try:  # pragma: no cover - JSONB available only on Postgres
    JSONType = JSONB
except Exception:  # pragma: no cover
    JSONType = JSON


class Device(Base):
    __tablename__ = "devices"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    device_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(128), index=True)
    location: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)
    tags: Mapped[list[str]] = mapped_column(JSONType, default=list, nullable=False)
    api_key_hash: Mapped[str] = mapped_column(String(256))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    last_seen: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    last_metrics: Mapped[Optional[dict[str, Any]]] = mapped_column(JSONType, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

