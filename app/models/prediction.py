from datetime import datetime
from typing import Optional
from sqlalchemy import String, ForeignKey, Float, DateTime
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.types import JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship
from ..db.session import Base

try:
    JSONType = JSONB  # use JSONB if Postgres
except Exception:
    JSONType = JSON


class PredictionEvent(Base):
    __tablename__ = "predictions"
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    user_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("users.id"), nullable=True
    )
    image_path: Mapped[str] = mapped_column(String(512))
    predicted_label: Mapped[str] = mapped_column(String(128), index=True)
    predicted_confidence: Mapped[float] = mapped_column(Float, index=True)
    probabilities: Mapped[dict] = mapped_column(JSONType)  # {class_name: prob}
    crop: Mapped[str] = mapped_column(String(128), index=True)
    tags: Mapped[list[str]] = mapped_column(JSONType, nullable=False, default=list)
    origin: Mapped[str] = mapped_column(String(32), default="server_web", index=True)
    device_id: Mapped[Optional[str]] = mapped_column(
        ForeignKey("devices.device_id"), nullable=True, index=True
    )
    device_local_timestamp: Mapped[Optional[datetime]] = mapped_column(
        DateTime, nullable=True
    )
    confirmed: Mapped[Optional[bool]] = mapped_column(default=None)  # None=not reviewed
    corrected_label: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow, index=True)
    confirmed_at: Mapped[Optional[datetime]] = mapped_column(nullable=True)

    user = relationship("User")
    device = relationship(
        "Device",
        primaryjoin="Device.device_id==PredictionEvent.device_id",
        foreign_keys=[device_id],
        viewonly=True,
    )
