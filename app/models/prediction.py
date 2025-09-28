from datetime import datetime
from typing import Optional
from sqlalchemy import String, DateTime, ForeignKey, Float
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
    predicted_label: Mapped[str] = mapped_column(String(128))
    predicted_confidence: Mapped[float] = mapped_column(Float)
    probabilities: Mapped[dict] = mapped_column(JSONType)  # {class_name: prob}
    confirmed: Mapped[Optional[bool]] = mapped_column(default=None)  # None=not reviewed
    corrected_label: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)
    confirmed_at: Mapped[Optional[datetime]] = mapped_column(nullable=True)

    user = relationship("User")
