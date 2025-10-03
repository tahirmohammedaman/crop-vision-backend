from datetime import datetime
from typing import Optional, List, Generic, TypeVar
from pydantic import BaseModel, Field
from pydantic.generics import GenericModel

T = TypeVar("T")


class PredictionResponse(BaseModel):
    id: int
    predicted_class: str
    confidence: float
    classes: List[str]
    probabilities: List[float]
    image_url: str


class PredictionHistoryItem(BaseModel):
    id: int
    created_at: datetime
    predicted_label: str
    predicted_confidence: float
    corrected_label: Optional[str] = None
    confirmed: Optional[bool] = None
    image_url: str
    crop: str
    tags: List[str] = Field(default_factory=list)


class PaginatedResponse(GenericModel, Generic[T]):
    total: int
    items: List[T]


class FeedbackRequest(BaseModel):
    is_correct: bool
    corrected_label: Optional[str] = Field(
        default=None, description="Required if not correct"
    )


class StatsResponse(BaseModel):
    total: int
    reviewed: int
    corrected: int
    accuracy_ratio: float
