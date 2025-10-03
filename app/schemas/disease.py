from typing import List, Optional
from pydantic import BaseModel, Field

class DiseaseOut(BaseModel):
    slug: str
    crop: str
    name: str
    description: str
    symptoms: List[str] = []
    images: List[str] = []
    recommendations: List[str] = []
    pesticide_notes: Optional[str] = Field(default=None, description="Herbicide/Pesticide recommendations and PHI notes")