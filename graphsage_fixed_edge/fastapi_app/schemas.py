from pydantic import BaseModel
from typing import List, Dict

# 추천 API 스키마
class RecommendRequest(BaseModel):
    embedding_path: str
    target_concept_id: int
    top_k: int
  
class RecommendResponse(BaseModel):
    recommendations: List[Dict] = []
