from typing import Optional

from pydantic import BaseModel


class RAGRequest(BaseModel):
    query: str
    limit: int = 3
    min_score: Optional[float] = None


class RAGResponse(BaseModel):
    query: str
    answer: str
    metadata: list[dict]
