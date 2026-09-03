from typing import Any, Dict, Optional

from models.search import SearchResponse, SearchResult
from qdrant_client import QdrantClient, models

from services.embeddings import EmbeddingService


class SearchService:
    def __init__(self, qdrant_url: str, qdrant_api_key: str, collection_name: str):
        self.qdrant = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        self.collection_name = collection_name
        self.embedding_service = EmbeddingService()

    def _build_qdrant_filter(self, filters: Optional[Dict[str, Any]]) -> Optional[Dict]:
        if not filters:
            return None

        must_conditions = []
        for key, value in filters.items():
            must_conditions.append(
                {"key": f"metadata.{key}", "match": {"value": value}}
            )
        return {"must": must_conditions}

    def search(
        self,
        query: str,
        limit: int = 3,
        filter: Optional[Dict[str, Any]] = None,
        min_score: Optional[float] = None,
    ):
        query_dense, query_sparse, query_colbert = self.embedding_service.embed_query(
            query
        )

        query_filter = self._build_qdrant_filter(filter)

        results = self.qdrant.query_points(
            collection_name=self.collection_name,
            prefetch=[
                {
                    "prefetch": [
                        {"query": query_dense, "using": "dense", "limit": 20},
                        {"query": query_sparse, "using": "sparse", "limit": 20},
                    ],
                    "query": models.FusionQuery(fusion=models.Fusion.RRF),
                    "limit": 15,
                }
            ],
            query=query_colbert,
            using="colbert",
            limit=limit,
            query_filter=query_filter,
        )

        points = results.points

        if min_score is not None:
            points = [point for point in points if point.score >= min_score]

        # Nothing passed the filter or the query returned nothing: an empty
        # list instead of blowing up in max(). The caller decides what the
        # silence means.
        if not points:
            return SearchResponse(results=[])

        max_score = max(point.score for point in points)

        search_results = [
            SearchResult(
                score=point.score / max_score,
                raw_score=point.score,
                text=point.payload["text"],
                metadata=point.payload["metadata"],
            )
            for point in points
        ]

        return SearchResponse(results=search_results)
