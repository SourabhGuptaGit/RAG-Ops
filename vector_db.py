from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from qdrant_client.http.exceptions import UnexpectedResponse
from custom_types import RAGSearchResult, RAGUpsertResult
from qdrant_client.models import Prefetch


class QdrantStorage:
    def __init__(self, url="http://localhost:6333", collection="docs", dim=3072):
        self.client = QdrantClient(url=url, timeout=30)
        self.collection = collection

        try:
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
            )
        except UnexpectedResponse:
            pass  # collection already exists
    
    def upsert(self, ids, vectors, payloads):
        points = [PointStruct(id=ids[i], vector=vectors[i], payload=payloads[i]) for i in range(len(ids))]
        self.client.upsert(self.collection, points=points)
        
    def search(self, query_vector, top_k: int = 5):
        results = self.client.query_points(
            collection_name=self.collection,
            query=query_vector,        # 🔑 REQUIRED
            prefetch=[
                Prefetch(
                    query=query_vector,
                    limit=top_k
                )
            ],
            with_payload=True,
            limit=top_k
        )

        contexts = []
        sources = set()

        for point in results.points:
            payload = point.payload or {}
            text = payload.get("text")
            source = payload.get("source")

            if text:
                contexts.append(text)
                if source:
                    sources.add(source)

        return {
            "contexts": contexts,
            "sources": list(sources)
        }
