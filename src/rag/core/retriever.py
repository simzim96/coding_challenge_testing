from dataclasses import dataclass
from typing import List


@dataclass
class Document:
    content: str
    metadata: dict


class Retriever:
    def __init__(self, embedder) -> None:
        self.embedder = embedder
        self.documents: List[Document] = []

    def index(self, chunks: List[str]) -> None:
        self.documents = [
            Document(content=c, metadata={"chunk_id": i}) for i, c in enumerate(chunks)
        ]
        self.embedder.fit([d.content for d in self.documents])

    def retrieve(self, query: str, k: int = 3) -> List[Document]:
        if not self.documents:
            return []
        indices = self.embedder.top_k_similar(
            query, [d.content for d in self.documents], k=k
        )
        return [self.documents[i] for i in indices]
