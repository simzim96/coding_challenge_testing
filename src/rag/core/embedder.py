from typing import List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class SimpleTFIDFEmbedder:
    """Fit a TF-IDF vectorizer and provide embedding + similarity utilities.

    This is intentionally simple and fast to avoid heavyweight model downloads.
    """

    def __init__(self) -> None:
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self._matrix = None

    def fit(self, texts: List[str]) -> None:
        self._matrix = self.vectorizer.fit_transform(texts)

    def embed(self, texts: List[str]):
        return self.vectorizer.transform(texts)

    def top_k_similar(self, query: str, corpus: List[str], k: int = 3) -> List[int]:
        if self._matrix is None:
            raise RuntimeError("Embedder must be fit before retrieval.")
        query_vec = self.embed([query])
        sims = cosine_similarity(query_vec, self._matrix)[0]
        top_indices = np.argsort(-sims)[:k]
        return top_indices.tolist()
