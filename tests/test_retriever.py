from pathlib import Path

from src.rag.core.chunker import split_text_into_chunks
from src.rag.core.embedder import SimpleTFIDFEmbedder
from src.rag.core.retriever import Retriever


def test_retriever_returns_relevant_chunk(tmp_path: Path) -> None:
    text = (
        "Python is a programming language.\n"
        "It emphasizes readability.\n"
        "RAG combines retrieval and generation.\n"
    )

    chunks = split_text_into_chunks(text, max_chars=60, overlap=10)
    retriever = Retriever(SimpleTFIDFEmbedder())
    retriever.index(chunks)

    results = retriever.retrieve("What is RAG?", k=1)
    assert results
    assert "RAG combines retrieval and generation" in results[0].content
