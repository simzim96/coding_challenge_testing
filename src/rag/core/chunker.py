from typing import List


def split_text_into_chunks(
    text: str, max_chars: int = 800, overlap: int = 100
) -> List[str]:
    """Simple character-based chunker with overlap.

    Args:
        text: Full input text to split.
        max_chars: Maximum characters per chunk.
        overlap: Overlapping characters between consecutive chunks.

    Returns:
        List of chunk strings.
    """
    if not text:
        return []

    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    if overlap < 0:
        raise ValueError("overlap must be non-negative")
    if overlap >= max_chars:
        # Avoid infinite or zero-progress steps
        overlap = max_chars // 4

    chunks: List[str] = []
    start_index = 0
    length = len(text)
    while start_index < length:
        end_index = min(start_index + max_chars, length)
        chunk = text[start_index:end_index].strip()
        if chunk:
            chunks.append(chunk)
        if end_index == length:
            break
        start_index = end_index - overlap
    return chunks
