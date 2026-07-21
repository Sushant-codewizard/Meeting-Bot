from __future__ import annotations
import logging

logger = logging.getLogger("meetingbot.embeddings")

_model = None

def load_model() -> None:
    """Mock load function to avoid loading heavy sentence-transformers into memory."""
    logger.info("event=embedding_model_loading_skipped_for_lightweight_mode")

def get_model():
    return None

def embed_texts(texts: list[str]) -> list[list[float]]:
    """Return clean dummy 384-dim zero vectors to bypass heavy embedding processing."""
    logger.info("event=embed_texts_skipped_for_lightweight_mode")
    return [[0.0] * 384 for _ in texts]

def chunk_text(text: str, chunk_words: int = 500, overlap_words: int = 50) -> list[str]:
    """Split into ~chunk_words word windows with overlap_words of overlap."""
    words = text.split()
    if not words:
        return []

    step = chunk_words - overlap_words
    chunks: list[str] = []
    start = 0
    while start < len(words):
        window = words[start : start + chunk_words]
        chunks.append(" ".join(window))
        if start + chunk_words >= len(words):
            break
        start += step
    return chunks

def _demo() -> None:
    words = [f"w{i}" for i in range(1200)]
    text = " ".join(words)
    chunks = chunk_text(text, chunk_words=500, overlap_words=50)
    assert len(chunks) == 3, chunks
    assert chunks[0].split()[0] == "w0"
    assert chunks[1].split()[0] == "w450"
    assert chunks[-1].split()[-1] == "w1199"
    assert chunk_text("") == []
    print("embeddings chunk_text self-check passed")

if __name__ == "__main__":
    _demo()
