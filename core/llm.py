from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

from core.config import (
    EMBEDDING_MODEL,
    GROQ_API_KEY,
    GROQ_MODEL,
)


@lru_cache(maxsize=1)
def get_llm() -> ChatGroq:
    """Create the Groq chat model."""
    return ChatGroq(
        model=GROQ_MODEL,
        groq_api_key=GROQ_API_KEY,
        temperature=0,
        max_tokens=1200,
    )


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    """Create a local embedding model for semantic retrieval."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )