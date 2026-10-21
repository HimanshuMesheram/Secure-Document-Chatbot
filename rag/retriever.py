from langchain_core.documents import Document

from core.config import RETRIEVAL_K
from core.security import check_document_access
from rag.ingest import get_vector_store


def retrieve_documents(
    question: str,
    owner_id: str,
    k: int = RETRIEVAL_K,
) -> list[Document]:
    """Retrieve semantically relevant chunks for one user."""
    if not owner_id:
        raise PermissionError("A valid owner ID is required.")

    store = get_vector_store()

    # This filter is applied by Chroma, before results are returned.
    documents = store.similarity_search(
        query=question,
        k=k,
        filter={"owner_id": owner_id},
    )

    # Defense in depth: verify every returned chunk's ownership.
    authorized = [
        document
        for document in documents
        if check_document_access(
            owner_id,
            str(document.metadata.get("owner_id", "")),
        )
    ]

    return authorized