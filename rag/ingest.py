import io
import uuid
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from core.config import CHROMA_DIR, UPLOAD_DIR
from core.llm import get_embeddings
from core.security import (
    safe_display_name,
    validate_upload,
)


def get_vector_store() -> Chroma:
    """Open the persistent Chroma collection."""
    return Chroma(
        collection_name="secure_documents",
        persist_directory=str(CHROMA_DIR),
        embedding_function=get_embeddings(),
    )


def ingest_document(
    owner_id: str,
    filename: str,
    content: bytes,
) -> dict:
    """Validate, parse, split and index one uploaded document."""
    if not owner_id:
        raise ValueError("An authenticated owner ID is required.")

    digest = validate_upload(filename, content)
    display_name = safe_display_name(filename)
    document_id = uuid.uuid4().hex

    # Save under a random filename rather than the user-provided path.
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    suffix = Path(display_name).suffix.lower()
    saved_path = UPLOAD_DIR / f"{document_id}{suffix}"
    saved_path.write_bytes(content)

    try:
        if suffix == ".txt":
            text = content.decode("utf-8")
            if not text.strip():
                raise ValueError("The text file has no readable content.")

            pages = [
                Document(
                    page_content=text,
                    metadata={"page": 1},
                )
            ]
        else:
            # PyPDFLoader accepts a filesystem path.
            pages = PyPDFLoader(str(saved_path)).load()

            if not any(page.page_content.strip() for page in pages):
                raise ValueError(
                    "No extractable text found. Scanned PDFs require OCR."
                )

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=900,
            chunk_overlap=150,
            add_start_index=True,
        )
        chunks = splitter.split_documents(pages)

        if not chunks:
            raise ValueError("No text chunks could be created.")

        for chunk in chunks:
            chunk.metadata.update(
                {
                    "owner_id": owner_id,
                    "document_id": document_id,
                    "source": display_name,
                    "content_sha256": digest,
                }
            )

        vector_store = get_vector_store()

        # Unique IDs prevent collisions between chunks from different files.
        vector_store.add_documents(
            documents=chunks,
            ids=[uuid.uuid4().hex for _ in chunks],
        )

        return {
            "document_id": document_id,
            "filename": display_name,
            "chunks_indexed": len(chunks),
        }

    except Exception:
        # Remove the staged upload if parsing/indexing fails.
        saved_path.unlink(missing_ok=True)
        raise


def delete_document(owner_id: str, document_id: str) -> None:
    """Delete only chunks belonging to the specified owner and document."""
    store = get_vector_store()

    # Filter on both owner and document ID.
    results = store.get(
        where={
            "$and": [
                {"owner_id": {"$eq": owner_id}},
                {"document_id": {"$eq": document_id}},
            ]
        },
        include=["metadatas"],
    )

    ids = results.get("ids", [])

    if ids:
        store.delete(ids=ids)

    # The original file is intentionally not deleted here unless a
    # trusted mapping from owner + document ID to its stored path exists.
    # Implement that mapping in production so deletion covers both stores.