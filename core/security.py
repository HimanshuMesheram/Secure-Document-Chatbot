import hashlib
import re
import uuid
from pathlib import Path

from core.config import MAX_QUESTION_CHARS, MAX_UPLOAD_MB

ALLOWED_EXTENSIONS = {".pdf", ".txt"}


def validate_question(question: str) -> str:
    """Validate a question before sending it to the LLM."""
    if not isinstance(question, str):
        raise ValueError("Question must be text.")

    question = question.strip()

    if not question:
        raise ValueError("Question cannot be empty.")

    if len(question) > MAX_QUESTION_CHARS:
        raise ValueError(
            f"Question exceeds {MAX_QUESTION_CHARS} characters."
        )

    if any(ord(char) < 32 and char not in "\n\t\r" for char in question):
        raise ValueError("Question contains invalid control characters.")

    return question


def validate_upload(filename: str, content: bytes) -> str:
    """Validate the file type, size and basic content."""
    if not filename or not isinstance(filename, str):
        raise ValueError("A filename is required.")

    suffix = Path(filename).suffix.lower()

    if suffix not in ALLOWED_EXTENSIONS:
        raise ValueError("Only PDF and TXT files are supported.")

    if not content:
        raise ValueError("The uploaded file is empty.")

    max_bytes = MAX_UPLOAD_MB * 1024 * 1024

    if len(content) > max_bytes:
        raise ValueError(
            f"File exceeds the {MAX_UPLOAD_MB} MB upload limit."
        )

    if suffix == ".pdf" and not content.startswith(b"%PDF-"):
        raise ValueError("The uploaded file is not a valid PDF header.")

    if suffix == ".txt":
        try:
            content.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("TXT files must use UTF-8 encoding.") from exc

    # Never use the original filename as a filesystem path.
    return hashlib.sha256(content).hexdigest()


def safe_owner_id() -> str:
    """Generate an opaque identity for a local browser session."""
    return uuid.uuid4().hex


def check_document_access(owner_id: str, document_owner_id: str) -> bool:
    """Check ownership before a document is returned."""
    return (
        bool(owner_id)
        and bool(document_owner_id)
        and owner_id == document_owner_id
    )


def safe_document_id() -> str:
    """Generate a random document identifier."""
    return uuid.uuid4().hex


def safe_display_name(filename: str) -> str:
    """Return a short, printable display name."""
    name = Path(filename).name
    name = re.sub(r"[\x00-\x1f\x7f]", "", name)
    return name[:150]