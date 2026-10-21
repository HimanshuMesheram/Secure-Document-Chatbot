import os
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from the project-root .env file.
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

CHROMA_DIR = Path(
    os.getenv("CHROMA_DIR", str(BASE_DIR / "data" / "chroma"))
)
UPLOAD_DIR = Path(
    os.getenv("UPLOAD_DIR", str(BASE_DIR / "data" / "uploads"))
)
HISTORY_DB = Path(
    os.getenv("HISTORY_DB", str(BASE_DIR / "data" / "history.sqlite3"))
)

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)

MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "10"))
MAX_QUESTION_CHARS = int(os.getenv("MAX_QUESTION_CHARS", "2000"))
RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "4"))

# Only enable tracing when a key has actually been configured.
LANGSMITH_TRACING = (
    os.getenv("LANGSMITH_TRACING", "false").lower() == "true"
)
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY", "")
LANGSMITH_PROJECT = os.getenv(
    "LANGSMITH_PROJECT", "secure-rag-chatbot"
)
LANGSMITH_ENDPOINT = os.getenv(
    "LANGSMITH_ENDPOINT", "https://api.smith.langchain.com"
)


def initialize_directories() -> None:
    """Create the directories required by the local application."""
    for directory in (CHROMA_DIR, UPLOAD_DIR, HISTORY_DB.parent):
        directory.mkdir(parents=True, exist_ok=True)


def validate_configuration() -> None:
    """Fail early if the required Groq API key is missing."""
    if not GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY is missing. Add it to your .env file."
        )

    if MAX_UPLOAD_MB < 1:
        raise RuntimeError("MAX_UPLOAD_MB must be at least 1.")

    if MAX_QUESTION_CHARS < 1:
        raise RuntimeError("MAX_QUESTION_CHARS must be positive.")

    if RETRIEVAL_K < 1:
        raise RuntimeError("RETRIEVAL_K must be positive.")

    if LANGSMITH_TRACING and not LANGSMITH_API_KEY:
        raise RuntimeError(
            "Tracing is enabled, but LANGSMITH_API_KEY is missing."
        )

    initialize_directories()

    # Map the project configuration to LangSmith's standard environment.
    if LANGSMITH_TRACING:
        os.environ["LANGSMITH_TRACING"] = "true"
        os.environ["LANGSMITH_API_KEY"] = LANGSMITH_API_KEY
        os.environ["LANGSMITH_PROJECT"] = LANGSMITH_PROJECT
        os.environ["LANGSMITH_ENDPOINT"] = LANGSMITH_ENDPOINT
    else:
        os.environ["LANGSMITH_TRACING"] = "false"