import logging
import sys
from pathlib import Path

import streamlit as st

project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from core.config import validate_configuration
from core.history import load_history
from core.security import safe_owner_id
from rag.graph import ask_question
from rag.ingest import ingest_document

st.set_page_config(
    page_title="Secure Document Chatbot",
    page_icon="🔐",
    layout="wide",
)

st.title("🔐 Secure Document Chatbot")
st.caption("RAG · LangChain · LangGraph · Groq API")

try:
    validate_configuration()
except Exception as exc:
    st.error(str(exc))
    st.stop()

# Demo identity: this is not a replacement for production authentication.
if "owner_id" not in st.session_state:
    st.session_state.owner_id = safe_owner_id()

if "thread_id" not in st.session_state:
    st.session_state.thread_id = safe_owner_id()

if "messages" not in st.session_state:
    st.session_state.messages = load_history(
        st.session_state.owner_id,
        st.session_state.thread_id,
    )

with st.sidebar:
    st.header("Document management")

    uploaded_files = st.file_uploader(
        "Upload PDF or TXT documents",
        type=["pdf", "txt"],
        accept_multiple_files=True,
    )

    if st.button("Index uploaded documents", use_container_width=True):
        if not uploaded_files:
            st.info("Choose at least one document.")
        else:
            for uploaded_file in uploaded_files:
                try:
                    result = ingest_document(
                        owner_id=st.session_state.owner_id,
                        filename=uploaded_file.name,
                        content=uploaded_file.getvalue(),
                    )
                    st.success(
                        f"Indexed {result['filename']}: "
                        f"{result['chunks_indexed']} chunks"
                    )
                except Exception as exc:
                    st.error(
                        f"Could not index {uploaded_file.name}: {exc}"
                    )

    if st.button("Start a new conversation", use_container_width=True):
        st.session_state.thread_id = safe_owner_id()
        st.session_state.messages = []
        st.rerun()

    st.caption("Supported: text-based PDF and UTF-8 TXT.")

# Render previous turns.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input("Ask a question about your documents...")

if question:
    with st.chat_message("user"):
        st.markdown(question)

    try:
        with st.chat_message("assistant"):
            with st.spinner("Searching documents and asking Groq..."):
                result = ask_question(
                    question=question,
                    owner_id=st.session_state.owner_id,
                    thread_id=st.session_state.thread_id,
                )

            st.markdown(result["answer"])

            if result["sources"]:
                with st.expander("View retrieved source excerpts"):
                    for index, source in enumerate(
                        result["sources"], start=1
                    ):
                        st.markdown(
                            f"**[S{index}] {source['source']} — "
                            f"page {source['page']}**"
                        )
                        st.write(source["content"])

        # The history is also persisted in SQLite by ask_question().
        st.session_state.messages.extend(
            [
                {"role": "user", "content": question},
                {"role": "assistant", "content": result["answer"]},
            ]
        )

    except Exception as exc:
        logging.exception("Document chat request failed")
        st.error(
            f"Request failed ({type(exc).__name__}). "
            "Check the Streamlit terminal for details."
        )