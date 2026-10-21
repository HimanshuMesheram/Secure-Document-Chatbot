from typing import TypedDict

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from core.history import load_history, save_history
from core.llm import get_llm
from core.security import validate_question
from rag.prompts import SYSTEM_PROMPT
from rag.retriever import retrieve_documents


class RAGState(TypedDict, total=False):
    owner_id: str
    thread_id: str
    question: str
    history: list[dict]
    documents: list[Document]
    answer: str


def retrieve_node(state: RAGState) -> dict:
    """Find relevant document chunks."""
    documents = retrieve_documents(
        question=state["question"],
        owner_id=state["owner_id"],
    )
    return {"documents": documents}


def generate_node(state: RAGState) -> dict:
    """Generate an answer from the retrieved context."""
    documents = state.get("documents", [])

    context_parts = []

    for index, document in enumerate(documents, start=1):
        source = document.metadata.get("source", "unknown")
        page = document.metadata.get("page", "?")
        context_parts.append(
            f"[S{index}] Source: {source}, page: {page}\n"
            f"{document.page_content}"
        )

    context = "\n\n".join(context_parts)

    if not context:
        return {
            "answer": (
                "I could not find relevant information in your uploaded "
                "documents. Please upload a relevant document or rephrase "
                "your question."
            )
        }

    history_lines = []

    for item in state.get("history", [])[-10:]:
        role = item.get("role", "")
        content = item.get("content", "")

        if role in {"user", "assistant"} and isinstance(content, str):
            history_lines.append(f"{role}: {content[:2000]}")

    history_text = "\n".join(history_lines) or "(No previous messages.)"

    prompt = SYSTEM_PROMPT.format(
        history=history_text,
        context=context,
        question=state["question"],
    )

    response = get_llm().invoke(
        [
            SystemMessage(content=prompt),
            HumanMessage(content=state["question"]),
        ]
    )

    answer = (
        response.content
        if isinstance(response.content, str)
        else str(response.content)
    )

    return {"answer": answer}


def build_graph():
    """Build and compile the LangGraph RAG workflow."""
    builder = StateGraph(RAGState)

    builder.add_node("retrieve", retrieve_node)
    builder.add_node("generate", generate_node)

    builder.add_edge(START, "retrieve")
    builder.add_edge("retrieve", "generate")
    builder.add_edge("generate", END)

    return builder.compile()


rag_graph = build_graph()


def ask_question(
    question: str,
    owner_id: str,
    thread_id: str,
) -> dict:
    """Run RAG and persist the resulting conversation."""
    question = validate_question(question)

    if not owner_id or not thread_id:
        raise PermissionError("Owner and conversation IDs are required.")

    history = load_history(owner_id, thread_id)

    result = rag_graph.invoke(
        {
            "owner_id": owner_id,
            "thread_id": thread_id,
            "question": question,
            "history": history,
        },
        config={
            "run_name": "secure_document_rag",
            "tags": ["rag", "document-chatbot"],
            "metadata": {
                "thread_id": thread_id,
            },
        },
    )

    answer = result["answer"]

    # Store conversation turns separately from the vector database.
    updated_history = history + [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]
    save_history(owner_id, thread_id, updated_history)

    return {
        "answer": answer,
        "sources": [
            {
                "source": document.metadata.get("source", "unknown"),
                "page": document.metadata.get("page", "?"),
                "content": document.page_content,
            }
            for document in result.get("documents", [])
        ],
    }