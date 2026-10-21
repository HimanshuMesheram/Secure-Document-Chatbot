# Secure Document Chatbot

A document question-answering application built with Streamlit,
LangChain, LangGraph, Groq, Chroma, SQLite, RAGAS and LangSmith.

## Features

- PDF and UTF-8 TXT document ingestion
- Semantic retrieval using local sentence-transformer embeddings
- Groq-powered answer generation
- LangGraph retrieval and generation workflow
- SQLite conversation history
- RAGAS evaluation
- Optional LangSmith tracing
- Basic input validation and document ownership filtering
- Docker deployment

## Prerequisites

- Python 3.11
- A Groq API key with model access
- Internet access for API requests and initial model download

## Local installation

Create and activate a virtual environment, install requirements, and
set `GROQ_API_KEY` in the existing project-root `.env` file.

Run the UI:

    streamlit run app/main.py

## Evaluation

Edit `evaluation/dataset.jsonl`, then run:

    python -m evaluation.evaluate

## Tests

    pytest -q


## Security warning

The local owner ID is generated for a Streamlit browser session.
It is not production authentication. Add trusted authentication,
server-side identity verification, persistent user records,
document lifecycle management, HTTPS and appropriate access policies
before using confidential or multi-user documents.

## Model configuration

Set GROQ_API_KEY and GROQ_MODEL in the deployment environment.
The embedding model runs locally; it does not use the Groq API.