import pytest

from core.security import (
    check_document_access,
    validate_question,
    validate_upload,
)


def test_valid_question():
    assert validate_question("What is RAG?") == "What is RAG?"


@pytest.mark.parametrize("question", ["", "   ", "\n\t"])
def test_empty_question_rejected(question):
    with pytest.raises(ValueError):
        validate_question(question)


def test_question_length_limit():
    with pytest.raises(ValueError):
        validate_question("x" * 10000)


def test_txt_upload():
    digest = validate_upload("notes.txt", b"Hello, RAG!")
    assert len(digest) == 64


def test_unsupported_extension():
    with pytest.raises(ValueError):
        validate_upload("script.py", b"print('hello')")


def test_invalid_pdf_header():
    with pytest.raises(ValueError):
        validate_upload("document.pdf", b"not a PDF")


def test_document_access():
    assert check_document_access("user-a", "user-a")
    assert not check_document_access("user-a", "user-b")
    assert not check_document_access("", "user-a")