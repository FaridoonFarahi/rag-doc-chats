from __future__ import annotations

from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.document_loaders.word_document import Docx2txtLoader


def load_documents(file_paths: List[Path]) -> List[Document]:
    """
    Takes a list of file paths and returns LangChain Document objects.
    Each Document contains:
      - page_content: the text
      - metadata: info like source filename, page number, etc.
    """
    docs: List[Document] = []

    for path in file_paths:
        suffix = path.suffix.lower()

        if suffix == ".pdf":
            # PyPDFLoader splits by page and adds metadata like page numbers.
            loader = PyPDFLoader(str(path))
            docs.extend(loader.load())

        elif suffix == ".txt":
            # TextLoader loads whole file as one Document (metadata includes source).
            loader = TextLoader(str(path), encoding="utf-8")
            docs.extend(loader.load())

        elif suffix == ".docx":
            # Docx2txtLoader extracts raw text from Word docs.
            loader = Docx2txtLoader(str(path))
            docs.extend(loader.load())

        else:
            # Skip unsupported formats (you can add more later).
            continue

    return docs
