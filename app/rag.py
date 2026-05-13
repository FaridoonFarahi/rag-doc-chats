from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple, Dict, Any

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from app.config import (
    UPLOAD_DIR,
    CHROMA_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    EMBEDDING_MODEL,
    CHAT_MODEL,
    TOP_K,
    MAX_UPLOAD_BYTES,
    MAX_CHUNKS_PER_INGEST,
)
from app.loaders import load_documents


class IngestError(Exception):
    """Raised when ingestion cannot proceed (oversized file, too many chunks, etc.)."""


def _require_api_key() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Copy .env.example to .env and add your key."
        )


def save_uploads(uploaded_files) -> List[Path]:
    """
    Save Streamlit UploadedFile objects to disk so loaders can read them reliably.
    Filenames are sanitized to prevent path traversal, and per-file size is capped.
    """
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    saved_paths: List[Path] = []
    for uf in uploaded_files:
        # Strip any directory components — only keep the basename.
        safe_name = Path(uf.name).name
        if not safe_name or safe_name in {".", ".."}:
            raise IngestError(f"Invalid filename: {uf.name!r}")

        buf = uf.getbuffer()
        if len(buf) > MAX_UPLOAD_BYTES:
            raise IngestError(
                f"{safe_name} exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB upload limit."
            )

        save_path = UPLOAD_DIR / safe_name
        save_path.write_bytes(buf)
        saved_paths.append(save_path)

    return saved_paths


def ingest_files(file_paths: List[Path], collection_name: str) -> Tuple[int, int]:
    """
    Ingest pipeline: load → split → embed → persist in Chroma under collection_name.
    Returns (num_documents_loaded, num_chunks_indexed).
    """
    _require_api_key()

    docs = load_documents(file_paths)
    if not docs:
        raise IngestError("No supported documents found in upload (PDF / DOCX / TXT only).")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(docs)

    if len(chunks) > MAX_CHUNKS_PER_INGEST:
        raise IngestError(
            f"Document(s) produced {len(chunks)} chunks, exceeding the "
            f"{MAX_CHUNKS_PER_INGEST}-chunk safety cap. Split the upload or raise "
            f"MAX_CHUNKS_PER_INGEST in app/config.py."
        )

    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR),
        collection_name=collection_name,
    )

    return (len(docs), len(chunks))


def get_db_and_retriever(collection_name: str):
    """Open an existing persisted Chroma collection and return (db, retriever)."""
    _require_api_key()

    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    db = Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings,
        collection_name=collection_name,
    )
    retriever = db.as_retriever(search_kwargs={"k": TOP_K})
    return db, retriever


def build_chain():
    """Build the prompt | llm | parser chain once; reusable across questions."""
    _require_api_key()

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a helpful assistant. Answer using ONLY the provided context. "
         "If the answer is not in the context, say you don't know. "
         "Cite sources using bracket numbers like [1], [2]."),
        ("human", "Question:\n{question}\n\nContext:\n{context}"),
    ])
    llm = ChatOpenAI(model=CHAT_MODEL, temperature=0)
    return prompt | llm | StrOutputParser()


def _format_context(retrieved_docs) -> str:
    blocks = []
    for i, d in enumerate(retrieved_docs, start=1):
        src = d.metadata.get("source", "unknown")
        page = d.metadata.get("page", None)
        page_str = f", page {page}" if page is not None else ""
        blocks.append(f"[{i}] Source: {src}{page_str}\n{d.page_content}")
    return "\n\n".join(blocks)


def _format_sources(retrieved_docs) -> List[Dict[str, Any]]:
    sources = []
    for i, d in enumerate(retrieved_docs, start=1):
        snippet = d.page_content
        if len(snippet) > 350:
            snippet = snippet[:350] + "…"
        sources.append({
            "id": i,
            "source": d.metadata.get("source", "unknown"),
            "page": d.metadata.get("page", None),
            "snippet": snippet,
        })
    return sources


def ask_question(question: str, retriever, chain=None) -> Dict[str, Any]:
    """
    Retrieve top-k chunks, send (question, context) through the cached chain,
    return {answer, sources}. Pass `chain` from session cache to avoid rebuilding.
    """
    if chain is None:
        chain = build_chain()

    retrieved_docs = retriever.invoke(question)
    context = _format_context(retrieved_docs)
    answer = chain.invoke({"question": question, "context": context})

    return {"answer": answer, "sources": _format_sources(retrieved_docs)}
