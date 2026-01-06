from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Dict, Any

from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from app.config import (
    UPLOAD_DIR,
    CHROMA_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    EMBEDDING_MODEL,
    TOP_K,
)
from app.loaders import load_documents


def save_uploads(uploaded_files) -> List[Path]:
    """
    Save Streamlit UploadedFile objects to disk so loaders can read them reliably.
    Returns a list of saved file paths.
    """
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    saved_paths: List[Path] = []
    for uf in uploaded_files:
        save_path = UPLOAD_DIR / uf.name
        save_path.write_bytes(uf.getbuffer())
        saved_paths.append(save_path)

    return saved_paths


def ingest_files(file_paths: List[Path], collection_name: str) -> Tuple[int, int]:
    """
    Ingest pipeline:
      1) Load docs from file paths
      2) Split into chunks
      3) Embed chunks
      4) Store in Chroma vector DB (persisted on disk) under collection_name

    Returns: (num_documents_loaded, num_chunks_indexed)
    """
    load_dotenv()

    docs = load_documents(file_paths)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    # Create / overwrite a collection with this name (Chroma manages collections internally)
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR),
        collection_name=collection_name,
    )

    return (len(docs), len(chunks))


def get_db_and_retriever(collection_name: str):
    """
    Open an existing persisted Chroma DB and return (db, retriever).
    We keep this separate so the UI can cache it in session_state.
    """
    load_dotenv()

    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

    db = Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings,
        collection_name=collection_name,
    )

    retriever = db.as_retriever(search_kwargs={"k": TOP_K})
    return db, retriever


def ask_question(question: str, retriever) -> Dict[str, Any]:
    """
    Question pipeline:
      1) Retrieve top-k relevant chunks
      2) Send question + retrieved chunks to LLM
      3) Return answer + sources
    """
    load_dotenv()

    # Newer LangChain retrievers use invoke()
    retrieved_docs = retriever.invoke(question)

    # Build context with bracketed ids so the model can cite [1], [2], ...
    context_blocks = []
    for i, d in enumerate(retrieved_docs, start=1):
        src = d.metadata.get("source", "unknown")
        page = d.metadata.get("page", None)
        page_str = f", page {page}" if page is not None else ""
        context_blocks.append(f"[{i}] Source: {src}{page_str}\n{d.page_content}")

    context = "\n\n".join(context_blocks)

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a helpful assistant. Answer using ONLY the provided context. "
         "If the answer is not in the context, say you don't know. "
         "Cite sources using bracket numbers like [1], [2]."),
        ("human", "Question:\n{question}\n\nContext:\n{context}")
    ])

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    chain = prompt | llm | StrOutputParser()

    answer = chain.invoke({"question": question, "context": context})

    sources = []
    for i, d in enumerate(retrieved_docs, start=1):
        sources.append({
            "id": i,
            "source": d.metadata.get("source", "unknown"),
            "page": d.metadata.get("page", None),
            "snippet": (d.page_content[:350] + "…") if len(d.page_content) > 350 else d.page_content
        })

    return {"answer": answer, "sources": sources}
