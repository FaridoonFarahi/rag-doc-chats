import sys
from pathlib import Path
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv

# Ensure project root is importable (Windows + Streamlit friendly)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Load .env once, at process start, before any module that needs the key.
load_dotenv(PROJECT_ROOT / ".env")

from app.rag import (  # noqa: E402  (import after sys.path + dotenv setup)
    save_uploads,
    ingest_files,
    ask_question,
    get_db_and_retriever,
    build_chain,
    IngestError,
)

st.set_page_config(page_title="Doc Chat (RAG)", layout="wide")
st.title("Doc Chat (RAG)")


def new_collection_name() -> str:
    # A new name per ingest, so we never delete files (which fails on Windows due to file locks).
    return "rag_docs_" + datetime.now().strftime("%Y%m%d_%H%M%S")


# ---------------- Sidebar ----------------
st.sidebar.header("Documents")

uploaded_files = st.sidebar.file_uploader(
    "Upload PDF, DOCX, or TXT",
    type=["pdf", "docx", "txt"],
    accept_multiple_files=True,
)

ingest_col, clear_col = st.sidebar.columns(2)
ingest_clicked = ingest_col.button("Ingest", type="primary", disabled=not uploaded_files)
clear_clicked = clear_col.button("Clear")

# ---------------- Session state ----------------
if "chat" not in st.session_state:
    st.session_state.chat = []

if "collection" not in st.session_state:
    st.session_state.collection = new_collection_name()

if "chain" not in st.session_state:
    try:
        st.session_state.chain = build_chain()
    except RuntimeError as e:
        st.session_state.chain = None
        st.sidebar.error(str(e))

# Clear switches to a new empty collection instead of deleting files.
if clear_clicked:
    st.session_state.chat = []
    st.session_state.pop("retriever", None)
    st.session_state.pop("db", None)
    st.session_state.collection = new_collection_name()
    st.sidebar.success("Cleared. Switched to a new empty index.")
    st.rerun()

# ---------------- Ingest flow ----------------
if ingest_clicked:
    try:
        with st.spinner("Saving and indexing documents..."):
            saved_paths = save_uploads(uploaded_files)
            st.session_state.collection = new_collection_name()
            num_docs, num_chunks = ingest_files(saved_paths, st.session_state.collection)
        db, retriever = get_db_and_retriever(st.session_state.collection)
        st.session_state["db"] = db
        st.session_state["retriever"] = retriever
        st.sidebar.success(f"Indexed {num_chunks} chunks from {num_docs} docs/pages.")
    except IngestError as e:
        st.sidebar.error(f"Ingest rejected: {e}")
    except Exception as e:  # noqa: BLE001
        st.sidebar.error(f"Ingest failed: {e}")

st.sidebar.divider()

if "retriever" in st.session_state:
    st.sidebar.success("Index: ready")
else:
    st.sidebar.warning("Index: not ready (upload and ingest first)")

st.sidebar.caption("Tip: ingest once, then ask as many questions as you like.")

# ---------------- Main Q&A ----------------
st.subheader("Ask a question")

question = st.text_input(
    "Your question",
    placeholder="e.g., Summarize the key points in this document.",
)

ask_col, reset_col = st.columns([1, 1])
ask_clicked = ask_col.button("Ask", type="primary", disabled=not question)

if reset_col.button("Reset chat"):
    st.session_state.chat = []
    st.rerun()

if ask_clicked:
    if "retriever" not in st.session_state:
        st.error("No index found. Upload documents and click Ingest first.")
    elif st.session_state.chain is None:
        st.error("OpenAI API key is not configured. Set OPENAI_API_KEY in .env and restart.")
    else:
        try:
            with st.spinner("Retrieving relevant chunks and generating an answer..."):
                result = ask_question(
                    question,
                    st.session_state["retriever"],
                    chain=st.session_state.chain,
                )
            st.session_state.chat.append({
                "question": question,
                "answer": result["answer"],
                "sources": result["sources"],
            })
        except Exception as e:  # noqa: BLE001
            st.error(f"Question failed: {e}")

# Chat history, oldest first
for turn in st.session_state.chat:
    st.markdown(f"### Question\n{turn['question']}")
    st.markdown(f"### Answer\n{turn['answer']}")

    with st.expander("Sources (retrieved chunks)"):
        for source in turn["sources"]:
            page = f"page {source['page']}" if source["page"] is not None else "page ?"
            st.markdown(f"**[{source['id']}] {source['source']} ({page})**")
            st.write(source["snippet"])
    st.divider()
