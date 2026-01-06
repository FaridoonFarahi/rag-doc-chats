import sys
from pathlib import Path
from datetime import datetime

import streamlit as st

# Ensure project root is importable (Windows + Streamlit friendly)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.rag import save_uploads, ingest_files, ask_question, get_db_and_retriever
from app.config import CHROMA_DIR

st.set_page_config(page_title="Doc Chat (RAG)", layout="wide")
st.title("📄 Doc Chat (RAG)")

# ---------------- Helpers ----------------
def new_collection_name() -> str:
    # unique name each time (prevents Windows lock issues from deletion)
    return "rag_docs_" + datetime.now().strftime("%Y%m%d_%H%M%S")


def chroma_exists() -> bool:
    return CHROMA_DIR.exists() and any(CHROMA_DIR.iterdir())


# ---------------- Sidebar ----------------
st.sidebar.header("Documents")

uploaded_files = st.sidebar.file_uploader(
    "Upload PDF, DOCX, or TXT",
    type=["pdf", "docx", "txt"],
    accept_multiple_files=True
)

colA, colB = st.sidebar.columns(2)
ingest_clicked = colA.button("📥 Ingest", type="primary", disabled=not uploaded_files)
clear_clicked = colB.button("🧹 Clear")

# Initialize session state
if "chat" not in st.session_state:
    st.session_state.chat = []

if "collection" not in st.session_state:
    # If an existing DB exists but we don't know collection, start fresh
    # (In practice you can persist the last used collection name later.)
    st.session_state.collection = new_collection_name()

# Clear = switch to a new empty collection (no file deletion)
if clear_clicked:
    st.session_state.chat = []
    st.session_state.pop("retriever", None)
    st.session_state.pop("db", None)
    st.session_state.collection = new_collection_name()
    st.sidebar.success("Cleared: switched to a new empty index.")
    st.rerun()

# Ingest flow
if ingest_clicked:
    with st.spinner("Saving & indexing documents..."):
        saved_paths = save_uploads(uploaded_files)
        # New collection each ingest keeps things clean and avoids Windows file lock deletion issues
        st.session_state.collection = new_collection_name()
        num_docs, num_chunks = ingest_files(saved_paths, st.session_state.collection)

    # Cache db+retriever for fast Q&A
    db, retriever = get_db_and_retriever(st.session_state.collection)
    st.session_state["db"] = db
    st.session_state["retriever"] = retriever

    st.sidebar.success(f"Indexed {num_chunks} chunks from {num_docs} docs/pages.")

st.sidebar.divider()

# Status
if "retriever" in st.session_state:
    st.sidebar.success("Index: ready ✅")
else:
    st.sidebar.warning("Index: not ready (upload + ingest)")

st.sidebar.caption("Tip: Ingest once → ask many questions.")

# ---------------- Main Q&A ----------------
st.subheader("Ask a question")

question = st.text_input(
    "Your question",
    placeholder="e.g., Summarize the key points in this document."
)

col1, col2 = st.columns([1, 1])
ask_clicked = col1.button("Ask", type="primary", disabled=not question)

if col2.button("Reset chat"):
    st.session_state.chat = []
    st.rerun()

if ask_clicked:
    if "retriever" not in st.session_state:
        st.error("No index found. Upload documents and click Ingest first.")
    else:
        with st.spinner("Retrieving relevant info and generating answer..."):
            result = ask_question(question, st.session_state["retriever"])

        st.session_state.chat.append({
            "question": question,
            "answer": result["answer"],
            "sources": result["sources"],
        })

# Display chat history (oldest → newest)
for turn in st.session_state.chat:
    st.markdown(f"### 🙋 Question\n{turn['question']}")
    st.markdown(f"### 🤖 Answer\n{turn['answer']}")

    with st.expander("Sources (retrieved chunks)"):
        for s in turn["sources"]:
            page = f"page {s['page']}" if s["page"] is not None else "page ?"
            st.markdown(f"**[{s['id']}] {s['source']} ({page})**")
            st.write(s["snippet"])
    st.divider()
