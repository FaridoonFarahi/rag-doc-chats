# 📄 Docs Chat (RAG) — End-to-End Retrieval-Augmented Generation System

A **production-oriented Retrieval-Augmented Generation (RAG) system** that allows users to upload documents (PDF / DOCX / TXT) and ask natural-language questions. Answers are generated **strictly from retrieved source content** and returned with **explicit citations and source snippets**.

This project demonstrates **end-to-end ML / AI system design**, including ingestion, embedding, vector indexing, semantic retrieval, controlled LLM generation, UI state management, and a deployment-ready architecture.

---

## 🔍 What This Project Solves

Large Language Models (LLMs) cannot reliably answer questions about **private or user-provided documents** without hallucination. Naively uploading files into chat interfaces provides limited transparency, no persistence, and no control over retrieval behavior.

This project implements a **true RAG architecture**, where:

- Retrieval is explicit and inspectable  
- Generation is constrained and grounded  
- Data is persistent and reusable  
- System behavior is deterministic and auditable  

---

## 🚀 Key Features

- Upload and index documents (PDF, DOCX, TXT)
- Persistent vector indexing using embeddings
- Semantic similarity search (vector retrieval)
- Source-grounded LLM answers with citations
- Expandable source snippets for explainability
- Session-cached retrieval for low-latency Q&A
- Windows-safe index lifecycle management
- Modular architecture designed for deployment

---

## 🧠 Why RAG (and Not Just Upload a PDF to ChatGPT?)

Uploading a document to ChatGPT is a **convenience feature**, not a system.

| Capability | ChatGPT Upload | This RAG System |
|-----------|---------------|----------------|
| Persistence | ❌ Session-only | ✅ Persistent |
| Retrieval Control | ❌ Hidden | ✅ Explicit |
| Chunking Strategy | ❌ None | ✅ Configurable |
| Explainability | ❌ Limited | ✅ Full |
| Auditability | ❌ No | ✅ Yes |
| Scalability | ❌ Low | ✅ High |
| Automation | ❌ Manual | ✅ Programmatic |

> **ChatGPT upload = smart reader**  
> **This project = search engine + controlled reasoning layer**

---

## 🏗️ System Architecture

The system follows a **modular RAG pipeline**:

```
Ingestion (offline) → Vector Index → Retrieval (runtime) → Generation (runtime)
```

### End-to-End Flow

1. User uploads documents via the web UI  
2. Documents are persisted to disk  
3. Text is extracted from files  
4. Text is split into semantic chunks  
5. Each chunk is embedded into vector space  
6. Embeddings are stored in a persistent vector database  
7. User questions are embedded at runtime  
8. Top-K semantically similar chunks are retrieved  
9. LLM generates an answer using **only retrieved context**  
10. Answer is returned with citations and source snippets  

---

## 🔬 Ingestion Pipeline (Offline)

**Implementation:** `app/rag.py → ingest_files()`

### 1. Document Loading
- PDF → `PyPDFLoader`
- DOCX → `Docx2txtLoader`
- TXT → `TextLoader`

Each file is converted into `Document` objects containing:
- `page_content` (text)
- `metadata` (source file path, page number)

### 2. Chunking
- Uses `RecursiveCharacterTextSplitter`
- Configurable chunk size and overlap
- Prevents context window overflow
- Improves retrieval precision

### 3. Embedding
- Each chunk is converted into a dense vector
- Embeddings encode **semantic meaning**, not keywords
- Enables similarity-based retrieval

### 4. Vector Storage
- Stored in **Chroma** (persistent vector database)
- Each ingestion uses a **new collection namespace**
- Avoids Windows file-locking (`WinError 32`)
- Enables safe index resets without deleting files

---

## 🔍 Retrieval Layer

Retrieval is **fully decoupled from generation**.

### Retrieval Process

1. User question → embedding  
2. Vector similarity search (cosine distance)  
3. Top-K relevant chunks retrieved  
4. Chunks labeled `[1], [2], …`  

### Key Design Principle

The LLM **does not search documents**.  
It only receives retrieved context.

---

## ✍️ Generation Layer

The LLM is used strictly for **synthesis**, not retrieval.

### Prompt Controls

- Enforced system prompt:
  - Use only provided context
  - Do not hallucinate
  - Cite sources using `[1][2]`
- Context constructed only from retrieved chunks

This dramatically reduces unsupported answers.

---

## 📎 Source Attribution & Explainability

Every answer includes:
- Source file name
- Page number (when available)
- Raw retrieved text snippet

This enables:
- Auditing
- Debugging retrieval quality
- Trust and transparency

---

## ⚙️ State & Performance Optimizations

### Session Caching
- Vector DB and retriever cached in `st.session_state`
- Prevents repeated DB reloads
- Improves latency for multi-question sessions

### Windows Compatibility
- No deletion of active vector DB files
- Collection switching instead of filesystem deletion
- Eliminates Windows file-lock crashes

---

## 📁 Project Structure

```
rag-doc-chat/
├── app/
│   ├── ui.py              # Streamlit UI
│   ├── rag.py             # Ingestion, retrieval, Q&A logic
│   ├── loaders.py         # Document loaders
│   ├── config.py          # Chunking, paths, model config
│   └── __init__.py
├── data/
│   ├── uploads/           # Saved uploads
│   └── chroma_db/         # Persistent vector store
├── .env                   # API keys (not committed)
├── .gitignore
└── requirements.txt
```