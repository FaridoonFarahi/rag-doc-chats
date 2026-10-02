# Docs Chat (RAG): an end-to-end retrieval-augmented generation system

A production-oriented retrieval-augmented generation (RAG) system. You upload documents (PDF, DOCX, or TXT) and ask questions in plain language. Answers come only from the retrieved source text, and each one comes back with citations and the source snippets it used.

The project covers the whole ML/AI system: ingestion, embedding, vector indexing, semantic retrieval, controlled LLM generation, UI state management, and an architecture that is ready to deploy.

## What problem it solves

Large language models (LLMs) can't reliably answer questions about private or user-provided documents without hallucinating. Uploading files into a chat interface gives you little transparency, no persistence, and no control over how retrieval works.

This project uses a full RAG architecture:

- Retrieval is explicit, and you can inspect what was retrieved.
- Generation is constrained to the retrieved context.
- The data persists and can be reused.
- System behavior is deterministic and auditable.

## Features

- Upload and index documents (PDF, DOCX, TXT)
- Persistent vector index built from embeddings
- Semantic similarity search (vector retrieval)
- LLM answers grounded in the sources, with citations
- Expandable source snippets so you can see where an answer came from
- Retrieval cached per session for low-latency Q&A
- Index lifecycle that is safe on Windows
- Modular architecture designed for deployment

## Why RAG instead of uploading a PDF to ChatGPT

Uploading a document to ChatGPT is a convenience feature. Here is how it compares with this project:

| Capability | ChatGPT upload | This RAG system |
|-----------|---------------|----------------|
| Persistence | Session only | Persistent |
| Retrieval control | Hidden | Explicit |
| Chunking strategy | None | Configurable |
| Explainability | Limited | Full |
| Auditability | No | Yes |
| Scalability | Low | High |
| Automation | Manual | Programmatic |

ChatGPT's upload works like a smart reader. This project is a search engine with a controlled reasoning layer on top.

## System architecture

The system is a modular RAG pipeline:

```
Ingestion (offline) → Vector Index → Retrieval (runtime) → Generation (runtime)
```

### End-to-end flow

1. The user uploads documents through the web UI.
2. The documents are saved to disk.
3. Text is extracted from the files.
4. The text is split into semantic chunks.
5. Each chunk is embedded into vector space.
6. The embeddings are stored in a persistent vector database.
7. Each user question is embedded at runtime.
8. The top-K most similar chunks are retrieved.
9. The LLM generates an answer using only the retrieved context.
10. The answer is returned with citations and source snippets.

## Ingestion pipeline (offline)

Implemented in `app/rag.py → ingest_files()`.

### 1. Document loading

- PDF: `PyPDFLoader`
- DOCX: `Docx2txtLoader`
- TXT: `TextLoader`

Each file is converted into `Document` objects containing:

- `page_content` (text)
- `metadata` (source file path, page number)

### 2. Chunking

Text is split with `RecursiveCharacterTextSplitter`, with configurable chunk size and overlap. Chunking keeps each piece inside the context window and improves retrieval precision.

### 3. Embedding

Each chunk is converted into a dense vector. The embeddings capture meaning rather than exact keywords, so retrieval works by similarity.

### 4. Vector storage

Chunks are stored in Chroma, a persistent vector database. Each ingestion uses a new collection namespace. This avoids Windows file locking (`WinError 32`) and lets you reset the index safely without deleting files.

## Retrieval layer

Retrieval is fully decoupled from generation.

### Retrieval process

1. The user's question is embedded.
2. A vector similarity search runs (cosine distance).
3. The top-K relevant chunks are retrieved.
4. The chunks are labeled `[1], [2], …`.

### Design principle

The LLM does not search the documents. It only receives the retrieved context.

## Generation layer

The LLM is used only to synthesize an answer from what retrieval returns.

### Prompt controls

- The system prompt tells the model to:
  - Use only the provided context
  - Not hallucinate
  - Cite sources using `[1][2]`
- The context is built only from retrieved chunks.

Together these sharply reduce unsupported answers.

## Source attribution and explainability

Every answer includes:

- The source file name
- The page number (when available)
- The raw retrieved text snippet

This lets you audit answers, debug retrieval quality, and see exactly where each answer came from.

## State and performance

### Session caching

The vector DB and retriever are cached in `st.session_state`, so the app doesn't reload the DB on every question. This lowers latency when you ask several questions in one session.

### Windows compatibility

The app never deletes active vector DB files. It switches to a new collection instead of deleting from the filesystem, which prevents Windows file-lock crashes.

## Project structure

```
rag-doc-chat/
├── app/
│   ├── ui.py              # Streamlit UI
│   ├── rag.py             # Ingestion, retrieval, Q&A logic
│   ├── loaders.py         # Document loaders
│   ├── config.py          # Chunking, paths, model config
│   └── __init__.py
├── data/
│   ├── uploads/           # Saved uploads (gitignored)
│   └── chroma_db/         # Persistent vector store (gitignored)
├── .env.example           # Template — copy to .env and fill in
├── .gitignore
└── requirements.txt
```

## Setup

```bash
# 1. Clone and enter the repo
git clone https://github.com/FaridoonFarahi/rag-doc-chats.git
cd rag-doc-chats

# 2. Create a virtualenv
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# 3. Install deps
pip install -r requirements.txt

# 4. Configure your OpenAI key
cp .env.example .env        # then edit .env and paste your key

# 5. Run the app
streamlit run app/ui.py
```

The `.env` file is gitignored. Never commit your API key.
