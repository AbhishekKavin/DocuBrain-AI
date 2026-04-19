# DocuBrain AI: Production-Grade RAG API

DocuBrain is an advanced Retrieval-Augmented Generation (RAG) system designed for high-performance document intelligence. It automates the transition from raw PDF data to an interactive, streaming AI service.

## 🚀 Key Engineering Highlights

- **Automated Ingestion Lifecycle**: Features a self-healing pipeline that reconciles local document directories with a FAISS vector store upon service initialization.

- **Entity-Aware Retrieval**: Implements dynamic entity extraction to prevent data leakage in multi-document environments (e.g., distinguishing between different candidate resumes).

- **Optimized Search (MMR)**: Utilizes Maximal Marginal Relevance to ensure high information density and diversity in retrieved context.

- **Asynchronous Streaming API**: Engineered with FastAPI to provide real-time, word-by-word generation via Server-Sent Events (SSE) for low-latency user experiences.

## 🛠️ Technical Stack

| Component | Technology |
|-----------|------------|
| Orchestration | LangChain |
| LLM & Embeddings | OpenAI (GPT-4o-mini & Text-Embedding-3-Small) |
| Vector Database | FAISS (Facebook AI Similarity Search) |
| Backend Framework | FastAPI (Asynchronous Python) |
| Document Processing | PyPDF & Recursive Character Splitting |

## 📦 Installation & Setup

### Clone & Environment

```bash
git clone https://github.com/Abhishek-Kavin/DocuBrain-AI.git
conda create -p venv python==3.11
conda activate .\venv
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Configuration

Create a `.env` file in the root directory with your OpenAI API key:

```bash
OPENAI_API_KEY=your_api_key_here
```

## Execution

```bash
uvicorn app.main:app --reload
```

## 📊 Operational Impact

This architecture mimics production-scale systems I've deployed, which have historically reduced diagnostic and manual retrieval times by up to 70%.
