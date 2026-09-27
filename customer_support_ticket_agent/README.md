# AI Customer Support Assistant (RAG & Multi-Turn Support Agent)

An end-to-end customer support assistant built with **FastAPI**, **Streamlit**, **LangChain/Groq LLM**, **FAISS Vector Store**, and a **Multi-Turn State Machine** for ticket generation.

---

## Architecture Overview

1. **RAG Pipeline (`src/rag/retriever.py`):**
   - Ingests internal domain knowledge documents.
   - Performs vector search using embeddings to ground LLM responses in factual context.

2. **Multi-Turn Support Agent (`src/pipeline.py`):**
   - Manages stateful user sessions via `SessionStore`.
   - Identifies ticket creation intent and performs conversational slot-filling (`email`, `name`, `description`).
   - Validates data against strict Pydantic schemas (`TicketCreate`) before calling `TicketRepository`.
   - Enforces duplicate ticket protection per session.

3. **FastAPI Server (`server.py`):**
   - Exposes REST endpoints (`/chat`, `/health`) handling JSON requests and returning structured `ChatResponse` objects.

4. **Streamlit UI (`app.py`):**
   - Provides an interactive chat interface, status indicators for backend components, and styled visual cards for generated support tickets.

---

## Project Structure

```text
├── app.py                   # Streamlit Frontend Interface
├── server.py                # FastAPI Backend Server
├── src/
│   ├── config.py            # Environment & Settings Configuration
│   ├── models.py            # Pydantic Schemas (TicketCreate, ChatResponse, etc.)
│   ├── pipeline.py          # Core Support Agent State Machine & Logic
│   ├── rag/                 # Knowledge Retriever & Vector Search
│   ├── sessions/            # In-Memory Session Storage
│   ├── tools/               # Ticket Repository & Mock Data Management
│   └── utils/               # Custom Errors & Logger
├── documents/               # Knowledge Base Markdown Files
├── requirements.txt         # Dependencies
└── README.md                # Documentation
```

## Project Structure
1. Prerequisites
Python 3.10 or higher

Groq API Key

2. Environment Variables
Create a .env file in the project root:

GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.3-70b-versatile

3. Installation
Install dependencies:

pip install -r requirements.txt

4. Running the Application
Step 1: Start the FastAPI Backend

uvicorn server:app --reload --port 8000

Step 2: Start the Streamlit Frontend
In a separate terminal window:

streamlit run app.py
Open your browser at http://localhost:8501.

Running Automated Tests
Run the unit and pipeline integration tests:

pytest tests/

## macOS / Linux (Bash / Zsh)

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn server:app --reload --port 8000
streamlit run app.py

## Architecture Diagram
┌─────────────────────────────────────────────────────────────────────────────────┐
│                                STREAMLIT FRONTEND                               │
│                                   (app.py)                                      │
└────────────────────────────────────────┬────────────────────────────────────────┘
                                         │ HTTP POST /chat
                                         ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                                 FASTAPI SERVER                                  │
│                                  (server.py)                                    │
└────────────────────────────────────────┬────────────────────────────────────────┘
                                         │
                                         ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                                SUPPORT PIPELINE                                 │
│                               (src/pipeline.py)                                 │
├────────────────────────────────────────┬────────────────────────────────────────┤
│                                        │                                        │
│  ┌──────────────────────────┐          │          ┌──────────────────────────┐  │
│  │      SESSION STORE       │          │          │      FAISS RETRIEVER     │  │
│  │   (src/sessions/store)   │          │          │    (src/rag/retriever)   │  │
│  └────────────┬─────────────┘          │          └────────────┬─────────────┘  │
│               │                        │                       │                │
│               ▼                        │                       ▼                │
│  ┌──────────────────────────┐          │          ┌──────────────────────────┐  │
│  │   TICKET STATE MACHINE   │          │          │   openai/gpt-oss-120b    │  │
│  │   (Slot Filling & Dups)  │          │          │ (llama-3.3-70b-versatile)│  │
│  └────────────┬─────────────┘          │          └──────────────────────────┘  │
│               │                        │                                        │
│               ▼                        │                                        │
│  ┌──────────────────────────┐          │                                        │
│  │    TICKET REPOSITORY     │          │                                        │
│  │ (src/tools/ticket_tool)  │          │                                        │
│  └──────────────────────────┘          │                                        │
└────────────────────────────────────────┴────────────────────────────────────────┘