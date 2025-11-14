# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**PhD Research Assistant v2.0** - AI-powered research management system for PhD students featuring intelligent document management, RAG-powered knowledge chat, agent-based paper discovery, and comprehensive research analytics.

**Tech Stack:**
- Backend: FastAPI, Python 3.9+, SQLModel, Pydantic, ChromaDB, LangChain/LangGraph
- Frontend: React 18, TypeScript, Vite, Tailwind CSS, TanStack Query
- LLM: OpenAI, Anthropic, DeepSeek (via LiteLLM)
- Document: PyMuPDF, python-docx
- Real-time: WebSockets (Socket.io)

**Key Features:**
- Multi-project research library management
- RAG-powered knowledge chat with source citations
- AI agent-based paper discovery (Semantic Scholar)
- Manual PDF upload and processing
- Vector similarity search with ChromaDB

## Environment Setup

### Virtual Environment
**CRITICAL**: Virtual environment is at ROOT level (not in backend/):
```bash
cd "/Users/alibakhtiari/Desktop/AI research assistant/phd_research_assistant"
source "../research_env/bin/activate"
```

### Backend Setup
```bash
cd backend
pip install -e ".[dev]"
cp .env.example .env
# Edit .env with API keys
```

### Frontend Setup
```bash
cd frontend
npm install
```

---

# Technical Architecture Guide

## Project Structure

```
phd_research_assistant/
├── backend/              # FastAPI REST API + AI services
│   ├── src/
│   │   ├── api/         # Route handlers (projects, papers, chat, recommendations, agents, settings)
│   │   ├── services/    # Business logic (project, paper, chat, recommendation)
│   │   ├── core/        # Foundation layer
│   │   │   ├── database/      # SQLite models and operations
│   │   │   ├── vector_store/  # ChromaDB embeddings
│   │   │   ├── llm/           # LLM clients (clients.py) and prompts (prompts.py)
│   │   │   ├── external_apis/ # Semantic Scholar, ArXiv
│   │   │   └── utils/         # PDF processing, text extraction
│   │   ├── models/      # Pydantic request/response models
│   │   ├── agents/      # LangGraph agent workflows (future)
│   │   ├── config.py    # Pydantic Settings configuration
│   │   ├── dependencies.py  # FastAPI dependency injection
│   │   └── main.py      # Application entry point
│   ├── tests/
│   └── pyproject.toml
├── frontend/             # React TypeScript SPA
│   ├── src/
│   │   ├── pages/       # Dashboard, Papers, Chat, Recommendations, Settings
│   │   ├── components/  # Reusable UI components
│   │   ├── services/    # API client (api.ts)
│   │   ├── contexts/    # React Context (ProjectContext)
│   │   ├── hooks/       # Custom React hooks
│   │   └── types/       # TypeScript interfaces
│   └── package.json
├── projects/             # User research projects (self-contained)
│   └── {project_name}/
│       ├── data/
│       │   ├── project.db        # SQLite (papers, chat sessions)
│       │   └── vector_store/     # ChromaDB collection
│       ├── papers/               # Original PDFs
│       └── README.md
├── archive/              # Archived Streamlit v1.0
├── shared/               # Shared configs
└── claude.log            # Architectural Decision Records (ADR)
```

## 3-Layer Backend Architecture

**Layer 1: API Routes** (`backend/src/api/`)
- Thin handlers with request validation (Pydantic models)
- Dependency injection for services
- No business logic

**Layer 2: Service Layer** (`backend/src/services/`)
- Business logic orchestration
- Transactional operations
- Service composition
- Error handling/retries

**Layer 3: Core** (`backend/src/core/`)
- Database operations (`database/`)
- Vector embeddings (`vector_store/`)
- LLM abstraction (`llm/`)
- External integrations (`external_apis/`)
- Utilities (`utils/`)

## Common Commands

### Environment Setup
```bash
# Virtual environment is at ROOT level (not in backend/)
cd "/Users/alibakhtiari/Desktop/AI research assistant/phd_research_assistant"
source "../research_env/bin/activate"
```

### Run Full Stack
```bash
# Terminal 1 - Backend
cd backend
../research_env/bin/python3 -m uvicorn src.main:app --reload --port 8000

# Terminal 2 - Frontend
cd frontend
npm run dev  # Runs on http://localhost:5173
```

### Testing
```bash
# Backend
cd backend
pytest                     # All tests
pytest tests/test_*.py     # Specific file
pytest -v --cov=src        # With coverage

# Frontend
cd frontend
npm test                   # When implemented
```

### Code Quality
```bash
# Backend
cd backend
black src/                 # Format
ruff check src/            # Lint
mypy src/                  # Type check

# Frontend
cd frontend
npm run lint               # ESLint
npm run build              # Type check + build
```

### API Documentation
When backend running: http://localhost:8000/api/docs

## Key Architectural Patterns

### Multi-Project System
Each project is self-contained in `projects/{project_name}/`:
- Separate SQLite database (papers, chat sessions)
- Separate ChromaDB collection (embeddings)
- Dedicated papers/ directory for PDFs
- No data shared between projects

### Dependency Injection
Services injected via FastAPI dependencies (`backend/src/dependencies.py`):
```python
from src.dependencies import get_project_service

@router.get("/projects")
async def list_projects(
    project_service: ProjectService = Depends(get_project_service)
):
    return project_service.list_projects()
```

### Configuration
All config in `backend/src/config.py` using Pydantic Settings:
- Loads from environment variables + `.env` file
- Cached with `@lru_cache()`
- Access via `get_settings()`

### LLM Provider Abstraction
`backend/src/core/llm/clients.py` provides unified interface:
- `get_llm_client(provider: str)` returns configured LLM
- Supports: OpenAI, Anthropic, DeepSeek
- Uses LiteLLM for unified API
- Prompts centralized in `prompts.py`

### RAG Pipeline (Chat Service)
1. **Query Processing** - Expand abbreviations, extract keywords
2. **Context Retrieval** - Vector similarity search (ChromaDB)
3. **Smart Ranking** - Deduplication, diversity scoring, recency
4. **Context Optimization** - Token limit management, sliding window
5. **LLM Generation** - Inject context + conversation history
6. **Source Attribution** - Track citations with page numbers

### Vector Store Strategy
- One ChromaDB collection per project
- **Embeddings:** OpenAI only (for best quality)
  - **text-embedding-3-small**: 1536-dim, $0.02/1M tokens - Recommended
  - **text-embedding-3-large**: 3072-dim, $0.13/1M tokens - Better quality, 5x cost
- **Caching:** SQLite-based fingerprint caching to reduce API costs
- Metadata stored: page_number, pdf_path, paper_id, chunk_index
- Supports location tracking (bounding boxes) for PDF navigation

### Embedding Configuration
Configure in `backend/.env`:
```bash
OPENAI_API_KEY=your_key_here  # REQUIRED
EMBEDDING_MODEL=text-embedding-3-small  # or text-embedding-3-large
EMBEDDING_CACHE_ENABLED=true
EMBEDDING_CACHE_PATH=./data/embeddings_cache.db
```

**No fallback:** System requires OpenAI API key - fails fast with clear error if not configured

## Critical Implementation Details

### PDF Processing
- PyMuPDF for text extraction
- Optional LLM-based cleaning (headers/footers/tables removal)
- Default: 4 pages per chunk
- Fallback to direct extraction if LLM cleaning fails

### Conversation Context Management
- Hybrid: sliding window (8-10 recent messages) + summarization
- Token estimation: ~1 token per 3.5 characters
- Conservative limit: 6000 tokens
- Emergency truncation if exceeded

### Database Schema
**Project**: id, name, description, created_at, db_path, vector_store_path, papers_dir
**Paper**: id, project_id, title, authors, year, abstract, file_path, metadata (JSON)
**ChatSession**: id, project_id, title, created_at, messages (JSON)

## Common Pitfalls

### Backend
1. **Imports**: Use `from src.` not relative imports
2. **Config changes**: Clear Pydantic cache or restart server
3. **Database**: Each project has separate SQLite file
4. **Vector store**: Must initialize collection before querying

### Frontend
1. **API URL**: Configured in Vite proxy, not hardcoded
2. **Types**: All API responses need TypeScript interfaces
3. **State**: Use TanStack Query for server state
4. **WebSockets**: Cleanup on component unmount

### Environment
1. **Virtual env**: Located at `research_env/` (ROOT level, not backend/)
2. **API keys**: Backend `.env` required for LLM features
3. **Ports**: Backend 8000, Frontend 5173
4. **CORS**: Frontend origin must be in backend `CORS_ORIGINS`

## Extension Points

### Add New LLM Provider
1. Add client in `backend/src/core/llm/clients.py`
2. Add config in `backend/src/config.py`
3. Update `get_llm_client()` factory

### Add New API Endpoint
1. Create router in `backend/src/api/`
2. Create service in `backend/src/services/`
3. Register in `backend/src/main.py`
4. Add models in `backend/src/models/`

### Add New Frontend Page
1. Create in `frontend/src/pages/`
2. Add route in `frontend/src/App.tsx`
3. Add API functions in `frontend/src/services/api.ts`

## Debugging Tips

### Backend Logs
- `INFO: Uvicorn running` = server started
- Watch for SQLite connection errors (permissions)
- ChromaDB warnings about collection reuse (normal)

### Frontend DevTools
- React DevTools for component inspection
- Network tab for API debugging
- Console for WebSocket status

### Common Errors
- **422 Unprocessable**: Check request body vs Pydantic model
- **CORS errors**: Verify frontend origin in backend config
- **Database locked**: Close other connections to project.db
- **Import errors**: Backend must be installed editable (`pip install -e .`)

## Migration Note

Original Streamlit v1.0 archived in `archive/streamlit_version/`. v2.0 is complete rewrite:
- Backend/frontend separation (was monolithic)
- REST API + WebSockets (was direct calls)
- React UI (was Streamlit)
- Service layer pattern (was direct DB access)

**Data compatibility**: Projects, databases, vector stores from v1.0 work in v2.0 without migration.
