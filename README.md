# PhD Research Assistant

AI-powered research management system with RAG chat, paper discovery, and smart PDF processing.

## Setup

### Prerequisites

- Python 3.9+
- Node.js 18+
- OpenAI API key (required for embeddings)
- DeepSeek API key (required for LLM)

### Installation

```bash
# Clone repository
git clone https://github.com/a-bakhtiari/phd_research_assistant.git
cd phd_research_assistant

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install backend
cd backend
pip install -e ".[dev]"

# Configure environment
cp .env.example .env
# Edit .env and add your API keys

# Install frontend
cd ../frontend
npm install
```

### Running

```bash
# Terminal 1 - Backend
cd backend
source ../.venv/bin/activate
uvicorn src.main:app --reload --port 8000

# Terminal 2 - Frontend
cd frontend
npm run dev
```

Access at http://localhost:5173

API docs at http://localhost:8000/api/docs
