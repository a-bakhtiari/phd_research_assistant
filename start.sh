#!/bin/bash

# PhD Research Assistant - Startup Script
# This script starts both the backend (FastAPI) and frontend (React/Vite)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  PhD Research Assistant - Starting    ${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Function to cleanup on exit
cleanup() {
    echo -e "\n${YELLOW}Shutting down servers...${NC}"
    kill $BACKEND_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    exit 0
}

# Trap CTRL+C and call cleanup
trap cleanup INT TERM

# Check if virtual environment exists
if [ ! -d "$SCRIPT_DIR/.venv" ]; then
    echo -e "${RED}Error: Virtual environment not found at .venv${NC}"
    echo -e "${YELLOW}Please create it first:${NC}"
    echo -e "  python3 -m venv .venv"
    echo -e "  source .venv/bin/activate"
    echo -e "  pip install -r backend/requirements.txt"
    exit 1
fi

# Check if node_modules exists
if [ ! -d "$SCRIPT_DIR/frontend/node_modules" ]; then
    echo -e "${YELLOW}Installing frontend dependencies...${NC}"
    cd "$SCRIPT_DIR/frontend"
    npm install
    if [ $? -ne 0 ]; then
        echo -e "${RED}Error: Failed to install frontend dependencies${NC}"
        exit 1
    fi
    cd "$SCRIPT_DIR"
fi

# Start Backend
echo -e "${GREEN}Starting Backend (FastAPI)...${NC}"
cd "$SCRIPT_DIR"
source .venv/bin/activate
cd backend

python -m uvicorn src.main:app --reload --host 0.0.0.0 --port 8000 > "$SCRIPT_DIR/backend.log" 2>&1 &
BACKEND_PID=$!

# Wait a moment for backend to start
sleep 2

# Check if backend started successfully
if ! kill -0 $BACKEND_PID 2>/dev/null; then
    echo -e "${RED}Error: Backend failed to start${NC}"
    echo -e "${YELLOW}Check backend.log for details${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Backend started successfully (PID: $BACKEND_PID)${NC}"
echo -e "${BLUE}  → API: http://localhost:8000${NC}"
echo -e "${BLUE}  → Docs: http://localhost:8000/api/docs${NC}"
echo -e "${BLUE}  → Logs: backend.log${NC}"
echo ""

# Start Frontend
echo -e "${GREEN}Starting Frontend (React/Vite)...${NC}"
cd "$SCRIPT_DIR/frontend"

npm run dev > "$SCRIPT_DIR/frontend.log" 2>&1 &
FRONTEND_PID=$!

# Wait a moment for frontend to start
sleep 3

# Check if frontend started successfully
if ! kill -0 $FRONTEND_PID 2>/dev/null; then
    echo -e "${RED}Error: Frontend failed to start${NC}"
    echo -e "${YELLOW}Check frontend.log for details${NC}"
    kill $BACKEND_PID
    exit 1
fi

echo -e "${GREEN}✓ Frontend started successfully (PID: $FRONTEND_PID)${NC}"
echo -e "${BLUE}  → App: http://localhost:5173${NC}"
echo -e "${BLUE}  → Logs: frontend.log${NC}"
echo ""

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  All services started successfully!   ${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "${YELLOW}Open your browser to: ${GREEN}http://localhost:5173${NC}"
echo ""
echo -e "${YELLOW}Press CTRL+C to stop all services${NC}"
echo ""

# Keep script running and tail both logs
tail -f "$SCRIPT_DIR/backend.log" "$SCRIPT_DIR/frontend.log" &
TAIL_PID=$!

# Wait for user interrupt
wait $BACKEND_PID $FRONTEND_PID

# Cleanup
kill $TAIL_PID 2>/dev/null
cleanup
