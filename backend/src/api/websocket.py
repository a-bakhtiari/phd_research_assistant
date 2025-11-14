"""
WebSocket API for real-time updates.

Provides real-time paper processing progress updates.
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from typing import Dict, Set
import logging
import json

logger = logging.getLogger(__name__)

router = APIRouter(tags=["websocket"])

# Connection manager to track active WebSocket connections per project
class ConnectionManager:
    def __init__(self):
        # project_id -> set of websockets
        self.active_connections: Dict[str, Set[WebSocket]] = {}
        # project_id -> paper_id -> processing state
        self.processing_state: Dict[str, Dict[int, dict]] = {}

    async def connect(self, websocket: WebSocket, project_id: str):
        """Accept WebSocket connection and add to project room."""
        await websocket.accept()
        if project_id not in self.active_connections:
            self.active_connections[project_id] = set()
        self.active_connections[project_id].add(websocket)
        logger.info(f"WebSocket connected for project {project_id}. Total connections: {len(self.active_connections[project_id])}")

    def disconnect(self, websocket: WebSocket, project_id: str):
        """Remove WebSocket connection from project room."""
        if project_id in self.active_connections:
            self.active_connections[project_id].discard(websocket)
            if not self.active_connections[project_id]:
                del self.active_connections[project_id]
        logger.info(f"WebSocket disconnected for project {project_id}")

    async def broadcast_to_project(self, project_id: str, message: dict):
        """Broadcast message to all connections in a project and update processing state."""
        # Update processing state
        if project_id not in self.processing_state:
            self.processing_state[project_id] = {}

        paper_id = message.get("paper_id")
        if paper_id is not None:
            if message["type"] == "processing_progress":
                # Store/update processing state
                self.processing_state[project_id][paper_id] = {
                    "paper_id": paper_id,
                    "title": message.get("title"),
                    "progress": message.get("progress", 0),
                    "step": message.get("step", "Processing..."),
                    "status": "processing"
                }
            elif message["type"] == "processing_complete":
                # Remove from processing state when complete
                self.processing_state[project_id].pop(paper_id, None)
            elif message["type"] == "processing_failed":
                # Remove from processing state when failed
                self.processing_state[project_id].pop(paper_id, None)

        # Broadcast to all connections
        if project_id not in self.active_connections:
            return

        disconnected = set()
        for websocket in self.active_connections[project_id]:
            try:
                await websocket.send_json(message)
            except Exception as e:
                logger.error(f"Error sending to WebSocket: {e}")
                disconnected.add(websocket)

        # Clean up disconnected websockets
        for websocket in disconnected:
            self.disconnect(websocket, project_id)

    def get_processing_state(self, project_id: str) -> list:
        """Get current processing state for a project."""
        if project_id not in self.processing_state:
            return []
        return list(self.processing_state[project_id].values())

manager = ConnectionManager()


@router.get("/processing/{project_id}")
async def get_processing_papers(project_id: str):
    """
    Get currently processing papers for a project.

    Returns list of papers currently being processed with their progress.
    """
    processing_papers = manager.get_processing_state(project_id)
    return {"processing_papers": processing_papers}


@router.websocket("/ws/{project_id}")
async def websocket_endpoint(websocket: WebSocket, project_id: str):
    """
    WebSocket endpoint for real-time project updates.

    Clients connect to /ws/{project_id} and receive real-time updates for:
    - Paper processing progress
    - Status changes
    - Completion notifications
    """
    await manager.connect(websocket, project_id)
    try:
        while True:
            # Keep connection alive and listen for client messages (ping/pong)
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_json({"type": "pong"})
    except WebSocketDisconnect:
        manager.disconnect(websocket, project_id)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket, project_id)
