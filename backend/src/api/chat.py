"""Chat API endpoints - handles knowledge chat and RAG."""

import logging
from typing import List, Annotated

from fastapi import APIRouter, Depends, Query, HTTPException, status

from src.models.schemas import (
    ChatSessionCreate,
    ChatSessionResponse,
    ChatMessageRequest,
    ChatMessageResponse
)
from src.services.chat_service import ChatService
from src.dependencies import get_chat_service

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/sessions", response_model=ChatSessionResponse, status_code=status.HTTP_201_CREATED)
async def create_chat_session(
    request: ChatSessionCreate,
    chat_service: Annotated[ChatService, Depends(get_chat_service)] = None
):
    """
    Create a new chat session.

    Args:
        request: Session creation request with project_id and optional title

    Returns:
        Created chat session

    Raises:
        HTTPException: If project not found or creation fails
    """
    try:
        session = chat_service.create_session(request)
        logger.info(f"Created chat session {session.session_id} for project {request.project_id}")
        return session
    except Exception as e:
        logger.error(f"Error creating chat session: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating chat session: {str(e)}"
        )


@router.get("/sessions", response_model=List[ChatSessionResponse])
async def list_chat_sessions(
    project_id: str = Query(..., description="Project ID"),
    chat_service: Annotated[ChatService, Depends(get_chat_service)] = None
):
    """
    List all chat sessions for a project.

    Args:
        project_id: Project identifier

    Returns:
        List of chat sessions, ordered by most recent first
    """
    try:
        sessions = chat_service.get_sessions()
        logger.info(f"Listed {len(sessions)} chat sessions for project {project_id}")
        return sessions
    except Exception as e:
        logger.error(f"Error listing chat sessions: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving chat sessions: {str(e)}"
        )


@router.get("/sessions/{session_id}", response_model=ChatSessionResponse)
async def get_chat_session(
    session_id: str,
    project_id: str = Query(..., description="Project ID"),
    chat_service: Annotated[ChatService, Depends(get_chat_service)] = None
):
    """
    Get details of a specific chat session.

    Args:
        session_id: Session ID
        project_id: Project identifier

    Returns:
        Chat session details

    Raises:
        HTTPException: If session not found
    """
    session = chat_service.get_session(session_id)

    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Chat session {session_id} not found"
        )

    return session


@router.post("/sessions/{session_id}/messages", response_model=ChatMessageResponse)
async def send_message(
    session_id: str,
    request: ChatMessageRequest,
    project_id: str = Query(..., description="Project ID"),
    chat_service: Annotated[ChatService, Depends(get_chat_service)] = None
):
    """
    Send a message in a chat session and get RAG response.

    This endpoint:
    - Retrieves relevant paper chunks from the vector store
    - Generates a contextualized answer using LLM
    - Provides source citations for the response
    - Saves the conversation to the session

    Args:
        session_id: Session ID
        request: Message request with query and optional max_sources
        project_id: Project identifier

    Returns:
        Assistant response with answer and source citations

    Raises:
        HTTPException: If session not found or processing fails
    """
    try:
        response = await chat_service.send_message(session_id, request)
        logger.info(f"Processed message in session {session_id}")
        return response

    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error sending message: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing message: {str(e)}"
        )


@router.get("/sessions/{session_id}/messages", response_model=List[ChatMessageResponse])
async def get_messages(
    session_id: str,
    project_id: str = Query(..., description="Project ID"),
    chat_service: Annotated[ChatService, Depends(get_chat_service)] = None
):
    """
    Get all messages in a chat session.

    Args:
        session_id: Session ID
        project_id: Project identifier

    Returns:
        List of messages (both user and assistant) in chronological order

    Raises:
        HTTPException: If session not found
    """
    try:
        messages = chat_service.get_messages(session_id)
        logger.info(f"Retrieved {len(messages)} messages from session {session_id}")
        return messages
    except Exception as e:
        logger.error(f"Error retrieving messages: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving messages: {str(e)}"
        )


@router.delete("/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_chat_session(
    session_id: str,
    project_id: str = Query(..., description="Project ID"),
    chat_service: Annotated[ChatService, Depends(get_chat_service)] = None
):
    """
    Delete a chat session.

    This will permanently delete the session and all its messages.

    Args:
        session_id: Session ID
        project_id: Project identifier

    Raises:
        HTTPException: If session not found or deletion fails
    """
    try:
        success = chat_service.delete_session(session_id)

        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Chat session {session_id} not found"
            )

        logger.info(f"Deleted chat session {session_id}")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting chat session {session_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting session: {str(e)}"
        )
