"""
Chat service for RAG-powered knowledge chat.

Wraps the existing RAG engine and chat manager.
"""

import logging
import json
from pathlib import Path
from typing import List, Optional
from datetime import datetime
import uuid

from src.core.database import DatabaseManager
from src.core.vector_store import VectorStoreManager
from src.core.llm import LLMManager, PromptManager
from src.modules.knowledge_chat import RAGEngine, ChatManager, ChatSession, ChatMessage
from src.models.schemas import (
    ChatSessionCreate,
    ChatSessionResponse,
    ChatMessageRequest,
    ChatMessageResponse,
    SourceCitation
)

logger = logging.getLogger(__name__)


class ChatService:
    """Service for managing chat sessions and RAG queries."""

    def __init__(
        self,
        db_manager: DatabaseManager,
        vector_manager: VectorStoreManager,
        llm_manager: LLMManager,
        prompt_manager: PromptManager,
        project_root: Path
    ):
        """
        Initialize chat service.

        Args:
            db_manager: Database manager instance
            vector_manager: Vector store manager instance
            llm_manager: LLM manager instance
            prompt_manager: Prompt manager instance
            project_root: Root directory of the project
        """
        self.db_manager = db_manager
        self.vector_manager = vector_manager
        self.llm_manager = llm_manager
        self.prompt_manager = prompt_manager
        self.project_root = project_root
        self.sessions_dir = project_root / "chat_sessions"
        self.sessions_dir.mkdir(parents=True, exist_ok=True)

        # Initialize RAG engine
        self.rag_engine = RAGEngine(
            vector_manager=vector_manager,
            llm_manager=llm_manager,
            prompt_manager=prompt_manager,
            db_manager=db_manager
        )

        # Initialize chat manager
        self.chat_manager = ChatManager(
            sessions_dir=str(self.sessions_dir)
        )

        logger.info("Initialized ChatService")

    def create_session(self, request: ChatSessionCreate) -> ChatSessionResponse:
        """
        Create a new chat session.

        Args:
            request: ChatSessionCreate request

        Returns:
            ChatSessionResponse
        """
        # Use ChatManager's create_session method
        session = self.chat_manager.create_session(
            title=request.title or "New Chat Session"
        )

        logger.info(f"Created chat session: {session.session_id}")

        return ChatSessionResponse(
            session_id=session.session_id,
            project_id=request.project_id,
            title=session.title,
            created_at=session.created_at,
            last_updated=session.last_updated,
            message_count=0,
            total_tokens_used=0
        )

    def get_sessions(self) -> List[ChatSessionResponse]:
        """
        Get all chat sessions.

        Returns:
            List of ChatSessionResponse objects
        """
        sessions = []

        for session_file in self.sessions_dir.glob("*.json"):
            try:
                session = self.chat_manager.get_session(session_file.stem)
                if session:
                    sessions.append(ChatSessionResponse(
                        session_id=session.session_id,
                        project_id="",  # TODO: Add project_id to session
                        title=session.title,
                        created_at=session.created_at,
                        last_updated=session.last_updated,
                        message_count=len(session.messages),
                        total_tokens_used=session.total_tokens_used
                    ))
            except Exception as e:
                logger.warning(f"Error loading session {session_file.stem}: {e}")
                continue

        return sorted(sessions, key=lambda x: x.last_updated, reverse=True)

    def get_session(self, session_id: str) -> Optional[ChatSessionResponse]:
        """
        Get a specific chat session.

        Args:
            session_id: Session ID

        Returns:
            ChatSessionResponse or None
        """
        session = self.chat_manager.get_session(session_id)
        if not session:
            return None

        return ChatSessionResponse(
            session_id=session.session_id,
            project_id="",
            title=session.title,
            created_at=session.created_at,
            last_updated=session.last_updated,
            message_count=len(session.messages),
            total_tokens_used=session.total_tokens_used
        )

    async def send_message(
        self,
        session_id: str,
        request: ChatMessageRequest
    ) -> ChatMessageResponse:
        """
        Send a message and get response (RAG or direct LLM).

        Args:
            session_id: Session ID
            request: ChatMessageRequest

        Returns:
            ChatMessageResponse with answer and sources (if RAG mode)
        """
        # Get session to retrieve conversation history
        session = self.chat_manager.get_session(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")

        # Add user message (ChatManager saves it)
        self.chat_manager.add_user_message(session_id, request.message)

        if request.use_rag:
            # Get RAG response (search papers + LLM)
            rag_response = self.rag_engine.answer_question(
                query=request.message,
                max_contexts=request.max_sources,
                conversation_history=session.get_conversation_history()
            )

            # Add assistant response (ChatManager saves it)
            self.chat_manager.add_assistant_response(session_id, rag_response)

            # Convert sources to SourceCitation models
            sources = []
            for source in rag_response.sources:
                sources.append(SourceCitation(
                    paper_id=source.paper_id,
                    paper_title=source.paper_title,
                    authors=source.authors,
                    year=source.year,
                    content=source.content,
                    page_number=source.page_number,
                    similarity_score=source.similarity_score
                ))

            logger.info(f"Processed RAG message in session {session_id}")

            return ChatMessageResponse(
                message_id=str(uuid.uuid4()),
                role="assistant",
                content=rag_response.answer,
                timestamp=rag_response.timestamp,
                sources=sources,
                tokens_used=rag_response.tokens_used,
                confidence_score=rag_response.confidence_score
            )
        else:
            # Direct LLM mode (no paper search)
            logger.info(f"Using direct LLM mode for session {session_id}")

            # Build conversation history for LLM
            history_messages = []
            for msg in session.get_conversation_history():
                history_messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })

            # Add current message
            history_messages.append({
                "role": "user",
                "content": request.message
            })

            # Get LLM response without RAG
            llm_response = self.llm_manager.generate_response(
                messages=history_messages,
                max_tokens=2000
            )

            # Create a mock RAG response structure for compatibility
            from src.modules.knowledge_chat.rag_engine import RAGResponse
            from datetime import datetime

            mock_rag_response = RAGResponse(
                answer=llm_response.content,
                sources=[],  # No sources in direct mode
                confidence_score=0.9,  # High confidence for direct LLM
                tokens_used=llm_response.token_usage,
                timestamp=datetime.now()
            )

            # Add assistant response
            self.chat_manager.add_assistant_response(session_id, mock_rag_response)

            logger.info(f"Processed direct LLM message in session {session_id}")

            return ChatMessageResponse(
                message_id=str(uuid.uuid4()),
                role="assistant",
                content=llm_response.content,
                timestamp=datetime.now(),
                sources=None,  # No sources in direct mode
                tokens_used=llm_response.token_usage,
                confidence_score=0.9
            )

    def get_messages(self, session_id: str) -> List[ChatMessageResponse]:
        """
        Get all messages in a session.

        Args:
            session_id: Session ID

        Returns:
            List of ChatMessageResponse objects
        """
        session = self.chat_manager.get_session(session_id)
        if not session:
            return []

        messages = []
        for msg in session.messages:
            sources = None
            if msg.sources:
                # Handle incomplete source data gracefully
                sources = []
                for s in msg.sources:
                    try:
                        sources.append(SourceCitation(**s))
                    except Exception as e:
                        logger.warning(f"Skipping invalid source citation: {e}")
                        continue

            messages.append(ChatMessageResponse(
                message_id=msg.message_id,
                role=msg.role,
                content=msg.content,
                timestamp=msg.timestamp,
                sources=sources if sources else None,
                tokens_used=msg.tokens_used,
                confidence_score=msg.confidence_score
            ))

        return messages

    def delete_session(self, session_id: str) -> bool:
        """
        Delete a chat session.

        Args:
            session_id: Session ID

        Returns:
            True if deleted, False if not found
        """
        session_file = self.sessions_dir / f"{session_id}.json"
        if not session_file.exists():
            return False

        try:
            session_file.unlink()
            logger.info(f"Deleted chat session: {session_id}")
            return True
        except Exception as e:
            logger.error(f"Error deleting session {session_id}: {e}")
            raise
