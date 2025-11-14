import logging
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass, asdict
from pathlib import Path
import uuid

from .rag_engine import RAGResponse

logger = logging.getLogger(__name__)


@dataclass
class ChatMessage:
    """Represents a single message in a chat session."""
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: datetime
    message_id: str
    sources: Optional[List[Dict[str, Any]]] = None  # For assistant messages
    tokens_used: Optional[Dict[str, int]] = None
    confidence_score: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary for serialization."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ChatMessage':
        """Create ChatMessage from dictionary."""
        data = data.copy()
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


@dataclass 
class ChatSession:
    """Represents a complete chat session with memory and metadata."""
    session_id: str
    title: str
    created_at: datetime
    last_updated: datetime
    messages: List[ChatMessage]
    total_tokens_used: int = 0
    session_metadata: Optional[Dict[str, Any]] = None
    
    def add_message(self, message: ChatMessage):
        """Add a message to the session."""
        self.messages.append(message)
        self.last_updated = datetime.now()
        
        # Update token count if available
        if message.tokens_used:
            self.total_tokens_used += message.tokens_used.get('total_tokens', 0)
    
    def get_conversation_history(self, max_messages: int = 10) -> List[Dict[str, str]]:
        """
        Get recent conversation history in format suitable for LLM context.
        
        Args:
            max_messages: Maximum number of recent messages to include
            
        Returns:
            List of messages in LLM format
        """
        recent_messages = self.messages[-max_messages:] if max_messages > 0 else self.messages
        
        history = []
        for message in recent_messages:
            history.append({
                "role": message.role,
                "content": message.content
            })
        
        return history
    
    def get_context_summary(self) -> Dict[str, Any]:
        """Get summary of conversation context for the session."""
        user_messages = [m for m in self.messages if m.role == 'user']
        assistant_messages = [m for m in self.messages if m.role == 'assistant']
        
        # Calculate average confidence
        confidences = [m.confidence_score for m in assistant_messages if m.confidence_score]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        return {
            'session_id': self.session_id,
            'title': self.title,
            'message_count': len(self.messages),
            'user_questions': len(user_messages),
            'assistant_responses': len(assistant_messages),
            'total_tokens_used': self.total_tokens_used,
            'avg_confidence': avg_confidence,
            'created_at': self.created_at.isoformat(),
            'last_updated': self.last_updated.isoformat(),
            'duration_minutes': (self.last_updated - self.created_at).total_seconds() / 60
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert session to dictionary for serialization."""
        return {
            'session_id': self.session_id,
            'title': self.title,
            'created_at': self.created_at.isoformat(),
            'last_updated': self.last_updated.isoformat(),
            'messages': [msg.to_dict() for msg in self.messages],
            'total_tokens_used': self.total_tokens_used,
            'session_metadata': self.session_metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ChatSession':
        """Create ChatSession from dictionary."""
        messages = [ChatMessage.from_dict(msg_data) for msg_data in data['messages']]
        
        return cls(
            session_id=data['session_id'],
            title=data['title'],
            created_at=datetime.fromisoformat(data['created_at']),
            last_updated=datetime.fromisoformat(data['last_updated']),
            messages=messages,
            total_tokens_used=data.get('total_tokens_used', 0),
            session_metadata=data.get('session_metadata')
        )


class ChatManager:
    """
    Manages chat sessions, conversation memory, and session persistence.
    """
    
    def __init__(self, sessions_dir: Optional[str] = None):
        """
        Initialize chat manager.
        
        Args:
            sessions_dir: Directory to store chat session files
        """
        self.active_sessions: Dict[str, ChatSession] = {}
        self.sessions_dir = Path(sessions_dir) if sessions_dir else None
        
        if self.sessions_dir:
            self.sessions_dir.mkdir(parents=True, exist_ok=True)
            self._load_existing_sessions()
        
        logger.info(f"Initialized chat manager with {len(self.active_sessions)} sessions")
    
    def create_session(self, title: Optional[str] = None) -> ChatSession:
        """
        Create a new chat session.
        
        Args:
            title: Optional title for the session
            
        Returns:
            New ChatSession object
        """
        session_id = str(uuid.uuid4())
        current_time = datetime.now()
        
        session = ChatSession(
            session_id=session_id,
            title=title or f"Chat Session {current_time.strftime('%Y-%m-%d %H:%M')}",
            created_at=current_time,
            last_updated=current_time,
            messages=[]
        )

        self.active_sessions[session_id] = session
        self._save_session(session)  # Persist to disk immediately
        logger.info(f"Created new chat session: {session_id}")

        return session
    
    def get_session(self, session_id: str) -> Optional[ChatSession]:
        """Get a chat session by ID."""
        return self.active_sessions.get(session_id)
    
    def add_user_message(self, session_id: str, message: str) -> ChatMessage:
        """
        Add a user message to a session.
        
        Args:
            session_id: Session ID
            message: User message content
            
        Returns:
            Created ChatMessage object
        """
        session = self.get_session(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        
        chat_message = ChatMessage(
            role='user',
            content=message,
            timestamp=datetime.now(),
            message_id=str(uuid.uuid4())
        )
        
        session.add_message(chat_message)
        self._save_session(session)
        
        return chat_message
    
    def add_assistant_response(self, session_id: str, rag_response: RAGResponse) -> ChatMessage:
        """
        Add an assistant response to a session.
        
        Args:
            session_id: Session ID
            rag_response: RAG response object
            
        Returns:
            Created ChatMessage object
        """
        session = self.get_session(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        
        # Convert sources to serializable format
        sources = []
        for source in rag_response.sources:
            sources.append({
                'paper_title': source.paper_title,
                'authors': source.authors,
                'year': source.year,
                'similarity_score': source.similarity_score,
                'source_info': source.source_info
            })
        
        chat_message = ChatMessage(
            role='assistant',
            content=rag_response.answer,
            timestamp=rag_response.timestamp,
            message_id=str(uuid.uuid4()),
            sources=sources,
            tokens_used=rag_response.tokens_used,
            confidence_score=rag_response.confidence_score
        )
        
        session.add_message(chat_message)
        self._save_session(session)
        
        return chat_message
    
    def get_conversation_context(self, 
                               session_id: str, 
                               max_messages: int = 6,
                               exclude_current: bool = True) -> List[Dict[str, str]]:
        """
        Get conversation context for LLM input.
        
        Args:
            session_id: Session ID
            max_messages: Maximum number of messages to include
            exclude_current: Whether to exclude the current (last) message
            
        Returns:
            Conversation history in LLM format
        """
        session = self.get_session(session_id)
        if not session:
            return []
        
        messages = session.messages[:-1] if exclude_current and session.messages else session.messages
        recent_messages = messages[-max_messages:] if max_messages > 0 else messages
        
        context = []
        for message in recent_messages:
            context.append({
                "role": message.role,
                "content": message.content
            })
        
        return context
    
    def list_sessions(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        List all chat sessions with summary information.
        
        Args:
            limit: Maximum number of sessions to return
            
        Returns:
            List of session summaries
        """
        sessions = list(self.active_sessions.values())
        sessions.sort(key=lambda x: x.last_updated, reverse=True)
        
        if limit:
            sessions = sessions[:limit]
        
        return [session.get_context_summary() for session in sessions]
    
    def delete_session(self, session_id: str) -> bool:
        """
        Delete a chat session.
        
        Args:
            session_id: Session ID to delete
            
        Returns:
            True if deleted successfully, False if not found
        """
        if session_id not in self.active_sessions:
            return False
        
        del self.active_sessions[session_id]
        
        # Remove session file if it exists
        if self.sessions_dir:
            session_file = self.sessions_dir / f"{session_id}.json"
            if session_file.exists():
                session_file.unlink()
        
        logger.info(f"Deleted chat session: {session_id}")
        return True
    
    def update_session_title(self, session_id: str, title: str) -> bool:
        """
        Update the title of a chat session.
        
        Args:
            session_id: Session ID
            title: New title
            
        Returns:
            True if updated successfully
        """
        session = self.get_session(session_id)
        if not session:
            return False
        
        session.title = title
        session.last_updated = datetime.now()
        self._save_session(session)
        
        return True
    
    def _save_session(self, session: ChatSession):
        """Save session to file if sessions directory is configured."""
        if not self.sessions_dir:
            return
        
        try:
            session_file = self.sessions_dir / f"{session.session_id}.json"
            with open(session_file, 'w', encoding='utf-8') as f:
                json.dump(session.to_dict(), f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error saving session {session.session_id}: {e}")
    
    def _load_existing_sessions(self):
        """Load existing sessions from files."""
        if not self.sessions_dir or not self.sessions_dir.exists():
            return
        
        for session_file in self.sessions_dir.glob("*.json"):
            try:
                with open(session_file, 'r', encoding='utf-8') as f:
                    session_data = json.load(f)
                
                session = ChatSession.from_dict(session_data)
                self.active_sessions[session.session_id] = session
                
            except Exception as e:
                logger.error(f"Error loading session from {session_file}: {e}")
    
    def export_session(self, session_id: str, format: str = 'json') -> Optional[str]:
        """
        Export a session in the specified format.
        
        Args:
            session_id: Session ID to export
            format: Export format ('json', 'markdown', 'txt')
            
        Returns:
            Exported content as string, or None if session not found
        """
        session = self.get_session(session_id)
        if not session:
            return None
        
        if format == 'json':
            return json.dumps(session.to_dict(), indent=2, ensure_ascii=False)
        
        elif format == 'markdown':
            lines = [f"# {session.title}\n"]
            lines.append(f"**Created:** {session.created_at.strftime('%Y-%m-%d %H:%M')}")
            lines.append(f"**Messages:** {len(session.messages)}")
            lines.append(f"**Tokens Used:** {session.total_tokens_used}\n")
            
            for message in session.messages:
                timestamp = message.timestamp.strftime('%H:%M')
                if message.role == 'user':
                    lines.append(f"## [{timestamp}] User")
                    lines.append(f"{message.content}\n")
                else:
                    lines.append(f"## [{timestamp}] Assistant")
                    lines.append(f"{message.content}")
                    
                    if message.sources:
                        lines.append("\n**Sources:**")
                        for i, source in enumerate(message.sources, 1):
                            lines.append(f"{i}. {source['source_info']}")
                    lines.append("")
            
            return "\n".join(lines)
        
        elif format == 'txt':
            lines = [f"{session.title}"]
            lines.append("=" * len(session.title))
            lines.append(f"Created: {session.created_at.strftime('%Y-%m-%d %H:%M')}")
            lines.append(f"Messages: {len(session.messages)}")
            lines.append("")
            
            for message in session.messages:
                timestamp = message.timestamp.strftime('%H:%M')
                role = "USER" if message.role == 'user' else "ASSISTANT"
                lines.append(f"[{timestamp}] {role}: {message.content}")
                lines.append("")
            
            return "\n".join(lines)
        
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about all chat sessions."""
        if not self.active_sessions:
            return {}
        
        sessions = list(self.active_sessions.values())
        total_messages = sum(len(s.messages) for s in sessions)
        total_tokens = sum(s.total_tokens_used for s in sessions)
        
        # Calculate date range
        created_dates = [s.created_at for s in sessions]
        updated_dates = [s.last_updated for s in sessions]
        
        return {
            'total_sessions': len(sessions),
            'total_messages': total_messages,
            'total_tokens_used': total_tokens,
            'avg_messages_per_session': total_messages / len(sessions) if sessions else 0,
            'avg_tokens_per_session': total_tokens / len(sessions) if sessions else 0,
            'earliest_session': min(created_dates).isoformat() if created_dates else None,
            'latest_activity': max(updated_dates).isoformat() if updated_dates else None
        }