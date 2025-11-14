"""
Knowledge Chat Module

This module implements RAG (Retrieval-Augmented Generation) functionality
for conversational interaction with the research paper knowledge base.
"""

from .knowledge_chat import KnowledgeChat
from .rag_engine import RAGEngine, RetrievedContext, RAGResponse
from .chat_manager import ChatSession, ChatManager, ChatMessage
from .context_retriever import ContextRetriever, ContextScore

__all__ = [
    "KnowledgeChat", 
    "RAGEngine", 
    "RAGResponse",
    "RetrievedContext",
    "ChatSession", 
    "ChatManager", 
    "ChatMessage",
    "ContextRetriever",
    "ContextScore"
]