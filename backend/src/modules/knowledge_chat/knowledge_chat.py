import logging
from typing import Dict, List, Any, Optional, Union
from pathlib import Path

from .rag_engine import RAGEngine, RAGResponse
from .chat_manager import ChatManager, ChatSession, ChatMessage
from .context_retriever import ContextRetriever
from src.core.vector_store import VectorStoreManager
from src.core.llm import LLMManager, PromptManager
from src.core.database import DatabaseManager
from src.core.utils import DirectoryManager

logger = logging.getLogger(__name__)


class KnowledgeChat:
    """
    Main interface for knowledge chat functionality.
    
    This class provides a high-level API for conversational interaction
    with the research paper knowledge base using RAG (Retrieval-Augmented Generation).
    """
    
    def __init__(self,
                 db_manager: DatabaseManager,
                 vector_manager: VectorStoreManager,
                 llm_manager: LLMManager,
                 prompt_manager: PromptManager,
                 directory_manager: DirectoryManager,
                 sessions_dir: Optional[str] = None,
                 max_context_length: int = 4000):
        """
        Initialize knowledge chat system.
        
        Args:
            db_manager: Database manager
            vector_manager: Vector store manager
            llm_manager: LLM manager
            prompt_manager: Prompt manager
            directory_manager: Directory manager for project structure
            sessions_dir: Directory to store chat sessions
            max_context_length: Maximum context length for retrieval
        """
        # Initialize core components
        self.db_manager = db_manager
        self.vector_manager = vector_manager
        self.llm_manager = llm_manager
        self.prompt_manager = prompt_manager
        
        # Initialize RAG engine
        self.rag_engine = RAGEngine(
            vector_manager=vector_manager,
            llm_manager=llm_manager,
            prompt_manager=prompt_manager,
            db_manager=db_manager
        )
        
        # Initialize context retriever
        self.context_retriever = ContextRetriever(
            vector_manager=vector_manager,
            db_manager=db_manager,
            max_context_length=max_context_length,
            llm_manager=llm_manager,
            enable_relevance_filtering=False,  # Default off, can be enabled via settings
            relevance_threshold=0.8
        )
        
        # Initialize chat manager
        chat_sessions_dir = None
        if sessions_dir:
            chat_sessions_dir = sessions_dir
        elif directory_manager:
            chat_sessions_dir = str(directory_manager.project_root / "chat_sessions")
        
        self.chat_manager = ChatManager(sessions_dir=chat_sessions_dir)
        
        logger.info("Initialized KnowledgeChat system")
    
    def start_new_conversation(self, title: Optional[str] = None) -> str:
        """
        Start a new conversation session.
        
        Args:
            title: Optional title for the conversation
            
        Returns:
            Session ID for the new conversation
        """
        session = self.chat_manager.create_session(title=title)
        logger.info(f"Started new conversation: {session.session_id}")
        return session.session_id
    
    def ask_question(self,
                    session_id: str,
                    question: str,
                    max_contexts: int = 5,
                    similarity_threshold: float = 0.3,
                    llm_provider: Optional[str] = None,
                    use_conversation_history: bool = True) -> Dict[str, Any]:
        """
        Ask a question and get an answer from the knowledge base.
        
        Args:
            session_id: Chat session ID
            question: User question
            max_contexts: Maximum number of context pieces to use
            similarity_threshold: Minimum similarity threshold
            llm_provider: Specific LLM provider to use
            use_conversation_history: Whether to include conversation history
            
        Returns:
            Dictionary containing the response and metadata
        """
        try:
            # Validate session
            session = self.chat_manager.get_session(session_id)
            if not session:
                raise ValueError(f"Session {session_id} not found")
            
            # Add user message to session
            user_message = self.chat_manager.add_user_message(session_id, question)
            
            # Get conversation history if requested
            conversation_history = None
            if use_conversation_history and len(session.messages) > 1:
                conversation_history = self.chat_manager.get_conversation_context(
                    session_id=session_id,
                    max_messages=6,
                    exclude_current=True
                )
            
            # Generate response using RAG
            rag_response = self.rag_engine.answer_question(
                query=question,
                max_contexts=max_contexts,
                similarity_threshold=similarity_threshold,
                conversation_history=conversation_history,
                llm_provider=llm_provider
            )
            
            # Add assistant response to session
            assistant_message = self.chat_manager.add_assistant_response(
                session_id, rag_response
            )
            
            # Format response
            response = {
                'session_id': session_id,
                'question': question,
                'answer': rag_response.answer,
                'sources': self._format_sources(rag_response.sources),
                'confidence_score': rag_response.confidence_score,
                'timestamp': rag_response.timestamp.isoformat(),
                'model_used': rag_response.model_used,
                'tokens_used': rag_response.tokens_used,
                'message_id': assistant_message.message_id,
                'context_summary': self.context_retriever.get_context_summary(rag_response.sources)
            }
            
            logger.info(f"Answered question in session {session_id}")
            return response
            
        except Exception as e:
            logger.error(f"Error answering question: {e}")
            return {
                'session_id': session_id,
                'question': question,
                'answer': f"I encountered an error while processing your question: {str(e)}",
                'sources': [],
                'confidence_score': 0.0,
                'timestamp': None,
                'error': str(e)
            }
    
    def _format_sources(self, sources: List[Any]) -> List[Dict[str, Any]]:
        """Format sources for response output."""
        formatted_sources = []
        
        for i, source in enumerate(sources, 1):
            formatted_source = {
                'source_number': i,
                'paper_title': source.paper_title,
                'authors': source.authors,
                'year': source.year,
                'similarity_score': round(source.similarity_score, 3),
                'citation': source.source_info,
                'content_preview': source.content[:200] + "..." if len(source.content) > 200 else source.content
            }
            
            # Add PDF location data if available
            if hasattr(source, 'pdf_path') and source.pdf_path:
                formatted_source['pdf_path'] = source.pdf_path
            if hasattr(source, 'page_number') and source.page_number is not None:
                formatted_source['page_number'] = source.page_number
            if hasattr(source, 'bbox') and source.bbox:
                formatted_source['bbox'] = source.bbox
                
            formatted_sources.append(formatted_source)
        
        return formatted_sources
    
    def get_conversation_history(self, session_id: str) -> Dict[str, Any]:
        """
        Get the full conversation history for a session.
        
        Args:
            session_id: Session ID
            
        Returns:
            Dictionary with conversation history and metadata
        """
        session = self.chat_manager.get_session(session_id)
        if not session:
            return {'error': f'Session {session_id} not found'}
        
        messages = []
        for message in session.messages:
            msg_data = {
                'role': message.role,
                'content': message.content,
                'timestamp': message.timestamp.isoformat(),
                'message_id': message.message_id
            }
            
            if message.role == 'assistant' and message.sources:
                msg_data['sources'] = self._format_sources_simple(message.sources)
                msg_data['confidence_score'] = message.confidence_score
                msg_data['tokens_used'] = message.tokens_used
            
            messages.append(msg_data)
        
        return {
            'session_id': session_id,
            'title': session.title,
            'created_at': session.created_at.isoformat(),
            'last_updated': session.last_updated.isoformat(),
            'messages': messages,
            'summary': session.get_context_summary()
        }
    
    def _format_sources_simple(self, sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format sources for conversation history display."""
        return [
            {
                'paper_title': source['paper_title'],
                'authors': source['authors'],
                'year': source['year'],
                'citation': source['source_info']
            }
            for source in sources
        ]
    
    def list_conversations(self, limit: Optional[int] = 20) -> List[Dict[str, Any]]:
        """
        List recent conversations.
        
        Args:
            limit: Maximum number of conversations to return
            
        Returns:
            List of conversation summaries
        """
        return self.chat_manager.list_sessions(limit=limit)
    
    def delete_conversation(self, session_id: str) -> bool:
        """
        Delete a conversation session.
        
        Args:
            session_id: Session ID to delete
            
        Returns:
            True if successful, False otherwise
        """
        return self.chat_manager.delete_session(session_id)
    
    def rename_conversation(self, session_id: str, new_title: str) -> bool:
        """
        Rename a conversation session.
        
        Args:
            session_id: Session ID
            new_title: New title for the conversation
            
        Returns:
            True if successful, False otherwise
        """
        return self.chat_manager.update_session_title(session_id, new_title)
    
    def export_conversation(self, 
                          session_id: str, 
                          format: str = 'markdown') -> Optional[str]:
        """
        Export a conversation in the specified format.
        
        Args:
            session_id: Session ID to export
            format: Export format ('json', 'markdown', 'txt')
            
        Returns:
            Exported content as string, or None if session not found
        """
        return self.chat_manager.export_session(session_id, format=format)
    
    def get_knowledge_base_stats(self) -> Dict[str, Any]:
        """Get statistics about the knowledge base and chat system."""
        try:
            # Get RAG engine stats
            knowledge_stats = self.rag_engine.get_knowledge_stats()
            
            # Get chat manager stats
            chat_stats = self.chat_manager.get_stats()
            
            # Combine statistics
            return {
                'knowledge_base': knowledge_stats,
                'chat_system': chat_stats,
                'system_info': {
                    'max_context_length': self.context_retriever.max_context_length,
                    'available_llm_providers': self.llm_manager.list_providers(),
                    'default_llm_provider': self.llm_manager.default_client
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting system stats: {e}")
            return {'error': str(e)}
    
    def search_knowledge_base(self,
                            query: str,
                            max_results: int = 10,
                            similarity_threshold: float = 0.3) -> Dict[str, Any]:
        """
        Search the knowledge base without generating a conversational response.
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
            similarity_threshold: Minimum similarity threshold
            
        Returns:
            Search results with context information
        """
        try:
            # Use context retriever for advanced search
            contexts = self.context_retriever.retrieve_and_rank_contexts(
                query=query,
                max_contexts=max_results,
                similarity_threshold=similarity_threshold
            )
            
            # Format results
            results = []
            for context in contexts:
                results.append({
                    'paper_title': context.paper_title,
                    'authors': context.authors,
                    'year': context.year,
                    'similarity_score': round(context.similarity_score, 3),
                    'content_preview': context.content[:300] + "..." if len(context.content) > 300 else context.content,
                    'citation': context.source_info
                })
            
            return {
                'query': query,
                'total_results': len(results),
                'results': results,
                'search_summary': self.context_retriever.get_context_summary(contexts)
            }
            
        except Exception as e:
            logger.error(f"Error searching knowledge base: {e}")
            return {
                'query': query,
                'total_results': 0,
                'results': [],
                'error': str(e)
            }
    
    def quick_ask(self,
                 question: str,
                 max_contexts: int = 3,
                 similarity_threshold: float = 0.4,
                 llm_provider: Optional[str] = None) -> Dict[str, Any]:
        """
        Ask a quick question without maintaining conversation history.
        
        Args:
            question: User question
            max_contexts: Maximum number of context pieces
            similarity_threshold: Minimum similarity threshold
            llm_provider: Specific LLM provider to use
            
        Returns:
            Quick response without session management
        """
        try:
            # Generate response directly using RAG engine
            rag_response = self.rag_engine.answer_question(
                query=question,
                max_contexts=max_contexts,
                similarity_threshold=similarity_threshold,
                conversation_history=None,
                llm_provider=llm_provider
            )
            
            return {
                'question': question,
                'answer': rag_response.answer,
                'sources': self._format_sources(rag_response.sources),
                'confidence_score': rag_response.confidence_score,
                'timestamp': rag_response.timestamp.isoformat(),
                'model_used': rag_response.model_used,
                'tokens_used': rag_response.tokens_used
            }
            
        except Exception as e:
            logger.error(f"Error in quick ask: {e}")
            return {
                'question': question,
                'answer': f"I encountered an error: {str(e)}",
                'sources': [],
                'confidence_score': 0.0,
                'error': str(e)
            }
    
    def configure_relevance_filtering(self, 
                                    enable: bool = True, 
                                    threshold: float = 0.8) -> Dict[str, Any]:
        """
        Configure relevance filtering settings for context retrieval.
        
        Args:
            enable: Whether to enable LLM-based relevance filtering
            threshold: Minimum confidence threshold for relevance validation
            
        Returns:
            Configuration status and impact information
        """
        try:
            old_setting = self.context_retriever.enable_relevance_filtering
            old_threshold = self.context_retriever.relevance_threshold
            
            # Update settings
            self.context_retriever.enable_relevance_filtering = enable
            self.context_retriever.relevance_threshold = threshold
            
            # Estimate impact on API costs
            if enable and not old_setting:
                api_impact = "⚠️ Enabling relevance filtering will increase LLM API usage by ~20-30% but improve answer quality"
            elif not enable and old_setting:
                api_impact = "✅ Disabling relevance filtering will reduce LLM API usage"
            else:
                api_impact = f"Threshold changed from {old_threshold} to {threshold}"
            
            logger.info(f"Relevance filtering configured: enabled={enable}, threshold={threshold}")
            
            return {
                'success': True,
                'previous_enabled': old_setting,
                'previous_threshold': old_threshold,
                'current_enabled': enable,
                'current_threshold': threshold,
                'api_cost_impact': api_impact,
                'expected_quality_improvement': "Higher thresholds filter more aggressively but may miss relevant context"
            }
            
        except Exception as e:
            logger.error(f"Error configuring relevance filtering: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_relevance_filtering_status(self) -> Dict[str, Any]:
        """Get current relevance filtering configuration and statistics."""
        try:
            return {
                'enabled': self.context_retriever.enable_relevance_filtering,
                'threshold': self.context_retriever.relevance_threshold,
                'llm_available': self.context_retriever.llm_manager is not None,
                'batch_size': 5,  # From implementation
                'description': {
                    'enabled': 'LLM validates each context chunk before including in responses',
                    'threshold': 'Minimum confidence score (0.0-1.0) for context inclusion',
                    'impact': 'Higher threshold = more focused but potentially fewer contexts'
                }
            }
        except Exception as e:
            logger.error(f"Error getting relevance filtering status: {e}")
            return {'error': str(e)}
    
    def test_relevance_filtering(self, 
                               test_query: str = "attention mechanisms in transformers",
                               max_contexts: int = 10) -> Dict[str, Any]:
        """
        Test relevance filtering with and without filtering to show the difference.
        
        Args:
            test_query: Query to test with
            max_contexts: Maximum contexts to retrieve
            
        Returns:
            Comparison results showing filtering impact
        """
        try:
            # Save current settings
            original_enabled = self.context_retriever.enable_relevance_filtering
            original_threshold = self.context_retriever.relevance_threshold
            
            # Test without filtering
            self.context_retriever.enable_relevance_filtering = False
            contexts_unfiltered = self.context_retriever.retrieve_and_rank_contexts(
                query=test_query,
                max_contexts=max_contexts
            )
            
            # Test with filtering
            self.context_retriever.enable_relevance_filtering = True
            self.context_retriever.relevance_threshold = 0.8
            contexts_filtered = self.context_retriever.retrieve_and_rank_contexts(
                query=test_query,
                max_contexts=max_contexts
            )
            
            # Restore original settings
            self.context_retriever.enable_relevance_filtering = original_enabled
            self.context_retriever.relevance_threshold = original_threshold
            
            # Analysis
            unfiltered_count = len(contexts_unfiltered)
            filtered_count = len(contexts_filtered)
            reduction_percentage = ((unfiltered_count - filtered_count) / unfiltered_count * 100) if unfiltered_count > 0 else 0
            
            # Get relevance scores for filtered contexts
            filtered_confidences = []
            for ctx in contexts_filtered:
                if hasattr(ctx, 'metadata') and ctx.metadata and 'relevance_confidence' in ctx.metadata:
                    filtered_confidences.append(ctx.metadata['relevance_confidence'])
            
            avg_confidence = sum(filtered_confidences) / len(filtered_confidences) if filtered_confidences else 0
            
            return {
                'test_query': test_query,
                'unfiltered_contexts': unfiltered_count,
                'filtered_contexts': filtered_count,
                'contexts_removed': unfiltered_count - filtered_count,
                'reduction_percentage': f"{reduction_percentage:.1f}%",
                'average_confidence': f"{avg_confidence:.3f}" if avg_confidence > 0 else "N/A",
                'quality_improvement': "Filtered contexts should be more directly relevant to the query",
                'sample_filtered_titles': [ctx.paper_title for ctx in contexts_filtered[:3]],
                'recommendation': (
                    "Good filtering impact" if 10 <= reduction_percentage <= 50 
                    else "Consider adjusting threshold" if reduction_percentage > 50
                    else "Minimal filtering - threshold may be too low"
                )
            }
            
        except Exception as e:
            logger.error(f"Error testing relevance filtering: {e}")
            return {
                'test_query': test_query,
                'error': str(e)
            }