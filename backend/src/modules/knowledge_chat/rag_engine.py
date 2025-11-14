import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import re
from datetime import datetime

from src.core.vector_store import VectorStoreManager
from src.core.llm import LLMManager, PromptManager
from src.core.database import DatabaseManager, Paper
from sqlmodel import select

logger = logging.getLogger(__name__)


@dataclass
class RetrievedContext:
    """Represents a piece of context retrieved from the knowledge base."""
    content: str
    paper_id: int
    paper_title: str
    authors: List[str]
    year: Optional[int]
    chunk_index: Optional[int]
    similarity_score: float
    source_info: str  # Formatted citation info
    
    # PDF location data
    pdf_path: Optional[str] = None
    page_number: Optional[int] = None
    bbox: Optional[tuple] = None


@dataclass
class RAGResponse:
    """Response from the RAG engine containing answer and sources."""
    answer: str
    sources: List[RetrievedContext]
    confidence_score: float
    query: str
    timestamp: datetime
    model_used: str
    tokens_used: Optional[Dict[str, int]] = None


class RAGEngine:
    """
    Core RAG (Retrieval-Augmented Generation) engine that orchestrates
    the retrieval of relevant context and generation of responses.
    """
    
    def __init__(self,
                 vector_manager: VectorStoreManager,
                 llm_manager: LLMManager,
                 prompt_manager: PromptManager,
                 db_manager: DatabaseManager):
        """
        Initialize RAG engine with required managers.
        
        Args:
            vector_manager: Vector store manager for similarity search
            llm_manager: LLM manager for response generation
            prompt_manager: Prompt manager for template handling
            db_manager: Database manager for paper metadata
        """
        self.vector_manager = vector_manager
        self.llm_manager = llm_manager
        self.prompt_manager = prompt_manager
        self.db_manager = db_manager
        
        logger.info("Initialized RAG engine")
    
    def process_query(self, query: str) -> str:
        """
        Clean and prepare user query for processing.
        
        Args:
            query: Raw user query
            
        Returns:
            Cleaned query string
        """
        # Remove extra whitespace and normalize
        query = re.sub(r'\s+', ' ', query.strip())
        
        # Handle common query patterns
        query = self._expand_abbreviations(query)
        
        return query
    
    def _expand_abbreviations(self, query: str) -> str:
        """Expand common academic abbreviations for better search."""
        abbreviations = {
            r'\bml\b': 'machine learning',
            r'\bdl\b': 'deep learning',
            r'\bnlp\b': 'natural language processing',
            r'\bai\b': 'artificial intelligence',
            r'\bnn\b': 'neural network',
            r'\bcnn\b': 'convolutional neural network',
            r'\brnn\b': 'recurrent neural network',
            r'\bgpt\b': 'generative pre-trained transformer'
        }
        
        for abbr, expansion in abbreviations.items():
            query = re.sub(abbr, expansion, query, flags=re.IGNORECASE)
        
        return query
    
    def retrieve_context(self,
                        query: str,
                        max_contexts: int = 5,
                        similarity_threshold: float = 0.3) -> List[RetrievedContext]:
        """
        Retrieve relevant context from the knowledge base.
        
        Args:
            query: User query
            max_contexts: Maximum number of contexts to retrieve
            similarity_threshold: Minimum similarity score threshold
            
        Returns:
            List of retrieved context objects
        """
        try:
            # Search for similar documents
            search_results = self.vector_manager.search_similar_papers(
                query=query,
                n_results=max_contexts * 2  # Get more results for filtering
            )
            
            # Filter by similarity threshold
            filtered_results = [
                result for result in search_results
                if result["similarity"] >= similarity_threshold
            ]
            
            # Convert to RetrievedContext objects with paper metadata
            contexts = []
            for result in filtered_results[:max_contexts]:
                paper_id = result["metadata"]["paper_id"]
                
                # Get paper metadata from database
                with self.db_manager.get_session() as session:
                    paper = session.get(Paper, paper_id)
                    
                    if paper:
                        # Extract PDF location data from metadata
                        metadata = result["metadata"]
                        pdf_path = metadata.get("pdf_path")
                        page_number = metadata.get("page_number")
                        bbox = None
                        if all(key in metadata for key in ["bbox_x0", "bbox_y0", "bbox_x1", "bbox_y1"]):
                            bbox = (
                                metadata["bbox_x0"],
                                metadata["bbox_y0"], 
                                metadata["bbox_x1"],
                                metadata["bbox_y1"]
                            )
                        
                        context = RetrievedContext(
                            content=result["document"],
                            paper_id=paper_id,
                            paper_title=paper.title,
                            authors=paper.get_authors_list(),
                            year=paper.year,
                            chunk_index=result["metadata"].get("chunk_index"),
                            similarity_score=result["similarity"],
                            source_info=self._format_citation(paper),
                            pdf_path=pdf_path,
                            page_number=page_number,
                            bbox=bbox
                        )
                        contexts.append(context)
            
            logger.info(f"Retrieved {len(contexts)} contexts for query")
            return contexts
            
        except Exception as e:
            logger.error(f"Error retrieving context: {e}")
            return []
    
    def _format_citation(self, paper: Paper) -> str:
        """Format paper citation for source attribution."""
        authors = paper.get_authors_list()
        author_str = ", ".join(authors[:2])  # First two authors
        if len(authors) > 2:
            author_str += " et al."
        
        year_str = f"({paper.year})" if paper.year else ""
        
        return f"{author_str} {year_str}. {paper.title}"
    
    def generate_response(self,
                         query: str,
                         contexts: List[RetrievedContext],
                         conversation_history: Optional[List[Dict[str, str]]] = None,
                         llm_provider: Optional[str] = None) -> RAGResponse:
        """
        Generate response using retrieved contexts and LLM.
        
        Args:
            query: User query
            contexts: Retrieved context objects
            conversation_history: Previous conversation messages
            llm_provider: Specific LLM provider to use
            
        Returns:
            RAG response with answer and sources
        """
        try:
            # Prepare context text
            if not contexts:
                context_text = "No relevant information found in the knowledge base."
                confidence_score = 0.1
            else:
                # Format contexts with source attribution
                context_parts = []
                for i, ctx in enumerate(contexts, 1):
                    context_parts.append(
                        f"[Source {i}: {ctx.source_info}]\n{ctx.content}\n"
                    )
                context_text = "\n---\n".join(context_parts)
                
                # Calculate confidence based on similarity scores
                avg_similarity = sum(ctx.similarity_score for ctx in contexts) / len(contexts)
                confidence_score = min(avg_similarity * 1.2, 1.0)  # Boost confidence slightly
            
            # Prepare messages for LLM
            messages = self.prompt_manager.format_prompt(
                "research_chat",
                context=context_text,
                question=query
            )
            
            # Add conversation history if provided
            if conversation_history:
                # Insert history before the current question
                messages = messages[:-1] + conversation_history + [messages[-1]]
            
            # Generate response
            llm_response = self.llm_manager.generate_response(
                messages=messages,
                provider=llm_provider,
                temperature=0.7,
                max_tokens=1000
            )
            
            # Create RAG response
            rag_response = RAGResponse(
                answer=llm_response.content,
                sources=contexts,
                confidence_score=confidence_score,
                query=query,
                timestamp=datetime.now(),
                model_used=llm_response.model or "unknown",
                tokens_used=llm_response.usage
            )
            
            logger.info(f"Generated response with {len(contexts)} sources")
            return rag_response
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            
            # Return fallback response
            return RAGResponse(
                answer="I apologize, but I encountered an error while processing your question. Please try again.",
                sources=[],
                confidence_score=0.0,
                query=query,
                timestamp=datetime.now(),
                model_used="error",
                tokens_used=None
            )
    
    def answer_question(self,
                       query: str,
                       max_contexts: int = 5,
                       similarity_threshold: float = 0.3,
                       conversation_history: Optional[List[Dict[str, str]]] = None,
                       llm_provider: Optional[str] = None) -> RAGResponse:
        """
        Complete RAG pipeline: process query, retrieve context, generate response.
        
        Args:
            query: User question
            max_contexts: Maximum number of context pieces to retrieve
            similarity_threshold: Minimum similarity threshold for context
            conversation_history: Previous conversation for context
            llm_provider: Specific LLM provider to use
            
        Returns:
            Complete RAG response with answer and sources
        """
        logger.info(f"Processing question: {query[:100]}...")
        
        try:
            # Step 1: Process and clean query
            processed_query = self.process_query(query)
            
            # Step 2: Retrieve relevant context
            contexts = self.retrieve_context(
                query=processed_query,
                max_contexts=max_contexts,
                similarity_threshold=similarity_threshold
            )
            
            # Step 3: Generate response with context
            response = self.generate_response(
                query=processed_query,
                contexts=contexts,
                conversation_history=conversation_history,
                llm_provider=llm_provider
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Error in RAG pipeline: {e}")
            return RAGResponse(
                answer="I encountered an error processing your question. Please try again.",
                sources=[],
                confidence_score=0.0,
                query=query,
                timestamp=datetime.now(),
                model_used="error"
            )
    
    def get_knowledge_stats(self) -> Dict[str, Any]:
        """Get statistics about the knowledge base."""
        try:
            vector_stats = self.vector_manager.get_collection_stats()
            
            # Get paper count from database
            with self.db_manager.get_session() as session:
                paper_count = session.exec(select(Paper)).all()
                paper_count = len(paper_count)
            
            return {
                "total_papers": paper_count,
                "total_chunks": vector_stats.get("total_documents", 0),
                "embedding_dimension": vector_stats.get("embedding_dimension", 0),
                "document_types": vector_stats.get("document_types", {}),
                "last_updated": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting knowledge stats: {e}")
            return {}