import logging
from typing import List, Dict, Any, Set, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import re
from difflib import SequenceMatcher

from src.core.vector_store import VectorStoreManager
from src.core.database import DatabaseManager, Paper
from .rag_engine import RetrievedContext

logger = logging.getLogger(__name__)


@dataclass 
class ContextScore:
    """Scoring information for context ranking."""
    base_similarity: float
    recency_boost: float
    diversity_penalty: float  
    length_score: float
    final_score: float


class ContextRetriever:
    """
    Advanced context retrieval system with smart ranking, deduplication,
    and optimization for RAG responses.
    """
    
    def __init__(self,
                 vector_manager: VectorStoreManager,
                 db_manager: DatabaseManager,
                 max_context_length: int = 4000,
                 llm_manager=None,
                 enable_relevance_filtering: bool = False,
                 relevance_threshold: float = 0.8):
        """
        Initialize context retriever.
        
        Args:
            vector_manager: Vector store manager for similarity search
            db_manager: Database manager for paper metadata
            max_context_length: Maximum total context length in characters
            llm_manager: LLM manager for relevance validation
            enable_relevance_filtering: Whether to enable LLM-based relevance filtering
            relevance_threshold: Minimum confidence threshold for relevance validation
        """
        self.vector_manager = vector_manager
        self.db_manager = db_manager
        self.max_context_length = max_context_length
        self.llm_manager = llm_manager
        self.enable_relevance_filtering = enable_relevance_filtering
        self.relevance_threshold = relevance_threshold
        
        logger.info(f"Initialized context retriever (relevance filtering: {enable_relevance_filtering})")
    
    def retrieve_and_rank_contexts(self,
                                  query: str,
                                  max_contexts: int = 5,
                                  similarity_threshold: float = 0.3,
                                  diversity_weight: float = 0.2,
                                  recency_weight: float = 0.1) -> List[RetrievedContext]:
        """
        Retrieve, rank, and optimize contexts for the given query.
        
        Args:
            query: User query
            max_contexts: Maximum number of contexts to return
            similarity_threshold: Minimum similarity score
            diversity_weight: Weight for paper diversity in ranking
            recency_weight: Weight for paper recency in ranking
            
        Returns:
            Ranked and optimized list of contexts
        """
        try:
            # Step 1: Initial retrieval with more candidates
            raw_results = self.vector_manager.search_similar_papers(
                query=query,
                n_results=max_contexts * 3  # Get more for better selection
            )
            
            # Step 2: Filter by similarity threshold
            filtered_results = [
                result for result in raw_results
                if result["similarity"] >= similarity_threshold
            ]
            
            if not filtered_results:
                logger.info("No results above similarity threshold")
                return []
            
            # Step 3: Convert to RetrievedContext objects
            contexts = self._create_context_objects(filtered_results)
            
            # Step 3.5: Apply relevance filtering if enabled
            if self.enable_relevance_filtering and self.llm_manager:
                contexts = self._filter_contexts_by_relevance(contexts, query)
            
            # Step 4: Remove duplicates
            contexts = self._deduplicate_contexts(contexts)
            
            # Step 5: Rank contexts with advanced scoring
            ranked_contexts = self._rank_contexts(
                contexts, query, diversity_weight, recency_weight
            )
            
            # Step 6: Optimize for context window
            optimized_contexts = self._optimize_context_window(
                ranked_contexts, max_contexts
            )
            
            logger.info(f"Retrieved and ranked {len(optimized_contexts)} contexts")
            return optimized_contexts
            
        except Exception as e:
            logger.error(f"Error in context retrieval and ranking: {e}")
            return []
    
    def _create_context_objects(self, search_results: List[Dict[str, Any]]) -> List[RetrievedContext]:
        """Convert search results to RetrievedContext objects with metadata."""
        contexts = []
        
        # Get paper metadata in batch
        paper_ids = {result["metadata"]["paper_id"] for result in search_results}
        paper_cache = {}
        
        with self.db_manager.get_session() as session:
            papers = session.query(Paper).filter(Paper.id.in_(paper_ids)).all()
            paper_cache = {paper.id: paper for paper in papers}
        
        for result in search_results:
            paper_id = result["metadata"]["paper_id"]
            paper = paper_cache.get(paper_id)
            
            if paper:
                context = RetrievedContext(
                    content=result["document"],
                    paper_id=paper_id,
                    paper_title=paper.title,
                    authors=paper.get_authors_list(),
                    year=paper.year,
                    chunk_index=result["metadata"].get("chunk_index"),
                    similarity_score=result["similarity"],
                    source_info=self._format_citation(paper)
                )
                contexts.append(context)
        
        return contexts
    
    def _format_citation(self, paper: Paper) -> str:
        """Format paper citation for source attribution."""
        authors = paper.get_authors_list()
        author_str = ", ".join(authors[:2])  # First two authors
        if len(authors) > 2:
            author_str += " et al."
        
        year_str = f"({paper.year})" if paper.year else ""
        
        return f"{author_str} {year_str}. {paper.title}"
    
    def _deduplicate_contexts(self, contexts: List[RetrievedContext]) -> List[RetrievedContext]:
        """Remove duplicate or highly similar contexts."""
        if not contexts:
            return contexts
        
        deduplicated = []
        seen_content = set()
        
        for context in contexts:
            # Create a content signature for duplicate detection
            content_signature = self._create_content_signature(context.content)
            
            # Check for exact duplicates
            if content_signature in seen_content:
                continue
            
            # Check for high similarity with existing contexts
            is_duplicate = False
            for existing in deduplicated:
                similarity = self._calculate_text_similarity(
                    context.content, existing.content
                )
                if similarity > 0.85:  # Very similar content
                    # Keep the one with higher similarity score
                    if context.similarity_score <= existing.similarity_score:
                        is_duplicate = True
                        break
                    else:
                        # Replace existing with current (higher similarity)
                        deduplicated.remove(existing)
                        break
            
            if not is_duplicate:
                deduplicated.append(context)
                seen_content.add(content_signature)
        
        logger.info(f"Deduplicated {len(contexts)} -> {len(deduplicated)} contexts")
        return deduplicated
    
    def _create_content_signature(self, content: str) -> str:
        """Create a signature for content duplicate detection."""
        # Remove whitespace and punctuation, keep key words
        words = re.findall(r'\b\w+\b', content.lower())
        # Keep only words longer than 3 characters
        key_words = [w for w in words if len(w) > 3]
        # Sort and take first 10 words as signature
        signature = " ".join(sorted(key_words)[:10])
        return signature
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two text strings."""
        return SequenceMatcher(None, text1, text2).ratio()
    
    def _rank_contexts(self,
                      contexts: List[RetrievedContext],
                      query: str,
                      diversity_weight: float,
                      recency_weight: float) -> List[RetrievedContext]:
        """
        Rank contexts using multiple factors.
        
        Args:
            contexts: List of contexts to rank
            query: Original query for relevance scoring
            diversity_weight: Weight for diversity factor
            recency_weight: Weight for recency factor
            
        Returns:
            Contexts sorted by final score (highest first)
        """
        if not contexts:
            return contexts
        
        # Group contexts by paper for diversity calculation
        paper_groups = defaultdict(list)
        for ctx in contexts:
            paper_groups[ctx.paper_id].append(ctx)
        
        # Calculate scores for each context
        scored_contexts = []
        
        for context in contexts:
            score = self._calculate_context_score(
                context, query, paper_groups, diversity_weight, recency_weight
            )
            scored_contexts.append((context, score))
        
        # Sort by final score (descending)
        scored_contexts.sort(key=lambda x: x[1].final_score, reverse=True)
        
        # Return sorted contexts
        return [ctx for ctx, score in scored_contexts]
    
    def _calculate_context_score(self,
                               context: RetrievedContext,
                               query: str,
                               paper_groups: Dict[int, List[RetrievedContext]],
                               diversity_weight: float,
                               recency_weight: float) -> ContextScore:
        """Calculate comprehensive score for a context."""
        
        # Base similarity score (0-1)
        base_similarity = context.similarity_score
        
        # Recency boost (newer papers get slight boost)
        recency_boost = 0.0
        if context.year:
            # Boost papers from last 5 years
            years_old = max(0, 2024 - context.year)
            if years_old <= 5:
                recency_boost = (5 - years_old) / 50  # Max boost of 0.1
        
        # Diversity penalty (penalize multiple chunks from same paper)
        diversity_penalty = 0.0
        paper_chunk_count = len(paper_groups[context.paper_id])
        if paper_chunk_count > 1:
            # Small penalty for papers with multiple chunks selected
            diversity_penalty = (paper_chunk_count - 1) * 0.05
        
        # Content length score (prefer moderate length chunks)
        content_length = len(context.content)
        if 200 <= content_length <= 1000:  # Sweet spot
            length_score = 0.02
        elif content_length < 100:  # Too short
            length_score = -0.05
        elif content_length > 2000:  # Too long
            length_score = -0.03
        else:
            length_score = 0.0
        
        # Calculate final score
        final_score = (
            base_similarity +
            (recency_boost * recency_weight) -
            (diversity_penalty * diversity_weight) +
            length_score
        )
        
        return ContextScore(
            base_similarity=base_similarity,
            recency_boost=recency_boost,
            diversity_penalty=diversity_penalty,
            length_score=length_score,
            final_score=final_score
        )
    
    def _optimize_context_window(self,
                               ranked_contexts: List[RetrievedContext],
                               max_contexts: int) -> List[RetrievedContext]:
        """
        Optimize contexts to fit within context window constraints.
        
        Args:
            ranked_contexts: Contexts sorted by relevance
            max_contexts: Maximum number of contexts
            
        Returns:
            Optimized list of contexts
        """
        if not ranked_contexts:
            return ranked_contexts
        
        optimized = []
        total_length = 0
        
        for context in ranked_contexts:
            # Check if adding this context would exceed limits
            context_length = len(context.content)
            
            if (len(optimized) >= max_contexts or 
                total_length + context_length > self.max_context_length):
                
                # Try to fit a shorter context if available
                if len(optimized) < max_contexts:
                    # Look for shorter contexts in remaining list
                    for remaining_context in ranked_contexts[len(optimized):]:
                        remaining_length = len(remaining_context.content)
                        if (total_length + remaining_length <= self.max_context_length and
                            remaining_length < context_length):
                            optimized.append(remaining_context)
                            total_length += remaining_length
                            break
                break
            
            optimized.append(context)
            total_length += context_length
        
        logger.info(f"Optimized to {len(optimized)} contexts, total length: {total_length}")
        return optimized
    
    def _filter_contexts_by_relevance(self, contexts: List[RetrievedContext], query: str) -> List[RetrievedContext]:
        """
        Filter contexts using LLM-based relevance validation.
        
        Args:
            contexts: List of contexts to filter
            query: User query to validate against
            
        Returns:
            Filtered list of contexts that meet relevance threshold
        """
        if not self.llm_manager or not contexts:
            return contexts
        
        logger.info(f"Filtering {len(contexts)} contexts for relevance (threshold: {self.relevance_threshold})")
        
        try:
            # Process contexts in batches for efficiency
            batch_size = 5
            filtered_contexts = []
            total_filtered = 0
            
            for i in range(0, len(contexts), batch_size):
                batch = contexts[i:i + batch_size]
                batch_results = self._validate_context_batch_relevance(batch, query)
                
                for j, (context, validation_result) in enumerate(zip(batch, batch_results)):
                    if validation_result and validation_result.get('relevant', False):
                        confidence = validation_result.get('confidence', 0.0)
                        if confidence >= self.relevance_threshold:
                            # Add relevance metadata to context
                            context.metadata = context.metadata or {}
                            context.metadata['relevance_confidence'] = confidence
                            context.metadata['relevance_reason'] = validation_result.get('reason', '')
                            filtered_contexts.append(context)
                        else:
                            total_filtered += 1
                            logger.debug(f"Filtered context with confidence {confidence:.3f} < {self.relevance_threshold}")
                    else:
                        total_filtered += 1
                        logger.debug("Filtered context - not relevant")
            
            logger.info(f"Relevance filtering: kept {len(filtered_contexts)}, filtered {total_filtered}")
            return filtered_contexts
            
        except Exception as e:
            logger.error(f"Error in relevance filtering: {e}")
            # Return original contexts if filtering fails
            return contexts
    
    def _validate_context_batch_relevance(self, contexts: List[RetrievedContext], query: str) -> List[Dict]:
        """
        Validate relevance of a batch of contexts using LLM.
        
        Args:
            contexts: Batch of contexts to validate
            query: User query
            
        Returns:
            List of validation results
        """
        try:
            # Create prompt for batch validation
            validation_prompt = f"""
            You are evaluating whether text chunks help answer a research question. For each chunk, determine if it's relevant and your confidence.
            
            Research Question: "{query}"
            
            Text Chunks:
            """
            
            for i, context in enumerate(contexts):
                # Truncate context for efficiency
                content = context.content[:500] + "..." if len(context.content) > 500 else context.content
                validation_prompt += f"\n--- Chunk {i+1} ---\nTitle: {context.paper_title}\nContent: {content}\n"
            
            validation_prompt += """
            
            For each chunk, respond with a JSON object containing:
            {
                "chunk_1": {"relevant": true/false, "confidence": 0.0-1.0, "reason": "brief explanation"},
                "chunk_2": {"relevant": true/false, "confidence": 0.0-1.0, "reason": "brief explanation"},
                ...
            }
            
            Confidence scale:
            - 1.0: Directly answers the question
            - 0.8-0.9: Highly relevant supporting information
            - 0.6-0.7: Somewhat relevant context
            - 0.4-0.5: Tangentially related
            - 0.0-0.3: Not relevant
            
            Return only the JSON object, no other text.
            """
            
            # Get LLM response
            messages = [{"role": "user", "content": validation_prompt}]
            response = self.llm_manager.generate_response(
                messages=messages,
                temperature=0.1,  # Low temperature for consistency
                max_tokens=1000
            )
            
            # Parse JSON response
            import json
            import re
            
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
            if json_match:
                validation_data = json.loads(json_match.group())
                
                # Convert to list format
                results = []
                for i in range(len(contexts)):
                    chunk_key = f"chunk_{i+1}"
                    if chunk_key in validation_data:
                        results.append(validation_data[chunk_key])
                    else:
                        # Default if parsing fails
                        results.append({"relevant": True, "confidence": 0.5, "reason": "parsing_failed"})
                
                return results
            else:
                # Fallback if JSON parsing fails
                logger.warning("Failed to parse relevance validation JSON")
                return [{"relevant": True, "confidence": 0.5, "reason": "parsing_failed"} for _ in contexts]
                
        except Exception as e:
            logger.error(f"Error validating context relevance: {e}")
            # Return default results if validation fails
            return [{"relevant": True, "confidence": 0.5, "reason": "validation_failed"} for _ in contexts]
    
    def get_context_summary(self, contexts: List[RetrievedContext]) -> Dict[str, Any]:
        """Get summary statistics about retrieved contexts."""
        if not contexts:
            return {}
        
        # Paper distribution
        paper_counts = defaultdict(int)
        years = []
        similarities = []
        total_length = 0
        
        for ctx in contexts:
            paper_counts[ctx.paper_title] += 1
            if ctx.year:
                years.append(ctx.year)
            similarities.append(ctx.similarity_score)
            total_length += len(ctx.content)
        
        return {
            "total_contexts": len(contexts),
            "unique_papers": len(paper_counts),
            "paper_distribution": dict(paper_counts),
            "year_range": [min(years), max(years)] if years else None,
            "avg_similarity": sum(similarities) / len(similarities),
            "total_context_length": total_length,
            "avg_context_length": total_length / len(contexts)
        }