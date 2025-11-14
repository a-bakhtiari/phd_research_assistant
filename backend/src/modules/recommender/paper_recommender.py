import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

from src.core.database import DatabaseManager, Paper
from src.core.vector_store import VectorStoreManager
from src.core.llm import LLMManager, PromptManager
from src.core.external_apis.semantic_scholar_client import SemanticScholarClient
try:
    from core.external_apis import SciHubClient, BrowserDownloadClient
except ImportError:
    SciHubClient = None
    BrowserDownloadClient = None

logger = logging.getLogger(__name__)


@dataclass
class PaperRecommendation:
    """Container for paper recommendation data."""
    title: str
    authors: List[str]
    year: int
    summary: str
    relevance_score: float
    reason: str
    source: str  # "perplexity", "vector_store"
    priority: str  # "high", "medium", "low"
    category: str  # "methodology", "related_work", "background", etc.
    doi: Optional[str] = None
    url: Optional[str] = None
    downloadable: bool = False
    download_attempted: bool = False
    download_status: Optional[str] = None  # "success", "failed", "pending"


@dataclass
class RecommendationReport:
    """Container for complete recommendation analysis."""
    query: str
    recommendations: List[PaperRecommendation]
    gap_analysis: str
    current_literature_summary: str
    search_strategy: str
    timestamp: datetime


class PaperRecommender:
    """AI-powered paper recommendation system using multiple sources."""
    
    def __init__(self,
                 db_manager: DatabaseManager,
                 vector_manager: VectorStoreManager,
                 llm_manager: LLMManager,
                 prompt_manager: PromptManager,
                 scihub_client: Optional[SciHubClient] = None,
                 browser_download_client: Optional[BrowserDownloadClient] = None):
        """Initialize paper recommender with all required components."""
        self.db_manager = db_manager
        self.vector_manager = vector_manager
        self.llm_manager = llm_manager
        self.prompt_manager = prompt_manager
        self.scihub_client = scihub_client
        self.browser_download_client = browser_download_client
        
        # Initialize Semantic Scholar client for metadata enrichment
        self.semantic_scholar_client = SemanticScholarClient()
        
        logger.info("Initialized Paper Recommender with DeepSeek + Semantic Scholar integration")
    
    def get_recommendations_from_analysis(self,
                                        document_analysis,
                                        recommendation_context,
                                        max_recommendations: int = 10) -> RecommendationReport:
        """
        Get paper recommendations based on document analysis and context.
        
        Args:
            document_analysis: DocumentAnalysis from DocumentProcessor
            recommendation_context: RecommendationContext with gaps and priorities
            max_recommendations: Maximum number of recommendations
            
        Returns:
            RecommendationReport with targeted recommendations
        """
        logger.info(f"Getting recommendations for document: {document_analysis.title}")
        
        # Create search queries based on document analysis
        search_queries = []
        
        # Add research questions as queries
        search_queries.extend(document_analysis.research_questions[:2])
        
        # Add literature gaps
        search_queries.extend(document_analysis.literature_gaps[:2])
        
        # Add key concepts
        search_queries.extend([f"recent advances in {concept}" for concept in document_analysis.key_concepts[:3]])
        
        # Add priority topics from context
        search_queries.extend([f"{topic} methodology" for topic in recommendation_context.priority_topics[:2]])
        
        # Get recommendations for each query
        all_recommendations = []
        current_papers = self._get_current_literature()
        
        for query in search_queries[:5]:  # Limit to top 5 queries
            try:
                # Get Perplexity recommendations for this specific query
                perplexity_recs = self._get_perplexity_recommendations(
                    document_analysis.content[:1000],  # First 1000 chars as context
                    query,
                    current_papers
                )
                all_recommendations.extend(perplexity_recs)
                
                # Add vector-based recommendations
                vector_recs = self._identify_vector_gaps(query, [p.summary for p in current_papers if p.summary])
                all_recommendations.extend(vector_recs[:2])  # Top 2 per query
                
            except Exception as e:
                logger.warning(f"Error getting recommendations for query '{query}': {e}")
                continue
        
        # Prioritize recommendations based on document needs
        prioritized_recommendations = self._prioritize_by_document_needs(
            all_recommendations, 
            document_analysis,
            recommendation_context
        )
        
        # Deduplicate and rank
        ranked_recommendations = self._rank_and_deduplicate_recommendations(
            prioritized_recommendations, max_recommendations
        )
        
        # Enrich with Semantic Scholar metadata (this includes rate limiting)
        logger.info("Enriching recommendations with Semantic Scholar metadata...")
        final_recommendations = self._enrich_recommendations_with_semantic_scholar(ranked_recommendations)
        
        # Generate targeted gap analysis
        gap_analysis = self._generate_document_gap_analysis(
            document_analysis, recommendation_context, final_recommendations
        )
        
        # Create literature summary focused on document content
        lit_summary = self._create_document_literature_summary(
            current_papers, document_analysis
        )
        
        return RecommendationReport(
            query=f"Document-based recommendations for: {document_analysis.title}",
            recommendations=final_recommendations,
            gap_analysis=gap_analysis,
            current_literature_summary=lit_summary,
            search_strategy="Document analysis + multi-query search",
            timestamp=datetime.now()
        )
    
    def get_recommendations_for_document(self,
                                       document_text: str,
                                       research_question: str,
                                       max_recommendations: int = 10) -> RecommendationReport:
        """
        Get paper recommendations based on current document and research question.
        
        Args:
            document_text: Current document being written
            research_question: Main research question/focus
            max_recommendations: Maximum number of recommendations
            
        Returns:
            RecommendationReport with recommendations and analysis
        """
        logger.info(f"Getting recommendations for research question: {research_question}")
        
        # Step 1: Analyze current literature in database
        current_papers = self._get_current_literature()
        current_summaries = [paper.summary for paper in current_papers if paper.summary]
        
        # Step 2: Identify gaps using vector similarity analysis
        vector_gaps = self._identify_vector_gaps(research_question, current_summaries)
        
        # Step 3: Get external recommendations using Perplexity
        perplexity_recommendations = self._get_perplexity_recommendations(
            document_text, research_question, current_papers
        )
        
        # Step 4: Semantic Scholar removed - using Perplexity and vector store only
        semantic_recommendations = []
        
        # Step 5: Combine and rank all recommendations
        all_recommendations = (
            vector_gaps + 
            perplexity_recommendations + 
            semantic_recommendations
        )
        
        # Step 6: Deduplicate and rank recommendations
        final_recommendations = self._rank_and_deduplicate_recommendations(
            all_recommendations, max_recommendations
        )
        
        # Step 7: Generate gap analysis using LLM
        gap_analysis = self._generate_gap_analysis(
            current_papers, final_recommendations, research_question
        )
        
        return RecommendationReport(
            query=research_question,
            recommendations=final_recommendations,
            gap_analysis=gap_analysis,
            current_literature_summary=self._summarize_current_literature(current_papers),
            search_strategy=self._describe_search_strategy(),
            timestamp=datetime.now()
        )
    
    def find_similar_papers(self, 
                           paper_title: str, 
                           max_results: int = 5) -> List[PaperRecommendation]:
        """
        Find papers similar to a given paper title.
        
        Args:
            paper_title: Title of the reference paper
            max_results: Maximum number of similar papers
            
        Returns:
            List of similar paper recommendations
        """
        recommendations = []
        
        # Search using vector similarity in existing library
        vector_results = self.vector_manager.search_similar_papers(
            paper_title, n_results=max_results
        )
        
        for result in vector_results:
            metadata = result["metadata"]
            recommendations.append(PaperRecommendation(
                title=metadata.get("title", "Unknown"),
                authors=[metadata.get("authors", "Unknown")],
                year=metadata.get("year", 0),
                summary=result["document"][:200] + "...",
                relevance_score=result["similarity"],
                reason=f"Vector similarity to '{paper_title}' (score: {result['similarity']:.3f})",
                source="vector_store",
                priority="medium",
                category="related_work"
            ))
        
        # Search using Perplexity for external papers
        try:
            perplexity_result = self.perplexity_client.search_academic_papers(
                query=f"papers similar to: {paper_title}",
                max_results=max_results
            )
            
            # Parse Perplexity results (would need custom parsing based on response format)
            perplexity_recommendations = self._parse_perplexity_search_results(
                perplexity_result, "related_work"
            )
            recommendations.extend(perplexity_recommendations)
            
        except Exception as e:
            logger.error(f"Error getting Perplexity recommendations: {e}")
        
        return recommendations[:max_results]
    
    def get_methodological_recommendations(self,
                                        methodology_description: str,
                                        research_area: str) -> List[PaperRecommendation]:
        """
        Get recommendations for methodological approaches.
        
        Args:
            methodology_description: Description of current methodology
            research_area: Research area/domain
            
        Returns:
            List of methodological paper recommendations
        """
        query = f"methodological approaches for {methodology_description} in {research_area}"
        
        try:
            perplexity_result = self.perplexity_client.search_academic_papers(
                query=query,
                research_area=research_area,
                max_results=8
            )
            
            recommendations = self._parse_perplexity_search_results(
                perplexity_result, "methodology"
            )
            
            # Enhance with LLM analysis
            enhanced_recommendations = self._enhance_recommendations_with_llm(
                recommendations, methodology_description, "methodology"
            )
            
            return enhanced_recommendations
            
        except Exception as e:
            logger.error(f"Error getting methodological recommendations: {e}")
            return []
    
    def _get_current_literature(self) -> List[Paper]:
        """Get current papers from the database."""
        return self.db_manager.get_all_papers()
    
    def get_intelligent_recommendations(self, 
                                      document_analysis,
                                      user_search_terms: Optional[str] = None) -> List:
        """
        Get intelligent paper recommendations using DeepSeek analysis + Semantic Scholar.
        
        Args:
            document_analysis: DocumentAnalysis from DocumentProcessor
            user_search_terms: Optional additional search terms from user
            
        Returns:
            List of SemanticScholarPaper objects
        """
        logger.info(f"Getting intelligent recommendations for: {document_analysis.title}")
        
        # 1. Get existing papers from database
        existing_papers = self._get_current_literature()
        logger.info(f"Found {len(existing_papers)} existing papers in collection")
        
        # 2. DeepSeek analyzes for TRUE research gaps
        research_gaps = self._identify_research_gaps_with_deepseek(
            document_analysis, existing_papers
        )
        logger.info(f"Identified research gaps: {research_gaps[:100]}...")
        
        # 3. DeepSeek generates optimal search query
        search_query = self._generate_academic_search_query(
            document_analysis, research_gaps, user_search_terms
        )
        logger.info(f"Generated search query: {search_query}")
        
        # 4. Single Semantic Scholar call
        try:
            papers = self.semantic_scholar_client.search_papers(query=search_query, limit=10)  # Get more papers initially
            logger.info(f"Found {len(papers)} papers from Semantic Scholar with AI-generated query")

            # Fallback: If AI query returns 0 results, try user's original query
            if len(papers) == 0 and user_search_terms:
                logger.warning(f"AI-generated query returned 0 results. Falling back to user query: {user_search_terms}")
                papers = self.semantic_scholar_client.search_papers(query=user_search_terms, limit=10)
                logger.info(f"Found {len(papers)} papers from Semantic Scholar with user query")

            # 5. Filter out duplicates (papers already in database)
            filtered_papers = self._filter_duplicate_papers(papers, existing_papers)
            logger.info(f"After duplicate filtering: {len(filtered_papers)} unique papers remain")

            # Return top 5 unique papers
            return filtered_papers[:5]
        except Exception as e:
            logger.error(f"Error searching Semantic Scholar: {e}")
            return []
    
    def _filter_duplicate_papers(self, candidate_papers, existing_papers):
        """
        Filter out papers that already exist in the database.
        
        Args:
            candidate_papers: List of SemanticScholarPaper objects from search
            existing_papers: List of Paper objects from database
            
        Returns:
            List of SemanticScholarPaper objects not already in database
        """
        if not candidate_papers:
            return []
        
        # Create sets of existing paper identifiers for fast lookup
        existing_titles = {paper.title.lower().strip() for paper in existing_papers if paper.title}
        existing_dois = {paper.doi.lower().strip() for paper in existing_papers if paper.doi}
        existing_ss_ids = {paper.semantic_scholar_id for paper in existing_papers if paper.semantic_scholar_id}
        
        filtered_papers = []
        for paper in candidate_papers:
            # Check if paper already exists by various criteria
            is_duplicate = False
            
            # Check by title (fuzzy matching)
            if paper.title and paper.title.lower().strip() in existing_titles:
                logger.debug(f"Duplicate found by title: {paper.title}")
                is_duplicate = True
            
            # Check by DOI
            elif paper.doi and paper.doi.lower().strip() in existing_dois:
                logger.debug(f"Duplicate found by DOI: {paper.doi}")
                is_duplicate = True
                
            # Check by Semantic Scholar ID
            elif paper.paper_id and paper.paper_id in existing_ss_ids:
                logger.debug(f"Duplicate found by SS ID: {paper.paper_id}")
                is_duplicate = True
            
            # Check by title similarity (for slight variations)
            elif paper.title:
                paper_title_clean = paper.title.lower().strip()
                for existing_title in existing_titles:
                    # Simple similarity check - if 90% of words match
                    paper_words = set(paper_title_clean.split())
                    existing_words = set(existing_title.split())
                    if len(paper_words) > 0 and len(existing_words) > 0:
                        overlap = len(paper_words.intersection(existing_words))
                        similarity = overlap / max(len(paper_words), len(existing_words))
                        if similarity > 0.9:
                            logger.debug(f"Duplicate found by title similarity: {paper.title}")
                            is_duplicate = True
                            break
            
            if not is_duplicate:
                filtered_papers.append(paper)
        
        return filtered_papers
    
    def _identify_research_gaps_with_deepseek(self, document_analysis, existing_papers) -> str:
        """Use DeepSeek to identify actual research gaps (not just language gaps)."""
        try:
            # All papers - title + abstract only
            papers_metadata = [
                f"'{paper.title}' ({paper.year}) - {paper.summary}"
                for paper in existing_papers
            ]
            
            prompt = f"""
Analyze this research document and existing paper collection to identify actual RESEARCH GAPS.

DOCUMENT BEING WRITTEN:
Title: {getattr(document_analysis, 'title', 'Unknown')}
Content: {getattr(document_analysis, 'content', '')}
Key Concepts: {getattr(document_analysis, 'key_concepts', [])}
Research Questions: {getattr(document_analysis, 'research_questions', [])}

EXISTING PAPERS IN COLLECTION (title + abstract):
{chr(10).join(papers_metadata)}

TASK: Identify actual RESEARCH GAPS - what knowledge, methodologies, or research directions are missing from this collection that would be valuable for the document being written?

Focus on:
- Missing research methodologies
- Unanswered research questions
- Unexplored application areas  
- Recent developments not covered
- Different theoretical approaches

Ignore:
- Different terminology for same concepts
- Language variations
- Minor implementation differences

Provide a concise analysis of the key research gaps that should be addressed.
"""

            messages = [{"role": "user", "content": prompt}]
            response = self.llm_manager.generate_response(messages)
            return response.content
            
        except Exception as e:
            logger.error(f"Error identifying research gaps with DeepSeek: {e}")
            return "Unable to analyze research gaps"
    
    def _generate_academic_search_query(self, document_analysis, research_gaps, user_search_terms) -> str:
        """Use DeepSeek to generate optimal academic search query for Semantic Scholar."""
        try:
            prompt = f"""
Create an optimal academic search query for Semantic Scholar API based on this analysis.

DOCUMENT FOCUS:
Key Concepts: {getattr(document_analysis, 'key_concepts', [])}
Research Questions: {getattr(document_analysis, 'research_questions', [])}

RESEARCH GAPS IDENTIFIED:
{research_gaps}

USER SEARCH TERMS: {user_search_terms or 'None provided'}

TASK: Generate a focused academic search query that will find papers addressing these research gaps.

Guidelines:
- Use scholarly terminology
- Focus on the most important gaps
- Include relevant methodological terms
- Optimize for academic paper search
- Keep query focused but comprehensive
- Combine concepts intelligently

Provide only the search query, no explanation.
"""

            messages = [{"role": "user", "content": prompt}]
            response = self.llm_manager.generate_response(messages)
            return response.content.strip().strip('"').strip("'")
            
        except Exception as e:
            logger.error(f"Error generating search query with DeepSeek: {e}")
            # Fallback to simple query
            concepts = getattr(document_analysis, 'key_concepts', [])
            return ' '.join(concepts[:3]) if concepts else 'machine learning'
    
    def _identify_vector_gaps(self, 
                            research_question: str, 
                            current_summaries: List[str]) -> List[PaperRecommendation]:
        """Identify gaps using vector similarity analysis."""
        try:
            gaps = self.vector_manager.find_literature_gaps(
                current_papers=current_summaries,
                query_context=research_question,
                similarity_threshold=0.4
            )
            
            recommendations = []
            for gap in gaps[:5]:  # Top 5 gaps
                metadata = gap["metadata"]
                recommendations.append(PaperRecommendation(
                    title=metadata.get("title", "Unknown"),
                    authors=[metadata.get("authors", "Unknown")],
                    year=metadata.get("year", 0),
                    summary=gap["document"][:200] + "...",
                    relevance_score=gap["gap_score"],
                    reason=f"Identified as literature gap (gap score: {gap['gap_score']:.3f})",
                    source="vector_store",
                    priority="high" if gap["gap_score"] > 0.7 else "medium",
                    category="gap_filling"
                ))
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error identifying vector gaps: {e}")
            return []
    
    def _get_perplexity_recommendations(self,
                                      document_text: str,
                                      research_question: str,
                                      current_papers: List[Paper]) -> List[PaperRecommendation]:
        """Get recommendations using Perplexity API."""
        try:
            # Prepare existing references for Perplexity
            existing_refs = [f"{paper.title} ({paper.year})" for paper in current_papers]
            
            # Get recommendations from Perplexity using the working search method
            perplexity_result = self.perplexity_client.search_academic_papers(
                query=research_question,
                research_area="machine learning",
                max_results=8
            )
            
            # Parse and convert to recommendations
            recommendations = self._parse_perplexity_recommendations(perplexity_result)
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error getting Perplexity recommendations: {e}")
            return []
    
    
    def _parse_perplexity_recommendations(self, 
                                        result) -> List[PaperRecommendation]:
        """Parse Perplexity API results into recommendations."""
        recommendations = []
        
        try:
            # Extract JSON from Perplexity's structured response
            answer = result.answer
            
            # Look for JSON block in the response
            import json
            import re
            
            # Find JSON block between ```json and ```
            json_pattern = r'```json\s*(\[.*?\])\s*```'
            json_match = re.search(json_pattern, answer, re.DOTALL)
            
            if json_match:
                json_text = json_match.group(1)
                
                try:
                    papers_data = json.loads(json_text)
                    
                    for paper in papers_data:
                        if isinstance(paper, dict) and 'title' in paper:
                            # Extract DOI from doi or link field
                            doi = paper.get('doi')
                            link = paper.get('link')
                            
                            recommendations.append(PaperRecommendation(
                                title=paper.get('title', 'Unknown Title'),
                                authors=paper.get('authors', ['Unknown']),
                                year=paper.get('year', 2024),
                                summary=paper.get('summary', 'No summary available'),
                                relevance_score=0.9,
                                reason=f"Recommended by Perplexity AI for research. Published in {paper.get('venue', 'Unknown venue')}.",
                                source="perplexity",
                                priority="high",
                                category="external_search",
                                doi=doi,
                                url=link
                            ))
                            
                    logger.info(f"Successfully parsed {len(recommendations)} papers from Perplexity JSON response")
                    
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse JSON from Perplexity: {e}")
                    recommendations = self._fallback_parse_perplexity(result)
            else:
                logger.warning("No JSON block found in Perplexity response, trying fallback parsing")
                recommendations = self._fallback_parse_perplexity(result)
                
        except Exception as e:
            logger.error(f"Error parsing Perplexity recommendations: {e}")
            recommendations = self._fallback_parse_perplexity(result)
        
        return recommendations
    
    def _fallback_parse_perplexity(self, result) -> List[PaperRecommendation]:
        """Fallback parser for Perplexity results when table parsing fails."""
        recommendations = []
        
        try:
            # Use LLM to extract paper titles from text
            parsing_prompt = f"""
            Extract paper recommendations from this text. Return ONLY a JSON list format like this:
            [
              {{
                "title": "Exact Paper Title",
                "authors": ["Author 1", "Author 2"],
                "year": 2024,
                "summary": "Brief summary"
              }}
            ]
            
            Text to parse:
            {result.answer[:2000]}
            
            Extract only real academic papers mentioned in the text. Return valid JSON only.
            """
            
            messages = [{"role": "user", "content": parsing_prompt}]
            llm_response = self.llm_manager.generate_response(messages, temperature=0.1)
            
            # Try to parse JSON response
            import json
            import re
            
            # Extract JSON from response
            json_match = re.search(r'\[.*\]', llm_response, re.DOTALL)
            if json_match:
                papers_data = json.loads(json_match.group())
                
                for paper in papers_data[:5]:  # Limit to 5 papers
                    if isinstance(paper, dict) and 'title' in paper:
                        recommendations.append(PaperRecommendation(
                            title=paper.get('title', 'Unknown Title'),
                            authors=paper.get('authors', ['Unknown']),
                            year=paper.get('year', 2024),
                            summary=paper.get('summary', 'No summary available'),
                            relevance_score=0.7,
                            reason="Extracted from Perplexity search results",
                            source="perplexity",
                            priority="medium",
                            category="external_search"
                        ))
            
        except Exception as e:
            logger.error(f"Fallback parsing failed: {e}")
            
        return recommendations
    
    def _parse_perplexity_search_results(self, 
                                       result,
                                       category: str) -> List[PaperRecommendation]:
        """Parse general Perplexity search results."""
        # Similar to above but for general search results
        return self._parse_perplexity_recommendations(result)
    
    def _rank_and_deduplicate_recommendations(self,
                                            recommendations: List[PaperRecommendation],
                                            max_results: int) -> List[PaperRecommendation]:
        """Rank and deduplicate recommendations."""
        # Remove duplicates based on title similarity
        unique_recommendations = []
        seen_titles = set()
        
        for rec in recommendations:
            title_lower = rec.title.lower().strip()
            if title_lower not in seen_titles:
                seen_titles.add(title_lower)
                unique_recommendations.append(rec)
        
        # Sort by relevance score (descending) and priority
        priority_weights = {"high": 3, "medium": 2, "low": 1}
        
        unique_recommendations.sort(
            key=lambda x: (priority_weights.get(x.priority, 1), x.relevance_score),
            reverse=True
        )
        
        return unique_recommendations[:max_results]
    
    def _generate_gap_analysis(self,
                             current_papers: List[Paper],
                             recommendations: List[PaperRecommendation],
                             research_question: str) -> str:
        """Generate gap analysis using LLM."""
        try:
            current_titles = [paper.title for paper in current_papers]
            recommended_titles = [rec.title for rec in recommendations]
            
            messages = self.prompt_manager.format_prompt(
                "find_gaps",
                research_context=research_question,
                current_papers="\n".join(current_titles[:10]),
                candidate_papers="\n".join(recommended_titles[:10])
            )
            
            response = self.llm_manager.generate_response(messages, temperature=0.5)
            return response.content
            
        except Exception as e:
            logger.error(f"Error generating gap analysis: {e}")
            return "Gap analysis could not be generated."
    
    def _summarize_current_literature(self, papers: List[Paper]) -> str:
        """Summarize current literature in the database."""
        if not papers:
            return "No papers currently in the literature database."
        
        summary = f"Current literature contains {len(papers)} papers:\n"
        summary += f"- Years covered: {min(p.year for p in papers if p.year)} to {max(p.year for p in papers if p.year)}\n"
        summary += f"- Status: {len([p for p in papers if p.status == 'read'])} read, "
        summary += f"{len([p for p in papers if p.status == 'unread'])} unread\n"
        
        return summary
    
    def _describe_search_strategy(self) -> str:
        """Describe the search strategy used."""
        strategy = "Multi-source recommendation strategy:\n"
        strategy += "1. Vector similarity analysis of existing literature\n"
        strategy += "2. Perplexity AI search for recent academic papers\n"
        strategy += "3. LLM-powered gap analysis and ranking\n"
        
        return strategy
    
    def _enhance_recommendations_with_llm(self,
                                        recommendations: List[PaperRecommendation],
                                        context: str,
                                        category: str) -> List[PaperRecommendation]:
        """Enhance recommendations with LLM analysis."""
        # This could add more detailed reasoning or re-rank based on LLM analysis
        return recommendations
    
    def download_recommended_papers(self, 
                                  recommendations: List[PaperRecommendation],
                                  max_downloads: int = 5,
                                  download_directory: str = None) -> Dict[str, Any]:
        """
        Download recommended papers for review.
        Papers are downloaded to a review directory, NOT automatically added to library.
        
        Args:
            recommendations: List of paper recommendations
            max_downloads: Maximum number of papers to download
            download_directory: Optional custom download directory
            
        Returns:
            Dictionary with download results and review queue information
        """
        if not self.scihub_client:
            logger.warning("SciHub client not available for downloading")
            return {"error": "SciHub client not configured"}
        
        if download_directory:
            self.scihub_client.set_download_directory(download_directory)
        
        download_results = {}
        downloaded_count = 0
        
        # Sort recommendations by priority and relevance
        sorted_recs = sorted(
            recommendations,
            key=lambda x: (
                {"high": 3, "medium": 2, "low": 1}.get(x.priority, 1),
                x.relevance_score
            ),
            reverse=True
        )
        
        for rec in sorted_recs:
            if downloaded_count >= max_downloads:
                break
            
            # Try to download using DOI first, then title
            download_result = None
            
            if rec.doi:
                download_result = self.scihub_client.download_paper(
                    rec.doi,
                    custom_filename=f"{rec.authors[0].split()[-1]}_{rec.year}_{rec.title[:30].replace(' ', '_')}"
                )
            
            if not download_result or not download_result.success:
                # Try with title if DOI failed
                download_result = self.scihub_client.search_and_download(
                    rec.title,
                    rec.authors,
                    rec.year
                )
            
            # Update recommendation with download status
            rec.download_attempted = True
            rec.download_status = "success" if download_result.success else "failed"
            rec.downloadable = download_result.success
            
            download_results[rec.title] = {
                "success": download_result.success,
                "file_path": download_result.file_path,
                "error": download_result.error_message,
                "recommendation": rec
            }
            
            if download_result.success:
                downloaded_count += 1
                logger.info(f"Downloaded paper for review: {rec.title}")
            else:
                logger.warning(f"Failed to download: {rec.title} - {download_result.error_message}")
        
        return {
            "total_attempted": len([r for r in sorted_recs[:max_downloads]]),
            "successful_downloads": downloaded_count,
            "download_results": download_results,
            "review_queue_info": f"{downloaded_count} papers ready for review"
        }
    
    def browser_assisted_download(self, 
                                recommendations: List[PaperRecommendation],
                                max_downloads: int = 5,
                                downloads_folder: str = None,
                                monitor_downloads: bool = True) -> Dict[str, Any]:
        """
        Browser-assisted download of recommended papers.
        Opens download links in browser and monitors downloads folder.
        
        Args:
            recommendations: List of paper recommendations
            max_downloads: Maximum number of papers to process
            downloads_folder: User's downloads folder (default: ~/Downloads)
            monitor_downloads: Whether to monitor downloads folder
            
        Returns:
            Dictionary with download links and monitoring results
        """
        if not self.browser_download_client:
            logger.warning("Browser download client not available")
            return {"error": "Browser download client not configured"}
        
        # Use default downloads folder if not specified
        if not downloads_folder:
            downloads_folder = str(Path.home() / "Downloads")
        
        download_results = []
        
        # Sort recommendations by priority and relevance
        sorted_recs = sorted(
            recommendations,
            key=lambda x: (
                {"high": 3, "medium": 2, "low": 1}.get(x.priority, 1),
                x.relevance_score
            ),
            reverse=True
        )
        
        logger.info(f"Processing {min(max_downloads, len(sorted_recs))} papers for browser download")
        
        # Process each recommendation
        for i, rec in enumerate(sorted_recs[:max_downloads]):
            logger.info(f"Processing paper {i+1}/{max_downloads}: {rec.title}")
            
            # Find download links and open browser
            result = self.browser_download_client.download_paper(
                title=rec.title,
                authors=rec.authors,
                doi=rec.doi,
                year=rec.year,
                auto_open_browser=True
            )
            
            download_results.append(result)
            
            # Small delay between opening tabs
            import time
            time.sleep(2)
        
        # Create summary of opened links
        total_links = sum(len(r.links_found) for r in download_results)
        browser_tabs_opened = sum(1 for r in download_results if r.browser_opened)
        
        summary = {
            "total_papers_processed": len(download_results),
            "browser_tabs_opened": browser_tabs_opened,
            "total_links_found": total_links,
            "download_results": download_results,
            "monitoring_info": "Download monitoring can be started separately",
            "instructions": [
                "Browser tabs have been opened with download links",
                "Download PDFs from the opened pages",
                "Downloaded files will be monitored and moved to new_papers folder",
                "Use monitor_downloads_folder() to track downloaded files"
            ]
        }
        
        # Optional: Start monitoring downloads folder
        if monitor_downloads:
            summary["monitoring_started"] = True
            summary["downloads_folder"] = downloads_folder
            logger.info(f"Download monitoring can be started for: {downloads_folder}")
        
        return summary
    
    def monitor_downloads_folder(self, 
                               downloads_folder: str = None,
                               check_interval: int = 5,
                               timeout: int = 300) -> List[str]:
        """
        Monitor downloads folder for new PDF files.
        
        Args:
            downloads_folder: Downloads folder to monitor
            check_interval: Seconds between checks
            timeout: Maximum monitoring time
            
        Returns:
            List of files that were moved to new_papers folder
        """
        if not self.browser_download_client:
            logger.warning("Browser download client not available")
            return []
        
        if not downloads_folder:
            downloads_folder = str(Path.home() / "Downloads")
        
        logger.info(f"Starting download monitoring for {timeout}s...")
        
        moved_files = self.browser_download_client.monitor_downloads_folder(
            check_interval=check_interval,
            timeout=timeout
        )
        
        logger.info(f"Download monitoring completed. Moved {len(moved_files)} files.")
        return moved_files
    
    def create_review_summary(self, 
                            recommendations: List[PaperRecommendation],
                            research_context: str) -> str:
        """
        Create a summary of recommended papers for user review.
        
        Args:
            recommendations: List of paper recommendations
            research_context: Context of the research
            
        Returns:
            Formatted summary for user review
        """
        try:
            summary_prompt = f"""
            Create a structured review summary of the following paper recommendations for the research context: "{research_context}"
            
            For each paper, provide:
            1. Title and authors
            2. Why it's relevant (in 1-2 sentences)
            3. Priority level and suggested use
            4. Download status
            
            Papers to review:
            """
            
            for i, rec in enumerate(recommendations, 1):
                summary_prompt += f"""
                {i}. "{rec.title}" by {', '.join(rec.authors[:3])} ({rec.year})
                   Relevance: {rec.reason}
                   Priority: {rec.priority}
                   Category: {rec.category}
                   Download Status: {rec.download_status or 'Not attempted'}
                """
            
            summary_prompt += """
            
            Provide a clear, actionable review summary that helps the researcher decide which papers to add to their library.
            """
            
            messages = [{"role": "user", "content": summary_prompt}]
            response = self.llm_manager.generate_response(messages, temperature=0.5)
            
            return response.content
            
        except Exception as e:
            logger.error(f"Error creating review summary: {e}")
            return f"Error generating review summary: {e}"
    
    def approve_papers_for_library(self,
                                 approved_papers: List[str],
                                 downloaded_papers: Dict[str, Any]) -> List[Paper]:
        """
        Move approved papers from review queue to main library.
        
        Args:
            approved_papers: List of paper titles that user approved
            downloaded_papers: Dictionary of downloaded papers from download_recommended_papers
            
        Returns:
            List of Paper objects added to the library
        """
        added_papers = []
        
        for title in approved_papers:
            if title in downloaded_papers["download_results"]:
                result = downloaded_papers["download_results"][title]
                
                if result["success"] and result["file_path"]:
                    try:
                        # Move file to main library processing
                        from pathlib import Path
                        review_file = Path(result["file_path"])
                        
                        if review_file.exists():
                            # Move to new_papers directory for processing
                            new_papers_dir = Path(self.scihub_client.download_directory).parent / "_NEW_PAPERS"
                            new_papers_dir.mkdir(exist_ok=True)
                            
                            final_path = new_papers_dir / review_file.name
                            review_file.rename(final_path)
                            
                            logger.info(f"Moved approved paper to processing: {final_path}")
                            
                            # The paper will be processed by the regular file management pipeline
                            # This ensures consistent metadata extraction and organization
                            
                    except Exception as e:
                        logger.error(f"Error moving approved paper {title}: {e}")
        
        return added_papers
    
    def get_recommendation_statistics(self) -> Dict[str, Any]:
        """Get statistics about recommendations and downloads."""
        try:
            stats = {
                "perplexity_available": self.perplexity_client is not None,
                "browser_download_available": self.browser_download_client is not None
            }
            
            # No download statistics from browser client (it doesn't track like SciHub did)
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting recommendation statistics: {e}")
            return {}
    
    def _prioritize_by_document_needs(self, 
                                    recommendations: List[PaperRecommendation],
                                    document_analysis,
                                    recommendation_context) -> List[PaperRecommendation]:
        """Prioritize recommendations based on document analysis needs."""
        prioritized = []
        
        for rec in recommendations:
            # Calculate priority score based on document needs
            priority_score = rec.relevance_score
            
            # Boost if matches literature gaps
            for gap in document_analysis.literature_gaps:
                if any(word.lower() in rec.title.lower() for word in gap.split() if len(word) > 3):
                    priority_score += 0.2
                    rec.reason = f"{rec.reason} | Addresses literature gap: {gap}"
            
            # Boost if matches key concepts
            for concept in document_analysis.key_concepts:
                if any(word.lower() in rec.title.lower() for word in concept.split() if len(word) > 3):
                    priority_score += 0.1
            
            # Boost if matches missing areas from context
            for area in recommendation_context.missing_areas:
                if any(word.lower() in rec.title.lower() for word in area.split() if len(word) > 3):
                    priority_score += 0.15
                    rec.reason = f"{rec.reason} | Fills missing area: {area}"
            
            # Update relevance score with priority boost
            rec.relevance_score = min(1.0, priority_score)
            prioritized.append(rec)
        
        return prioritized
    
    def _generate_document_gap_analysis(self, 
                                      document_analysis,
                                      recommendation_context, 
                                      recommendations: List[PaperRecommendation]) -> str:
        """Generate gap analysis specific to the document."""
        try:
            analysis_prompt = f"""
Based on the document analysis and recommended papers, provide a gap analysis:

DOCUMENT: {document_analysis.title}
DOCUMENT TYPE: {document_analysis.document_type}
KEY CONCEPTS: {', '.join(document_analysis.key_concepts)}
RESEARCH QUESTIONS: {'; '.join(document_analysis.research_questions)}
IDENTIFIED GAPS: {'; '.join(document_analysis.literature_gaps)}

RECOMMENDED PAPERS:
{chr(10).join([f"- {rec.title} ({rec.year}): {rec.reason}" for rec in recommendations[:5]])}

Provide a concise analysis of:
1. How these recommendations address the document's literature gaps
2. Which research questions would benefit most from these papers
3. Any remaining gaps that still need attention

Keep the analysis focused and actionable (max 300 words).
"""
            
            response = self.llm_manager.generate_response(
                messages=[{"role": "user", "content": analysis_prompt}],
                max_tokens=400
            )
            
            return response.content
            
        except Exception as e:
            logger.error(f"Error generating document gap analysis: {e}")
            return f"Gap analysis for {document_analysis.title}: The recommended papers address key areas in {', '.join(document_analysis.key_concepts)} and should help fill identified literature gaps."
    
    def _create_document_literature_summary(self, 
                                          current_papers: List[Paper], 
                                          document_analysis) -> str:
        """Create literature summary focused on document topics."""
        relevant_papers = []
        
        # Find papers relevant to document concepts
        for paper in current_papers:
            if paper.summary:
                for concept in document_analysis.key_concepts:
                    if any(word.lower() in paper.summary.lower() for word in concept.split() if len(word) > 3):
                        relevant_papers.append(paper)
                        break
        
        if not relevant_papers:
            return f"Current literature collection has limited coverage of the topics in '{document_analysis.title}'. The recommended papers should significantly enhance your research foundation."
        
        # Summarize relevant papers
        topics = {}
        for paper in relevant_papers[:10]:  # Top 10 relevant papers
            for concept in document_analysis.key_concepts:
                if any(word.lower() in paper.summary.lower() for word in concept.split() if len(word) > 3):
                    if concept not in topics:
                        topics[concept] = []
                    topics[concept].append(paper.title)
        
        summary_parts = []
        for topic, papers in topics.items():
            summary_parts.append(f"{topic}: {len(papers)} papers including {papers[0]}")
        
        return f"Current literature covers: {'; '.join(summary_parts)}. Recommended papers will expand coverage in identified gap areas."
    
    def _enrich_recommendations_with_semantic_scholar(self, recommendations: List[PaperRecommendation]) -> List[PaperRecommendation]:
        """Enrich paper recommendations with detailed metadata from Semantic Scholar."""
        enriched_recommendations = []
        
        for rec in recommendations:
            try:
                logger.info(f"Enriching recommendation: {rec.title[:50]}...")
                
                # Search for the paper by title in Semantic Scholar
                semantic_paper = self.semantic_scholar_client.search_paper_by_title(rec.title)
                
                if semantic_paper:
                    # Create enriched recommendation with Semantic Scholar data
                    enriched_rec = PaperRecommendation(
                        title=semantic_paper.title or rec.title,
                        authors=semantic_paper.authors or rec.authors,
                        year=semantic_paper.year or rec.year,
                        summary=semantic_paper.abstract or rec.summary,
                        relevance_score=rec.relevance_score,
                        reason=rec.reason,
                        source=f"{rec.source} + Semantic Scholar",
                        priority=rec.priority,
                        category=rec.category,
                        doi=semantic_paper.doi,
                        url=semantic_paper.url,
                        downloadable=bool(semantic_paper.open_access_pdf or semantic_paper.is_open_access)
                    )
                    
                    # Add enhanced metadata to reason
                    enhanced_reason = rec.reason
                    if semantic_paper.citation_count:
                        enhanced_reason += f" | {semantic_paper.citation_count} citations"
                    if semantic_paper.influential_citation_count:
                        enhanced_reason += f" ({semantic_paper.influential_citation_count} influential)"
                    if semantic_paper.venue:
                        enhanced_reason += f" | Published in {semantic_paper.venue}"
                    if semantic_paper.is_open_access:
                        enhanced_reason += " | Open Access Available"
                    if semantic_paper.tldr:
                        enhanced_reason += f" | TLDR: {semantic_paper.tldr[:100]}..."
                    
                    enriched_rec.reason = enhanced_reason
                    
                    # Store additional metadata for display
                    enriched_rec.__dict__['semantic_scholar_data'] = {
                        'citation_count': semantic_paper.citation_count,
                        'influential_citation_count': semantic_paper.influential_citation_count,
                        'venue': semantic_paper.venue,
                        'is_open_access': semantic_paper.is_open_access,
                        'open_access_pdf': semantic_paper.open_access_pdf,
                        'fields_of_study': semantic_paper.fields_of_study,
                        'tldr': semantic_paper.tldr
                    }
                    
                    enriched_recommendations.append(enriched_rec)
                    logger.info(f"✅ Enriched with {semantic_paper.citation_count} citations")
                
                else:
                    # Keep original recommendation if Semantic Scholar lookup fails
                    enriched_recommendations.append(rec)
                    logger.warning(f"❌ Could not find in Semantic Scholar: {rec.title[:50]}")
                    
            except Exception as e:
                logger.error(f"Error enriching recommendation {rec.title[:50]}: {e}")
                # Keep original recommendation on error
                enriched_recommendations.append(rec)
        
        return enriched_recommendations# force reload
