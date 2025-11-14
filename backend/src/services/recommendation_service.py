"""Recommendation service - wraps PaperRecommender."""

import logging
from pathlib import Path
from datetime import datetime
from typing import List
from dataclasses import dataclass

from src.core.database import DatabaseManager
from src.core.vector_store import VectorStoreManager
from src.core.llm import LLMManager, PromptManager
from src.modules.recommender import PaperRecommender
from src.models.schemas import RecommendationQueryRequest, RecommendationResponse, PaperRecommendationItem

logger = logging.getLogger(__name__)


@dataclass
class SimpleDocumentAnalysis:
    """Simple document analysis for recommendation queries."""
    title: str
    content: str
    research_questions: List[str]
    key_concepts: List[str]
    literature_gaps: List[str]


class RecommendationService:
    """Service for paper recommendations."""

    def __init__(
        self,
        db_manager: DatabaseManager,
        vector_manager: VectorStoreManager,
        llm_manager: LLMManager,
        prompt_manager: PromptManager
    ):
        self.db_manager = db_manager
        self.vector_manager = vector_manager
        self.llm_manager = llm_manager
        self.prompt_manager = prompt_manager

        self.recommender = PaperRecommender(
            db_manager=db_manager,
            vector_manager=vector_manager,
            llm_manager=llm_manager,
            prompt_manager=prompt_manager
        )

        logger.info("Initialized RecommendationService")

    async def get_recommendations(self, request: RecommendationQueryRequest) -> RecommendationResponse:
        """
        Get paper recommendations based on query.

        Args:
            request: Recommendation request with query and parameters

        Returns:
            RecommendationResponse with list of recommended papers
        """
        logger.info(f"Getting recommendations for query: {request.query}")

        # Create a simple document analysis from the query
        document_analysis = SimpleDocumentAnalysis(
            title=f"Research query: {request.query[:100]}",
            content=request.query,
            research_questions=[request.query],
            key_concepts=self._extract_key_concepts(request.query),
            literature_gaps=[]
        )

        try:
            # Get recommendations using the intelligent recommender
            semantic_scholar_papers = self.recommender.get_intelligent_recommendations(
                document_analysis=document_analysis,
                user_search_terms=request.query
            )

            logger.info(f"Found {len(semantic_scholar_papers)} recommendations")

            # Convert Semantic Scholar papers to PaperRecommendationItem
            recommendations = []
            for idx, paper in enumerate(semantic_scholar_papers[:request.max_recommendations]):
                recommendations.append(PaperRecommendationItem(
                    title=paper.title,
                    authors=paper.authors,
                    year=paper.year,
                    summary=paper.abstract or "No abstract available",
                    relevance_score=1.0 - (idx * 0.1),  # Simple relevance scoring
                    reason=self._generate_recommendation_reason(paper, request.query),
                    doi=paper.doi,
                    url=paper.url,
                    semantic_scholar_id=paper.paper_id,
                    citation_count=paper.citation_count,
                    venue=paper.venue
                ))

            # Generate gap analysis
            gap_analysis = self._generate_simple_gap_analysis(
                request.query,
                len(recommendations),
                semantic_scholar_papers
            )

            return RecommendationResponse(
                query=request.query,
                recommendations=recommendations,
                gap_analysis=gap_analysis,
                timestamp=datetime.now()
            )

        except Exception as e:
            logger.error(f"Error getting recommendations: {e}")
            # Return empty result instead of failing
            return RecommendationResponse(
                query=request.query,
                recommendations=[],
                gap_analysis=f"Error retrieving recommendations: {str(e)}",
                timestamp=datetime.now()
            )

    def _extract_key_concepts(self, query: str) -> List[str]:
        """
        Extract key concepts from query.

        Simple implementation - split on common words and take capitalized terms.
        """
        # Split and filter
        words = query.split()
        concepts = [w for w in words if len(w) > 4 and w[0].isupper()]
        return concepts[:5] if concepts else [query[:30]]

    def _generate_recommendation_reason(self, paper, query: str) -> str:
        """Generate a reason why this paper was recommended."""
        reasons = []

        # Check if query terms appear in title
        query_words = set(query.lower().split())
        title_words = set(paper.title.lower().split())
        overlap = query_words & title_words

        if overlap:
            reasons.append(f"Matches query terms: {', '.join(list(overlap)[:3])}")

        # Check citation count
        if paper.citation_count and paper.citation_count > 50:
            reasons.append(f"Highly cited ({paper.citation_count} citations)")

        # Check recency
        if paper.year and paper.year >= datetime.now().year - 2:
            reasons.append("Recent publication")

        # Check venue
        if paper.venue:
            reasons.append(f"Published in {paper.venue}")

        return " | ".join(reasons) if reasons else "Relevant to your research area"

    def _generate_simple_gap_analysis(self, query: str, count: int, papers) -> str:
        """Generate a simple gap analysis."""
        if count == 0:
            return f"No recommendations found for '{query}'. Try broadening your search terms."

        analysis = f"Found {count} relevant papers for your query '{query}'.\n\n"

        if papers:
            # Get year distribution
            years = [p.year for p in papers if p.year]
            if years:
                recent = len([y for y in years if y >= datetime.now().year - 3])
                analysis += f"- {recent} recent papers (last 3 years)\n"

            # Get citation stats
            citations = [p.citation_count for p in papers if p.citation_count]
            if citations:
                highly_cited = len([c for c in citations if c > 50])
                analysis += f"- {highly_cited} highly cited papers (>50 citations)\n"

            analysis += f"\nConsider reviewing these papers to understand current research trends and identify potential gaps."

        return analysis
