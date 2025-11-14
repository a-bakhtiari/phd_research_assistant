import requests
import json
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class PerplexitySearchResult:
    """Container for Perplexity search results."""
    query: str
    answer: str
    citations: List[Dict[str, Any]]
    usage: Dict[str, int]


class PerplexityClient:
    """Client for Perplexity API to search for academic papers and research."""
    
    def __init__(self, api_key: str):
        """
        Initialize Perplexity client.
        
        Args:
            api_key: Perplexity API key
        """
        self.api_key = api_key
        self.base_url = "https://api.perplexity.ai"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        logger.info("Initialized Perplexity client")
    
    def search_academic_papers(self, 
                              query: str, 
                              research_area: str = "",
                              year_range: str = "2020-2024",
                              max_results: int = 10) -> PerplexitySearchResult:
        """
        Search for academic papers using Perplexity AI.
        
        Args:
            query: Search query
            research_area: Specific research area/domain
            year_range: Year range for papers (e.g., "2020-2024")
            max_results: Maximum number of results
            
        Returns:
            PerplexitySearchResult with papers and analysis
        """
        # Construct academic-focused search prompt
        academic_prompt = self._build_academic_search_prompt(
            query, research_area, year_range, max_results
        )
        
        try:
            response = self._make_request(academic_prompt)
            
            return PerplexitySearchResult(
                query=query,
                answer=response.get("choices", [{}])[0].get("message", {}).get("content", ""),
                citations=self._extract_citations(response),
                usage=response.get("usage", {})
            )
            
        except Exception as e:
            logger.error(f"Error searching academic papers: {e}")
            raise
    
    def find_literature_gaps(self, 
                           current_literature: List[str],
                           research_question: str,
                           research_area: str = "") -> PerplexitySearchResult:
        """
        Use Perplexity to identify gaps in current literature.
        
        Args:
            current_literature: List of current paper titles/summaries
            research_question: Main research question
            research_area: Research domain
            
        Returns:
            PerplexitySearchResult with gap analysis
        """
        gap_analysis_prompt = self._build_gap_analysis_prompt(
            current_literature, research_question, research_area
        )
        
        try:
            response = self._make_request(gap_analysis_prompt)
            
            return PerplexitySearchResult(
                query=f"Gap analysis for: {research_question}",
                answer=response.get("choices", [{}])[0].get("message", {}).get("content", ""),
                citations=self._extract_citations(response),
                usage=response.get("usage", {})
            )
            
        except Exception as e:
            logger.error(f"Error in literature gap analysis: {e}")
            raise
    
    def get_paper_recommendations(self,
                                current_document: str,
                                existing_references: List[str],
                                research_focus: str = "") -> PerplexitySearchResult:
        """
        Get paper recommendations based on current document and existing references.
        
        Args:
            current_document: Text of current document being written
            existing_references: List of already cited papers
            research_focus: Specific research focus
            
        Returns:
            PerplexitySearchResult with recommendations
        """
        recommendation_prompt = self._build_recommendation_prompt(
            current_document, existing_references, research_focus
        )
        
        try:
            response = self._make_request(recommendation_prompt)
            
            return PerplexitySearchResult(
                query=f"Recommendations for: {research_focus}",
                answer=response.get("choices", [{}])[0].get("message", {}).get("content", ""),
                citations=self._extract_citations(response),
                usage=response.get("usage", {})
            )
            
        except Exception as e:
            logger.error(f"Error getting paper recommendations: {e}")
            raise
    
    def _build_academic_search_prompt(self, 
                                    query: str, 
                                    research_area: str,
                                    year_range: str,
                                    max_results: int) -> str:
        """Build academic search prompt for Perplexity."""
        prompt = f"""Find {max_results} recent academic papers related to: "{query}"

Research area: {research_area if research_area else "General academic research"}
Time period: {year_range}

Return ONLY the JSON array of papers. No analysis, no explanation, just the JSON:

```json
[
  {{
    "title": "Exact Paper Title",
    "authors": ["First Author", "Second Author"],
    "year": 2024,
    "venue": "Journal/Conference Name",
    "summary": "Brief summary",
    "doi": "10.1234/example or arXiv:1234.5678 or null",
    "link": "https://paper-url or null"
  }}
]
```

Requirements:
- Return exactly {max_results} high-impact papers from 2020-2024
- Use exact paper titles and author names
- Ensure valid JSON format
- Keep summaries under 50 words"""
        
        return prompt
    
    def _build_gap_analysis_prompt(self,
                                 current_literature: List[str],
                                 research_question: str,
                                 research_area: str) -> str:
        """Build literature gap analysis prompt."""
        literature_summary = "\n".join([f"- {lit}" for lit in current_literature[:10]])  # Limit to avoid token limits
        
        prompt = f"""Conduct a literature gap analysis for the following research question:
"{research_question}"

Current literature being reviewed:
{literature_summary}

Research area: {research_area if research_area else "Not specified"}

Please identify:
1. What aspects of the research question are well-covered by the current literature
2. What important gaps exist in the current literature coverage
3. Suggest 5-7 specific papers or research areas that would fill these gaps
4. Explain how each suggested paper would strengthen the literature review

For each suggested paper, provide:
- Title and authors
- Brief summary of relevance
- Specific gap it addresses
- Why it's important for the research question

Focus on recent, high-quality research from reputable sources."""
        
        return prompt
    
    def _build_recommendation_prompt(self,
                                   current_document: str,
                                   existing_references: List[str],
                                   research_focus: str) -> str:
        """Build paper recommendation prompt."""
        # Truncate document to avoid token limits
        document_excerpt = current_document[:2000] + "..." if len(current_document) > 2000 else current_document
        references_summary = "\n".join([f"- {ref}" for ref in existing_references[:15]])
        
        prompt = f"""I am writing a research paper on: "{research_focus}"

Current document excerpt:
{document_excerpt}

Already referenced papers:
{references_summary}

Please recommend 5-8 additional papers that would strengthen this work by:
1. Supporting key arguments made in the document
2. Providing methodological foundations
3. Offering contrasting perspectives or alternative approaches
4. Filling gaps in the current reference list

For each recommended paper:
- Provide full citation (title, authors, year, journal)
- Explain specific relevance to the document
- Indicate where in the paper it would be most useful (introduction, methods, discussion, etc.)
- Note its importance level (essential, very useful, supportive)

Focus on recent, peer-reviewed publications that directly relate to the research."""
        
        return prompt
    
    def _make_request(self, prompt: str) -> Dict[str, Any]:
        """Make request to Perplexity API."""
        payload = {
            "model": "sonar",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a research assistant specializing in academic literature search and analysis."
                },
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            "max_tokens": 2000,
            "temperature": 0.3
        }
        
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=self.headers,
            json=payload,
            timeout=60
        )
        
        if response.status_code != 200:
            raise Exception(f"Perplexity API error {response.status_code}: {response.text}")
        
        return response.json()
    
    def _extract_citations(self, response: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract citations from Perplexity response."""
        citations = []
        
        # Perplexity may include citations in the response metadata
        if "citations" in response:
            citations.extend(response["citations"])
        
        # Also check for citations in the message content
        message_content = response.get("choices", [{}])[0].get("message", {})
        if "citations" in message_content:
            citations.extend(message_content["citations"])
        
        return citations
    
    def test_connection(self) -> bool:
        """Test Perplexity API connection."""
        try:
            test_prompt = "What is machine learning? (Brief answer in one sentence)"
            response = self._make_request(test_prompt)
            return len(response.get("choices", [])) > 0
        except Exception as e:
            logger.error(f"Perplexity connection test failed: {e}")
            return False