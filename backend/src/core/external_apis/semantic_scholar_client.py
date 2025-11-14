import requests
import json
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import time
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


@dataclass
class SemanticScholarPaper:
    """Container for Semantic Scholar paper data."""
    paper_id: str
    title: str
    authors: List[str]
    year: int
    abstract: str
    url: str
    citation_count: int
    venue: str
    doi: Optional[str] = None
    # Enhanced fields
    influential_citation_count: Optional[int] = None
    is_open_access: bool = False
    open_access_pdf: Optional[str] = None
    fields_of_study: Optional[List[str]] = None
    tldr: Optional[str] = None


class SemanticScholarClient:
    """Enhanced client for Semantic Scholar API with rich metadata support."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Semantic Scholar client.

        Args:
            api_key: Optional Semantic Scholar API key (increases rate limits)
                    If not provided, API will work with default rate limits (100 requests/5min)
        """
        self.api_key = api_key
        self.base_url = "https://api.semanticscholar.org/graph/v1"

        # Only add API key header if one is provided
        self.headers = {}
        if self.api_key:
            self.headers["x-api-key"] = self.api_key
            logger.info("Initialized Enhanced Semantic Scholar client with API key")
        else:
            logger.info("Initialized Enhanced Semantic Scholar client (using default rate limits)")
    
    def search_paper_by_title(self, title: str, max_results: int = 5, max_retries: int = 2) -> Optional[SemanticScholarPaper]:
        """
        Search for papers by title and return the best match with enhanced metadata.
        
        Args:
            title: Paper title to search for
            max_results: Maximum results to consider
            max_retries: Number of retry attempts for failed requests
            
        Returns:
            SemanticScholarPaper with rich metadata or None if not found
        """
        for attempt in range(max_retries + 1):
            try:
                # Apply rate limiting (1 request per second)
                time.sleep(1.0)
                
                search_url = f"{self.base_url}/paper/search"
                
                # Enhanced fields for rich metadata
                fields = [
                    "paperId", "externalIds", "url", "title", "abstract", 
                    "venue", "publicationVenue", "year", "publicationDate",
                    "referenceCount", "citationCount", "influentialCitationCount",
                    "isOpenAccess", "openAccessPdf", "fieldsOfStudy", "s2FieldsOfStudy",
                    "publicationTypes", "journal", "authors", "tldr"
                ]
                
                params = {
                    "query": title,
                    "limit": max_results,
                    "fields": ",".join(fields)
                }
                
                response = requests.get(search_url, params=params, headers=self.headers, timeout=30)
                
                # Handle rate limiting
                if response.status_code == 429:
                    logger.warning("Rate limit reached, waiting and retrying...")
                    time.sleep(2.0)
                    response = requests.get(search_url, params=params, headers=self.headers, timeout=30)
                
                if response.status_code == 400:
                    # Fallback with minimal fields
                    params["fields"] = "paperId,title,abstract,authors,year,citationCount,url"
                    response = requests.get(search_url, params=params, headers=self.headers, timeout=30)
                
                response.raise_for_status()
                data = response.json()
                
                results = data.get("data", [])
                if results:
                    # Get the best match (first result)
                    best_match = results[0]
                    
                    # Fetch complete details if we have a paper ID
                    if best_match.get('paperId'):
                        return self.get_paper_details(best_match['paperId'])
                    else:
                        return self._parse_paper_data(best_match)
                
                return None
                
            except Exception as e:
                if attempt < max_retries:
                    logger.warning(f"Attempt {attempt + 1} failed for '{title}': {e}. Retrying...")
                    time.sleep(2.0)  # Wait before retry
                    continue
                else:
                    logger.error(f"All {max_retries + 1} attempts failed for '{title}': {e}")
                    return None
        
        return None
    
    def get_paper_details(self, paper_id: str) -> Optional[SemanticScholarPaper]:
        """Get detailed paper information with enhanced metadata."""
        try:
            # Apply rate limiting
            time.sleep(1.0)
            
            paper_url = f"{self.base_url}/paper/{paper_id}"
            
            # Request all available fields
            fields = [
                "paperId", "corpusId", "externalIds", "url", "title", "abstract", 
                "venue", "publicationVenue", "year", "publicationDate", 
                "referenceCount", "citationCount", "influentialCitationCount",
                "isOpenAccess", "openAccessPdf", "fieldsOfStudy", "s2FieldsOfStudy",
                "publicationTypes", "journal",
                "authors", "authors.authorId", "authors.name", "authors.url",
                "authors.affiliations", "authors.homepage", "authors.paperCount",
                "authors.citationCount", "authors.hIndex",
                "tldr"
            ]
            
            params = {"fields": ",".join(fields)}
            
            response = requests.get(paper_url, params=params, headers=self.headers, timeout=30)
            
            if response.status_code == 429:
                logger.warning("Rate limit reached, waiting and retrying...")
                time.sleep(2.0)
                response = requests.get(paper_url, params=params, headers=self.headers, timeout=30)
            
            response.raise_for_status()
            return self._parse_paper_data(response.json())
            
        except Exception as e:
            logger.error(f"Error getting paper details: {e}")
            # Try with basic fields as fallback
            try:
                basic_fields = ["paperId", "title", "abstract", "authors", "year", 
                               "citationCount", "url", "openAccessPdf", "externalIds"]
                params = {"fields": ",".join(basic_fields)}
                response = requests.get(f"{self.base_url}/paper/{paper_id}", 
                                      params=params, headers=self.headers, timeout=30)
                response.raise_for_status()
                return self._parse_paper_data(response.json())
            except:
                return None
    
    def search_papers(self, 
                     query: str, 
                     limit: int = 10,
                     year_range: Optional[str] = None,
                     fields: List[str] = None,
                     max_retries: int = 2) -> List[SemanticScholarPaper]:
        """
        Search for papers using Semantic Scholar API with retry logic.
        
        Args:
            query: Search query
            limit: Maximum number of results (up to 100)
            year_range: Year range like "2020-2024"
            fields: Fields to retrieve
            max_retries: Number of retry attempts for failed requests
            
        Returns:
            List of SemanticScholarPaper objects
        """
        if fields is None:
            fields = [
                "paperId", "title", "authors", "year", "abstract", 
                "url", "citationCount", "venue", "externalIds"
            ]
        
        params = {
            "query": query,
            "limit": min(limit, 100),
            "fields": ",".join(fields)
        }
        
        if year_range:
            params["year"] = year_range
        
        for attempt in range(max_retries + 1):
            try:
                response = requests.get(
                    f"{self.base_url}/paper/search",
                    headers=self.headers,
                    params=params,
                    timeout=30
                )
                
                if response.status_code == 429:  # Rate limited
                    time.sleep(2.0)
                    response = requests.get(
                        f"{self.base_url}/paper/search",
                        headers=self.headers,
                        params=params,
                        timeout=30
                    )
                
                response.raise_for_status()
                data = response.json()
                
                papers = []
                for paper_data in data.get("data", []):
                    paper = self._parse_paper_data(paper_data)
                    if paper:
                        papers.append(paper)
                
                return papers
                
            except Exception as e:
                if attempt < max_retries:
                    logger.warning(f"Search attempt {attempt + 1} failed for query '{query}': {e}. Retrying...")
                    time.sleep(2.0)  # Wait before retry
                    continue
                else:
                    logger.error(f"All {max_retries + 1} search attempts failed for query '{query}': {e}")
                    return []
        
        return []
    
    
    def get_citations(self, paper_id: str, limit: int = 10) -> List[SemanticScholarPaper]:
        """Get papers that cite the given paper."""
        fields = [
            "paperId", "title", "authors", "year", "abstract",
            "url", "citationCount", "venue"
        ]
        
        try:
            response = requests.get(
                f"{self.base_url}/paper/{paper_id}/citations",
                headers=self.headers,
                params={
                    "fields": ",".join(fields),
                    "limit": limit
                },
                timeout=30
            )
            
            response.raise_for_status()
            data = response.json()
            
            papers = []
            for citation_data in data.get("data", []):
                paper_data = citation_data.get("citingPaper", {})
                paper = self._parse_paper_data(paper_data)
                if paper:
                    papers.append(paper)
            
            return papers
            
        except Exception as e:
            logger.error(f"Error getting citations: {e}")
            return []
    
    def get_references(self, paper_id: str, limit: int = 10) -> List[SemanticScholarPaper]:
        """Get papers referenced by the given paper."""
        fields = [
            "paperId", "title", "authors", "year", "abstract",
            "url", "citationCount", "venue"
        ]
        
        try:
            response = requests.get(
                f"{self.base_url}/paper/{paper_id}/references",
                headers=self.headers,
                params={
                    "fields": ",".join(fields),
                    "limit": limit
                },
                timeout=30
            )
            
            response.raise_for_status()
            data = response.json()
            
            papers = []
            for reference_data in data.get("data", []):
                paper_data = reference_data.get("citedPaper", {})
                paper = self._parse_paper_data(paper_data)
                if paper:
                    papers.append(paper)
            
            return papers
            
        except Exception as e:
            logger.error(f"Error getting references: {e}")
            return []
    
    def _parse_paper_data(self, paper_data: Dict[str, Any]) -> Optional[SemanticScholarPaper]:
        """Parse paper data from Semantic Scholar API response with enhanced fields."""
        try:
            # Extract authors
            authors = []
            for author_data in paper_data.get("authors", []):
                name = author_data.get("name", "")
                if name:
                    authors.append(name)
            
            # Extract DOI
            doi = None
            external_ids = paper_data.get("externalIds", {})
            if external_ids:
                doi = external_ids.get("DOI")
            
            # Extract venue - prioritize publicationVenue over venue
            venue = ""
            if paper_data.get("publicationVenue"):
                venue = paper_data["publicationVenue"].get("name", "")
            if not venue:
                venue = paper_data.get("venue", "")
            
            # Extract open access PDF URL
            open_access_pdf = None
            if paper_data.get("openAccessPdf"):
                open_access_pdf = paper_data["openAccessPdf"].get("url")
            
            # Extract fields of study
            fields_of_study = []
            if paper_data.get("fieldsOfStudy"):
                fields_of_study = paper_data["fieldsOfStudy"]
            elif paper_data.get("s2FieldsOfStudy"):
                fields_of_study = [field.get("category", "") for field in paper_data["s2FieldsOfStudy"]]
            
            # Extract TLDR
            tldr = None
            if paper_data.get("tldr"):
                tldr = paper_data["tldr"].get("text", "")
            
            return SemanticScholarPaper(
                paper_id=paper_data.get("paperId", ""),
                title=paper_data.get("title", ""),
                authors=authors,
                year=paper_data.get("year", 0),
                abstract=paper_data.get("abstract", ""),
                url=paper_data.get("url", ""),
                citation_count=paper_data.get("citationCount", 0),
                venue=venue,
                doi=doi,
                # Enhanced fields
                influential_citation_count=paper_data.get("influentialCitationCount"),
                is_open_access=paper_data.get("isOpenAccess", False),
                open_access_pdf=open_access_pdf,
                fields_of_study=fields_of_study,
                tldr=tldr
            )
            
        except Exception as e:
            logger.error(f"Error parsing paper data: {e}")
            return None
    
    def test_connection(self) -> bool:
        """Test Semantic Scholar API connection."""
        try:
            response = requests.get(
                f"{self.base_url}/paper/search",
                headers=self.headers,
                params={"query": "machine learning", "limit": 1},
                timeout=10
            )
            return response.status_code == 200
        except Exception:
            return False
    
    # Paper enrichment methods (consolidated from SemanticScholarEnrichmentService)
    
    def enrich_paper(self, paper) -> Dict[str, Any]:
        """
        Enrich a paper with Semantic Scholar metadata.
        
        Args:
            paper: Paper object to enrich
            
        Returns:
            Dict of fields to update in the paper record
        """
        logger.info(f"Enriching paper with Semantic Scholar: {paper.title[:50]}...")
        
        try:
            # Add rate limiting delay
            time.sleep(1.0)
            
            # Search for paper by title
            semantic_paper = self.search_paper_by_title(paper.title)
            
            if not semantic_paper:
                logger.warning(f"No Semantic Scholar data found for: {paper.title[:50]}")
                return self._create_empty_enrichment()
            
            # Convert Semantic Scholar data to database fields
            enrichment_data = self._convert_semantic_scholar_data(semantic_paper)
            
            logger.info(f"Successfully enriched paper: {paper.title[:50]} "
                       f"(Citations: {enrichment_data.get('citation_count', 'N/A')})")
            
            return enrichment_data
            
        except Exception as e:
            logger.error(f"Failed to enrich paper {paper.title[:50]}: {e}")
            return self._create_empty_enrichment()
    
    def _convert_semantic_scholar_data(self, semantic_paper: SemanticScholarPaper) -> Dict[str, Any]:
        """Convert SemanticScholarPaper to database fields."""
        # Convert fields of study list to JSON string
        fields_of_study_json = None
        if semantic_paper.fields_of_study:
            fields_of_study_json = json.dumps(semantic_paper.fields_of_study)
        
        return {
            'citation_count': semantic_paper.citation_count,
            'influential_citation_count': semantic_paper.influential_citation_count,
            'venue': semantic_paper.venue,
            'is_open_access': semantic_paper.is_open_access,
            'open_access_pdf': semantic_paper.open_access_pdf,
            'semantic_scholar_id': semantic_paper.paper_id,
            'semantic_scholar_url': semantic_paper.url,
            'tldr_summary': semantic_paper.tldr,
            'fields_of_study': fields_of_study_json,
            'ss_last_updated': datetime.utcnow()
        }
    
    def _create_empty_enrichment(self) -> Dict[str, Any]:
        """Create enrichment data for when Semantic Scholar lookup fails."""
        return {
            'citation_count': None,
            'influential_citation_count': None,
            'venue': None,
            'is_open_access': None,
            'open_access_pdf': None,
            'semantic_scholar_id': None,
            'semantic_scholar_url': None,
            'tldr_summary': None,
            'fields_of_study': None,
            'ss_last_updated': datetime.utcnow()  # Mark as attempted
        }
    
    def should_update_paper(self, paper, max_age_days: int = 30) -> bool:
        """
        Check if a paper should be updated with fresh Semantic Scholar data.
        
        Args:
            paper: Paper to check
            max_age_days: Maximum age of data before refresh (default 30 days)
            
        Returns:
            True if paper should be updated
        """
        # Always update if never enriched
        if paper.ss_last_updated is None:
            return True
        
        # Update if data is older than max_age_days
        cutoff_date = datetime.utcnow() - timedelta(days=max_age_days)
        if paper.ss_last_updated < cutoff_date:
            return True
        
        # Update if we have an ID but missing core data (indicates previous failure)
        if paper.semantic_scholar_id and not paper.citation_count:
            return True
        
        return False
    
    def get_fields_of_study_list(self, paper) -> list:
        """
        Parse fields of study JSON string into list.
        
        Args:
            paper: Paper with fields_of_study JSON string
            
        Returns:
            List of field names
        """
        if not paper.fields_of_study:
            return []
        
        try:
            return json.loads(paper.fields_of_study)
        except json.JSONDecodeError:
            logger.warning(f"Invalid fields_of_study JSON for paper {paper.id}")
            return []