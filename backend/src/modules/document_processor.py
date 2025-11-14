import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import re

from docx import Document
from docx.shared import Inches

from src.core.database import DatabaseManager, Paper
from src.core.vector_store import VectorStoreManager
from src.core.llm import LLMManager, PromptManager

logger = logging.getLogger(__name__)


@dataclass
class DocumentAnalysis:
    """Container for document analysis results."""
    title: str
    content: str
    word_count: int
    research_questions: List[str]
    key_concepts: List[str]
    literature_gaps: List[str]
    citations_present: List[str]
    citations_needed: List[str]
    document_type: str  # "research_paper", "literature_review", "thesis_chapter", "other"
    sections: Dict[str, str]  # section_name -> content
    confidence_score: float


@dataclass
class RecommendationContext:
    """Context for generating document-based recommendations."""
    document_analysis: DocumentAnalysis
    missing_areas: List[str]
    research_focus: str
    priority_topics: List[str]
    suggested_search_terms: List[str]


class DocumentProcessor:
    """Process .docx documents and analyze them for paper recommendations."""
    
    def __init__(self,
                 db_manager: DatabaseManager,
                 vector_manager: VectorStoreManager,
                 llm_manager: LLMManager,
                 prompt_manager: PromptManager):
        """Initialize document processor."""
        self.db_manager = db_manager
        self.vector_manager = vector_manager
        self.llm_manager = llm_manager
        self.prompt_manager = prompt_manager
        
        logger.info("Initialized Document Processor")
    
    def analyze_document(self, docx_path: Path) -> DocumentAnalysis:
        """
        Analyze a .docx document to understand its content and research needs.
        
        Args:
            docx_path: Path to the .docx file
            
        Returns:
            DocumentAnalysis object with comprehensive analysis
        """
        logger.info(f"Analyzing document: {docx_path.name}")
        
        try:
            # Extract text from .docx
            document_text, sections = self._extract_docx_content(docx_path)
            
            if not document_text.strip():
                raise ValueError("Document appears to be empty")
            
            # Basic document info
            word_count = len(document_text.split())
            
            # Use LLM to analyze the document
            analysis_prompt = self._create_document_analysis_prompt(document_text)
            llm_response = self.llm_manager.generate_response(
                messages=[{"role": "user", "content": analysis_prompt}],
                max_tokens=2000
            )
            
            # Parse LLM response into structured analysis
            analysis = self._parse_analysis_response(llm_response.content, document_text, sections, word_count)
            
            logger.info(f"Document analysis completed: {analysis.title}")
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing document {docx_path}: {e}")
            # Return minimal analysis with error info
            return DocumentAnalysis(
                title=f"Error analyzing {docx_path.name}",
                content="",
                word_count=0,
                research_questions=[],
                key_concepts=[],
                literature_gaps=[],
                citations_present=[],
                citations_needed=[],
                document_type="other",
                sections={},
                confidence_score=0.0
            )
    
    def create_recommendation_context(self, analysis: DocumentAnalysis) -> RecommendationContext:
        """
        Create recommendation context from document analysis.
        
        Args:
            analysis: Document analysis results
            
        Returns:
            RecommendationContext for targeted paper suggestions
        """
        # Check current paper database to identify gaps
        existing_papers = self.db_manager.get_all_papers()
        existing_topics = set()
        
        for paper in existing_papers:
            if paper.summary:
                # Extract key topics from existing papers (simple keyword extraction)
                words = re.findall(r'\w+', paper.summary.lower())
                existing_topics.update([w for w in words if len(w) > 4])
        
        # Identify missing areas
        missing_areas = []
        for concept in analysis.key_concepts:
            concept_words = concept.lower().split()
            if not any(word in existing_topics for word in concept_words):
                missing_areas.append(concept)
        
        # Add literature gaps from analysis
        missing_areas.extend(analysis.literature_gaps)
        
        # Create research focus
        research_focus = self._determine_research_focus(analysis)
        
        # Generate priority topics
        priority_topics = analysis.key_concepts[:5]  # Top 5 concepts
        
        # Create search terms
        search_terms = self._generate_search_terms(analysis)
        
        return RecommendationContext(
            document_analysis=analysis,
            missing_areas=missing_areas,
            research_focus=research_focus,
            priority_topics=priority_topics,
            suggested_search_terms=search_terms
        )
    
    def _extract_docx_content(self, docx_path: Path) -> Tuple[str, Dict[str, str]]:
        """Extract text content from .docx file."""
        try:
            doc = Document(docx_path)
            
            full_text = []
            sections = {}
            current_section = "Introduction"
            current_content = []
            
            for paragraph in doc.paragraphs:
                text = paragraph.text.strip()
                if not text:
                    continue
                
                # Check if this might be a section header
                if self._is_section_header(text):
                    # Save previous section
                    if current_content:
                        sections[current_section] = " ".join(current_content)
                    
                    # Start new section
                    current_section = text
                    current_content = []
                else:
                    current_content.append(text)
                    full_text.append(text)
            
            # Save last section
            if current_content:
                sections[current_section] = " ".join(current_content)
            
            return " ".join(full_text), sections
            
        except Exception as e:
            logger.error(f"Error extracting content from {docx_path}: {e}")
            raise
    
    def _is_section_header(self, text: str) -> bool:
        """Determine if text is likely a section header."""
        # Simple heuristics for section headers
        if len(text) > 100:  # Too long to be a header
            return False
        
        # Common section patterns
        section_patterns = [
            r'^(abstract|introduction|background|literature review|methodology|methods|results|discussion|conclusion|references)$',
            r'^\d+\.?\s+[A-Z][a-z]',  # "1. Introduction", "2 Methods"
            r'^[A-Z][A-Z\s]+$',  # ALL CAPS
        ]
        
        text_lower = text.lower().strip()
        for pattern in section_patterns:
            if re.match(pattern, text_lower):
                return True
        
        return False
    
    def _create_document_analysis_prompt(self, document_text: str) -> str:
        """Create prompt for LLM document analysis."""
        prompt = f"""
Analyze the following academic document and provide a structured analysis in JSON format:

DOCUMENT TEXT:
{document_text[:3000]}{'...' if len(document_text) > 3000 else ''}

Please analyze and return a JSON response with these fields:
1. "title": Inferred title of the document
2. "research_questions": List of research questions or objectives (max 3)
3. "key_concepts": List of main concepts/topics discussed (max 5)
4. "literature_gaps": Areas where more literature review is needed (max 3)
5. "citations_present": Citations or references mentioned in the text (max 5)
6. "citations_needed": Areas that would benefit from citations (max 3)
7. "document_type": One of ["research_paper", "literature_review", "thesis_chapter", "proposal", "other"]
8. "confidence_score": Your confidence in this analysis (0.0 to 1.0)

Focus on identifying what additional papers would be most helpful for this document.

Response format:
```json
{{
    "title": "Document Title",
    "research_questions": ["Question 1", "Question 2"],
    "key_concepts": ["concept1", "concept2"],
    "literature_gaps": ["gap1", "gap2"],
    "citations_present": ["citation1", "citation2"],
    "citations_needed": ["area1", "area2"],
    "document_type": "research_paper",
    "confidence_score": 0.85
}}
```
"""
        return prompt
    
    def _parse_analysis_response(self, llm_response: str, full_text: str, sections: Dict[str, str], word_count: int) -> DocumentAnalysis:
        """Parse LLM analysis response into DocumentAnalysis object."""
        try:
            # Extract JSON from response
            import json
            
            # Find JSON block
            json_start = llm_response.find('{')
            json_end = llm_response.rfind('}') + 1
            
            if json_start == -1 or json_end == 0:
                # Fallback parsing if JSON not found
                return self._create_fallback_analysis(full_text, sections, word_count)
            
            json_text = llm_response[json_start:json_end]
            parsed = json.loads(json_text)
            
            return DocumentAnalysis(
                title=parsed.get('title', 'Untitled Document'),
                content=full_text,
                word_count=word_count,
                research_questions=parsed.get('research_questions', []),
                key_concepts=parsed.get('key_concepts', []),
                literature_gaps=parsed.get('literature_gaps', []),
                citations_present=parsed.get('citations_present', []),
                citations_needed=parsed.get('citations_needed', []),
                document_type=parsed.get('document_type', 'other'),
                sections=sections,
                confidence_score=parsed.get('confidence_score', 0.5)
            )
            
        except Exception as e:
            logger.error(f"Error parsing analysis response: {e}")
            return self._create_fallback_analysis(full_text, sections, word_count)
    
    def _create_fallback_analysis(self, full_text: str, sections: Dict[str, str], word_count: int) -> DocumentAnalysis:
        """Create basic analysis when LLM parsing fails."""
        # Simple keyword extraction for concepts
        words = re.findall(r'\b[a-zA-Z]{4,}\b', full_text.lower())
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # Top concepts by frequency
        key_concepts = [word for word, _ in sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5]]
        
        return DocumentAnalysis(
            title=f"Document ({word_count} words)",
            content=full_text,
            word_count=word_count,
            research_questions=[],
            key_concepts=key_concepts,
            literature_gaps=[],
            citations_present=[],
            citations_needed=[],
            document_type="other",
            sections=sections,
            confidence_score=0.3
        )
    
    def _determine_research_focus(self, analysis: DocumentAnalysis) -> str:
        """Determine the main research focus of the document."""
        if analysis.research_questions:
            return analysis.research_questions[0]
        elif analysis.key_concepts:
            return f"Research on {', '.join(analysis.key_concepts[:3])}"
        else:
            return "General research document"
    
    def _generate_search_terms(self, analysis: DocumentAnalysis) -> List[str]:
        """Generate effective search terms for paper recommendations."""
        search_terms = []
        
        # Add key concepts
        search_terms.extend(analysis.key_concepts)
        
        # Add literature gaps
        search_terms.extend(analysis.literature_gaps)
        
        # Add terms from research questions
        for question in analysis.research_questions:
            # Extract key terms from questions
            words = re.findall(r'\b[a-zA-Z]{4,}\b', question.lower())
            search_terms.extend(words[:2])  # Top 2 terms per question
        
        # Remove duplicates and return top terms
        unique_terms = list(dict.fromkeys(search_terms))
        return unique_terms[:10]  # Top 10 search terms