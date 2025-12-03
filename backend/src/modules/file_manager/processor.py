import logging
import json
import asyncio
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

from src.core.database import DatabaseManager, Paper
from src.core.vector_store import VectorStoreManager
from src.core.llm import LLMManager, PromptManager, METADATA_SCHEMA
from src.core.utils import PDFProcessor, DirectoryManager
from src.core.external_apis.semantic_scholar_client import SemanticScholarClient

# PDF generation imports
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable
    from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
    from reportlab.lib.colors import HexColor
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    logger.warning("ReportLab not available - PDF generation will be disabled")

logger = logging.getLogger(__name__)


class IntegratedPDFGenerator:
    """Integrated PDF summary generator for the paper processing pipeline."""
    
    def __init__(self, project_root: Path):
        """Initialize the PDF generator.

        Args:
            project_root: Path to the data directory (passed from DirectoryManager.base_path)
        """
        self.project_root = project_root
        # Store summaries inside PAPERS/summaries (project_root is already data/)
        self.summaries_dir = project_root / "PAPERS" / "summaries"
        self.summaries_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize PDF styles if ReportLab is available
        if REPORTLAB_AVAILABLE:
            self.styles = self._create_styles()
        else:
            self.styles = None
    
    def _create_styles(self) -> Dict:
        """Create custom styles for PDF formatting."""
        if not REPORTLAB_AVAILABLE:
            return {}
            
        styles = getSampleStyleSheet()
        
        return {
            'title': ParagraphStyle(
                'CustomTitle',
                parent=styles['Title'],
                fontSize=18,
                spaceAfter=8,
                alignment=TA_CENTER,
                textColor=HexColor('#1B365D'),
                fontName='Helvetica-Bold'
            ),
            'authors': ParagraphStyle(
                'CustomAuthors',
                parent=styles['Normal'],
                fontSize=11,
                spaceAfter=20,
                alignment=TA_CENTER,
                textColor=HexColor('#6B7280')
            ),
            'section_header': ParagraphStyle(
                'SectionHeader',
                parent=styles['Heading2'],
                fontSize=11,
                spaceBefore=18,
                spaceAfter=8,
                textColor=HexColor('#1B365D'),
                fontName='Helvetica-Bold'
            ),
            'body': ParagraphStyle(
                'CustomBody',
                parent=styles['Normal'],
                fontSize=11,
                spaceAfter=8,
                alignment=TA_JUSTIFY,
                leading=13,
                textColor=HexColor('#1F2937')
            ),
            'bullet_item': ParagraphStyle(
                'BulletItem',
                parent=styles['Normal'],
                fontSize=11,
                spaceAfter=8,
                spaceBefore=3,
                alignment=TA_JUSTIFY,
                leading=13,
                leftIndent=0.25*inch,
                bulletIndent=0.15*inch,
                textColor=HexColor('#1F2937')
            ),
            'executive_summary': ParagraphStyle(
                'ExecutiveSummary',
                parent=styles['Normal'],
                fontSize=11,
                spaceAfter=12,
                alignment=TA_JUSTIFY,
                leading=13,
                textColor=HexColor('#1F2937'),
                borderWidth=1,
                borderColor=HexColor('#E5E7EB'),
                borderPadding=8,
                backColor=HexColor('#F9FAFB')
            )
        }
    
    def generate_pdf_summary(self, paper: Paper) -> Optional[str]:
        """Generate PDF summary for a paper.
        
        Args:
            paper: Paper object with AI summary data
            
        Returns:
            Path to generated PDF file, or None if generation failed
        """
        if not REPORTLAB_AVAILABLE:
            logger.warning("ReportLab not available - skipping PDF generation")
            return None
            
        if not self._has_ai_summary(paper):
            logger.warning(f"Paper {paper.filename} missing AI summary - skipping PDF generation")
            return None
        
        try:
            # Create output filename
            pdf_filename = f"{paper.filename.replace('.pdf', '')}_summary.pdf"
            pdf_path = self.summaries_dir / pdf_filename
            
            logger.info(f"📄 Generating PDF summary: {pdf_filename}")
            
            # Create PDF document
            doc = SimpleDocTemplate(
                str(pdf_path),
                pagesize=letter,
                rightMargin=0.75*inch,
                leftMargin=0.75*inch,
                topMargin=0.75*inch,
                bottomMargin=0.75*inch
            )
            
            # Build content
            content = self._build_pdf_content(paper)
            
            # Generate PDF
            doc.build(content)
            
            # Get file size
            file_size = pdf_path.stat().st_size / 1024  # KB
            
            logger.info(f"✅ PDF summary generated: {pdf_filename} ({file_size:.1f} KB)")
            return str(pdf_path)
            
        except Exception as e:
            logger.error(f"❌ Failed to generate PDF summary for {paper.filename}: {e}")
            return None
    
    def _has_ai_summary(self, paper: Paper) -> bool:
        """Check if paper has sufficient AI summary data for PDF generation."""
        return (
            paper.ai_executive_summary and 
            (paper.ai_methodology or paper.ai_major_findings_contributions)
        )
    
    def _build_pdf_content(self, paper: Paper) -> List:
        """Build PDF content from paper data."""
        content = []
        
        # Header with date
        header_text = f"AI-Generated Summary • {datetime.now().strftime('%B %Y')}"
        content.append(Paragraph(header_text, self.styles['authors']))
        content.append(Spacer(1, 0.1*inch))
        
        # Paper metadata box
        metadata_text = self._format_metadata(paper)
        content.append(Paragraph(metadata_text, self.styles['authors']))
        content.append(Spacer(1, 0.1*inch))
        
        # Paper title and authors
        content.append(Paragraph(paper.title, self.styles['title']))
        
        # Format authors
        try:
            if isinstance(paper.authors, str):
                authors_list = json.loads(paper.authors)
            else:
                authors_list = paper.authors or []
        except:
            authors_list = []
        
        authors_text = ", ".join(authors_list) if authors_list else "Unknown"
        year_text = f" | ({paper.year})" if paper.year else ""
        content.append(Paragraph(f"{authors_text}{year_text}", self.styles['authors']))
        
        # Executive Summary
        if paper.ai_executive_summary:
            content.append(Paragraph("1. Executive Summary", self.styles['section_header']))
            content.append(Paragraph(paper.ai_executive_summary, self.styles['executive_summary']))
        
        # Purpose & Research Question
        if paper.ai_purpose_rationale_research_question:
            content.append(Paragraph("2. Purpose & Research Question", self.styles['section_header']))
            content.extend(self._format_json_content(paper.ai_purpose_rationale_research_question))
        
        # Theoretical Framework
        if paper.ai_theory_framework:
            content.append(Paragraph("3. Theoretical Framework", self.styles['section_header']))
            content.extend(self._format_json_content(paper.ai_theory_framework))
        
        # Methodology
        if paper.ai_methodology:
            content.append(Paragraph("4. Methodology", self.styles['section_header']))
            content.extend(self._format_json_content(paper.ai_methodology))
        
        # Major Findings & Contributions
        if paper.ai_major_findings_contributions:
            content.append(Paragraph("5. Major Findings & Contributions", self.styles['section_header']))
            content.extend(self._format_json_content(paper.ai_major_findings_contributions))
        
        # Study Limitations & Gaps
        if paper.ai_study_limitations_gaps:
            content.append(Paragraph("6. Study Limitations & Gaps", self.styles['section_header']))
            content.extend(self._format_json_content(paper.ai_study_limitations_gaps))
        
        # Study Implications
        if paper.ai_study_implications:
            content.append(Paragraph("7. Study Implications", self.styles['section_header']))
            content.extend(self._format_json_content(paper.ai_study_implications))
        
        return content
    
    def _format_metadata(self, paper: Paper) -> str:
        """Format paper metadata for display."""
        pages = f"Pages: {paper.page_count}" if paper.page_count else "Pages: null"
        venue = f"Venue: {paper.venue}" if hasattr(paper, 'venue') and paper.venue else "Venue: null"
        citations = f"Citations: {paper.citation_count}" if hasattr(paper, 'citation_count') and paper.citation_count else "Citations: null"
        access = f"Access: {'Open' if getattr(paper, 'is_open_access', None) else 'null'}"
        fields = f"Fields: {getattr(paper, 'fields_of_study', 'null')}"

        
        return f"<b>Paper Metadata</b><br/>{pages} • {venue} • {citations} • {access} • {fields}"
    
    def _format_json_content(self, json_data) -> List:
        """Format JSON array content for PDF display."""
        if not json_data:
            return [Paragraph("No data available.", self.styles['body'])]
        
        # Parse JSON if it's a string
        if isinstance(json_data, str):
            try:
                json_data = json.loads(json_data)
            except json.JSONDecodeError:
                return [Paragraph(json_data, self.styles['body'])]
        
        paragraphs = []
        
        if isinstance(json_data, list):
            for item in json_data:
                if isinstance(item, dict) and 'content' in item:
                    content = item['content']
                    tag = item.get('tag', 'Unknown')
                    
                    # Format as: • Content (Tag)
                    formatted_text = f"• {content} <i><font color='#666666'>({tag})</font></i>"
                    paragraphs.append(Paragraph(formatted_text, self.styles['bullet_item']))
                    
                elif isinstance(item, str):
                    formatted_text = f"• {item}"
                    paragraphs.append(Paragraph(formatted_text, self.styles['bullet_item']))
        
        elif isinstance(json_data, str):
            paragraphs.append(Paragraph(json_data, self.styles['body']))
        
        else:
            paragraphs.append(Paragraph(str(json_data), self.styles['body']))
        
        return paragraphs


class JSONRepairService:
    """Service for repairing malformed JSON using LLM calls."""
    
    def __init__(self, llm_manager):
        """
        Initialize the JSON repair service.
        
        Args:
            llm_manager: LLM manager instance for making repair calls
        """
        self.llm_manager = llm_manager
        self.repair_stats = {
            "attempts": 0,
            "successes": 0,
            "failures": 0
        }
    
    def repair_malformed_json(self, 
                             malformed_json: str, 
                             expected_schema: Dict[str, Any],
                             max_attempts: int = 2) -> Optional[Dict[str, Any]]:
        """
        Attempt to repair malformed JSON using LLM.
        
        Args:
            malformed_json: The malformed JSON string that failed to parse
            expected_schema: Dictionary describing the expected JSON structure
            max_attempts: Maximum number of repair attempts
            
        Returns:
            Parsed JSON dictionary if repair successful, None if failed
        """
        self.repair_stats["attempts"] += 1
        
        logger.info(f"🔧 Attempting JSON repair for {len(malformed_json)} character response")
        
        # First, try to identify the specific parsing error
        parsing_error = self._identify_parsing_error(malformed_json)
        logger.info(f"🔍 Parsing error identified: {parsing_error}")
        
        for attempt in range(max_attempts):
            try:
                logger.info(f"📞 JSON repair attempt {attempt + 1}/{max_attempts}")
                
                # Create repair prompt
                repair_prompt = self._create_repair_prompt(
                    malformed_json, 
                    expected_schema, 
                    parsing_error
                )
                
                # Call LLM for repair
                response = self.llm_manager.generate_response(
                    messages=repair_prompt,
                    temperature=0,  # Use zero temperature for consistent repairs
                    max_tokens=len(malformed_json) + 500  # Allow some extra space
                )
                
                # Extract repaired JSON from response
                repaired_json_str = self._extract_json_from_response(response.content)
                
                # Try to parse the repaired JSON
                repaired_json = json.loads(repaired_json_str)
                
                # Validate the repaired JSON structure
                if self._validate_repaired_json(repaired_json, expected_schema):
                    logger.info(f"✅ JSON repair successful on attempt {attempt + 1}")
                    self.repair_stats["successes"] += 1
                    return repaired_json
                else:
                    logger.warning(f"⚠️ Repaired JSON failed structure validation on attempt {attempt + 1}")
                    
            except json.JSONDecodeError as e:
                logger.warning(f"❌ Repair attempt {attempt + 1} still produced invalid JSON: {e}")
            except Exception as e:
                logger.error(f"❌ Error during repair attempt {attempt + 1}: {e}")
        
        logger.error(f"💥 JSON repair failed after {max_attempts} attempts")
        self.repair_stats["failures"] += 1
        return None
    
    def _identify_parsing_error(self, malformed_json: str) -> str:
        """Try to parse the JSON and identify the specific error."""
        try:
            json.loads(malformed_json)
            return "No error - JSON appears valid"
        except json.JSONDecodeError as e:
            error_context = self._get_error_context(malformed_json, e.pos if hasattr(e, 'pos') else 0)
            return f"{e.msg} at position {e.pos if hasattr(e, 'pos') else 'unknown'}: {error_context}"
        except Exception as e:
            return f"Unexpected error: {str(e)}"
    
    def _get_error_context(self, text: str, error_pos: int, context_length: int = 50) -> str:
        """Get context around the error position."""
        start = max(0, error_pos - context_length)
        end = min(len(text), error_pos + context_length)
        
        context = text[start:end]
        # Mark the error position within the context
        relative_pos = error_pos - start
        if 0 <= relative_pos < len(context):
            context = context[:relative_pos] + "<<<ERROR>>>" + context[relative_pos:]
        
        return context.strip()
    
    def _create_repair_prompt(self, 
                             malformed_json: str, 
                             expected_schema: Dict[str, Any], 
                             parsing_error: str) -> list:
        """Create the prompt for LLM JSON repair."""
        schema_description = self._describe_schema(expected_schema)
        
        system_prompt = """You are a JSON repair expert. Your job is to fix malformed JSON while preserving ALL original content exactly as written.

RULES:
1. Preserve ALL original text content exactly - do not change any words, phrases, or meanings
2. Only fix JSON formatting issues (missing quotes, brackets, commas, etc.)
3. Ensure the repaired JSON matches the expected structure
4. Return ONLY the repaired JSON, no explanations or markdown
5. If content is truncated, keep it truncated - do not add missing content"""
        
        user_prompt = f"""Fix this malformed JSON response. The parsing error was: {parsing_error}

EXPECTED JSON STRUCTURE:
{schema_description}

MALFORMED JSON TO FIX:
{malformed_json}

Return the repaired JSON (no markdown, no explanations):"""
        
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    
    def _describe_schema(self, schema: Dict[str, Any]) -> str:
        """Create a human-readable description of the expected JSON schema."""
        return """Expected structure for AI paper summary:
{
    "title": "string",
    "executive_summary": "string", 
    "purpose_rationale_research_question": [{"tag": "string", "content": "string"}],
    "theory_framework": [{"tag": "string", "content": "string"}],
    "methodology": [{"tag": "string", "content": "string"}],
    "major_findings_contributions": [{"tag": "string", "content": "string"}],
    "study_limitations_gaps_that_remain": [{"tag": "string", "content": "string"}],
    "study_implications_for_research_practice_policy": [{"tag": "string", "content": "string"}]
}"""
    
    def _extract_json_from_response(self, response_content: str) -> str:
        """Extract JSON from LLM response, removing any markdown or explanations."""
        content = response_content.strip()
        
        # Remove markdown code blocks if present
        if content.startswith('```json'):
            content = content[7:]
        elif content.startswith('```'):
            content = content[3:]
        
        if content.endswith('```'):
            content = content[:-3]
        
        content = content.strip()
        
        # If there's explanatory text, try to extract just the JSON part
        if not content.startswith('{'):
            # Look for the start of JSON
            json_start = content.find('{')
            if json_start != -1:
                content = content[json_start:]
        
        return content
    
    def _validate_repaired_json(self, 
                               repaired_json: Dict[str, Any], 
                               expected_schema: Dict[str, Any]) -> bool:
        """Validate that the repaired JSON matches the expected structure."""
        try:
            # Check for required top-level keys
            required_keys = ["title", "executive_summary"]
            for key in required_keys:
                if key not in repaired_json:
                    logger.warning(f"Missing required key: {key}")
                    return False
                if not repaired_json[key]:
                    logger.warning(f"Empty required key: {key}")
                    return False
            
            # Check that array fields are actually arrays
            array_fields = [
                "purpose_rationale_research_question",
                "theory_framework", 
                "methodology",
                "major_findings_contributions",
                "study_limitations_gaps_that_remain",
                "study_implications_for_research_practice_policy"
            ]
            
            for field in array_fields:
                if field in repaired_json:
                    if not isinstance(repaired_json[field], list):
                        logger.warning(f"Field {field} should be an array but is {type(repaired_json[field])}")
                        return False
            
            logger.info("✅ Repaired JSON structure validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Error validating repaired JSON: {e}")
            return False
    
    def get_repair_stats(self) -> Dict[str, Any]:
        """Get statistics about JSON repair operations."""
        total_attempts = self.repair_stats["attempts"]
        if total_attempts > 0:
            success_rate = (self.repair_stats["successes"] / total_attempts) * 100
        else:
            success_rate = 0
        
        return {
            **self.repair_stats,
            "success_rate": success_rate
        }


class PaperProcessor:
    """Processes new PDF papers: extracts metadata, creates embeddings, organizes files."""
    
    def __init__(self,
                 db_manager: DatabaseManager,
                 vector_manager: VectorStoreManager,
                 llm_manager: LLMManager,
                 prompt_manager: PromptManager,
                 directory_manager: DirectoryManager,
                 pdf_processor: PDFProcessor):
        """Initialize paper processor with all required managers."""
        self.db_manager = db_manager
        self.vector_manager = vector_manager
        self.llm_manager = llm_manager
        self.prompt_manager = prompt_manager
        self.directory_manager = directory_manager
        self.pdf_processor = pdf_processor
        
        # Initialize Semantic Scholar client
        self.ss_client = SemanticScholarClient()
        
        # Initialize JSON repair service
        self.json_repair_service = JSONRepairService(llm_manager)
        logger.info("✅ JSON repair service initialized for AI summary generation")
        
        # Initialize integrated PDF generator
        project_root = self.directory_manager.base_path
        self.pdf_generator = IntegratedPDFGenerator(project_root)
        logger.info("✅ Integrated PDF generator initialized for summary generation")
    
    async def process_new_paper(self, pdf_path: Path, force_clean: bool = False, progress_callback=None) -> Optional[Paper]:
        """
        Process a new PDF paper: extract metadata, create summary, add to database and vector store.

        Args:
            pdf_path: Path to the PDF file
            force_clean: Whether to force LLM cleaning even if over threshold (default: False)
            progress_callback: Optional async callback function(progress: int, step: str) for progress updates

        Returns:
            Paper object if successful, None if failed
        """
        logger.info(f"Processing new paper: {pdf_path.name}")

        try:
            # Step 1: Move file to processing directory
            processing_path = self.directory_manager.move_file(
                pdf_path, "processing", pdf_path.name
            )

            # Step 1.5: Check for duplicate paper (before expensive processing)
            is_duplicate, existing_filename = self._is_duplicate_paper(pdf_path.name)
            if is_duplicate:
                reason = f"Duplicate paper: already exists as {existing_filename}"
                logger.info(f"Skipping duplicate paper: {pdf_path.name} -> {reason}")
                self._move_to_failed(processing_path, reason)
                return None

            # Step 2: Extract text from PDF
            if progress_callback:
                await progress_callback(25, "Extracting text from PDF...")
            extracted_text = await self.pdf_processor.extract_text_from_pdf(processing_path, force_clean=force_clean)

            if not extracted_text.first_pages_text.strip():
                logger.error(f"No text extracted from {pdf_path.name}")
                self._move_to_failed(processing_path, "No text extracted")
                return None

            # Step 3: Extract metadata using LLM
            if progress_callback:
                await progress_callback(40, "Extracting metadata...")
            metadata = self._extract_metadata_with_llm(extracted_text.first_pages_text)
            
            if not metadata or not metadata.get("title"):
                logger.error(f"Failed to extract metadata from {pdf_path.name}")
                self._move_to_failed(processing_path, "Metadata extraction failed")
                return None
            
            # Step 3.5: Enrich with Semantic Scholar metadata BEFORE AI summary
            enrichment_data = self._enrich_with_semantic_scholar_early(metadata)

            # Step 3.75: Use cleaned text for AI processing if available
            # Note: PDF cleaning happens in PDFProcessor.extract_text_from_pdf()
            text_for_ai = extracted_text.cleaned_full_text if extracted_text.cleaned_full_text else extracted_text.full_text

            if extracted_text.cleaned_full_text:
                reduction = len(extracted_text.full_text) - len(extracted_text.cleaned_full_text)
                percent = (reduction / len(extracted_text.full_text)) * 100
                logger.info(f"🧹 Using cleaned text: {reduction} chars removed ({percent:.1f}% reduction)")
            else:
                logger.info("📄 Using uncleaned text (cleaning disabled or failed)")

            # Step 4: Generate paper summary
            summary = self._generate_summary(metadata, text_for_ai)

            # Step 4.5: Generate structured AI summary
            if progress_callback:
                await progress_callback(60, "Generating AI summary...")
            logger.info(f"Starting AI structured summary generation for: {pdf_path.name}")
            structured_summary = self._generate_structured_summary(text_for_ai)
            
            if structured_summary:
                logger.info(f"✅ AI structured summary generated successfully for: {pdf_path.name}")
                logger.info(f"   - Executive summary length: {len(structured_summary.get('executive_summary', ''))}")
                logger.info(f"   - Context/problem sections: {len(structured_summary.get('context_and_problem_statement', []))}")
                logger.info(f"   - Research questions: {len(structured_summary.get('research_questions_or_hypotheses', []))}")
                logger.info(f"   - Methodology sections: {len(structured_summary.get('methodology', []))}")
                logger.info(f"   - Key findings: {len(structured_summary.get('key_findings_and_results', []))}")
                logger.info(f"   - Primary contributions: {len(structured_summary.get('primary_contributions', []))}")
            else:
                logger.warning(f"❌ AI structured summary generation failed for: {pdf_path.name}")
            
            # Step 5: Create standardized filename
            new_filename = self._create_standardized_filename(metadata) + ".pdf"
            
            # Step 6: Move to final papers directory
            final_path = self.directory_manager.move_file(
                processing_path, "papers", new_filename
            )
            
            # Step 7: Add to database
            paper_data = {
                "filename": new_filename,
                "original_filename": pdf_path.name,
                "authors": metadata.get("authors", []),
                "year": metadata.get("year"),
                "title": metadata["title"],
                "summary": summary,
                "status": "unread",
                "file_path": str(final_path),
                "doi": metadata.get("doi"),
                "journal": metadata.get("journal"),
                "page_count": extracted_text.page_count,
                "abstract": metadata.get("abstract"),
                "cleaned_full_text": extracted_text.cleaned_full_text  # LLM-cleaned text
            }
            
            # Add Semantic Scholar enrichment data if available
            logger.info(f"🔍 DEBUG: enrichment_data = {enrichment_data}")
            if enrichment_data:
                logger.info(f"🔍 DEBUG: About to update paper_data with enrichment_data")
                logger.info(f"🔍 DEBUG: paper_data keys before update: {list(paper_data.keys())}")
                paper_data.update(enrichment_data)
                logger.info(f"💾 Applied Semantic Scholar enrichment to paper_data")
                logger.info(f"🔍 DEBUG: paper_data keys after update: {list(paper_data.keys())}")
                logger.info(f"🔍 DEBUG: SS fields in paper_data: venue={paper_data.get('venue')}, citations={paper_data.get('citation_count')}")
            else:
                logger.info(f"⚠️  No enrichment_data to apply to paper_data")
            
            # Add structured AI summary fields if available
            if structured_summary:
                logger.info(f"💾 Storing AI structured summary data for: {pdf_path.name}")
                paper_data.update({
                    "ai_executive_summary": json.dumps(structured_summary.get("executive_summary")) if isinstance(structured_summary.get("executive_summary"), (dict, list)) else structured_summary.get("executive_summary"),
                    "ai_purpose_rationale_research_question": json.dumps(structured_summary.get("purpose_rationale_research_question", [])),
                    "ai_theory_framework": json.dumps(structured_summary.get("theory_framework", [])),
                    "ai_methodology": json.dumps(structured_summary.get("methodology", [])),
                    "ai_major_findings_contributions": json.dumps(structured_summary.get("major_findings_contributions", [])),
                    "ai_study_limitations_gaps": json.dumps(structured_summary.get("study_limitations_gaps_that_remain", [])),
                    "ai_study_implications": json.dumps(structured_summary.get("study_implications_for_research_practice_policy", [])),
                    "ai_summary_generated_at": datetime.utcnow()
                })
                logger.info(f"✅ AI summary data added to paper_data for database storage")
            else:
                logger.info(f"⚠️  No AI structured summary to store for: {pdf_path.name}")
            
            # Convert authors list to JSON string
            if isinstance(paper_data["authors"], list):
                paper_data["authors"] = json.dumps(paper_data["authors"])
            
            paper = self.db_manager.add_paper(paper_data)
            
            # Log AI summary storage confirmation
            if structured_summary:
                logger.info(f"🎉 Paper saved with AI structured summary! Paper ID: {paper.id}")
                logger.info(f"   Database verification: ai_summary_generated_at = {paper.ai_summary_generated_at}")
                
                # Step 7.5: Generate PDF summary (only if AI summary is available)
                logger.info(f"📄 Starting PDF summary generation for: {paper.filename}")
                pdf_path = self.pdf_generator.generate_pdf_summary(paper)
                if pdf_path:
                    logger.info(f"✅ PDF summary generated: {pdf_path}")
                    # Update paper record with PDF path if desired
                    try:
                        self.db_manager.update_paper(paper.id, {"pdf_summary_path": pdf_path})
                        logger.info(f"💾 PDF path stored in database for paper ID: {paper.id}")
                    except Exception as e:
                        logger.warning(f"⚠️  Could not store PDF path in database: {e}")
                else:
                    logger.warning(f"❌ PDF summary generation failed for: {paper.filename}")
            else:
                logger.info(f"📝 Paper saved without AI structured summary. Paper ID: {paper.id}")
                logger.info(f"⏭️  Skipping PDF generation (no AI summary available)")
            
            # Step 8: Add to vector store (SS enrichment now done early in Step 3.5)
            if progress_callback:
                await progress_callback(80, "Creating embeddings...")
            self._add_to_vector_store(paper, extracted_text, metadata, final_path)
            
            logger.info(f"Successfully processed paper: {paper.title} (ID: {paper.id})")
            return paper
            
        except Exception as e:
            logger.error(f"Error processing paper {pdf_path.name}: {e}")
            try:
                # Try to move to failed directory
                if pdf_path.exists():
                    self._move_to_failed(pdf_path, str(e))
            except Exception:
                pass
            return None
    
    def _extract_metadata_with_llm(self, paper_text: str) -> Optional[Dict[str, Any]]:
        """Extract metadata from paper text using LLM with Semantic Scholar verification."""
        try:
            # Step 1: Extract metadata first (without SS context) to get proper title
            logger.info("📄 Step 1: Extracting metadata to get proper title...")
            metadata = self._extract_basic_metadata(paper_text)
            
            if not metadata or not metadata.get("title"):
                logger.error("Failed to extract basic metadata from paper")
                self._reset_verification_result()
                return None
            
            # Step 2: Search Semantic Scholar with the properly extracted title
            logger.info("🔍 Step 2: Searching Semantic Scholar with extracted title...")
            semantic_paper = self._search_semantic_scholar(metadata["title"])
            
            # Step 3: If SS match found, do LLM verification
            if semantic_paper:
                logger.info("✅ Step 3: SS match found - performing LLM verification...")
                verification_result = self._verify_semantic_scholar_match(paper_text, semantic_paper)
                self._verification_result = verification_result
            else:
                logger.info("⚠️ No Semantic Scholar match found - skipping verification")
                self._reset_verification_result()
            
            # Return clean metadata (no verification fields)
            metadata_for_db = {k: v for k, v in metadata.items() 
                             if k not in ["verified_semantic_scholar_match", "use_semantic_scholar_data"]}
            
            return metadata_for_db
            
        except Exception as e:
            logger.error(f"Error extracting metadata with LLM: {e}")
            self._reset_verification_result()
            return None
    
    def _extract_basic_metadata(self, paper_text: str) -> Optional[Dict[str, Any]]:
        """Extract basic metadata without Semantic Scholar context."""
        try:
            messages = self.prompt_manager.format_prompt(
                "extract_metadata",
                paper_text=paper_text[:8000],  # Limit text length
                semantic_scholar_context=""  # No SS context for initial extraction
            )
            
            metadata = self.llm_manager.generate_structured_response(
                messages=messages,
                schema=METADATA_SCHEMA,
                temperature=0.0
            )
            
            # Validate and clean metadata
            metadata = self._validate_and_clean_metadata(metadata)
            
            logger.info(f"✅ Basic metadata extracted: title='{metadata.get('title', 'N/A')[:50]}...'")
            return metadata
            
        except Exception as e:
            logger.error(f"Error extracting basic metadata: {e}")
            return None
    
    def _search_semantic_scholar(self, title: str) -> Optional[object]:
        """Search Semantic Scholar using the properly extracted title."""
        try:
            logger.info(f"🔍 Searching Semantic Scholar for: '{title[:50]}...'")
            semantic_paper = self.ss_client.search_paper_by_title(title)
            
            if semantic_paper:
                logger.info(f"✅ Found SS match: '{semantic_paper.title[:50]}...'")
                logger.info(f"   Authors: {semantic_paper.authors[:2]}{'...' if len(semantic_paper.authors) > 2 else ''}")
                logger.info(f"   Year: {semantic_paper.year}, Venue: {semantic_paper.venue}")
                logger.info(f"   Citations: {semantic_paper.citation_count}")
                return semantic_paper
            else:
                logger.info("⚠️ No Semantic Scholar candidate found")
                return None
                
        except Exception as e:
            logger.warning(f"Error during Semantic Scholar search: {e}")
            return None
    
    def _verify_semantic_scholar_match(self, paper_text: str, semantic_paper: object) -> Dict[str, bool]:
        """Use LLM to verify if the Semantic Scholar match is correct."""
        try:
            # Create SS context for verification
            semantic_scholar_context = f"""
SEMANTIC SCHOLAR CANDIDATE FOUND:
Title: {semantic_paper.title}
Authors: {', '.join(semantic_paper.authors)}
Year: {semantic_paper.year}
Venue: {semantic_paper.venue}
Abstract: {semantic_paper.abstract[:300] if semantic_paper.abstract else 'No abstract'}...
Citation Count: {semantic_paper.citation_count}

Please verify if this Semantic Scholar entry matches the paper you're analyzing."""
            
            messages = self.prompt_manager.format_prompt(
                "extract_metadata",
                paper_text=paper_text[:8000],
                semantic_scholar_context=semantic_scholar_context
            )
            
            verification_response = self.llm_manager.generate_structured_response(
                messages=messages,
                schema=METADATA_SCHEMA,
                temperature=0.0
            )
            
            # Extract verification fields
            verification_result = {
                "verified_semantic_scholar_match": verification_response.get("verified_semantic_scholar_match", False),
                "use_semantic_scholar_data": verification_response.get("use_semantic_scholar_data", False)
            }
            
            logger.info(f"✅ LLM verification complete:")
            logger.info(f"   Verified match: {verification_result['verified_semantic_scholar_match']}")
            logger.info(f"   Use SS data: {verification_result['use_semantic_scholar_data']}")
            
            return verification_result
            
        except Exception as e:
            logger.error(f"Error during LLM verification: {e}")
            return {
                "verified_semantic_scholar_match": False,
                "use_semantic_scholar_data": False
            }
    
    def _validate_and_clean_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and clean extracted metadata."""
        # Validate year
        if metadata.get("year") and metadata["year"] != "null":
            try:
                metadata["year"] = int(metadata["year"])
                # Validate year is reasonable
                current_year = datetime.now().year
                if metadata["year"] < 1900 or metadata["year"] > current_year + 5:
                    metadata["year"] = None
            except (ValueError, TypeError):
                metadata["year"] = None
        else:
            metadata["year"] = None
        
        # Ensure authors is a list
        if not isinstance(metadata.get("authors"), list):
            metadata["authors"] = []
        
        return metadata
    
    def _reset_verification_result(self):
        """Reset verification result to default state."""
        self._verification_result = {
            "verified_semantic_scholar_match": False,
            "use_semantic_scholar_data": False
        }
    
    def _enrich_with_semantic_scholar_early(self, metadata: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Enrich paper data with Semantic Scholar metadata before database save.
        Uses verification result from LLM to decide whether to proceed.
        
        Args:
            metadata: Extracted metadata dictionary containing title
            
        Returns:
            Dictionary of enrichment data to add to paper_data, or None if no enrichment
        """
        # Check if LLM verified the Semantic Scholar match
        verification_result = getattr(self, '_verification_result', {})
        use_semantic_scholar = verification_result.get('use_semantic_scholar_data', False)
        
        logger.info(f"🔍 DEBUG: verification_result = {verification_result}")
        logger.info(f"🔍 DEBUG: use_semantic_scholar = {use_semantic_scholar}")
        
        if not use_semantic_scholar:
            verified_match = verification_result.get('verified_semantic_scholar_match', False)
            if verified_match:
                logger.info(f"Skipping early SS enrichment (LLM found match but chose not to use SS data)")
            else:
                logger.info(f"Skipping early SS enrichment (no SS match found or LLM verification failed)")
            # Clean up verification result after use
            self._reset_verification_result()
            return None
        
        logger.info(f"Starting early Semantic Scholar enrichment (LLM verified match)")
        
        # Create temporary paper object for enrichment
        temp_paper = type('TempPaper', (), {})()
        temp_paper.title = metadata["title"]
        temp_paper.id = None  # No ID yet since not saved to DB
        
        # Get enrichment data from client
        logger.info(f"🔍 DEBUG: About to call ss_client.enrich_paper with title: {temp_paper.title}")
        enrichment_data = self.ss_client.enrich_paper(temp_paper)
        logger.info(f"🔍 DEBUG: enrichment_data returned: {enrichment_data}")
        
        # Log results
        if enrichment_data.get('citation_count') is not None:
            logger.info(f"✅ Early SS enrichment successful: Citations: {enrichment_data['citation_count']}, "
                       f"Venue: {enrichment_data.get('venue', 'N/A')}")
        else:
            logger.info(f"⚠️  No Semantic Scholar data found during early enrichment")
            
        # Clean up verification result after use
        self._reset_verification_result()
        
        return enrichment_data
    
    def _generate_summary(self, metadata: Dict[str, Any], full_text: str) -> str:
        """Generate summary using LLM."""
        try:
            # Use abstract if available, otherwise use first part of full text
            content_for_summary = metadata.get("abstract", "")
            if not content_for_summary.strip():
                content_for_summary = full_text[:5000]  # First 5000 chars
            
            messages = self.prompt_manager.format_prompt(
                "summarize_paper",
                title=metadata.get("title", ""),
                authors=", ".join(metadata.get("authors", [])),
                content=content_for_summary
            )
            
            
            response = self.llm_manager.generate_response(
                messages=messages,
                temperature=0.1,
                max_tokens=300
            )
            
            return response.content.strip()
            
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return "Summary generation failed."
    
    def _generate_structured_summary(self, full_text: str) -> Optional[Dict[str, Any]]:
        """Generate structured AI summary using LLM with specific JSON format."""
        try:
            logger.info("🤖 Generating structured AI summary...")

            # Add delay to prevent connection pool exhaustion and rate limiting
            # When processing multiple papers sequentially, this gives the HTTP connection
            # pool time to recover and prevents rapid-fire requests to DeepSeek API
            logger.info("⏳ Waiting 3 seconds before LLM request to prevent connection issues...")
            time.sleep(3)

            # Use FULL text as requested by user
            text_for_analysis = full_text
            logger.info(f"📄 Using {len(text_for_analysis)} characters for AI analysis (full paper text)")

            messages = self.prompt_manager.format_prompt(
                "structured_paper_analysis",
                paper_text=text_for_analysis
            )

            logger.info(f"📤 Sending request to LLM for structured analysis...")
            logger.debug(f"Prompt message count: {len(messages)}")

            # Use simple approach that works (like in the successful test)
            response = self.llm_manager.generate_response(
                messages=messages,
                temperature=0  # Zero temperature for maximum consistency in JSON output
            )
            
            logger.info("📥 Received response from LLM")
            logger.info(f"🔍 STRUCTURED SUMMARY RESPONSE DEBUG:")
            logger.info(f"Response type: {type(response)}")
            
            # Parse JSON from response content (like in the successful test)
            try:
                if hasattr(response, 'content'):
                    response_text = response.content
                else:
                    response_text = str(response)
                
                # Safe preview logging
                try:
                    response_preview = str(response_text)[:200] if response_text else "None"
                    logger.info(f"Response content preview: {response_preview}...")
                except Exception as e:
                    logger.info(f"Could not preview response content: {e}")
                
                # Clean response text and parse JSON
                import json
                
                # Ensure response_text is a string
                if not isinstance(response_text, str):
                    response_text = str(response_text)
                
                response_text = response_text.strip()
                
                # Remove any markdown code blocks if present
                if response_text.startswith('```json'):
                    response_text = response_text[7:]
                if response_text.endswith('```'):
                    response_text = response_text[:-3]
                
                response_text = response_text.strip()
                
                logger.info(f"📝 Cleaned response text length: {len(response_text)}")
                
                # Parse JSON
                structured_response = json.loads(response_text)
                
                logger.info("✅ Successfully parsed JSON response")
                
                
                # Validate response structure
                if structured_response and isinstance(structured_response, dict):
                    required_keys = ["title", "executive_summary", "purpose_rationale_research_question", 
                                   "theory_framework", "methodology", 
                                   "major_findings_contributions", "study_limitations_gaps_that_remain",
                                   "study_implications_for_research_practice_policy"]
                    missing_keys = [key for key in required_keys if key not in structured_response]
                    
                    if missing_keys:
                        logger.warning(f"⚠️  LLM response missing required keys: {missing_keys}")
                    else:
                        logger.info("✅ LLM response contains all required structured sections")
                        
                        # Log sample content for verification
                        logger.info(f"📋 Sample data preview:")
                        title = str(structured_response.get('title', 'N/A'))[:50]
                        exec_summary = str(structured_response.get('executive_summary', 'N/A'))[:50]
                        logger.info(f"   Title: {title}...")
                        logger.info(f"   Executive: {exec_summary}...")
                    
                    return structured_response
                else:
                    logger.error("❌ Invalid response structure from LLM")
                    return None
                    
            except json.JSONDecodeError as e:
                logger.warning(f"⚠️ Initial JSON parsing failed: {e}")
                try:
                    error_preview = str(response_text)[:500] if response_text else "None"
                    logger.warning(f"Raw response text: {error_preview}...")
                except Exception:
                    logger.warning("Could not preview response text for error logging")
                
                # Attempt JSON repair
                logger.info("🔧 Attempting JSON repair...")
                expected_schema = {
                    "title": "string",
                    "executive_summary": "string",
                    "purpose_rationale_research_question": "array",
                    "theory_framework": "array",
                    "methodology": "array",
                    "major_findings_contributions": "array",
                    "study_limitations_gaps_that_remain": "array",
                    "study_implications_for_research_practice_policy": "array"
                }
                
                repaired_response = self.json_repair_service.repair_malformed_json(
                    response_text, 
                    expected_schema, 
                    max_attempts=2
                )
                
                if repaired_response:
                    logger.info("🎉 JSON repair successful!")
                    
                    # Validate the repaired response structure (same as above)
                    if repaired_response and isinstance(repaired_response, dict):
                        required_keys = ["title", "executive_summary", "purpose_rationale_research_question", 
                                       "theory_framework", "methodology", 
                                       "major_findings_contributions", "study_limitations_gaps_that_remain",
                                       "study_implications_for_research_practice_policy"]
                        missing_keys = [key for key in required_keys if key not in repaired_response]
                        
                        if missing_keys:
                            logger.warning(f"⚠️  Repaired LLM response missing required keys: {missing_keys}")
                        else:
                            logger.info("✅ Repaired LLM response contains all required structured sections")
                            
                            # Log sample content for verification
                            logger.info(f"📋 Repaired sample data preview:")
                            title = str(repaired_response.get('title', 'N/A'))[:50]
                            exec_summary = str(repaired_response.get('executive_summary', 'N/A'))[:50]
                            logger.info(f"   Title: {title}...")
                            logger.info(f"   Executive: {exec_summary}...")
                        
                        return repaired_response
                    else:
                        logger.error("❌ Invalid repaired response structure from LLM")
                        return None
                else:
                    logger.error("💥 JSON repair failed - no valid structured summary generated")
                    return None
            
        except Exception as e:
            logger.error(f"❌ Error generating structured summary: {e}")
            return None
    
    def _create_standardized_filename(self, metadata: Dict[str, Any]) -> str:
        """Create standardized filename from metadata."""
        authors = metadata.get("authors", [])
        year = metadata.get("year") or 2023  # Default year if not found
        title = metadata.get("title", "Unknown")
        
        return self.pdf_processor.suggest_filename(authors, year, title)
    
    def _add_to_vector_store(self, paper: Paper, extracted_text, metadata: Dict[str, Any], final_path: Path):
        """Add paper to vector store."""
        try:
            # Add paper summary to vector store
            if paper.summary:
                # Clean metadata - ChromaDB doesn't accept None values
                clean_metadata = {
                    "title": paper.title or "Unknown",
                    "authors": paper.authors or "[]",
                    "year": paper.year or 0,
                    "filename": paper.filename or "unknown.pdf"
                }
                self.vector_manager.add_paper_summary(
                    paper_id=paper.id,
                    summary=paper.summary,
                    metadata=clean_metadata
                )
            
            # Add paper chunks to vector store
            if extracted_text.chunks:
                # Clean metadata - ChromaDB doesn't accept None values
                clean_metadata = {
                    "title": paper.title or "Unknown",
                    "authors": paper.authors or "[]",
                    "year": paper.year or 0,
                    "filename": paper.filename or "unknown.pdf",
                    "pdf_path": str(final_path)
                }
                self.vector_manager.add_paper_chunks(
                    paper_id=paper.id,
                    chunks=extracted_text.chunks,
                    metadata=clean_metadata,
                    enhanced_chunks=extracted_text.enhanced_chunks
                )
            
            logger.info(f"Added paper to vector store: {paper.title}")
            
        except Exception as e:
            logger.error(f"Error adding paper to vector store: {e}")
    
    def _move_to_failed(self, file_path: Path, reason: str):
        """Move file to failed directory with reason."""
        try:
            failed_path = self.directory_manager.move_file(
                file_path, "failed", f"{file_path.stem}_FAILED_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            )
            
            # Write reason to accompanying text file
            reason_file = failed_path.with_suffix(".txt")
            reason_file.write_text(f"Processing failed: {reason}\nTimestamp: {datetime.now()}")
            
            logger.warning(f"Moved file to failed directory: {failed_path}")
            
        except Exception as e:
            logger.error(f"Error moving file to failed directory: {e}")
    
    def reprocess_failed_papers(self) -> List[Paper]:
        """Attempt to reprocess papers in the failed directory."""
        failed_dir = self.directory_manager.get_directory("failed")
        failed_pdfs = list(failed_dir.glob("*_FAILED_*.pdf"))
        
        logger.info(f"Found {len(failed_pdfs)} failed papers to reprocess")
        
        reprocessed_papers = []
        for pdf_path in failed_pdfs:
            try:
                # Remove the FAILED timestamp from filename
                original_name = pdf_path.name.split("_FAILED_")[0] + ".pdf"
                
                # Move back to processing and try again
                processing_path = self.directory_manager.move_file(
                    pdf_path, "processing", original_name
                )
                
                # Remove associated reason file if it exists
                reason_file = pdf_path.with_suffix(".txt")
                if reason_file.exists():
                    reason_file.unlink()
                
                # Process the paper
                paper = self.process_new_paper(processing_path)
                if paper:
                    reprocessed_papers.append(paper)
                    logger.info(f"Successfully reprocessed: {original_name}")
                
            except Exception as e:
                logger.error(f"Error reprocessing {pdf_path.name}: {e}")
        
        return reprocessed_papers
    
    def _enrich_with_semantic_scholar(self, paper: Paper) -> None:
        """
        Enrich paper with Semantic Scholar metadata in a non-blocking way.
        Uses verification result from LLM to decide whether to proceed with enrichment.
        
        Args:
            paper: Paper object to enrich
        """
        try:
            # Check if we have a verification result from the metadata extraction
            verification_result = getattr(self, '_verification_result', {})
            use_semantic_scholar = verification_result.get('use_semantic_scholar_data', False)
            

            if not use_semantic_scholar:
                verified_match = verification_result.get('verified_semantic_scholar_match', False)
                if verified_match:
                    logger.info(f"Skipping Semantic Scholar enrichment for: {paper.title[:50]} "
                               f"(LLM found match but chose not to use Semantic Scholar data)")
                else:
                    logger.info(f"Skipping Semantic Scholar enrichment for: {paper.title[:50]} "
                               f"(No Semantic Scholar match found or LLM verification failed)")
                return
            
            logger.info(f"Starting Semantic Scholar enrichment for: {paper.title[:50]} "
                       f"(LLM verified match)")
            
            # Get enrichment data from client
            enrichment_data = self.ss_client.enrich_paper(paper)
            
            # Update paper in database with enrichment data
            self.db_manager.update_paper(paper.id, enrichment_data)
            
            # Log results
            if enrichment_data.get('citation_count') is not None:
                logger.info(f"Successfully enriched paper: {paper.title[:50]} "
                           f"(Citations: {enrichment_data['citation_count']}, "
                           f"Venue: {enrichment_data.get('venue', 'N/A')})")
            else:
                logger.info(f"No Semantic Scholar data found for: {paper.title[:50]}")
                
        except Exception as e:
            # Non-blocking: log error but don't fail paper processing
            logger.warning(f"Semantic Scholar enrichment failed for {paper.title[:50]}: {e}")
        finally:
            # Clean up verification result after use
            self._verification_result = {
                "verified_semantic_scholar_match": False,
                "use_semantic_scholar_data": False
            }
    
    def _is_duplicate_paper(self, original_filename: str) -> tuple[bool, Optional[str]]:
        """
        Check if paper already exists in database by exact filename match.
        
        Args:
            original_filename: The original filename to check
            
        Returns:
            Tuple of (is_duplicate, existing_filename) - existing_filename is None if not duplicate
        """
        try:
            # Query database for exact filename match using DatabaseManager method
            existing_paper = self.db_manager.get_paper_by_original_filename(original_filename)
            
            if existing_paper:
                logger.info(f"Duplicate detected: {original_filename} already exists as {existing_paper.filename}")
                return True, existing_paper.filename
            else:
                return False, None
                
        except Exception as e:
            logger.error(f"Error checking for duplicate paper: {e}")
            # If we can't check, assume it's not a duplicate to avoid blocking processing
            return False, None
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get statistics about paper processing."""
        stats = self.directory_manager.get_directory_stats()
        
        # Add database stats
        db_stats = self.db_manager.get_paper_stats()
        
        # Add vector store stats
        vector_stats = self.vector_manager.get_collection_stats()
        
        # Add JSON repair stats
        json_repair_stats = self.json_repair_service.get_repair_stats()
        
        # Add PDF generation stats
        pdf_stats = {
            "pdf_generator_available": REPORTLAB_AVAILABLE,
            "summaries_directory": str(self.pdf_generator.summaries_dir) if hasattr(self.pdf_generator, 'summaries_dir') else None
        }
        
        return {
            "directories": stats,
            "database": db_stats,
            "vector_store": vector_stats,
            "json_repair": json_repair_stats,
            "pdf_generation": pdf_stats
        }