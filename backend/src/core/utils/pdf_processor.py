import fitz  # PyMuPDF
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import re
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class TextChunk:
    """Container for a text chunk with location information."""
    text: str
    page_number: int
    bbox: Tuple[float, float, float, float]  # (x0, y0, x1, y1)
    text_start: int  # Character position in full text
    text_end: int

@dataclass
class ExtractedText:
    """Container for extracted PDF text and metadata."""
    full_text: str
    first_pages_text: str  # First 2-3 pages for metadata extraction
    page_count: int
    chunks: List[str]  # Legacy: plain text chunks
    enhanced_chunks: List[TextChunk]  # New: chunks with location data
    metadata: Dict[str, Any]
    cleaned_full_text: Optional[str] = None  # LLM-cleaned text (no headers/footers/tables/refs)


class PDFProcessor:
    """Handles PDF text extraction and processing for academic papers."""
    
    def __init__(self,
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200,
                 metadata_pages: int = 3,
                 enable_llm_cleaning: bool = True,
                 llm_manager: Optional[Any] = None,
                 pages_per_chunk: int = 1,
                 enable_reference_detection: bool = True,
                 auto_skip_threshold: Optional[int] = None,
                 enable_parallel: bool = True,
                 max_concurrent: Optional[int] = None):
        """
        Initialize PDF processor.

        Args:
            chunk_size: Target size for text chunks
            chunk_overlap: Overlap between chunks
            metadata_pages: Number of first pages to use for metadata extraction
            enable_llm_cleaning: Whether to use LLM-based PDF cleaning
            llm_manager: LLM manager instance for PDF cleaning
            pages_per_chunk: Pages per chunk for LLM cleaning (default: 1)
            enable_reference_detection: Whether to detect and remove references (default: True)
            auto_skip_threshold: Skip cleaning if PDF exceeds this many pages (default: None)
            enable_parallel: Use parallel PDF processing (default: True)
            max_concurrent: Maximum concurrent chunks (default: None = unlimited)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.metadata_pages = metadata_pages
        self.enable_llm_cleaning = enable_llm_cleaning
        self.llm_manager = llm_manager
        self.pages_per_chunk = pages_per_chunk
        self.enable_reference_detection = enable_reference_detection
        self.auto_skip_threshold = auto_skip_threshold
        self.enable_parallel = enable_parallel
        self.max_concurrent = max_concurrent
    
    async def extract_text_from_pdf(self, pdf_path: str, force_clean: bool = False) -> ExtractedText:
        """
        Extract text and metadata from PDF file with location information.

        Args:
            pdf_path: Path to PDF file
            force_clean: Whether to force LLM cleaning even if over threshold (default: False)

        Returns:
            ExtractedText object with full text, chunks, and metadata
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        try:
            doc = fitz.open(pdf_path)
            page_count = len(doc)
            
            # Extract text from all pages with location data
            full_text = ""
            first_pages_text = ""
            page_texts = []  # Store page-by-page text with metadata
            
            for page_num in range(page_count):
                page = doc[page_num]
                
                # Get text with detailed location information
                text_dict = page.get_text("dict")
                page_text, page_blocks = self._extract_text_with_coordinates(text_dict, page_num)
                
                # Clean up text
                page_text = self._clean_text(page_text)
                full_text += page_text + "\n\n"
                
                # Store page data for chunk creation
                page_texts.append({
                    'page_number': page_num,
                    'text': page_text,
                    'blocks': page_blocks,
                    'text_start': len(full_text) - len(page_text) - 2,  # Adjust for \n\n
                    'text_end': len(full_text) - 2
                })
                
                # Collect first few pages for metadata extraction
                if page_num < self.metadata_pages:
                    first_pages_text += page_text + "\n\n"
            
            doc.close()

            # LLM-based PDF cleaning (if enabled)
            cleaned_full_text = None
            text_for_chunking = full_text  # Default to uncleaned text

            logger.info(f"PDF cleaning check: enable_llm_cleaning={self.enable_llm_cleaning}, llm_manager={'Present' if self.llm_manager else 'None'}")

            if self.enable_llm_cleaning and self.llm_manager:
                logger.info(f"🧹 Starting LLM-based PDF cleaning for: {pdf_path.name}")
                from src.core.utils.pdf_cleaner import PDFCleaner

                cleaner = PDFCleaner()
                cleaned_full_text = await cleaner.clean_pdf_text(
                    pdf_path,
                    self.llm_manager,
                    pages_per_chunk=self.pages_per_chunk,
                    enable_reference_detection=self.enable_reference_detection,
                    auto_skip_threshold=self.auto_skip_threshold,
                    force_clean=force_clean,
                    enable_parallel=self.enable_parallel,
                    max_concurrent=self.max_concurrent
                )

                # Handle cleaning being skipped (returns None when threshold exceeded)
                if cleaned_full_text:
                    text_for_chunking = cleaned_full_text
                    logger.info(f"✅ PDF cleaning complete: {len(cleaned_full_text)} chars (vs {len(full_text)} original)")
                else:
                    logger.info("ℹ️ PDF cleaning was skipped - using original text for chunking")
                    text_for_chunking = full_text
            else:
                logger.info("PDF cleaning disabled or LLM manager not available")

            # Create both legacy chunks and enhanced chunks with location data
            # NOTE: Chunks are created from cleaned text if cleaning was successful
            chunks = self._create_chunks(text_for_chunking)
            enhanced_chunks = self._create_enhanced_chunks(text_for_chunking, page_texts, str(pdf_path))

            # Extract basic metadata from PDF
            pdf_metadata = self._extract_pdf_metadata(pdf_path)

            return ExtractedText(
                full_text=full_text.strip(),
                first_pages_text=first_pages_text.strip(),
                page_count=page_count,
                chunks=chunks,
                enhanced_chunks=enhanced_chunks,
                metadata=pdf_metadata,
                cleaned_full_text=cleaned_full_text
            )
            
        except Exception as e:
            logger.error(f"Error extracting text from {pdf_path}: {e}")
            raise
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize extracted text."""
        # Remove excessive whitespace
        text = re.sub(r'\n\s*\n', '\n\n', text)
        text = re.sub(r' +', ' ', text)
        
        # Remove page numbers and headers/footers (basic heuristics)
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            
            # Skip very short lines that might be page numbers
            if len(line) < 3:
                continue
            
            # Skip lines that are just numbers (likely page numbers)
            if line.isdigit():
                continue
            
            # Skip common header/footer patterns
            if re.match(r'^[A-Z\s]+$', line) and len(line) < 50:
                continue
            
            cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def _extract_text_with_coordinates(self, text_dict: Dict, page_num: int) -> Tuple[str, List[Dict]]:
        """
        Extract text with bounding box coordinates from PyMuPDF text dictionary.
        
        Args:
            text_dict: PyMuPDF text dictionary from page.get_text("dict")
            page_num: Page number
            
        Returns:
            Tuple of (page_text, list of text blocks with coordinates)
        """
        page_text = ""
        text_blocks = []
        
        for block in text_dict.get("blocks", []):
            if "lines" not in block:  # Skip image blocks
                continue
                
            block_text = ""
            block_bbox = block.get("bbox", [0, 0, 0, 0])
            
            for line in block["lines"]:
                line_text = ""
                for span in line.get("spans", []):
                    span_text = span.get("text", "")
                    line_text += span_text
                
                if line_text.strip():
                    block_text += line_text + "\n"
            
            if block_text.strip():
                page_text += block_text + "\n"
                text_blocks.append({
                    'text': block_text.strip(),
                    'bbox': block_bbox,
                    'page_number': page_num
                })
        
        return page_text, text_blocks
    
    def _create_enhanced_chunks(self, full_text: str, page_texts: List[Dict], pdf_path: str) -> List[TextChunk]:
        """
        Create enhanced text chunks with location information.
        
        Args:
            full_text: Complete text from PDF
            page_texts: List of page data with coordinates
            pdf_path: Path to source PDF
            
        Returns:
            List of TextChunk objects with location data
        """
        if not full_text.strip():
            return []
        
        enhanced_chunks = []
        current_chunk = ""
        current_page = 0
        current_bbox = [0, 0, 0, 0]
        chunk_start_pos = 0
        
        # Split by paragraphs and track positions
        paragraphs = []
        pos = 0
        for paragraph in full_text.split('\n\n'):
            if paragraph.strip():
                # Find which page this paragraph belongs to
                page_info = self._find_paragraph_page(pos, page_texts)
                paragraphs.append({
                    'text': paragraph.strip(),
                    'position': pos,
                    'page_info': page_info
                })
            pos += len(paragraph) + 2  # +2 for \n\n
        
        for i, para_data in enumerate(paragraphs):
            paragraph = para_data['text']
            page_info = para_data['page_info']
            
            # If adding this paragraph would exceed chunk size
            if len(current_chunk) + len(paragraph) > self.chunk_size and current_chunk:
                # Create chunk with current data
                enhanced_chunks.append(TextChunk(
                    text=current_chunk.strip(),
                    page_number=current_page,
                    bbox=tuple(current_bbox),
                    text_start=chunk_start_pos,
                    text_end=chunk_start_pos + len(current_chunk)
                ))
                
                # Start new chunk with overlap
                if self.chunk_overlap > 0:
                    overlap_text = current_chunk[-self.chunk_overlap:]
                    last_sentence = overlap_text.rfind('.')
                    if last_sentence > 0:
                        overlap_text = overlap_text[last_sentence + 1:].strip()
                    current_chunk = overlap_text + " " + paragraph
                else:
                    current_chunk = paragraph
                
                # Update tracking for new chunk
                chunk_start_pos = para_data['position']
                current_page = page_info['page_number']
                current_bbox = page_info['bbox']
                
            else:
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
                    chunk_start_pos = para_data['position']
                    current_page = page_info['page_number']
                    current_bbox = page_info['bbox']
        
        # Add the last chunk
        if current_chunk.strip():
            enhanced_chunks.append(TextChunk(
                text=current_chunk.strip(),
                page_number=current_page,
                bbox=tuple(current_bbox),
                text_start=chunk_start_pos,
                text_end=chunk_start_pos + len(current_chunk)
            ))
        
        return enhanced_chunks
    
    def _find_paragraph_page(self, text_position: int, page_texts: List[Dict]) -> Dict:
        """
        Find which page and approximate bounding box a text position belongs to.
        
        Args:
            text_position: Character position in full text
            page_texts: List of page data
            
        Returns:
            Dict with page_number and bbox information
        """
        for page_data in page_texts:
            if page_data['text_start'] <= text_position <= page_data['text_end']:
                # Use the first text block's bbox as approximation
                bbox = [0, 0, 0, 0]
                if page_data['blocks']:
                    bbox = page_data['blocks'][0]['bbox']
                
                return {
                    'page_number': page_data['page_number'],
                    'bbox': bbox
                }
        
        # Fallback: return first page
        return {
            'page_number': 0,
            'bbox': [0, 0, 0, 0]
        }
    
    def _create_chunks(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks for better context preservation.
        
        Args:
            text: Full text to chunk
            
        Returns:
            List of text chunks
        """
        if not text.strip():
            return []
        
        # Split by paragraphs first
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            # If adding this paragraph would exceed chunk size
            if len(current_chunk) + len(paragraph) > self.chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                
                # Start new chunk with overlap from previous chunk
                if self.chunk_overlap > 0:
                    overlap_text = current_chunk[-self.chunk_overlap:]
                    # Find a good break point for overlap
                    last_sentence = overlap_text.rfind('.')
                    if last_sentence > 0:
                        overlap_text = overlap_text[last_sentence + 1:].strip()
                    current_chunk = overlap_text + " " + paragraph
                else:
                    current_chunk = paragraph
            else:
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
        
        # Add the last chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _extract_pdf_metadata(self, pdf_path: Path) -> Dict[str, Any]:
        """Extract basic metadata from PDF file properties."""
        try:
            doc = fitz.open(pdf_path)
            metadata = doc.metadata
            doc.close()
            
            return {
                "file_size": pdf_path.stat().st_size,
                "file_name": pdf_path.name,
                "pdf_title": metadata.get("title", ""),
                "pdf_author": metadata.get("author", ""),
                "pdf_subject": metadata.get("subject", ""),
                "pdf_creator": metadata.get("creator", ""),
                "pdf_producer": metadata.get("producer", ""),
                "creation_date": metadata.get("creationDate", ""),
                "modification_date": metadata.get("modDate", "")
            }
        except Exception as e:
            logger.error(f"Error extracting PDF metadata: {e}")
            return {}
    
    def extract_abstract_section(self, text: str) -> Optional[str]:
        """
        Attempt to extract abstract section from paper text.
        
        Args:
            text: Full or partial paper text
            
        Returns:
            Abstract text if found, None otherwise
        """
        # Common abstract patterns
        abstract_patterns = [
            r'abstract\s*[:\-]?\s*(.*?)(?=\n\s*(?:keywords?|introduction|1\.|i\.|background))',
            r'summary\s*[:\-]?\s*(.*?)(?=\n\s*(?:keywords?|introduction|1\.|i\.|background))',
            r'^(.*?)(?=\n\s*(?:keywords?|introduction|1\.|i\.))',
        ]
        
        text_lower = text.lower()
        
        for pattern in abstract_patterns:
            match = re.search(pattern, text_lower, re.DOTALL | re.IGNORECASE)
            if match:
                abstract = match.group(1).strip()
                
                # Clean up the abstract
                abstract = re.sub(r'\n+', ' ', abstract)
                abstract = re.sub(r'\s+', ' ', abstract)
                
                # Filter out very short or very long abstracts
                if 50 <= len(abstract) <= 2000:
                    return abstract
        
        return None
    
    def suggest_filename(self, authors: List[str], year: int, title: str) -> str:
        """
        Generate standardized filename from paper metadata.
        
        Args:
            authors: List of author names
            year: Publication year
            title: Paper title
            
        Returns:
            Suggested filename (without .pdf extension)
        """
        # Get first author's last name
        if authors:
            first_author = authors[0].strip()
            # Extract last name (assume it's the last word)
            last_name = first_author.split()[-1]
            # Remove non-alphanumeric characters
            last_name = re.sub(r'[^\w\s]', '', last_name)
        else:
            last_name = "Unknown"
        
        # Clean title
        title_clean = re.sub(r'[^\w\s]', '', title)
        title_clean = re.sub(r'\s+', ' ', title_clean).strip()
        
        # Format: LastName_Year_Title
        filename = f"{last_name}_{year}_{title_clean.replace(' ', '_')}"
        
        # Remove any remaining problematic characters
        filename = re.sub(r'[<>:"/\\|?*]', '', filename)
        
        return filename
    
    def batch_process_pdfs(self, pdf_directory: str) -> List[Tuple[Path, ExtractedText]]:
        """
        Process multiple PDF files in a directory.
        
        Args:
            pdf_directory: Directory containing PDF files
            
        Returns:
            List of tuples (pdf_path, extracted_text)
        """
        pdf_dir = Path(pdf_directory)
        if not pdf_dir.exists():
            raise FileNotFoundError(f"Directory not found: {pdf_dir}")
        
        results = []
        pdf_files = list(pdf_dir.glob("*.pdf"))
        
        logger.info(f"Processing {len(pdf_files)} PDF files...")
        
        for pdf_file in pdf_files:
            try:
                extracted_text = self.extract_text_from_pdf(pdf_file)
                results.append((pdf_file, extracted_text))
                logger.info(f"Successfully processed: {pdf_file.name}")
            except Exception as e:
                logger.error(f"Failed to process {pdf_file.name}: {e}")
                continue
        
        logger.info(f"Successfully processed {len(results)}/{len(pdf_files)} PDF files")
        return results
    
    def validate_pdf(self, pdf_path: str) -> bool:
        """
        Validate if file is a readable PDF.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            True if valid PDF, False otherwise
        """
        try:
            doc = fitz.open(pdf_path)
            page_count = len(doc)
            doc.close()
            return page_count > 0
        except Exception:
            return False