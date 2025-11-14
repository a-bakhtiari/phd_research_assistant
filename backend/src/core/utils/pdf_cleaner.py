"""
LLM-based PDF Cleaning Module

Extracts text from PDFs and uses LLM to clean junk (headers, footers, page numbers,
tables, figures, references) while preserving actual paper content.

This module ports the 4-step pipeline from the test folder:
1. Extract PDF blocks with column detection
2. Create chunks by grouping pages
3. LLM processes chunks to remove junk and fix reading order
4. Stitch cleaned chunks together

Author: Ported from test folder pipeline
"""

import logging
import json
import time
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from collections import defaultdict

try:
    import fitz  # PyMuPDF
except ImportError:
    raise ImportError("PyMuPDF not installed. Install with: pip install PyMuPDF")

logger = logging.getLogger(__name__)


class PDFBlockExtractor:
    """Extracts text blocks from PDF with spatial metadata (Step 1)."""

    def extract_page_blocks(self, page: fitz.Page, page_num: int) -> Dict[str, Any]:
        """Extract blocks from a single page with column and position metadata."""
        page_width = page.rect.width

        # Get text blocks: (x0, y0, x1, y1, text, block_no, block_type)
        raw_blocks = page.get_text("blocks")

        # Filter to text blocks only (type 0) and enrich with minimal metadata
        enriched_blocks = []
        for block in raw_blocks:
            x0, y0, x1, y1, text, block_no, block_type = block

            # Skip non-text blocks (images, etc.)
            if block_type != 0:
                continue

            # Determine column based on x-position
            column = "left" if x0 < (page_width / 2) else "right"

            enriched_blocks.append({
                "column": column,
                "y": round(y0, 2),
                "text": text.strip()
            })

        # Sort by reading order: left column (by y), then right column (by y)
        left_blocks = [b for b in enriched_blocks if b["column"] == "left"]
        right_blocks = [b for b in enriched_blocks if b["column"] == "right"]

        left_blocks.sort(key=lambda b: b["y"])
        right_blocks.sort(key=lambda b: b["y"])

        # Combine left + right, assign indices
        sorted_blocks = left_blocks + right_blocks
        for idx, block in enumerate(sorted_blocks):
            block["index"] = idx

        return {
            "page_number": page_num,
            "page_width": round(page_width, 2),
            "block_count": len(sorted_blocks),
            "blocks": sorted_blocks
        }

    def process_pdf(self, pdf_path: Path) -> List[Dict[str, Any]]:
        """Process a PDF file and return per-page data."""
        try:
            doc = fitz.open(pdf_path)
            total_pages = len(doc)

            pages_data = []
            for page_num in range(total_pages):
                page = doc[page_num]
                page_data = self.extract_page_blocks(page, page_num + 1)
                page_data["source_file"] = pdf_path.name
                pages_data.append(page_data)

            doc.close()
            logger.info(f"Extracted {total_pages} pages from {pdf_path.name}")
            return pages_data

        except Exception as e:
            logger.error(f"Error extracting blocks from {pdf_path}: {e}")
            raise


class ReferenceSectionDetector:
    """
    Detects and truncates reference sections in PDFs using LLM verification.

    Uses a two-step approach:
    1. Find all "references" or "bibliography" keywords
    2. Verify with LLM (checking context) to avoid false positives
    """

    def _extract_words_from_blocks(self, blocks: List[Dict[str, Any]], max_words: int) -> str:
        """
        Extract up to max_words from a list of blocks.

        Args:
            blocks: List of block dictionaries with 'text' field
            max_words: Maximum number of words to extract

        Returns:
            String of extracted words
        """
        words = []
        for block in blocks:
            block_words = block['text'].split()
            words.extend(block_words)
            if len(words) >= max_words:
                break

        return ' '.join(words[:max_words])

    def extract_context(self,
                       pages_data: List[Dict[str, Any]],
                       target_page: int,
                       target_block_idx: int,
                       words_before: int = 100,
                       words_after: int = 100) -> str:
        """
        Extract context around a target block (100 words before and after).

        Args:
            pages_data: List of page data
            target_page: Page number containing the keyword (1-indexed)
            target_block_idx: Block index on that page
            words_before: Number of words to extract before keyword
            words_after: Number of words to extract after keyword

        Returns:
            Context string with keyword in the middle
        """
        # Find target page index (pages_data uses 1-indexed page_number)
        target_page_idx = None
        for idx, page in enumerate(pages_data):
            if page['page_number'] == target_page:
                target_page_idx = idx
                break

        if target_page_idx is None:
            return ""

        # Get the keyword block
        target_block = pages_data[target_page_idx]['blocks'][target_block_idx]
        keyword_text = target_block['text']

        # Collect blocks before the keyword
        blocks_before = []

        # Same page, blocks before target
        for i in range(target_block_idx):
            blocks_before.append(pages_data[target_page_idx]['blocks'][i])

        # Previous pages (in reverse order, then reverse the list)
        for page_idx in range(target_page_idx - 1, -1, -1):
            for block in reversed(pages_data[page_idx]['blocks']):
                blocks_before.insert(0, block)

        # Collect blocks after the keyword
        blocks_after = []

        # Same page, blocks after target
        for i in range(target_block_idx + 1, len(pages_data[target_page_idx]['blocks'])):
            blocks_after.append(pages_data[target_page_idx]['blocks'][i])

        # Following pages
        for page_idx in range(target_page_idx + 1, len(pages_data)):
            for block in pages_data[page_idx]['blocks']:
                blocks_after.append(block)

        # Extract words
        text_before = self._extract_words_from_blocks(list(reversed(blocks_before)), words_before)
        text_before = ' '.join(reversed(text_before.split()))  # Reverse back to correct order

        text_after = self._extract_words_from_blocks(blocks_after, words_after)

        # Combine context
        context = f"{text_before}\n\n{keyword_text}\n\n{text_after}"

        logger.debug(f"Extracted context: {len(text_before.split())} words before, "
                    f"{len(text_after.split())} words after")

        return context

    def verify_reference_start(self, context_text: str, llm_manager) -> bool:
        """
        Ask LLM to verify if the context looks like the start of a reference section.

        Args:
            context_text: Text context around potential reference heading
            llm_manager: LLM manager for making API calls

        Returns:
            True if LLM confirms this is a reference section start, False otherwise
        """
        prompt = f"""You are analyzing an academic research paper. I need to identify where the reference/bibliography section starts.

Below is text from the paper around a potential reference section heading:

---
{context_text}
---

Question: Is this the START of the reference/bibliography section where citations are listed?

Instructions:
- Answer ONLY "yes" or "no"
- "yes" if this is where the reference list begins (followed by citation entries like "[1] Smith, J. (2020)..." or "Author, A. (Year). Title...")
- "no" if this is just a mention of references in the main text

Answer:"""

        logger.info("📤 Sending context to LLM for verification...")
        logger.debug(f"Prompt:\n{prompt}")

        # Call LLM with properly formatted messages
        messages = [{"role": "user", "content": prompt}]
        llm_response = llm_manager.generate_response(messages, max_tokens=50, temperature=0.0)
        response = llm_response.content

        logger.info(f"📥 LLM response: {response.strip()}")

        # Parse response (case-insensitive)
        response_lower = response.lower().strip()

        if 'yes' in response_lower:
            return True
        elif 'no' in response_lower:
            return False
        else:
            # Unexpected response - log and treat as "no"
            logger.warning(f"⚠️ Unexpected LLM response (treating as 'no'): {response}")
            return False

    def detect_cutoff_page(self,
                          pages_data: List[Dict[str, Any]],
                          llm_manager) -> Optional[int]:
        """
        Detect page number where reference section starts using LLM verification.

        Args:
            pages_data: List of page data from PDFBlockExtractor
            llm_manager: LLM manager for verification calls

        Returns:
            Page number to stop at (1-indexed), or None if no cutoff detected
        """
        import re

        logger.info("🔍 Searching for 'references' or 'bibliography' keywords...")

        # Step 1: Find all candidates
        candidates = []

        for page_data in pages_data:
            page_num = page_data['page_number']

            for block_idx, block in enumerate(page_data['blocks']):
                text = block['text'].strip()

                # Check for whole word "references" or "bibliography" (case-insensitive)
                if re.search(r'\breferences\b', text, re.IGNORECASE) or \
                   re.search(r'\bbibliography\b', text, re.IGNORECASE):

                    candidates.append({
                        'page': page_num,
                        'block_idx': block_idx,
                        'text': text
                    })

                    logger.info(f"📌 Found candidate at page {page_num}: '{text[:50]}...'")

        if not candidates:
            logger.info("ℹ️ No 'references' or 'bibliography' keywords found")
            return None

        logger.info(f"Found {len(candidates)} candidate(s) on pages: "
                   f"{[c['page'] for c in candidates]}")

        # Step 2: Check from last to first
        logger.info("🔄 Checking candidates from last to first...")

        for candidate in reversed(candidates):
            logger.info(f"\n📍 Checking page {candidate['page']}...")

            # Extract context
            context = self.extract_context(
                pages_data,
                candidate['page'],
                candidate['block_idx'],
                words_before=100,
                words_after=100
            )

            # Ask LLM to verify
            is_ref_section = self.verify_reference_start(context, llm_manager)

            if is_ref_section:
                logger.info(f"✅ LLM confirmed: This IS the reference section start")
                logger.info(f"✂️ Truncating at page {candidate['page']}")
                return candidate['page']
            else:
                logger.info(f"❌ LLM rejected: Not a reference section (likely in-text mention)")

        # All candidates rejected
        logger.info("\n❌ No valid reference section found after LLM verification")
        logger.info("ℹ️ All 'references' mentions appear to be in-text, not section headers")
        return None

    def truncate_pages(self,
                      pages_data: List[Dict[str, Any]],
                      cutoff_page: int) -> List[Dict[str, Any]]:
        """
        Remove pages starting from cutoff_page.

        Args:
            pages_data: Original page data
            cutoff_page: Page number to stop at (1-indexed)

        Returns:
            Truncated page data
        """
        original_count = len(pages_data)
        truncated = [p for p in pages_data if p['page_number'] < cutoff_page]
        removed_count = original_count - len(truncated)

        logger.info(f"📊 Truncation summary:")
        logger.info(f"   Original pages: {original_count}")
        logger.info(f"   Kept pages: {len(truncated)} (1-{cutoff_page-1})")
        logger.info(f"   Removed pages: {removed_count} ({cutoff_page}-{original_count})")
        logger.info(f"   Reduction: {removed_count/original_count*100:.1f}%")

        return truncated


class ChunkCreator:
    """Groups pages into chunks for LLM processing (Step 2)."""

    def create_chunks(self, pages_data: List[Dict[str, Any]], pages_per_chunk: int) -> List[Dict[str, Any]]:
        """Combine multiple pages into chunks."""
        chunks = []

        for i in range(0, len(pages_data), pages_per_chunk):
            chunk_pages = pages_data[i:i + pages_per_chunk]
            chunk_id = (i // pages_per_chunk) + 1

            # Combine blocks from all pages in this chunk
            combined_blocks = []
            block_offset = 0
            page_numbers = []

            for page_data in chunk_pages:
                page_numbers.append(page_data['page_number'])

                for block in page_data['blocks']:
                    adjusted_block = block.copy()
                    adjusted_block['index'] = block['index'] + block_offset
                    adjusted_block['page'] = page_data['page_number']
                    combined_blocks.append(adjusted_block)

                block_offset += len(page_data['blocks'])

            # Create chunk data
            page_range = f"{min(page_numbers)}-{max(page_numbers)}" if len(page_numbers) > 1 else str(min(page_numbers))

            chunk_data = {
                "chunk_id": chunk_id,
                "page_range": page_range,
                "pages": sorted(page_numbers),
                "block_count": len(combined_blocks),
                "blocks": combined_blocks
            }

            chunks.append(chunk_data)

        logger.info(f"Created {len(chunks)} chunks from {len(pages_data)} pages")
        return chunks


class LLMChunkProcessor:
    """Processes chunks using LLM for cleanup, ordering, and paragraph grouping (Step 3)."""

    def __init__(self, llm_manager, max_retries: int = 3):
        """
        Initialize LLM chunk processor.

        Args:
            llm_manager: LLM manager instance for making calls
            max_retries: Maximum number of retries for LLM calls
        """
        self.llm_manager = llm_manager
        self.max_retries = max_retries

    def _format_blocks_for_prompt(self, blocks: List[Dict]) -> str:
        """Format blocks for LLM prompt."""
        return "\n".join([
            f"Index: {b['index']}, Column: {b['column']}, Y: {b['y']}, Text: \"{b['text'][:150]}...\""
            if len(b['text']) > 150 else
            f"Index: {b['index']}, Column: {b['column']}, Y: {b['y']}, Text: \"{b['text']}\""
            for b in blocks
        ])

    def _create_processing_prompt(self, page_range: str, blocks: List[Dict]) -> str:
        """Create the LLM prompt for page processing.

        IMPORTANT: This prompt is carefully engineered - DO NOT MODIFY.
        """
        blocks_str = self._format_blocks_for_prompt(blocks)

        return f"""You are processing PAGE {page_range} of an academic paper.

**CONTEXT:**
I am giving you a list of text blocks extracted from a PDF using PyMuPDF's get_text("blocks") method. Each block contains:
- **Text content**: The actual text from the PDF
- **Column position**: Whether the block is in the "left" or "right" column (for 2-column layouts)
- **Y-coordinate**: Vertical position on the page (lower Y = higher on page)
- The PDF is processed in chunks (groups of pages, e.g., 4 pages at a time) to manage size - you are seeing one chunk at a time.

**TASK 1 - REMOVE JUNK:**
Identify blocks that are:
- Page headers/footers (e.g., "Page 5", running headers at top of page)
- Page numbers
- Journal names, DOIs, URLs
- Author information, affiliations, email addresses
- Copyright notices
- "Available at..."
- **Tables**
- **Figures**
- **Schemes, Charts, Diagrams**
- **Equations**
- *References and Citations*
- *Appendix or anything that comes after the conclusion*

**IMPORTANT - DO NOT REMOVE:**
- **Section headers** (e.g., "Abstract", "1. Introduction", "2. Methods", "3.1. Participants", "4. Results", "5.1. Altruism")
- Title of the paper

**NOTE ON LINE NUMBERS:**
Some manuscripts/preprints have numbered lines (e.g., "1 This is text", "2 More text"). These line numbers are part of the PDF formatting and cannot be removed. When evaluating content, focus on the actual text content AFTER any line numbers - do not mistake manuscript line numbers for page numbers or treat them as junk to remove.

**TASK 2 - VERIFY ORDER:**
Blocks are pre-sorted in reading order: ALL left column blocks (top→bottom), THEN ALL right column blocks (top→bottom).

ONLY reorder if there's an obvious error (e.g., a section header appearing AFTER its content).

**ABSOLUTE RULE:** ALL "left" column blocks MUST come before ALL "right" column blocks. Never interleave columns.

**TASK 3 - GROUP INTO PARAGRAPHS:**
Group consecutive blocks that clearly belong to the same paragraph:
- Blocks that continue mid-sentence (start with lowercase, no period at end of previous block)
- Blocks separated by hyphenation (previous ends with hyphen like "exper-")
- Blocks that are semantically continuous (same topic, no topic break)

Each paragraph group should be a list of consecutive block indices.

**TASK 4 - CROSS-PAGE CONTINUITY:**
For each paragraph, determine:
- `continues_from_prev`: Does it start mid-sentence? (lowercase start, incomplete thought)
- `continues_to_next`: Does it end mid-sentence? (no period, or ends with hyphen)

**INPUT BLOCKS:**
```
{blocks_str}
```

**REQUIRED OUTPUT (JSON only, no explanation):**
{{
  "remove": [list of block indices to remove],
  "order": [reordered block indices - same as input if order is correct],
  "paragraphs": [
    {{
      "block_indices": [0, 1],
      "continues_from_prev": false,
      "continues_to_next": true
    }},
    ...
  ]
}}

**RULES:**
- Only include indices from INPUT BLOCKS in your output
- If order is correct, "order" should be [0, 1, 2, 3, ...] in sequence
- Each paragraph's block_indices must be consecutive in the final order
- Removed blocks should NOT appear in order or paragraphs
"""

    def _call_llm(self, prompt: str) -> Optional[Dict]:
        """Call LLM with retry logic."""
        for attempt in range(self.max_retries):
            try:
                response = self.llm_manager.generate_response(
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert academic document processor. Output only valid JSON."
                        },
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.0,
                    max_tokens=8000  # Allow longer responses for chunks with many blocks
                )

                # Extract JSON from response
                content = response.content.strip()

                # Remove markdown code blocks if present
                if content.startswith('```json'):
                    content = content[7:]
                elif content.startswith('```'):
                    content = content[3:]

                if content.endswith('```'):
                    content = content[:-3]

                content = content.strip()

                result = json.loads(content)

                # Validate response structure
                required_keys = ["remove", "order", "paragraphs"]
                if not all(key in result for key in required_keys):
                    logger.warning(f"Missing keys in LLM response. Retrying...")
                    if attempt < self.max_retries - 1:
                        time.sleep(2)
                    continue

                return result

            except json.JSONDecodeError as e:
                logger.warning(f"JSON decode error: {e}. Retrying...")
                if attempt < self.max_retries - 1:
                    time.sleep(2)

            except Exception as e:
                logger.warning(f"LLM call failed: {e}. Retrying...")
                if attempt < self.max_retries - 1:
                    time.sleep(2)

        return None

    def process_chunk(self, chunk_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process a single chunk."""
        blocks = chunk_data['blocks']
        chunk_id = chunk_data['chunk_id']
        page_range = chunk_data['page_range']

        if not blocks:
            logger.warning(f"No blocks found in chunk {chunk_id}, skipping")
            return None

        # Call LLM
        prompt = self._create_processing_prompt(page_range, blocks)
        result = self._call_llm(prompt)

        if not result:
            logger.error(f"LLM processing failed for chunk {chunk_id} after {self.max_retries} attempts")
            return None

        # Log results
        removed_count = len(result.get('remove', []))
        para_count = len(result.get('paragraphs', []))
        logger.info(f"Chunk {chunk_id}: Removed {removed_count} blocks, grouped into {para_count} paragraphs")

        return {
            "chunk_id": chunk_id,
            "page_range": page_range,
            "pages": chunk_data['pages'],
            "original_block_count": len(blocks),
            "processing_result": result,
            "original_blocks": blocks
        }

    def process_all_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process all chunks."""
        processed_chunks = []

        for chunk in chunks:
            processed = self.process_chunk(chunk)
            if processed:
                processed_chunks.append(processed)

        logger.info(f"Successfully processed {len(processed_chunks)}/{len(chunks)} chunks")
        return processed_chunks


class ParallelLLMChunkProcessor:
    """
    Parallel version of LLMChunkProcessor using asyncio.

    Processes multiple PDF chunks simultaneously while maintaining order.
    Designed for significant performance improvement on large PDFs.
    """

    def __init__(self, llm_manager, max_retries: int = 3, max_concurrent: Optional[int] = None):
        """
        Initialize parallel LLM chunk processor.

        Args:
            llm_manager: LLM manager instance for making calls
            max_retries: Maximum number of retries per chunk
            max_concurrent: Maximum concurrent API calls (None = unlimited)
        """
        self.llm_manager = llm_manager
        self.max_retries = max_retries
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent) if max_concurrent else None

        # Performance metrics
        self.total_chunks_processed = 0
        self.total_api_calls = 0
        self.total_retries = 0
        self.failed_chunks = 0

    def _format_blocks_for_prompt(self, blocks: List[Dict]) -> str:
        """Format blocks for LLM prompt."""
        return "\n".join([
            f"Index: {b['index']}, Column: {b['column']}, Y: {b['y']}, Text: \"{b['text'][:150]}...\""
            if len(b['text']) > 150 else
            f"Index: {b['index']}, Column: {b['column']}, Y: {b['y']}, Text: \"{b['text']}\""
            for b in blocks
        ])

    def _create_processing_prompt(self, page_range: str, blocks: List[Dict]) -> str:
        """
        Create the LLM prompt for page processing.

        IMPORTANT: This is identical to the sequential version.
        """
        blocks_str = self._format_blocks_for_prompt(blocks)

        return f"""You are processing PAGE {page_range} of an academic paper.

**CONTEXT:**
I am giving you a list of text blocks extracted from a PDF using PyMuPDF's get_text("blocks") method. Each block contains:
- **Text content**: The actual text from the PDF
- **Column position**: Whether the block is in the "left" or "right" column (for 2-column layouts)
- **Y-coordinate**: Vertical position on the page (lower Y = higher on page)
- The PDF is processed in chunks (groups of pages, e.g., 4 pages at a time) to manage size - you are seeing one chunk at a time.

**TASK 1 - REMOVE JUNK:**
Identify blocks that are:
- Page headers/footers (e.g., "Page 5", running headers at top of page)
- Page numbers
- Journal names, DOIs, URLs
- Author information, affiliations, email addresses
- Copyright notices
- "Available at..."
- **Tables**
- **Figures**
- **Schemes, Charts, Diagrams**
- **Equations**
- *References and Citations*
- *Appendix or anything that comes after the conclusion*

**IMPORTANT - DO NOT REMOVE:**
- **Section headers** (e.g., "Abstract", "1. Introduction", "2. Methods", "3.1. Participants", "4. Results", "5.1. Altruism")
- Title of the paper

**NOTE ON LINE NUMBERS:**
Some manuscripts/preprints have numbered lines (e.g., "1 This is text", "2 More text"). These line numbers are part of the PDF formatting and cannot be removed. When evaluating content, focus on the actual text content AFTER any line numbers - do not mistake manuscript line numbers for page numbers or treat them as junk to remove.

**TASK 2 - VERIFY ORDER:**
Blocks are pre-sorted in reading order: ALL left column blocks (top→bottom), THEN ALL right column blocks (top→bottom).

ONLY reorder if there's an obvious error (e.g., a section header appearing AFTER its content).

**ABSOLUTE RULE:** ALL "left" column blocks MUST come before ALL "right" column blocks. Never interleave columns.

**TASK 3 - GROUP INTO PARAGRAPHS:**
Group consecutive blocks that clearly belong to the same paragraph:
- Blocks that continue mid-sentence (start with lowercase, no period at end of previous block)
- Blocks separated by hyphenation (previous ends with hyphen like "exper-")
- Blocks that are semantically continuous (same topic, no topic break)

Each paragraph group should be a list of consecutive block indices.

**TASK 4 - CROSS-PAGE CONTINUITY:**
For each paragraph, determine:
- `continues_from_prev`: Does it start mid-sentence? (lowercase start, incomplete thought)
- `continues_to_next`: Does it end mid-sentence? (no period, or ends with hyphen)

**INPUT BLOCKS:**
```
{blocks_str}
```

**REQUIRED OUTPUT (JSON only, no explanation):**
{{
  "remove": [list of block indices to remove],
  "order": [reordered block indices - same as input if order is correct],
  "paragraphs": [
    {{
      "block_indices": [0, 1],
      "continues_from_prev": false,
      "continues_to_next": true
    }},
    ...
  ]
}}

**RULES:**
- Only include indices from INPUT BLOCKS in your output
- If order is correct, "order" should be [0, 1, 2, 3, ...] in sequence
- Each paragraph's block_indices must be consecutive in the final order
- Removed blocks should NOT appear in order or paragraphs
"""

    async def _call_llm_async(self, prompt: str, chunk_id: int) -> Optional[Dict]:
        """
        Call LLM asynchronously with exponential backoff retry logic.

        Args:
            prompt: The LLM prompt
            chunk_id: Chunk identifier for logging

        Returns:
            Parsed JSON response or None if all retries fail
        """
        for attempt in range(self.max_retries):
            try:
                self.total_api_calls += 1

                # Use semaphore if max_concurrent is set
                if self.semaphore:
                    async with self.semaphore:
                        response = await self._make_llm_call(prompt)
                else:
                    response = await self._make_llm_call(prompt)

                # Extract JSON from response
                content = response.content.strip()

                # Remove markdown code blocks if present
                if content.startswith('```json'):
                    content = content[7:]
                elif content.startswith('```'):
                    content = content[3:]

                if content.endswith('```'):
                    content = content[:-3]

                content = content.strip()

                result = json.loads(content)

                # Validate response structure
                required_keys = ["remove", "order", "paragraphs"]
                if not all(key in result for key in required_keys):
                    logger.warning(f"Chunk {chunk_id}: Missing keys in LLM response. Retrying...")
                    if attempt < self.max_retries - 1:
                        self.total_retries += 1
                        delay = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                        await asyncio.sleep(delay)
                    continue

                return result

            except json.JSONDecodeError as e:
                logger.warning(f"Chunk {chunk_id}: JSON decode error: {e}. Retrying...")
                if attempt < self.max_retries - 1:
                    self.total_retries += 1
                    delay = 2 ** attempt
                    await asyncio.sleep(delay)

            except Exception as e:
                logger.warning(f"Chunk {chunk_id}: LLM call failed: {e}. Retrying...")
                if attempt < self.max_retries - 1:
                    self.total_retries += 1
                    delay = 2 ** attempt
                    await asyncio.sleep(delay)

        return None

    async def _make_llm_call(self, prompt: str):
        """
        Make the actual LLM API call.

        This wraps the synchronous LLM manager call in asyncio.to_thread
        to prevent blocking the event loop.
        """
        # Run synchronous LLM call in thread pool to avoid blocking
        return await asyncio.to_thread(
            self.llm_manager.generate_response,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert academic document processor. Output only valid JSON."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=8000  # Allow longer responses for chunks with many blocks
        )

    async def process_chunk_async(self, chunk_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Process a single chunk asynchronously.

        Args:
            chunk_data: Dict containing chunk_id, blocks, page_range, pages

        Returns:
            Processed chunk data or None if processing fails
        """
        blocks = chunk_data['blocks']
        chunk_id = chunk_data['chunk_id']
        page_range = chunk_data['page_range']

        if not blocks:
            logger.warning(f"No blocks found in chunk {chunk_id}, skipping")
            return None

        # Call LLM
        prompt = self._create_processing_prompt(page_range, blocks)
        result = await self._call_llm_async(prompt, chunk_id)

        if not result:
            logger.error(f"LLM processing failed for chunk {chunk_id} after {self.max_retries} attempts")
            self.failed_chunks += 1
            return None

        # Log results
        removed_count = len(result.get('remove', []))
        para_count = len(result.get('paragraphs', []))
        logger.info(f"Chunk {chunk_id}: Removed {removed_count} blocks, grouped into {para_count} paragraphs")

        self.total_chunks_processed += 1

        return {
            "chunk_id": chunk_id,
            "page_range": page_range,
            "pages": chunk_data['pages'],
            "original_block_count": len(blocks),
            "processing_result": result,
            "original_blocks": blocks
        }

    async def process_all_chunks_parallel(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process all chunks in parallel using asyncio.gather().

        Args:
            chunks: List of chunk data dictionaries

        Returns:
            List of processed chunks, sorted by chunk_id to maintain order
        """
        start_time = time.time()

        logger.info(f"Starting parallel processing of {len(chunks)} chunks...")

        # Create async tasks for all chunks
        tasks = [self.process_chunk_async(chunk) for chunk in chunks]

        # Execute all tasks in parallel
        # return_exceptions=True prevents one failure from canceling others
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out None results and exceptions, sort by chunk_id
        processed_chunks = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Chunk processing raised exception: {result}")
                self.failed_chunks += 1
            elif result is not None:
                processed_chunks.append(result)

        # Sort by chunk_id to maintain original order
        processed_chunks.sort(key=lambda x: x['chunk_id'])

        elapsed = time.time() - start_time

        logger.info(f"Parallel processing completed in {elapsed:.2f}s")
        logger.info(f"Successfully processed {len(processed_chunks)}/{len(chunks)} chunks")
        logger.info(f"Total API calls: {self.total_api_calls}, Retries: {self.total_retries}, Failed: {self.failed_chunks}")

        return processed_chunks

    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        return {
            "total_chunks_processed": self.total_chunks_processed,
            "total_api_calls": self.total_api_calls,
            "total_retries": self.total_retries,
            "failed_chunks": self.failed_chunks,
            "success_rate": (
                self.total_chunks_processed / (self.total_chunks_processed + self.failed_chunks)
                if (self.total_chunks_processed + self.failed_chunks) > 0
                else 0.0
            )
        }


class PageStitcher:
    """Stitches processed chunks into final clean documents (Step 4)."""

    def get_block_text(self, blocks: List[Dict], index: int) -> str:
        """Get text from a block by index."""
        for block in blocks:
            if block['index'] == index:
                return block['text']
        return ""

    def stitch_chunks(self, processed_chunks: List[Dict[str, Any]]) -> str:
        """Assemble all chunks into final cleaned text."""
        paragraphs = []
        continuation_buffer = ""  # Holds text from incomplete paragraph

        for chunk_data in sorted(processed_chunks, key=lambda x: x['chunk_id']):
            chunk_id = chunk_data.get('chunk_id', '?')
            page_range = chunk_data.get('page_range', '?')
            result = chunk_data['processing_result']
            blocks = chunk_data['original_blocks']
            removed = set(result.get('remove', []))

            logger.debug(f"Stitching chunk {chunk_id} (pages {page_range}): {len(result.get('paragraphs', []))} paragraphs")

            for para in result.get('paragraphs', []):
                block_indices = para['block_indices']
                continues_from_prev = para.get('continues_from_prev', False)
                continues_to_next = para.get('continues_to_next', False)

                # Get text from blocks (excluding removed ones)
                texts = []
                for idx in block_indices:
                    if idx not in removed:
                        text = self.get_block_text(blocks, idx)
                        if text:
                            texts.append(text)

                if not texts:
                    continue

                # Join blocks within paragraph
                para_text = ' '.join(texts)

                # Handle cross-page continuation
                if continues_from_prev and continuation_buffer:
                    # Append to buffer (continuation from previous page)
                    continuation_buffer += ' ' + para_text
                elif continuation_buffer:
                    # Buffer exists but this para doesn't continue it - save buffer first
                    paragraphs.append(continuation_buffer)
                    continuation_buffer = ""
                    # Then handle current para
                    if continues_to_next:
                        continuation_buffer = para_text
                    else:
                        paragraphs.append(para_text)
                elif continues_from_prev and not continuation_buffer:
                    # Marked as continuation but no buffer - treat as new paragraph
                    if continues_to_next:
                        continuation_buffer = para_text
                    else:
                        paragraphs.append(para_text)
                else:
                    # Normal paragraph
                    if continues_to_next:
                        continuation_buffer = para_text
                    else:
                        paragraphs.append(para_text)

        # Don't forget last buffered paragraph
        if continuation_buffer:
            paragraphs.append(continuation_buffer)

        # Join all paragraphs with double newlines
        cleaned_text = '\n\n'.join(paragraphs)

        logger.info(f"Assembled {len(paragraphs)} paragraphs into final cleaned text ({len(cleaned_text)} chars)")
        return cleaned_text


class PDFCleaner:
    """
    Main PDF cleaner that orchestrates the 4-step pipeline.

    Usage:
        cleaner = PDFCleaner()
        cleaned_text = cleaner.clean_pdf_text(pdf_path, llm_manager, pages_per_chunk=4)
    """

    async def clean_pdf_text(self,
                      pdf_path: Path,
                      llm_manager,
                      pages_per_chunk: int = 1,
                      max_retries: int = 3,
                      enable_reference_detection: bool = True,
                      auto_skip_threshold: Optional[int] = None,
                      force_clean: bool = False,
                      enable_parallel: bool = True,
                      max_concurrent: Optional[int] = None) -> Optional[str]:
        """
        Clean PDF text using LLM-based processing pipeline.

        Args:
            pdf_path: Path to PDF file
            llm_manager: LLM manager for making API calls
            pages_per_chunk: Number of pages to group per chunk (default: 1 for optimal parallel performance)
            max_retries: Maximum retries for LLM calls (default: 3)
            enable_reference_detection: Whether to detect and truncate at references (default: True)
            auto_skip_threshold: Skip cleaning if PDF exceeds this many pages (default: None = no limit)
            force_clean: Force cleaning even if threshold exceeded (default: False)
            enable_parallel: Use parallel processing with asyncio (default: True)
            max_concurrent: Maximum concurrent chunks if parallel enabled (default: None = unlimited)

        Returns:
            Cleaned full text string, or None if cleaning was skipped due to threshold
        """
        logger.info(f"Starting PDF cleaning pipeline for: {pdf_path.name}")
        processing_mode = "PARALLEL" if enable_parallel else "SEQUENTIAL"
        logger.info(f"Configuration: mode={processing_mode}, pages_per_chunk={pages_per_chunk}, max_retries={max_retries}, "
                   f"reference_detection={enable_reference_detection}, auto_skip_threshold={auto_skip_threshold}, "
                   f"force_clean={force_clean}, max_concurrent={max_concurrent}")

        try:
            # Step 1: Extract blocks from PDF
            logger.info("Step 1/5: Extracting PDF blocks...")
            extractor = PDFBlockExtractor()
            pages_data = extractor.process_pdf(pdf_path)

            # Step 1.5: Detect and truncate at references with LLM verification
            if enable_reference_detection:
                logger.info("Step 1.5/5: Detecting reference sections with LLM verification...")
                detector = ReferenceSectionDetector()
                cutoff_page = detector.detect_cutoff_page(pages_data, llm_manager)

                if cutoff_page:
                    pages_data = detector.truncate_pages(pages_data, cutoff_page)
                    logger.info(f"✅ Reference detection successful - processing {len(pages_data)} pages instead of full document")
                else:
                    logger.info("ℹ️ No reference section detected - processing entire document")
            else:
                logger.info("Step 1.5/5: Reference detection disabled - processing entire document")

            # Step 1.6: Check page threshold (NEW!)
            current_page_count = len(pages_data)
            if auto_skip_threshold and current_page_count > auto_skip_threshold and not force_clean:
                # Auto-skip triggered - skip LLM cleaning to save time and cost
                logger.warning(f"\n{'='*80}")
                logger.warning(f"⚠️  AUTO-SKIP TRIGGERED")
                logger.warning(f"{'='*80}")
                logger.warning(f"PDF has {current_page_count} pages (after reference removal)")
                logger.warning(f"Threshold: {auto_skip_threshold} pages")
                logger.warning(f"Estimated cleaning time: ~{current_page_count * 8} seconds ({current_page_count * 8 / 60:.1f} minutes)")
                logger.warning(f"Estimated cost: ~${current_page_count * 0.001:.2f}")
                logger.warning(f"")
                logger.warning(f"✂️ Skipping LLM cleaning to save time and cost.")
                logger.warning(f"💡 To force cleaning, use force_clean=True parameter")
                logger.warning(f"ℹ️  References have still been removed")
                logger.warning(f"ℹ️  Text will still be extracted and embedded (searchable)")
                logger.warning(f"{'='*80}\n")
                return None
            elif auto_skip_threshold and current_page_count > auto_skip_threshold and force_clean:
                # Force clean override
                logger.info(f"\n{'='*80}")
                logger.info(f"🔧 FORCE CLEAN ENABLED")
                logger.info(f"{'='*80}")
                logger.info(f"PDF has {current_page_count} pages (exceeds threshold of {auto_skip_threshold})")
                logger.info(f"force_clean=True - proceeding with cleaning anyway")
                logger.info(f"Estimated time: ~{current_page_count * 8} seconds ({current_page_count * 8 / 60:.1f} minutes)")
                logger.info(f"Estimated cost: ~${current_page_count * 0.001:.2f}")
                logger.info(f"{'='*80}\n")

            # Step 2: Create chunks
            logger.info(f"Step 2/5: Creating chunks ({pages_per_chunk} pages per chunk)...")
            chunk_creator = ChunkCreator()
            chunks = chunk_creator.create_chunks(pages_data, pages_per_chunk)

            # Step 3: LLM processing (parallel or sequential)
            if enable_parallel:
                logger.info(f"Step 3/5: Processing chunks with LLM (PARALLEL mode, {len(chunks)} chunks)...")
                llm_processor = ParallelLLMChunkProcessor(llm_manager, max_retries, max_concurrent)
                processed_chunks = await llm_processor.process_all_chunks_parallel(chunks)

                # Log performance metrics
                metrics = llm_processor.get_metrics()
                logger.info(f"Parallel processing metrics: {metrics['success_rate']:.1%} success rate, "
                           f"{metrics['total_retries']} retries, {metrics['failed_chunks']} failed")
            else:
                logger.info(f"Step 3/5: Processing chunks with LLM (SEQUENTIAL mode, {len(chunks)} chunks)...")
                llm_processor = LLMChunkProcessor(llm_manager, max_retries)
                processed_chunks = llm_processor.process_all_chunks(chunks)

            if not processed_chunks:
                raise ValueError("LLM processing failed - no chunks were successfully processed")

            # Step 4: Stitch together
            logger.info("Step 4/5: Stitching cleaned chunks together...")
            stitcher = PageStitcher()
            cleaned_text = stitcher.stitch_chunks(processed_chunks)

            logger.info(f"✅ PDF cleaning complete: {len(cleaned_text)} characters")
            return cleaned_text

        except Exception as e:
            logger.error(f"PDF cleaning failed: {e}")
            raise
