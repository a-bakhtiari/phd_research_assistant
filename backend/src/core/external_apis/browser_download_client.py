import os
import webbrowser
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
import time
import urllib.parse

logger = logging.getLogger(__name__)


@dataclass
class DownloadLink:
    """Container for download link information."""
    url: str
    source: str  # "google_scholar", "arxiv", "publisher", "doi"
    type: str    # "pdf", "html", "preprint"
    title: str
    confidence: float  # 0.0 to 1.0


@dataclass
class DownloadResult:
    """Container for download operation results."""
    success: bool
    links_found: List[DownloadLink]
    browser_opened: bool = False
    file_path: Optional[str] = None
    error_message: Optional[str] = None
    doi: Optional[str] = None
    title: Optional[str] = None


class BrowserDownloadClient:
    """Client for browser-assisted paper downloading with automatic file monitoring."""
    
    def __init__(self, downloads_folder: str, new_papers_folder: str):
        """
        Initialize browser download client.
        
        Args:
            downloads_folder: User's default downloads folder to monitor
            new_papers_folder: Target folder for processed papers
        """
        self.downloads_folder = Path(downloads_folder)
        self.new_papers_folder = Path(new_papers_folder)
        
        # Ensure both folders exist
        self.downloads_folder.mkdir(parents=True, exist_ok=True)
        self.new_papers_folder.mkdir(parents=True, exist_ok=True)
        
        # Track download attempts for monitoring
        self.pending_downloads = {}  # title -> timestamp
        
        logger.info(f"Initialized browser download client")
        logger.info(f"  Downloads folder: {downloads_folder}")
        logger.info(f"  New papers folder: {new_papers_folder}")
    
    def find_download_links(self, 
                          title: str, 
                          authors: List[str] = None,
                          doi: str = None,
                          year: int = None) -> List[DownloadLink]:
        """
        Find download links for a paper from multiple sources.
        
        Args:
            title: Paper title
            authors: List of authors
            doi: DOI if available
            year: Publication year
            
        Returns:
            List of DownloadLink objects
        """
        # Just one simple Google search with paper title and author  
        google_query = self._build_google_search_query(title, authors, year)
        
        return [DownloadLink(
            url=google_query,
            source="google_search",
            type="search", 
            title=f"Google search: {title}",
            confidence=1.0
        )]
    
    def download_paper(self, 
                      title: str,
                      authors: List[str] = None,
                      doi: str = None,
                      year: int = None,
                      auto_open_browser: bool = True) -> DownloadResult:
        """
        Initiate paper download by finding links and optionally opening browser.
        
        Args:
            title: Paper title
            authors: List of authors
            doi: DOI if available
            year: Publication year
            auto_open_browser: Whether to automatically open browser
            
        Returns:
            DownloadResult with links and status
        """
        try:
            logger.info(f"Finding download links for: {title}")
            
            # Find all available download links
            links = self.find_download_links(title, authors, doi, year)
            
            if not links:
                return DownloadResult(
                    success=False,
                    links_found=[],
                    error_message="No download links found"
                )
            
            browser_opened = False
            if auto_open_browser and links:
                # Open the highest confidence link in browser
                best_link = links[0]
                logger.info(f"Opening browser for: {best_link.url}")
                webbrowser.open(best_link.url)
                browser_opened = True
                
                # Track this download attempt
                self.pending_downloads[title] = time.time()
            
            return DownloadResult(
                success=True,
                links_found=links,
                browser_opened=browser_opened,
                title=title,
                doi=doi
            )
            
        except Exception as e:
            logger.error(f"Error initiating download for {title}: {e}")
            return DownloadResult(
                success=False,
                links_found=[],
                error_message=str(e)
            )
    
    def monitor_downloads_folder(self, check_interval: int = 5, timeout: int = 300) -> List[str]:
        """
        Monitor downloads folder for new PDF files and move them to new_papers folder.
        
        Args:
            check_interval: Seconds between checks
            timeout: Maximum time to monitor
            
        Returns:
            List of files that were moved
        """
        moved_files = []
        start_time = time.time()
        initial_files = set(f.name for f in self.downloads_folder.glob("*.pdf"))
        
        logger.info(f"Monitoring downloads folder for new PDFs...")
        logger.info(f"Initial PDF files: {len(initial_files)}")
        
        while time.time() - start_time < timeout:
            try:
                current_files = set(f.name for f in self.downloads_folder.glob("*.pdf"))
                new_files = current_files - initial_files
                
                for filename in new_files:
                    source_path = self.downloads_folder / filename
                    target_path = self.new_papers_folder / filename
                    
                    # Make sure file is completely downloaded (not being written)
                    if self._is_file_ready(source_path):
                        try:
                            # Move file to new papers folder
                            source_path.rename(target_path)
                            moved_files.append(str(target_path))
                            logger.info(f"Moved downloaded paper: {filename}")
                            
                            # Remove from initial files so we don't move it again
                            initial_files.add(filename)
                            
                        except Exception as e:
                            logger.error(f"Error moving file {filename}: {e}")
                
                if new_files:
                    logger.info(f"Found {len(new_files)} new files: {list(new_files)}")
                
                time.sleep(check_interval)
                
            except Exception as e:
                logger.error(f"Error monitoring downloads: {e}")
                break
        
        logger.info(f"Download monitoring completed. Moved {len(moved_files)} files.")
        return moved_files
    
    def _is_file_ready(self, file_path: Path, stable_time: int = 2) -> bool:
        """Check if file is completely downloaded by monitoring size stability."""
        try:
            if not file_path.exists():
                return False
            
            initial_size = file_path.stat().st_size
            time.sleep(stable_time)
            final_size = file_path.stat().st_size
            
            return initial_size == final_size and final_size > 1024  # At least 1KB
        except:
            return False
    
    def _build_google_search_query(self, title: str, authors: List[str] = None, year: int = None) -> str:
        """Build regular Google search URL with paper title and authors."""
        query_parts = [f'"{title}"']
        
        # Add first author if available (keeps it simple)
        if authors and len(authors) > 0 and authors[0] != "Unknown":
            author = authors[0].replace(" et al.", "")  # Remove et al. suffix
            query_parts.append(author)
        
        query = " ".join(query_parts)
        encoded_query = urllib.parse.quote(query)
        
        return f"https://www.google.com/search?q={encoded_query}"
    
    def _build_arxiv_query(self, title: str, authors: List[str] = None) -> str:
        """Build arXiv search URL."""
        query_parts = [f'ti:"{title}"']
        
        if authors:
            # Add first author
            author_last = authors[0].split()[-1]  # Get last name
            query_parts.append(f'au:"{author_last}"')
        
        query = " AND ".join(query_parts)
        encoded_query = urllib.parse.quote(query)
        
        return f"https://arxiv.org/search/?query={encoded_query}&searchtype=all"
    
    def _build_semantic_scholar_query(self, title: str, authors: List[str] = None) -> str:
        """Build Semantic Scholar search URL."""
        encoded_title = urllib.parse.quote(title)
        return f"https://www.semanticscholar.org/search?q={encoded_title}"
    
    def _build_pubmed_query(self, title: str, authors: List[str] = None) -> str:
        """Build PubMed search URL."""
        query_parts = [f'"{title}"[Title]']
        
        if authors:
            author_last = authors[0].split()[-1]
            query_parts.append(f'"{author_last}"[Author]')
        
        query = " AND ".join(query_parts)
        encoded_query = urllib.parse.quote(query)
        
        return f"https://pubmed.ncbi.nlm.nih.gov/?term={encoded_query}"
    
    def _might_be_arxiv_paper(self, title: str, authors: List[str] = None) -> bool:
        """Heuristic to determine if paper might be on arXiv."""
        cs_ml_keywords = [
            "neural", "machine learning", "deep learning", "artificial intelligence",
            "computer vision", "natural language", "nlp", "transformer", "algorithm",
            "optimization", "reinforcement learning", "CNN", "RNN", "LSTM", "GAN"
        ]
        
        title_lower = title.lower()
        return any(keyword in title_lower for keyword in cs_ml_keywords)
    
    def _might_be_pubmed_paper(self, title: str, authors: List[str] = None) -> bool:
        """Heuristic to determine if paper might be biomedical."""
        biomedical_keywords = [
            "patient", "clinical", "medical", "disease", "treatment", "therapy",
            "diagnosis", "biomarker", "gene", "protein", "cell", "cancer",
            "drug", "pharmaceutical", "health", "medicine", "biology"
        ]
        
        title_lower = title.lower()
        return any(keyword in title_lower for keyword in biomedical_keywords)
    
    def get_pending_downloads(self) -> Dict[str, float]:
        """Get list of papers that were opened for download."""
        return self.pending_downloads.copy()
    
    def clear_pending_downloads(self):
        """Clear the pending downloads list."""
        self.pending_downloads.clear()
    
    def create_download_summary(self, results: List[DownloadResult]) -> str:
        """Create a formatted summary of download attempts."""
        if not results:
            return "No download attempts made."
        
        summary = f"📥 DOWNLOAD SUMMARY ({len(results)} papers)\n"
        summary += "=" * 50 + "\n\n"
        
        for i, result in enumerate(results, 1):
            status = "✅" if result.success else "❌"
            summary += f"{i}. {status} {result.title}\n"
            
            if result.success:
                summary += f"   🔗 Found {len(result.links_found)} download links:\n"
                for link in result.links_found[:3]:  # Show top 3 links
                    summary += f"      • {link.source}: {link.title}\n"
                
                if result.browser_opened:
                    summary += f"   🌐 Browser opened for best link\n"
            else:
                summary += f"   ❌ Error: {result.error_message}\n"
            
            summary += "\n"
        
        summary += "💡 Instructions:\n"
        summary += "1. Use the opened browser tabs to download papers\n"
        summary += "2. Downloaded PDFs will automatically move to new_papers folder\n"
        summary += "3. Papers in new_papers folder will be processed by the system\n"
        
        return summary