import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass
from scihub import SciHub

logger = logging.getLogger(__name__)


@dataclass
class DownloadResult:
    """Container for paper download results."""
    success: bool
    file_path: Optional[str] = None
    error_message: Optional[str] = None
    doi: Optional[str] = None
    title: Optional[str] = None


class SciHubClient:
    """Client for downloading academic papers using SciHub."""
    
    def __init__(self, download_directory: str):
        """
        Initialize SciHub client.
        
        Args:
            download_directory: Directory to save downloaded papers
        """
        self.download_directory = Path(download_directory)
        self.download_directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize SciHub
        self.sh = SciHub()
        
        logger.info(f"Initialized SciHub client with download directory: {download_directory}")
    
    def download_paper(self, 
                      identifier: str, 
                      custom_filename: str = None) -> DownloadResult:
        """
        Download a paper using DOI, title, or other identifier.
        
        Args:
            identifier: DOI, paper title, or other identifier
            custom_filename: Optional custom filename (without extension)
            
        Returns:
            DownloadResult with download status and file path
        """
        try:
            logger.info(f"Attempting to download paper: {identifier}")
            
            # The scihub library fetch method returns the PDF content
            # We need to handle the download ourselves
            pdf_content = self.sh.fetch(identifier)
            
            if pdf_content:
                # Determine filename
                if custom_filename:
                    filename = f"{custom_filename}.pdf"
                else:
                    # Generate filename from identifier
                    safe_identifier = "".join(c for c in identifier if c.isalnum() or c in "._-")[:50]
                    filename = f"{safe_identifier}.pdf"
                
                file_path = self.download_directory / filename
                
                # Write PDF content to file
                with open(file_path, 'wb') as f:
                    f.write(pdf_content)
                
                logger.info(f"Successfully downloaded paper to: {file_path}")
                
                return DownloadResult(
                    success=True,
                    file_path=str(file_path),
                    doi=identifier if self._is_doi(identifier) else None,
                    title=identifier if not self._is_doi(identifier) else None
                )
            else:
                logger.warning(f"No content received for paper: {identifier}")
                return DownloadResult(
                    success=False,
                    error_message="No content received from SciHub"
                )
                
        except Exception as e:
            logger.error(f"Error downloading paper {identifier}: {e}")
            return DownloadResult(
                success=False,
                error_message=str(e)
            )
    
    def batch_download_papers(self, 
                            identifiers: list, 
                            custom_filenames: list = None) -> Dict[str, DownloadResult]:
        """
        Download multiple papers in batch.
        
        Args:
            identifiers: List of DOIs, titles, or other identifiers
            custom_filenames: Optional list of custom filenames
            
        Returns:
            Dictionary mapping identifiers to DownloadResults
        """
        results = {}
        
        if custom_filenames and len(custom_filenames) != len(identifiers):
            logger.warning("Custom filenames list length doesn't match identifiers list")
            custom_filenames = None
        
        for i, identifier in enumerate(identifiers):
            custom_name = custom_filenames[i] if custom_filenames else None
            
            result = self.download_paper(identifier, custom_name)
            results[identifier] = result
            
            # Small delay between downloads to be respectful
            import time
            time.sleep(1)
        
        successful_downloads = sum(1 for r in results.values() if r.success)
        logger.info(f"Batch download completed: {successful_downloads}/{len(identifiers)} successful")
        
        return results
    
    def search_and_download(self, 
                           title: str, 
                           authors: list = None,
                           year: int = None) -> DownloadResult:
        """
        Search for a paper and download it.
        
        Args:
            title: Paper title
            authors: List of author names
            year: Publication year
            
        Returns:
            DownloadResult with download status
        """
        # Try different search strategies
        search_queries = [title]
        
        if authors and year:
            # Try with first author and year
            first_author = authors[0].split()[-1]  # Get last name
            search_queries.append(f"{title} {first_author} {year}")
        
        if year:
            search_queries.append(f"{title} {year}")
        
        for query in search_queries:
            result = self.download_paper(query)
            if result.success:
                return result
        
        # If all searches failed
        return DownloadResult(
            success=False,
            error_message=f"Could not find or download paper: {title}"
        )
    
    def _is_doi(self, identifier: str) -> bool:
        """Check if identifier is a DOI."""
        return identifier.startswith("10.") or "doi.org" in identifier.lower()
    
    def test_connection(self) -> bool:
        """Test SciHub connection by attempting a simple download."""
        try:
            # Try to download a well-known open access paper
            test_doi = "10.1371/journal.pone.0000001"  # First PLOS ONE paper
            
            # Don't actually download, just test if SciHub is accessible
            # We'll check if we can reach the service
            import requests
            test_url = "https://sci-hub.se/"
            
            response = requests.get(test_url, timeout=10)
            return response.status_code == 200
            
        except Exception as e:
            logger.error(f"SciHub connection test failed: {e}")
            return False
    
    def get_download_statistics(self) -> Dict[str, Any]:
        """Get statistics about downloaded papers."""
        try:
            pdf_files = list(self.download_directory.glob("*.pdf"))
            total_size = sum(f.stat().st_size for f in pdf_files)
            
            return {
                "total_papers": len(pdf_files),
                "total_size_mb": round(total_size / (1024 * 1024), 2),
                "download_directory": str(self.download_directory),
                "files": [f.name for f in pdf_files]
            }
        except Exception as e:
            logger.error(f"Error getting download statistics: {e}")
            return {}
    
    def cleanup_failed_downloads(self):
        """Clean up any partial or failed download files."""
        try:
            # Remove any temporary or partial files
            for pattern in ["*.tmp", "*.part", "*.download"]:
                for temp_file in self.download_directory.glob(pattern):
                    temp_file.unlink()
                    logger.info(f"Cleaned up temporary file: {temp_file}")
            
            # Remove empty or very small PDF files (likely failed downloads)
            for pdf_file in self.download_directory.glob("*.pdf"):
                if pdf_file.stat().st_size < 1024:  # Less than 1KB
                    pdf_file.unlink()
                    logger.info(f"Removed small/empty PDF: {pdf_file}")
                    
        except Exception as e:
            logger.error(f"Error cleaning up failed downloads: {e}")
    
    def set_download_directory(self, new_directory: str):
        """Change the download directory."""
        self.download_directory = Path(new_directory)
        self.download_directory.mkdir(parents=True, exist_ok=True)
        logger.info(f"Changed download directory to: {new_directory}")