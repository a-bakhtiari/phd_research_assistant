from .perplexity_client import PerplexityClient
from .semantic_scholar_client import SemanticScholarClient

# Optional imports for clients that may have missing dependencies
try:
    from .scihub_client import SciHubClient
except ImportError:
    SciHubClient = None

try:
    from .browser_download_client import BrowserDownloadClient
except ImportError:
    BrowserDownloadClient = None

__all__ = ["PerplexityClient", "SemanticScholarClient", "SciHubClient", "BrowserDownloadClient"]