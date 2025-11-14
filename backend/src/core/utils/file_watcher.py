import time
import logging
from pathlib import Path
from typing import Callable, Dict, Any, Optional
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileCreatedEvent, FileMovedEvent
import threading
from queue import Queue

logger = logging.getLogger(__name__)


class PaperFileHandler(FileSystemEventHandler):
    """File system event handler for monitoring new papers."""
    
    def __init__(self, callback: Callable[[Path], None]):
        """
        Initialize file handler.
        
        Args:
            callback: Function to call when new PDF is detected
        """
        super().__init__()
        self.callback = callback
        self.processing_queue = Queue()
        self.processing_thread = None
        self.stop_processing = False
        
        # Start processing thread
        self.start_processing_thread()
    
    def on_created(self, event):
        """Handle file creation events."""
        if not event.is_directory and self._is_pdf_file(event.src_path):
            logger.info(f"New PDF detected: {event.src_path}")
            self._queue_for_processing(event.src_path)
    
    def on_moved(self, event):
        """Handle file move events (e.g., downloads completing)."""
        if not event.is_directory and self._is_pdf_file(event.dest_path):
            logger.info(f"PDF moved to watched directory: {event.dest_path}")
            self._queue_for_processing(event.dest_path)
    
    def _is_pdf_file(self, file_path: str) -> bool:
        """Check if file is a PDF."""
        return Path(file_path).suffix.lower() == '.pdf'
    
    def _queue_for_processing(self, file_path: str):
        """Add file to processing queue."""
        self.processing_queue.put(file_path)
    
    def start_processing_thread(self):
        """Start the background processing thread."""
        if self.processing_thread is None or not self.processing_thread.is_alive():
            self.stop_processing = False
            self.processing_thread = threading.Thread(target=self._process_files, daemon=True)
            self.processing_thread.start()
            logger.info("Started file processing thread")
    
    def stop_processing_thread(self):
        """Stop the background processing thread."""
        self.stop_processing = True
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=5)
            logger.info("Stopped file processing thread")
    
    def _process_files(self):
        """Background thread for processing files."""
        while not self.stop_processing:
            try:
                # Wait for files with timeout
                if not self.processing_queue.empty():
                    file_path = self.processing_queue.get(timeout=1)
                    
                    # Wait a bit to ensure file is fully written
                    time.sleep(2)
                    
                    # Check if file still exists and is readable
                    path = Path(file_path)
                    if path.exists() and path.stat().st_size > 0:
                        try:
                            self.callback(path)
                        except Exception as e:
                            logger.error(f"Error processing file {file_path}: {e}")
                    else:
                        logger.warning(f"File not ready or missing: {file_path}")
                else:
                    time.sleep(1)  # No files to process, wait
                    
            except Exception as e:
                logger.error(f"Error in file processing thread: {e}")
                time.sleep(1)


class FileWatcher:
    """Monitors directories for new PDF files and triggers processing."""
    
    def __init__(self, 
                 watch_directories: Dict[str, str],
                 process_callback: Callable[[Path, str], None],
                 config_manager = None,
                 llm_manager = None,
                 new_papers_path: str = None,
                 workflow_coordinator = None):
        """
        Initialize file watcher.
        
        Args:
            watch_directories: Dict mapping directory names to paths
            process_callback: Function to call with (file_path, directory_type)
            config_manager: Configuration manager for settings
            llm_manager: LLM manager for academic verification
            new_papers_path: Path to NEW_PAPERS directory for file moving
            workflow_coordinator: Workflow coordinator for processing papers
        """
        self.watch_directories = watch_directories
        self.process_callback = process_callback
        self.observers = {}
        self.handlers = {}
        
        # Store managers for thread-safe access
        self.config_manager = config_manager
        self.llm_manager = llm_manager
        self.new_papers_path = Path(new_papers_path) if new_papers_path else None
        self.workflow_coordinator = workflow_coordinator
        
        # Snapshot of files in Downloads folder when monitoring starts
        self.initial_downloads_snapshot = set()
        
        # Validate directories and create initial snapshot
        for name, path in watch_directories.items():
            path_obj = Path(path)
            if not path_obj.exists():
                path_obj.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created watch directory: {path}")
        
        # Create snapshot of Downloads folder to avoid processing existing files
        self._create_initial_snapshot()
    
    def _create_initial_snapshot(self):
        """Create snapshot of files currently in Downloads folder."""
        for name, directory in self.watch_directories.items():
            if name == "downloads":
                download_path = Path(directory)
                if download_path.exists():
                    existing_pdfs = {f.name for f in download_path.glob("*.pdf")}
                    self.initial_downloads_snapshot = existing_pdfs
                    logger.info(f"Downloads snapshot created: {len(existing_pdfs)} existing PDFs will be ignored")
                    if existing_pdfs:
                        logger.debug(f"Existing files in Downloads: {list(existing_pdfs)}")
                else:
                    logger.info("Downloads folder does not exist - empty snapshot created")
                break  # Only need to check downloads folder
    
    def get_downloads_snapshot_info(self) -> dict:
        """Get information about the Downloads folder snapshot."""
        return {
            "snapshot_size": len(self.initial_downloads_snapshot),
            "ignored_files": list(self.initial_downloads_snapshot),
            "has_downloads_monitoring": "downloads" in self.watch_directories
        }
    
    def start_watching(self):
        """Start monitoring all configured directories."""
        for name, directory in self.watch_directories.items():
            self._start_watching_directory(name, directory)
        
        logger.info(f"Started watching {len(self.watch_directories)} directories")
    
    def stop_watching(self):
        """Stop monitoring all directories."""
        for name, observer in self.observers.items():
            observer.stop()
            observer.join()
            
            # Stop processing threads
            if name in self.handlers:
                self.handlers[name].stop_processing_thread()
        
        self.observers.clear()
        self.handlers.clear()
        logger.info("Stopped all file watchers")
    
    def _start_watching_directory(self, name: str, directory: str):
        """Start watching a specific directory."""
        def callback(file_path: Path):
            # Pass additional context including file watcher instance for snapshot access
            self.process_callback(file_path, name, self)
        
        # Create handler
        handler = PaperFileHandler(callback)
        self.handlers[name] = handler
        
        # Create observer
        observer = Observer()
        observer.schedule(handler, directory, recursive=False)
        observer.start()
        
        self.observers[name] = observer
        logger.info(f"Started watching directory '{name}': {directory}")
    
    def add_directory(self, name: str, directory: str):
        """Add a new directory to watch."""
        if name in self.watch_directories:
            logger.warning(f"Directory '{name}' already being watched")
            return
        
        # Create directory if it doesn't exist
        path_obj = Path(directory)
        if not path_obj.exists():
            path_obj.mkdir(parents=True, exist_ok=True)
        
        self.watch_directories[name] = directory
        self._start_watching_directory(name, directory)
    
    def remove_directory(self, name: str):
        """Stop watching a specific directory."""
        if name not in self.observers:
            logger.warning(f"Directory '{name}' not being watched")
            return
        
        # Stop observer
        self.observers[name].stop()
        self.observers[name].join()
        del self.observers[name]
        
        # Stop handler
        if name in self.handlers:
            self.handlers[name].stop_processing_thread()
            del self.handlers[name]
        
        # Remove from config
        if name in self.watch_directories:
            del self.watch_directories[name]
        
        logger.info(f"Stopped watching directory '{name}'")
    
    def get_watched_directories(self) -> Dict[str, str]:
        """Get currently watched directories."""
        return self.watch_directories.copy()
    
    def is_watching(self) -> bool:
        """Check if any directories are being watched."""
        return len(self.observers) > 0
    
    def scan_existing_files(self, directory_name: str = None):
        """Scan for existing PDF files in watched directories."""
        directories_to_scan = {}
        
        if directory_name:
            if directory_name in self.watch_directories:
                directories_to_scan[directory_name] = self.watch_directories[directory_name]
            else:
                logger.error(f"Directory '{directory_name}' not being watched")
                return
        else:
            directories_to_scan = self.watch_directories
        
        for name, directory in directories_to_scan.items():
            path_obj = Path(directory)
            pdf_files = list(path_obj.glob("*.pdf"))
            
            logger.info(f"Scanning {len(pdf_files)} existing PDF files in '{name}'")
            
            for pdf_file in pdf_files:
                try:
                    self.process_callback(pdf_file, name)
                except Exception as e:
                    logger.error(f"Error processing existing file {pdf_file}: {e}")


class DirectoryManager:
    """Manages the directory structure for organizing papers."""
    
    def __init__(self, base_path: str):
        """
        Initialize directory manager.
        
        Args:
            base_path: Base path for all paper directories
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # Standard directories
        self.directories = {
            "new_papers": self.base_path / "_NEW_PAPERS",
            "processing": self.base_path / "_PROCESSING", 
            "papers": self.base_path / "PAPERS",
            "failed": self.base_path / "_FAILED"
        }
        
        # Create all directories
        for directory in self.directories.values():
            directory.mkdir(parents=True, exist_ok=True)
    
    def get_directory(self, name: str) -> Path:
        """Get path to a specific directory."""
        if name not in self.directories:
            raise ValueError(f"Unknown directory: {name}. Available: {list(self.directories.keys())}")
        return self.directories[name]
    
    def move_file(self, source: Path, destination_dir: str, new_filename: str = None) -> Path:
        """
        Move file to a different directory.
        
        Args:
            source: Source file path
            destination_dir: Destination directory name
            new_filename: Optional new filename
            
        Returns:
            New file path
        """
        dest_directory = self.get_directory(destination_dir)
        
        if new_filename:
            dest_path = dest_directory / new_filename
        else:
            dest_path = dest_directory / source.name
        
        # Handle filename conflicts
        counter = 1
        original_dest = dest_path
        while dest_path.exists():
            stem = original_dest.stem
            suffix = original_dest.suffix
            dest_path = dest_directory / f"{stem}_{counter}{suffix}"
            counter += 1
        
        source.rename(dest_path)
        logger.info(f"Moved file: {source} -> {dest_path}")
        return dest_path
    
    def get_directory_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all directories."""
        stats = {}
        
        for name, directory in self.directories.items():
            pdf_files = list(directory.glob("*.pdf"))
            total_size = sum(f.stat().st_size for f in pdf_files)
            
            stats[name] = {
                "path": str(directory),
                "file_count": len(pdf_files),
                "total_size_mb": round(total_size / (1024 * 1024), 2),
                "files": [f.name for f in pdf_files]
            }
        
        return stats
    
    def cleanup_empty_directories(self):
        """Remove empty directories (except standard ones)."""
        for directory in self.base_path.rglob("*"):
            if (directory.is_dir() and 
                directory not in self.directories.values() and
                not any(directory.iterdir())):
                try:
                    directory.rmdir()
                    logger.info(f"Removed empty directory: {directory}")
                except OSError:
                    pass