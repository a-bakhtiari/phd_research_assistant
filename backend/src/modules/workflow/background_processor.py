import logging
import threading
import time
import pickle
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import json
from queue import Queue, Empty

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Status of a background task."""
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class BackgroundTask:
    """Represents a background processing task."""
    task_id: str
    task_type: str
    file_path: Optional[Path] = None  # Not required for recommendation tasks
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    status: TaskStatus = TaskStatus.QUEUED
    progress: float = 0.0
    error_message: Optional[str] = None
    result: Optional[Any] = None
    retry_count: int = 0
    max_retries: int = 2
    # Additional parameters for recommendation tasks
    document_paths: Optional[List[Path]] = None
    search_query: Optional[str] = None
    recommendation_params: Optional[Dict[str, Any]] = None


@dataclass
class ProcessingStats:
    """Statistics for background processing."""
    total_tasks: int = 0
    queued_tasks: int = 0
    processing_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    average_processing_time: float = 0.0
    last_activity: Optional[datetime] = None
    queue_start_time: datetime = field(default_factory=datetime.now)


class BackgroundProcessingQueue:
    """
    Background processing queue that operates independently of the UI.
    Handles paper processing in dedicated worker threads.
    """
    
    def __init__(self, 
                 project_root: Path,
                 num_workers: int = 2,
                 auto_start: bool = True):
        """
        Initialize the background processing queue.
        
        Args:
            project_root: Root directory for the project
            num_workers: Number of worker threads
            auto_start: Whether to start workers immediately
        """
        self.project_root = Path(project_root)
        self.num_workers = num_workers
        
        # Thread-safe queue and data structures
        self.task_queue = Queue()
        self.tasks: Dict[str, BackgroundTask] = {}
        self.stats = ProcessingStats()
        
        # Thread management
        self.workers: List[threading.Thread] = []
        self.shutdown_event = threading.Event()
        self.lock = threading.Lock()
        
        # Persistence
        self.queue_file = self.project_root / "tmp" / "processing_queue.pkl"
        self.queue_file.parent.mkdir(exist_ok=True)
        
        # Processors - will be set by integration coordinator
        self.paper_processor = None
        self.workflow_coordinator = None
        self.document_processor = None
        self.paper_recommender = None
        
        # Status callback for UI updates
        self.status_callback: Optional[Callable[[ProcessingStats], None]] = None
        
        # Load persisted queue
        self._load_queue()
        
        if auto_start:
            self.start_workers()
            
        logger.info(f"Background processing queue initialized with {num_workers} workers")
    
    def set_processors(self, paper_processor, workflow_coordinator, document_processor=None, paper_recommender=None):
        """Set the processors that will handle actual work."""
        self.paper_processor = paper_processor
        self.workflow_coordinator = workflow_coordinator
        self.document_processor = document_processor
        self.paper_recommender = paper_recommender
        
        # Log processor setup status
        logger.info("Background queue processors set:")
        logger.info(f"  Paper Processor: {type(paper_processor).__name__ if paper_processor else 'None'}")
        logger.info(f"  Workflow Coordinator: {type(workflow_coordinator).__name__ if workflow_coordinator else 'None'}")
        logger.info(f"  Document Processor: {type(document_processor).__name__ if document_processor else 'None'}")
        logger.info(f"  Paper Recommender: {type(paper_recommender).__name__ if paper_recommender else 'None'}")
        
        if not document_processor or not paper_recommender:
            logger.warning("Document processor or paper recommender not set - recommendation tasks will fail until these are provided via set_recommendation_processors()")
    
    def set_status_callback(self, callback: Callable[[ProcessingStats], None]):
        """Set callback for status updates."""
        self.status_callback = callback
    
    def start_workers(self):
        """Start the background worker threads."""
        if self.workers:
            logger.warning("Workers already started")
            return
            
        self.shutdown_event.clear()
        
        for i in range(self.num_workers):
            worker = threading.Thread(
                target=self._worker_loop,
                name=f"BackgroundWorker-{i+1}",
                daemon=True
            )
            worker.start()
            self.workers.append(worker)
            
        logger.info(f"Started {len(self.workers)} background worker threads")
    
    def stop_workers(self):
        """Stop all background worker threads."""
        logger.info("Stopping background workers...")
        self.shutdown_event.set()
        
        # Wait for workers to finish current tasks
        for worker in self.workers:
            worker.join(timeout=10.0)  # Wait up to 10 seconds
            
        self.workers.clear()
        self._save_queue()
        logger.info("Background workers stopped")
    
    def add_paper_task(self, file_path: Path, task_type: str = "process_paper") -> str:
        """
        Add a paper processing task to the queue.
        
        Args:
            file_path: Path to the paper file
            task_type: Type of processing task
            
        Returns:
            Task ID for tracking
        """
        task_id = f"{task_type}_{file_path.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        task = BackgroundTask(
            task_id=task_id,
            task_type=task_type,
            file_path=file_path
        )
        
        with self.lock:
            self.tasks[task_id] = task
            self.task_queue.put(task_id)
            self.stats.total_tasks += 1
            self.stats.queued_tasks += 1
            self.stats.last_activity = datetime.now()
        
        self._save_queue()
        self._notify_status_update()
        
        logger.info(f"Added paper task: {file_path.name} -> {task_id}")
        return task_id
    
    def add_recommendation_task(self, 
                              task_type: str, 
                              document_paths: Optional[List[Path]] = None,
                              search_query: str = "",
                              recommendation_params: Optional[Dict[str, Any]] = None) -> str:
        """
        Add a recommendation task to the queue.
        
        Args:
            task_type: Type of recommendation task (startup, documents, refresh)
            document_paths: List of document paths to analyze
            search_query: Optional search query for targeted recommendations
            recommendation_params: Additional parameters for the recommendation
            
        Returns:
            Task ID for tracking
        """
        task_id = f"recommend_{task_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create descriptive name for logging
        if document_paths:
            doc_names = [p.name for p in document_paths]
            desc = f"{len(document_paths)} documents: {', '.join(doc_names[:2])}"
            if len(doc_names) > 2:
                desc += f" and {len(doc_names)-2} more"
        else:
            desc = "all documents"
        
        task = BackgroundTask(
            task_id=task_id,
            task_type=f"recommend_{task_type}",
            file_path=None,  # Not used for recommendation tasks
            document_paths=document_paths,
            search_query=search_query,
            recommendation_params=recommendation_params or {}
        )
        
        with self.lock:
            self.tasks[task_id] = task
            self.task_queue.put(task_id)
            self.stats.total_tasks += 1
            self.stats.queued_tasks += 1
            self.stats.last_activity = datetime.now()
        
        self._save_queue()
        self._notify_status_update()
        
        logger.info(f"Added recommendation task ({task_type}): {desc} -> {task_id}")
        return task_id
    
    def get_task_status(self, task_id: str) -> Optional[BackgroundTask]:
        """Get the status of a specific task."""
        with self.lock:
            return self.tasks.get(task_id)
    
    def get_stats(self) -> ProcessingStats:
        """Get current processing statistics."""
        with self.lock:
            # Update current counts
            self.stats.queued_tasks = sum(1 for task in self.tasks.values() 
                                        if task.status == TaskStatus.QUEUED)
            self.stats.processing_tasks = sum(1 for task in self.tasks.values() 
                                            if task.status == TaskStatus.PROCESSING)
            self.stats.completed_tasks = sum(1 for task in self.tasks.values() 
                                           if task.status == TaskStatus.COMPLETED)
            self.stats.failed_tasks = sum(1 for task in self.tasks.values() 
                                        if task.status == TaskStatus.FAILED)
            
            # Calculate average processing time
            completed_times = []
            for task in self.tasks.values():
                if task.status == TaskStatus.COMPLETED and task.started_at and task.completed_at:
                    duration = (task.completed_at - task.started_at).total_seconds()
                    completed_times.append(duration)
            
            self.stats.average_processing_time = (
                sum(completed_times) / len(completed_times) if completed_times else 0.0
            )
            
            return ProcessingStats(
                total_tasks=self.stats.total_tasks,
                queued_tasks=self.stats.queued_tasks,
                processing_tasks=self.stats.processing_tasks,
                completed_tasks=self.stats.completed_tasks,
                failed_tasks=self.stats.failed_tasks,
                average_processing_time=self.stats.average_processing_time,
                last_activity=self.stats.last_activity,
                queue_start_time=self.stats.queue_start_time
            )
    
    def get_active_tasks(self) -> List[BackgroundTask]:
        """Get list of currently active (queued or processing) tasks."""
        with self.lock:
            return [
                task for task in self.tasks.values()
                if task.status in [TaskStatus.QUEUED, TaskStatus.PROCESSING]
            ]
    
    def get_recent_tasks(self, limit: int = 10) -> List[BackgroundTask]:
        """Get list of recent tasks, sorted by creation time."""
        with self.lock:
            return sorted(
                self.tasks.values(),
                key=lambda t: t.created_at,
                reverse=True
            )[:limit]
    
    def cancel_task(self, task_id: str) -> bool:
        """Cancel a queued task."""
        with self.lock:
            task = self.tasks.get(task_id)
            if task and task.status == TaskStatus.QUEUED:
                task.status = TaskStatus.CANCELLED
                self._save_queue()
                self._notify_status_update()
                logger.info(f"Cancelled task: {task_id}")
                return True
        return False
    
    def clear_completed_tasks(self):
        """Remove completed and failed tasks from memory."""
        with self.lock:
            to_remove = [
                task_id for task_id, task in self.tasks.items()
                if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]
            ]
            
            for task_id in to_remove:
                del self.tasks[task_id]
                
            logger.info(f"Cleared {len(to_remove)} completed/failed tasks")
        
        self._save_queue()
        self._notify_status_update()
    
    def _worker_loop(self):
        """Main loop for background worker threads."""
        worker_name = threading.current_thread().name
        logger.info(f"{worker_name} started")
        
        while not self.shutdown_event.is_set():
            try:
                # Get next task from queue (with timeout)
                try:
                    task_id = self.task_queue.get(timeout=1.0)
                except Empty:
                    continue
                
                # Get task details
                with self.lock:
                    task = self.tasks.get(task_id)
                
                if not task or task.status != TaskStatus.QUEUED:
                    self.task_queue.task_done()
                    continue
                
                # Process the task
                self._process_task(task, worker_name)
                self.task_queue.task_done()
                
            except Exception as e:
                logger.error(f"Error in {worker_name}: {e}")
                
        logger.info(f"{worker_name} stopped")
    
    def _process_task(self, task: BackgroundTask, worker_name: str):
        """Process a single background task."""
        task_desc = task.file_path.name if task.file_path else f"recommendation task {task.task_type}"
        logger.info(f"{worker_name} processing: {task_desc}")
        
        # Update task status
        with self.lock:
            task.status = TaskStatus.PROCESSING
            task.started_at = datetime.now()
            task.progress = 0.0
            self.stats.last_activity = datetime.now()
        
        self._save_queue()
        self._notify_status_update()
        
        try:
            # Actual processing
            if task.task_type == "process_paper":
                result = self._process_paper_task(task)
                
                with self.lock:
                    task.status = TaskStatus.COMPLETED
                    task.completed_at = datetime.now()
                    task.progress = 100.0
                    task.result = result
                    
                logger.info(f"{worker_name} completed: {task_desc}")
                
            elif task.task_type.startswith("recommend_"):
                result = self._process_recommendation_task(task)
                
                with self.lock:
                    task.status = TaskStatus.COMPLETED
                    task.completed_at = datetime.now()
                    task.progress = 100.0
                    task.result = result
                    
                logger.info(f"{worker_name} completed recommendation task: {task.task_type}")
                
            else:
                raise ValueError(f"Unknown task type: {task.task_type}")
                
        except Exception as e:
            task_desc = task.file_path.name if task.file_path else f"recommendation task {task.task_type}"
            
            # Enhanced error logging for recommendation tasks
            if task.task_type.startswith("recommend_"):
                logger.error(f"{worker_name} RECOMMENDATION TASK FAILED: {task_desc}")
                logger.error(f"  Task ID: {task.task_id}")
                logger.error(f"  Task Type: {task.task_type}")
                logger.error(f"  Error: {e}")
                logger.error(f"  Document Processor Available: {self.document_processor is not None}")
                logger.error(f"  Paper Recommender Available: {self.paper_recommender is not None}")
                if hasattr(task, 'document_paths') and task.document_paths:
                    logger.error(f"  Document Paths: {[str(p) for p in task.document_paths]}")
                else:
                    docs_dir = self.project_root / "documents"
                    doc_files = list(docs_dir.glob("*.docx")) if docs_dir.exists() else []
                    logger.error(f"  Documents directory exists: {docs_dir.exists()}")
                    logger.error(f"  Available .docx files: {[f.name for f in doc_files]}")
            else:
                logger.error(f"{worker_name} failed processing {task_desc}: {e}")
            
            with self.lock:
                task.status = TaskStatus.FAILED
                task.completed_at = datetime.now()
                task.error_message = str(e)
                task.retry_count += 1
                
                # Retry if under retry limit
                if task.retry_count <= task.max_retries:
                    task.status = TaskStatus.QUEUED
                    task.started_at = None
                    task.completed_at = None
                    task.progress = 0.0
                    self.task_queue.put(task.task_id)
                    logger.info(f"Retrying task {task.task_id} (attempt {task.retry_count}/{task.max_retries})")
        
        self._save_queue()
        self._notify_status_update()
    
    def _process_paper_task(self, task: BackgroundTask):
        """Process a paper file."""
        if not self.workflow_coordinator:
            raise RuntimeError("No workflow coordinator available")
        
        # Update progress
        with self.lock:
            task.progress = 25.0
        
        # Process through workflow coordinator
        result = self.workflow_coordinator.process_new_paper(task.file_path)
        
        # Update progress
        with self.lock:
            task.progress = 100.0
            
        return result
    
    def _process_recommendation_task(self, task: BackgroundTask):
        """Process a recommendation task."""
        if not self.document_processor or not self.paper_recommender:
            error_msg = f"Cannot process recommendation task: document_processor={type(self.document_processor).__name__ if self.document_processor else 'None'}, paper_recommender={type(self.paper_recommender).__name__ if self.paper_recommender else 'None'}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        recommendation_type = task.task_type.replace("recommend_", "")
        logger.info(f"Processing recommendation task: {recommendation_type}")
        
        # Update progress
        with self.lock:
            task.progress = 10.0
            
        # Determine document paths to analyze
        document_paths = task.document_paths
        if not document_paths:
            # For startup/refresh tasks, analyze all documents in the documents folder
            docs_dir = self.project_root / "documents"
            if docs_dir.exists():
                document_paths = list(docs_dir.glob("*.docx"))
            else:
                logger.warning("No documents directory found, skipping recommendation task")
                return None
        
        if not document_paths:
            logger.warning("No documents found for recommendation task")
            return None
        
        # Update progress
        with self.lock:
            task.progress = 20.0
        
        try:
            # Step 1: Analyze documents
            combined_analysis = None
            combined_context = ""
            
            for i, document_path in enumerate(document_paths):
                try:
                    logger.info(f"Analyzing document: {document_path.name}")
                    document_analysis = self.document_processor.analyze_document(document_path)
                    recommendation_context = self.document_processor.create_recommendation_context(document_analysis)
                    
                    if i == 0:
                        combined_analysis = document_analysis
                        combined_context = recommendation_context
                    else:
                        # Combine analyses
                        combined_analysis.key_concepts.extend(document_analysis.key_concepts)
                        combined_analysis.research_questions.extend(document_analysis.research_questions)
                        combined_analysis.literature_gaps.extend(document_analysis.literature_gaps)
                        combined_context += f"\n\nDocument {i+1} context: {recommendation_context}"
                        
                    # Update progress incrementally
                    progress = 20.0 + (i + 1) / len(document_paths) * 30.0
                    with self.lock:
                        task.progress = progress
                        
                except Exception as e:
                    logger.warning(f"Failed to process document {document_path}: {e}")
                    continue
            
            if not combined_analysis:
                raise RuntimeError("Failed to analyze any documents")
            
            # Step 2: Clean up analysis
            combined_analysis.key_concepts = list(set(combined_analysis.key_concepts))[:10]
            combined_analysis.research_questions = list(set(combined_analysis.research_questions))[:5]
            combined_analysis.literature_gaps = list(set(combined_analysis.literature_gaps))[:8]
            
            # Add search query if provided
            if task.search_query:
                combined_context += f"\n\nAdditional search focus: {task.search_query}"
            
            # Update progress
            with self.lock:
                task.progress = 60.0
            
            # Step 3: Generate recommendations
            logger.info("Generating paper recommendations...")
            max_recommendations = task.recommendation_params.get('max_recommendations', 5)
            
            recommendation_report = self.paper_recommender.get_recommendations_from_analysis(
                combined_analysis,
                combined_context,
                max_recommendations=max_recommendations
            )
            
            # Update progress
            with self.lock:
                task.progress = 90.0
            
            # Step 4: Save to cache (if this is a startup/refresh task)
            if recommendation_type in ['startup', 'refresh'] and recommendation_report.recommendations:
                try:
                    # Use the same caching logic as the UI
                    self._save_recommendations_to_cache(recommendation_report.recommendations)
                    logger.info(f"Saved {len(recommendation_report.recommendations)} recommendations to cache")
                except Exception as e:
                    logger.warning(f"Failed to save recommendations to cache: {e}")
            
            return {
                'recommendations': recommendation_report.recommendations,
                'document_count': len(document_paths),
                'recommendation_type': recommendation_type,
                'search_query': task.search_query
            }
            
        except Exception as e:
            logger.error(f"Error in recommendation processing: {e}")
            raise
    
    def _save_recommendations_to_cache(self, recommendations):
        """Save recommendations to persistent cache."""
        try:
            import pickle
            import hashlib
            from datetime import datetime
            
            # Calculate documents hash for cache key
            docs_dir = self.project_root / "documents" 
            current_docs_hash = ""
            if docs_dir.exists():
                doc_files = sorted(docs_dir.glob("*.docx"))
                for doc_file in doc_files:
                    current_docs_hash += f"{doc_file.name}_{doc_file.stat().st_mtime}_"
                current_docs_hash = hashlib.md5(current_docs_hash.encode()).hexdigest()[:16]
            
            # Save cache file
            tmp_dir = self.project_root / "tmp"
            tmp_dir.mkdir(exist_ok=True)
            cache_file = tmp_dir / f"recommendations_cache_{current_docs_hash}.pkl"
            
            cache_data = {
                'recommendations': recommendations,
                'timestamp': datetime.now(),
                'docs_hash': current_docs_hash
            }
            
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
                
            logger.info(f"Saved recommendations cache to {cache_file}")
            
        except Exception as e:
            logger.error(f"Error saving recommendations cache: {e}")
    
    def _save_queue(self):
        """Save queue state to disk."""
        try:
            queue_data = {
                'tasks': {task_id: {
                    'task_id': task.task_id,
                    'task_type': task.task_type,
                    'file_path': str(task.file_path) if task.file_path else None,
                    'created_at': task.created_at.isoformat(),
                    'started_at': task.started_at.isoformat() if task.started_at else None,
                    'completed_at': task.completed_at.isoformat() if task.completed_at else None,
                    'status': task.status.value,
                    'progress': task.progress,
                    'error_message': task.error_message,
                    'retry_count': task.retry_count,
                    'max_retries': task.max_retries,
                    'document_paths': [str(p) for p in task.document_paths] if task.document_paths else None,
                    'search_query': task.search_query,
                    'recommendation_params': task.recommendation_params
                } for task_id, task in self.tasks.items()},
                'stats': {
                    'total_tasks': self.stats.total_tasks,
                    'last_activity': self.stats.last_activity.isoformat() if self.stats.last_activity else None,
                    'queue_start_time': self.stats.queue_start_time.isoformat()
                }
            }
            
            with open(self.queue_file, 'w') as f:
                json.dump(queue_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving queue: {e}")
    
    def _load_queue(self):
        """Load queue state from disk."""
        if not self.queue_file.exists():
            return
            
        try:
            with open(self.queue_file, 'r') as f:
                queue_data = json.load(f)
            
            # Load tasks
            for task_id, task_data in queue_data.get('tasks', {}).items():
                # Handle both old and new task formats
                document_paths = None
                if task_data.get('document_paths'):
                    document_paths = [Path(p) for p in task_data['document_paths']]
                
                task = BackgroundTask(
                    task_id=task_data['task_id'],
                    task_type=task_data['task_type'],
                    file_path=Path(task_data['file_path']) if task_data['file_path'] else None,
                    created_at=datetime.fromisoformat(task_data['created_at']),
                    started_at=datetime.fromisoformat(task_data['started_at']) if task_data['started_at'] else None,
                    completed_at=datetime.fromisoformat(task_data['completed_at']) if task_data['completed_at'] else None,
                    status=TaskStatus(task_data['status']),
                    progress=task_data['progress'],
                    error_message=task_data['error_message'],
                    retry_count=task_data['retry_count'],
                    max_retries=task_data['max_retries'],
                    document_paths=document_paths,
                    search_query=task_data.get('search_query', ''),
                    recommendation_params=task_data.get('recommendation_params', {})
                )
                
                self.tasks[task_id] = task
                
                # Re-queue unfinished tasks
                if task.status in [TaskStatus.QUEUED, TaskStatus.PROCESSING]:
                    task.status = TaskStatus.QUEUED
                    task.started_at = None
                    task.progress = 0.0
                    self.task_queue.put(task_id)
            
            # Load stats
            stats_data = queue_data.get('stats', {})
            self.stats.total_tasks = stats_data.get('total_tasks', 0)
            if stats_data.get('last_activity'):
                self.stats.last_activity = datetime.fromisoformat(stats_data['last_activity'])
            if stats_data.get('queue_start_time'):
                self.stats.queue_start_time = datetime.fromisoformat(stats_data['queue_start_time'])
            
            logger.info(f"Loaded {len(self.tasks)} tasks from queue file")
            
        except Exception as e:
            logger.error(f"Error loading queue: {e}")
    
    def _notify_status_update(self):
        """Notify status callback of updates."""
        if self.status_callback:
            try:
                self.status_callback(self.get_stats())
            except Exception as e:
                logger.error(f"Error in status callback: {e}")
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        self.stop_workers()