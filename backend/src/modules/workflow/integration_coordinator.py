import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime
from dataclasses import dataclass, field
import threading
from enum import Enum

from src.core.database import DatabaseManager, Paper
from src.core.vector_store import VectorStoreManager
from src.core.llm import LLMManager, PromptManager
from src.modules.file_manager.processor import PaperProcessor
from src.modules.recommender.paper_recommender import PaperRecommender
from src.modules.knowledge_chat import KnowledgeChat
from src.modules.workflow.background_processor import BackgroundProcessingQueue, ProcessingStats
from src.core.external_apis.semantic_scholar_client import SemanticScholarClient

logger = logging.getLogger(__name__)


class WorkflowEvent(Enum):
    """Types of workflow events that can trigger actions."""
    PAPER_ADDED = "paper_added"
    PAPER_PROCESSED = "paper_processed"
    RECOMMENDATIONS_REQUESTED = "recommendations_requested"
    MODEL_CHANGED = "model_changed"
    BULK_PROCESSING_COMPLETE = "bulk_processing_complete"


@dataclass
class WorkflowStatus:
    """Represents the current status of the system workflow."""
    papers_in_queue: int = 0
    papers_processing: int = 0
    papers_processed_today: int = 0
    recommendations_last_updated: Optional[datetime] = None
    model_last_updated: Optional[datetime] = None
    system_health: str = "healthy"  # healthy, warning, error
    last_error: Optional[str] = None
    active_workflows: List[str] = field(default_factory=list)


@dataclass
class WorkflowMetrics:
    """Workflow performance metrics."""
    total_papers_processed: int = 0
    average_processing_time: float = 0.0
    recommendations_generated: int = 0
    chat_sessions_active: int = 0
    system_uptime: datetime = field(default_factory=datetime.now)
    error_count_24h: int = 0


class IntegrationCoordinator:
    """
    Coordinates the complete workflow from new papers through recommendations and chat.
    Ensures all system components work together seamlessly.
    """
    
    def __init__(self,
                 db_manager: DatabaseManager,
                 vector_manager: VectorStoreManager,
                 llm_manager: LLMManager,
                 prompt_manager: PromptManager,
                 paper_processor: PaperProcessor,
                 paper_recommender: PaperRecommender,
                 knowledge_chat: KnowledgeChat,
                 project_root: Path,
                 status_callback: Optional[Callable[[WorkflowStatus], None]] = None):
        """
        Initialize the integration coordinator.
        
        Args:
            db_manager: Database manager
            vector_manager: Vector store manager
            llm_manager: LLM manager
            prompt_manager: Prompt manager
            paper_processor: Paper processor
            paper_recommender: Paper recommender
            knowledge_chat: Knowledge chat system
            project_root: Project root directory for background queue
            status_callback: Optional callback for status updates
        """
        # Core managers
        self.db_manager = db_manager
        self.vector_manager = vector_manager
        self.llm_manager = llm_manager
        self.prompt_manager = prompt_manager
        
        # Processing modules
        self.paper_processor = paper_processor
        self.paper_recommender = paper_recommender
        self.knowledge_chat = knowledge_chat
        
        # Enrichment services
        self.ss_client = SemanticScholarClient()
        
        # Background processing queue
        self.background_queue = BackgroundProcessingQueue(
            project_root=project_root,
            num_workers=2,
            auto_start=True
        )
        self.background_queue.set_processors(paper_processor, self, None, None)  # document_processor and paper_recommender will be set separately
        
        # Status tracking
        self.status = WorkflowStatus()
        self.metrics = WorkflowMetrics()
        self.status_callback = status_callback
        
        # Event handlers
        self.event_handlers: Dict[WorkflowEvent, List[Callable]] = {
            event: [] for event in WorkflowEvent
        }
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Setup default event handlers
        self._setup_default_handlers()
        
        logger.info("Integration Coordinator initialized with background processing queue")
    
    def _setup_default_handlers(self):
        """Setup default event handlers for workflow coordination."""
        # When a paper is processed, update recommendations
        self.register_event_handler(
            WorkflowEvent.PAPER_PROCESSED,
            self._handle_paper_processed
        )
        
        # When recommendations are requested, ensure fresh data
        self.register_event_handler(
            WorkflowEvent.RECOMMENDATIONS_REQUESTED,
            self._handle_recommendations_requested
        )
        
        # When embedding model changes, handle re-processing
        self.register_event_handler(
            WorkflowEvent.MODEL_CHANGED,
            self._handle_model_changed
        )
        
        # When bulk processing completes, update system state
        self.register_event_handler(
            WorkflowEvent.BULK_PROCESSING_COMPLETE,
            self._handle_bulk_processing_complete
        )
    
    def register_event_handler(self, event: WorkflowEvent, handler: Callable):
        """Register an event handler for workflow events."""
        with self._lock:
            if event not in self.event_handlers:
                self.event_handlers[event] = []
            self.event_handlers[event].append(handler)
            logger.debug(f"Registered handler for {event}")
    
    def trigger_event(self, event: WorkflowEvent, **kwargs):
        """Trigger a workflow event and execute all registered handlers."""
        with self._lock:
            handlers = self.event_handlers.get(event, [])
            self.status.active_workflows.append(f"{event.value}_{datetime.now().isoformat()}")
        
        logger.info(f"Triggering workflow event: {event.value}")
        
        for handler in handlers:
            try:
                handler(**kwargs)
            except Exception as e:
                logger.error(f"Error in event handler for {event}: {e}")
                self._update_status(system_health="warning", last_error=str(e))
        
        # Clean up old active workflows
        with self._lock:
            self.status.active_workflows = [
                wf for wf in self.status.active_workflows[-10:]  # Keep last 10
            ]
        
        self._notify_status_update()
    
    def process_new_paper(self, pdf_path: Path) -> Optional[Paper]:
        """
        Process a new paper through the complete workflow.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Paper object if successful, None if failed
        """
        logger.info(f"Starting complete workflow for: {pdf_path.name}")
        
        start_time = datetime.now()
        
        with self._lock:
            self.status.papers_processing += 1
        
        try:
            # Step 1: Process the paper (extract metadata, create embeddings)
            paper = self.paper_processor.process_new_paper(pdf_path)
            
            if paper:
                # Step 2: Trigger post-processing workflows
                self.trigger_event(WorkflowEvent.PAPER_PROCESSED, paper=paper)
                
                # Update metrics
                processing_time = (datetime.now() - start_time).total_seconds()
                self._update_metrics(processing_time=processing_time)
                
                with self._lock:
                    self.status.papers_processed_today += 1
                    self.status.papers_processing -= 1
                
                logger.info(f"Successfully completed workflow for: {paper.title}")
                return paper
            else:
                with self._lock:
                    self.status.papers_processing -= 1
                
                self._update_status(
                    system_health="warning",
                    last_error=f"Failed to process {pdf_path.name}"
                )
                
                return None
                
        except Exception as e:
            with self._lock:
                self.status.papers_processing -= 1
            
            logger.error(f"Error in complete workflow for {pdf_path.name}: {e}")
            self._update_status(system_health="error", last_error=str(e))
            return None
    
    def _handle_paper_processed(self, paper: Paper, **kwargs):
        """Handle post-processing after a paper is successfully processed."""
        logger.info(f"Post-processing workflow for paper: {paper.title}")
        
        try:
            # If this is one of the first few papers, or periodically, 
            # we might want to refresh recommendations
            paper_count = self.db_manager.get_paper_stats()["total"]
            
            if paper_count <= 10 or paper_count % 5 == 0:
                logger.info("Triggering recommendation refresh due to new papers")
                # This will be handled by the recommendations_requested handler
                # when the user next visits the recommendations page
                with self._lock:
                    self.status.recommendations_last_updated = None  # Mark as stale
            
        except Exception as e:
            logger.error(f"Error in paper post-processing: {e}")
    
    def _handle_recommendations_requested(self, query: str = None, **kwargs):
        """Handle requests for paper recommendations."""
        logger.info("Handling recommendations request")
        
        try:
            # Check if we need to refresh recommendations
            current_time = datetime.now()
            last_updated = self.status.recommendations_last_updated
            
            if (not last_updated or 
                (current_time - last_updated).total_seconds() > 3600):  # 1 hour cache
                
                logger.info("Refreshing recommendation system")
                # The actual recommendation generation will be handled by the UI
                # We just mark that recommendations are fresh
                with self._lock:
                    self.status.recommendations_last_updated = current_time
                    
        except Exception as e:
            logger.error(f"Error handling recommendations request: {e}")
    
    def _handle_model_changed(self, old_model: str, new_model: str, **kwargs):
        """Handle embedding model changes and coordinate re-processing."""
        logger.info(f"Handling model change from {old_model} to {new_model}")
        
        try:
            # Mark that model has changed
            with self._lock:
                self.status.model_last_updated = datetime.now()
                self.status.recommendations_last_updated = None  # Invalidate recommendations
            
            # The actual re-processing will be handled by the UI migration tools
            # This just coordinates the workflow state
            
            logger.info("Model change workflow coordination complete")
            
        except Exception as e:
            logger.error(f"Error handling model change: {e}")
    
    def _handle_bulk_processing_complete(self, processed_count: int, **kwargs):
        """Handle completion of bulk processing operations."""
        logger.info(f"Bulk processing complete: {processed_count} papers")
        
        try:
            # Update system status
            with self._lock:
                self.status.recommendations_last_updated = None  # Refresh needed
                self.status.system_health = "healthy"
                
            # Clear any processing queues
            self.status.papers_in_queue = 0
            self.status.papers_processing = 0
            
            logger.info("Bulk processing workflow coordination complete")
            
        except Exception as e:
            logger.error(f"Error handling bulk processing completion: {e}")
    
    def get_workflow_status(self) -> WorkflowStatus:
        """Get current workflow status."""
        with self._lock:
            # Update real-time stats
            try:
                stats = self.db_manager.get_paper_stats()
                today = datetime.now().date()
                
                # Count papers processed today (approximate)
                papers_today = 0
                try:
                    recent_papers = self.db_manager.get_recent_papers(limit=50)
                    papers_today = sum(
                        1 for p in recent_papers 
                        if p.created_at and p.created_at.date() == today
                    )
                except Exception:
                    pass  # Fallback to existing count
                
                self.status.papers_processed_today = papers_today
                
                # Update system health based on errors
                if self.metrics.error_count_24h > 10:
                    self.status.system_health = "error"
                elif self.metrics.error_count_24h > 3:
                    self.status.system_health = "warning"
                elif self.status.last_error is None:
                    self.status.system_health = "healthy"
                
            except Exception as e:
                logger.error(f"Error updating workflow status: {e}")
            
            return WorkflowStatus(
                papers_in_queue=self.status.papers_in_queue,
                papers_processing=self.status.papers_processing,
                papers_processed_today=self.status.papers_processed_today,
                recommendations_last_updated=self.status.recommendations_last_updated,
                model_last_updated=self.status.model_last_updated,
                system_health=self.status.system_health,
                last_error=self.status.last_error,
                active_workflows=self.status.active_workflows.copy()
            )
    
    def get_workflow_metrics(self) -> WorkflowMetrics:
        """Get workflow performance metrics."""
        with self._lock:
            return WorkflowMetrics(
                total_papers_processed=self.metrics.total_papers_processed,
                average_processing_time=self.metrics.average_processing_time,
                recommendations_generated=self.metrics.recommendations_generated,
                chat_sessions_active=self.metrics.chat_sessions_active,
                system_uptime=self.metrics.system_uptime,
                error_count_24h=self.metrics.error_count_24h
            )
    
    def health_check(self) -> Dict[str, Any]:
        """Perform a comprehensive system health check."""
        health_report = {
            "timestamp": datetime.now().isoformat(),
            "overall_status": "healthy",
            "components": {}
        }
        
        try:
            # Check database connectivity
            health_report["components"]["database"] = self._check_database_health()
            
            # Check vector store
            health_report["components"]["vector_store"] = self._check_vector_store_health()
            
            # Check LLM connectivity
            health_report["components"]["llm"] = self._check_llm_health()
            
            # Check workflow status
            status = self.get_workflow_status()
            health_report["components"]["workflow"] = {
                "status": "healthy" if status.system_health == "healthy" else "warning",
                "papers_processing": status.papers_processing,
                "last_error": status.last_error
            }
            
            # Determine overall status
            component_statuses = [comp["status"] for comp in health_report["components"].values()]
            if "error" in component_statuses:
                health_report["overall_status"] = "error"
            elif "warning" in component_statuses:
                health_report["overall_status"] = "warning"
            
            return health_report
            
        except Exception as e:
            logger.error(f"Error performing health check: {e}")
            health_report["overall_status"] = "error"
            health_report["error"] = str(e)
            return health_report
    
    def _check_database_health(self) -> Dict[str, Any]:
        """Check database health."""
        try:
            stats = self.db_manager.get_paper_stats()
            return {
                "status": "healthy",
                "total_papers": stats["total"],
                "response_time": "< 100ms"
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    def _check_vector_store_health(self) -> Dict[str, Any]:
        """Check vector store health."""
        try:
            stats = self.vector_manager.get_collection_stats()
            return {
                "status": "healthy",
                "total_embeddings": stats.get("total_embeddings", 0),
                "model": self.vector_manager.embedding_model_name
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    def _check_llm_health(self) -> Dict[str, Any]:
        """Check LLM connectivity."""
        try:
            # Simple test prompt
            test_messages = [{"role": "user", "content": "Hello"}]
            response = self.llm_manager.generate_response(test_messages, max_tokens=10)
            
            return {
                "status": "healthy",
                "model": "connected",
                "test_response": len(response.content) > 0
            }
        except Exception as e:
            return {
                "status": "warning",
                "error": str(e)
            }
    
    def _update_status(self, **kwargs):
        """Update workflow status."""
        with self._lock:
            for key, value in kwargs.items():
                if hasattr(self.status, key):
                    setattr(self.status, key, value)
        
        self._notify_status_update()
    
    def _update_metrics(self, processing_time: float = None):
        """Update workflow metrics."""
        with self._lock:
            if processing_time:
                # Update average processing time
                total = self.metrics.total_papers_processed
                current_avg = self.metrics.average_processing_time
                
                new_avg = ((current_avg * total) + processing_time) / (total + 1)
                self.metrics.average_processing_time = new_avg
                self.metrics.total_papers_processed += 1
    
    def _notify_status_update(self):
        """Notify registered callbacks of status updates."""
        if self.status_callback:
            try:
                self.status_callback(self.get_workflow_status())
            except Exception as e:
                logger.error(f"Error in status callback: {e}")
    
    def queue_paper_for_processing(self, pdf_path: Path) -> str:
        """
        Queue a paper for background processing instead of processing immediately.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Task ID for tracking
        """
        logger.info(f"Queuing paper for background processing: {pdf_path.name}")
        return self.background_queue.add_paper_task(pdf_path, "process_paper")
    
    def get_processing_stats(self) -> ProcessingStats:
        """Get current background processing statistics."""
        return self.background_queue.get_stats()
    
    def get_active_processing_tasks(self):
        """Get list of currently active processing tasks."""
        return self.background_queue.get_active_tasks()
    
    def get_recent_processing_tasks(self, limit: int = 10):
        """Get list of recent processing tasks."""
        return self.background_queue.get_recent_tasks(limit)
    
    def cancel_processing_task(self, task_id: str) -> bool:
        """Cancel a queued processing task."""
        return self.background_queue.cancel_task(task_id)
    
    def clear_completed_tasks(self):
        """Clear completed and failed tasks from memory."""
        self.background_queue.clear_completed_tasks()
    
    def set_recommendation_processors(self, document_processor, paper_recommender):
        """Set additional processors for recommendation tasks."""
        if hasattr(self, 'background_queue'):
            if document_processor is None or paper_recommender is None:
                logger.error(f"Cannot set processors - document_processor: {type(document_processor)}, paper_recommender: {type(paper_recommender)}")
                return False
                
            self.background_queue.document_processor = document_processor
            self.background_queue.paper_recommender = paper_recommender
            logger.info(f"Successfully set recommendation processors - document_processor: {type(document_processor).__name__}, paper_recommender: {type(paper_recommender).__name__}")
            return True
        else:
            logger.error("Background queue not available for setting processors")
            return False
    
    def validate_recommendation_setup(self):
        """
        Validate that the recommendation system is properly configured.
        
        Returns:
            tuple: (is_valid, list_of_issues)
        """
        issues = []
        
        if not hasattr(self, 'background_queue'):
            issues.append("Background queue not initialized")
            return False, issues
            
        if not self.background_queue.document_processor:
            issues.append("Document processor not set in background queue")
            
        if not self.background_queue.paper_recommender:
            issues.append("Paper recommender not set in background queue")
            
        # Check if workers are running
        if not self.background_queue.workers:
            issues.append("Background worker threads not started")
        elif not any(worker.is_alive() for worker in self.background_queue.workers):
            issues.append("Background worker threads not running")
            
        # Check if there are documents to analyze
        docs_dir = Path(self.background_queue.project_root) / "documents"
        if not docs_dir.exists():
            issues.append("Documents directory does not exist")
        else:
            doc_files = list(docs_dir.glob("*.docx"))
            if not doc_files:
                issues.append("No .docx files found in documents directory")
        
        is_valid = len(issues) == 0
        return is_valid, issues
    
    def queue_recommendation_task(self, 
                                task_type: str,
                                document_paths: Optional[List[Path]] = None,
                                search_query: str = "",
                                max_recommendations: int = 5) -> str:
        """
        Queue a recommendation task for background processing.
        
        Args:
            task_type: Type of recommendation (startup, documents, refresh)
            document_paths: Optional list of specific documents to analyze
            search_query: Optional search query for targeted recommendations
            max_recommendations: Maximum number of recommendations to generate
            
        Returns:
            Task ID for tracking
        """
        recommendation_params = {'max_recommendations': max_recommendations}
        
        logger.info(f"Queuing recommendation task: {task_type}")
        return self.background_queue.add_recommendation_task(
            task_type=task_type,
            document_paths=document_paths,
            search_query=search_query,
            recommendation_params=recommendation_params
        )
    
    def get_recommendation_tasks(self, task_type: Optional[str] = None):
        """Get recommendation tasks, optionally filtered by type."""
        all_tasks = self.get_recent_processing_tasks(limit=20)
        
        if task_type:
            return [task for task in all_tasks if task.task_type == f"recommend_{task_type}"]
        else:
            return [task for task in all_tasks if task.task_type.startswith("recommend_")]
    
    def shutdown(self):
        """Clean shutdown of the integration coordinator."""
        logger.info("Shutting down Integration Coordinator")
        
        with self._lock:
            self.status.system_health = "shutting_down"
            self.status.active_workflows.clear()
        
        # Stop background processing queue
        if hasattr(self, 'background_queue'):
            self.background_queue.stop_workers()
        
        # Any cleanup operations would go here
        logger.info("Integration Coordinator shutdown complete")