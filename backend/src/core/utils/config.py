from typing import Dict, Any, Optional, List
from pathlib import Path
import yaml
import os
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMConfig(BaseSettings):
    """Configuration for LLM providers."""
    
    # OpenAI Configuration
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-3.5-turbo"
    openai_base_url: Optional[str] = None
    
    # Anthropic Configuration
    anthropic_api_key: Optional[str] = None
    anthropic_model: str = "claude-3-sonnet-20240229"
    
    # DeepSeek Configuration
    deepseek_api_key: Optional[str] = None
    deepseek_model: str = "deepseek-chat"
    
    # Default provider
    default_provider: str = "openai"
    
    model_config = SettingsConfigDict(
        env_ignore_empty=True,
        extra='forbid'  # Only allow defined fields
    )
    
    def __init__(self, global_config: Dict[str, Any] = None, **kwargs):
        # Load API keys from global config if not provided
        if global_config and 'api_keys' in global_config:
            api_keys = global_config['api_keys']
            if 'openai_api_key' not in kwargs:
                kwargs['openai_api_key'] = api_keys.get('openai')
            if 'anthropic_api_key' not in kwargs:
                kwargs['anthropic_api_key'] = api_keys.get('anthropic')
            if 'deepseek_api_key' not in kwargs:
                kwargs['deepseek_api_key'] = api_keys.get('deepseek')
                
            # Also load model settings
            if 'llm' in global_config:
                llm_config = global_config['llm']
                if 'default_provider' not in kwargs:
                    kwargs['default_provider'] = llm_config.get('default_provider', 'openai')
                if 'models' in llm_config:
                    models = llm_config['models']
                    if 'openai_model' not in kwargs:
                        kwargs['openai_model'] = models.get('openai', 'gpt-3.5-turbo')
                    if 'anthropic_model' not in kwargs:
                        kwargs['anthropic_model'] = models.get('anthropic', 'claude-3-sonnet-20240229')
                    if 'deepseek_model' not in kwargs:
                        kwargs['deepseek_model'] = models.get('deepseek', 'deepseek-chat')
        
        super().__init__(**kwargs)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for LLMManager."""
        config = {}
        
        if self.openai_api_key:
            config["openai"] = {
                "api_key": self.openai_api_key,
                "model": self.openai_model,
                "base_url": self.openai_base_url
            }
        
        if self.anthropic_api_key:
            config["anthropic"] = {
                "api_key": self.anthropic_api_key,
                "model": self.anthropic_model
            }
        
        if self.deepseek_api_key:
            config["deepseek"] = {
                "api_key": self.deepseek_api_key,
                "model": self.deepseek_model
            }
        
        return config


class ProjectConfig(BaseSettings):
    """Main project configuration."""
    
    # Project Identification
    project_name: str = "PhD Research Project"
    project_description: str = ""
    research_area: str = ""
    
    # Paths
    project_root: str = Field(..., description="Root directory for project data")
    papers_directory: str = "data/PAPERS"
    new_papers_directory: str = "data/_NEW_PAPERS"
    processing_directory: str = "data/_PROCESSING"
    failed_directory: str = "data/_FAILED"
    database_file: str = "data/project.db"
    vector_store_directory: str = "data/vector_store"
    prompts_directory: str = "prompts"
    
    # PDF Processing
    pdf_chunk_size: int = 1000
    pdf_chunk_overlap: int = 200
    pdf_metadata_pages: int = 3
    
    # Vector Store
    embedding_model: str = "all-MiniLM-L6-v2"
    vector_collection_name: str = "papers"
    
    # File Watching
    enable_file_watching: bool = True
    scan_existing_files_on_startup: bool = True
    monitor_downloads: bool = True
    verify_academic_papers: bool = True
    
    # UI Settings
    ui_title: str = "PhD Research Assistant"
    ui_theme: str = "light"
    show_debug_info: bool = False
    
    # Academic Search APIs
    semantic_scholar_api_key: Optional[str] = None
    perplexity_api_key: Optional[str] = None
    
    model_config = SettingsConfigDict(
        env_file=".env", 
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        extra='ignore'  # Ignore extra environment variables
    )
    
    def get_absolute_path(self, relative_path: str) -> Path:
        """Convert relative path to absolute path based on project root."""
        return Path(self.project_root) / relative_path
    
    def get_all_paths(self) -> Dict[str, Path]:
        """Get all configured paths as absolute Path objects."""
        return {
            "project_root": Path(self.project_root),
            "papers": self.get_absolute_path(self.papers_directory),
            "new_papers": self.get_absolute_path(self.new_papers_directory),
            "processing": self.get_absolute_path(self.processing_directory),
            "failed": self.get_absolute_path(self.failed_directory),
            "database": self.get_absolute_path(self.database_file),
            "vector_store": self.get_absolute_path(self.vector_store_directory),
            "prompts": self.get_absolute_path(self.prompts_directory)
        }


class ConfigManager:
    """Manages configuration loading and saving."""
    
    def __init__(self, config_path: str = None):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to configuration file (YAML)
        """
        self.config_path = Path(config_path) if config_path else None
        self.project_config = None
        
        # Load global config from root directory
        global_config_path = Path(__file__).parent.parent.parent.parent / "config.yaml"
        self.global_config = self._load_global_config(global_config_path)
        
        # LLM config loaded with global config
        self.llm_config = LLMConfig(global_config=self.global_config)
        
        if self.config_path and self.config_path.exists():
            self.load_config()
        else:
            self._initialize_default_config()
    
    def _load_global_config(self, config_path: Path) -> Dict[str, Any]:
        """Load global configuration from YAML file."""
        try:
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    return yaml.safe_load(f) or {}
            else:
                print(f"Warning: Global config file not found at {config_path}")
                return {}
        except Exception as e:
            print(f"Error loading global config: {e}")
            return {}
    
    def _initialize_default_config(self):
        """Initialize with default configuration."""
        self.project_config = ProjectConfig(project_root=str(Path.cwd()))
    
    def load_config(self):
        """Load configuration from YAML file."""
        if not self.config_path or not self.config_path.exists():
            self._initialize_default_config()
            return
        
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f) or {}
            
            # Load project configuration
            project_data = config_data.get("project", {})
            if "project_root" not in project_data:
                project_data["project_root"] = str(self.config_path.parent)
            
            self.project_config = ProjectConfig(**project_data)
            
            # LLM configuration is always loaded from environment, not from project files
            
        except Exception as e:
            print(f"Error loading config from {self.config_path}: {e}")
            self._initialize_default_config()
    
    def save_config(self):
        """Save current configuration to YAML file."""
        if not self.config_path:
            raise ValueError("No config path specified")
        
        # Ensure directory exists
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        
        config_data = {
            "project": self.project_config.model_dump()
            # LLM configuration is not saved to project files
        }
        
        with open(self.config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_data, f, default_flow_style=False, indent=2)
    
    def update_project_config(self, **kwargs):
        """Update project configuration parameters."""
        current_data = self.project_config.model_dump()
        current_data.update(kwargs)
        self.project_config = ProjectConfig(**current_data)
    
    # LLM config is managed at application level, not project level
    
    def create_project_directories(self):
        """Create all required project directories."""
        paths = self.project_config.get_all_paths()
        
        for name, path in paths.items():
            if name == "database":
                # Create parent directory for database file
                path.parent.mkdir(parents=True, exist_ok=True)
            else:
                # Create directory
                path.mkdir(parents=True, exist_ok=True)
    
    def validate_config(self) -> List[str]:
        """Validate configuration and return list of issues."""
        issues = []
        
        # Check if at least one LLM provider is configured
        llm_dict = self.llm_config.to_dict()
        if not llm_dict:
            issues.append("No LLM providers configured. Please set API keys.")
        
        # Check if project root exists
        project_root = Path(self.project_config.project_root)
        if not project_root.exists():
            issues.append(f"Project root directory does not exist: {project_root}")
        
        return issues
    
    def get_watch_directories(self) -> Dict[str, str]:
        """Get directories to watch for new files."""
        paths = self.project_config.get_all_paths()
        watch_dirs = {
            "new_papers": str(paths["new_papers"])
        }

        # Add Downloads folder if monitoring is enabled
        if self.project_config.monitor_downloads:
            downloads_path = str(Path.home() / "Downloads")
            watch_dirs["downloads"] = downloads_path

        return watch_dirs

    def get_pdf_processing_config(self) -> Dict[str, Any]:
        """Get PDF processing configuration settings."""
        return self.global_config.get('pdf_processing', {
            'enable_llm_cleaning': True,
            'pages_per_chunk': 4,
            'max_retries': 3
        })
    
    @classmethod
    def create_template_config(cls, project_path: str) -> "ConfigManager":
        """Create a template configuration for a new project."""
        config_path = Path(project_path) / "config.yaml"
        
        # Create basic project structure
        project_root = Path(project_path)
        project_root.mkdir(parents=True, exist_ok=True)
        
        # Initialize config manager
        config_manager = cls(str(config_path))
        config_manager.project_config.project_root = str(project_root)
        config_manager.project_config.project_name = project_root.name
        
        # Create directories
        config_manager.create_project_directories()
        
        # Save config
        config_manager.save_config()
        
        return config_manager
    
    def export_env_template(self) -> str:
        """Export environment variables template."""
        template = """# PhD Research Assistant Environment Variables
# Copy this to .env and fill in your API keys

# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Anthropic Configuration  
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# DeepSeek Configuration (optional)
DEEPSEEK_API_KEY=your_deepseek_api_key_here

# Semantic Scholar API (optional)
SEMANTIC_SCHOLAR_API_KEY=your_semantic_scholar_api_key_here
"""
        return template