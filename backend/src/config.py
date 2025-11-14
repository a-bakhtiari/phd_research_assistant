"""
Application configuration using Pydantic Settings.

Loads configuration from environment variables and .env file.
"""

from functools import lru_cache
from pathlib import Path
from typing import List, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.

    All settings can be overridden via environment variables or .env file.
    """

    # API Keys
    openai_api_key: str = Field(default="", description="OpenAI API key")
    anthropic_api_key: str = Field(default="", description="Anthropic API key")
    deepseek_api_key: str = Field(default="", description="DeepSeek API key")
    perplexity_api_key: str = Field(default="", description="Perplexity API key")
    semantic_scholar_api_key: str = Field(default="", description="Semantic Scholar API key")

    # LLM Configuration
    default_llm_provider: str = Field(
        default="deepseek",
        description="Default LLM provider (openai, anthropic, deepseek)"
    )
    openai_model: str = Field(default="gpt-3.5-turbo", description="OpenAI model name")
    anthropic_model: str = Field(
        default="claude-3-sonnet-20240229",
        description="Anthropic model name"
    )
    deepseek_model: str = Field(default="deepseek-chat", description="DeepSeek model name")

    # Application Settings
    debug: bool = Field(default=False, description="Enable debug mode")
    log_level: str = Field(default="INFO", description="Logging level")
    cors_origins: List[str] = Field(
        default=["http://localhost:5173", "http://localhost:3000"],
        description="Allowed CORS origins"
    )

    # Database
    database_url: str = Field(
        default="sqlite:///./data/phd_assistant.db",
        description="Database connection URL"
    )

    # Vector Store
    vector_store_path: str = Field(
        default="./data/vector_store",
        description="Path to vector store directory"
    )

    # Embedding Configuration (OpenAI only)
    embedding_model: str = Field(
        default="text-embedding-3-small",
        description="OpenAI embedding model (text-embedding-3-small or text-embedding-3-large)"
    )
    embedding_cache_enabled: bool = Field(
        default=True,
        description="Enable SQLite caching for embeddings"
    )
    embedding_cache_path: str = Field(
        default="./data/embeddings_cache.db",
        description="Path to embeddings cache database"
    )
    embedding_batch_size: int = Field(
        default=64,
        description="Batch size for embedding API calls"
    )
    embedding_max_retries: int = Field(
        default=5,
        description="Maximum retries for embedding API calls"
    )

    # PDF Processing
    enable_llm_pdf_cleaning: bool = Field(
        default=True,
        description="Enable LLM-based PDF cleaning"
    )
    enable_parallel_pdf_processing: bool = Field(
        default=True,
        description="Enable parallel PDF chunk processing using asyncio"
    )
    pdf_pages_per_chunk: int = Field(
        default=1,
        description="Number of pages per chunk for PDF processing (1 recommended for parallel)"
    )
    max_concurrent_pdf_chunks: Optional[int] = Field(
        default=None,
        description="Maximum concurrent PDF chunks (None = unlimited)"
    )
    pdf_max_retries: int = Field(
        default=3,
        description="Maximum retries for PDF processing operations"
    )

    # Server
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8000, description="Server port")
    reload: bool = Field(default=False, description="Enable auto-reload (dev only)")

    # Security (for future use)
    secret_key: str = Field(
        default="change_this_to_a_random_secret_key",
        description="Secret key for JWT tokens"
    )
    access_token_expire_minutes: int = Field(
        default=30,
        description="Access token expiration time in minutes"
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.

    Uses lru_cache to ensure settings are loaded only once.

    Returns:
        Settings: Application settings instance
    """
    return Settings()
