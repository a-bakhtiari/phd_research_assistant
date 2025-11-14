"""Settings service for application configuration."""

import logging
from src.models.schemas import EmbeddingModelInfo, LLMSettingsResponse

logger = logging.getLogger(__name__)


class SettingsService:
    """Service for managing settings."""

    def get_embedding_models(self):
        """Get available embedding models."""
        return [
            EmbeddingModelInfo(
                name="all-MiniLM-L6-v2",
                description="Fast and efficient",
                dimension=384,
                is_current=True
            )
        ]

    def get_llm_settings(self) -> LLMSettingsResponse:
        """Get LLM settings."""
        return LLMSettingsResponse(
            default_provider="deepseek",
            available_providers=["openai", "anthropic", "deepseek"],
            models={"openai": "gpt-3.5-turbo", "deepseek": "deepseek-chat"}
        )
