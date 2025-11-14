"""
Main FastAPI application entry point for PhD Research Assistant.

This module initializes the FastAPI app, configures middleware, and registers all API routes.
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api import projects, papers, chat, recommendations, agents, documents, queue, websocket
from src.api import settings as settings_api
from src.config import get_settings

# Configure logging (only if not already configured)
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.

    Handles startup and shutdown events for the application.
    """
    # Startup
    logger.info("Starting PhD Research Assistant API...")
    settings = get_settings()
    logger.info(f"Debug mode: {settings.debug}")
    logger.info(f"Default LLM provider: {settings.default_llm_provider}")

    yield

    # Shutdown
    logger.info("Shutting down PhD Research Assistant API...")


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.

    Returns:
        FastAPI: Configured FastAPI application instance
    """
    settings = get_settings()

    app = FastAPI(
        title="PhD Research Assistant API",
        description="AI-powered research management and knowledge discovery system",
        version="2.0.0",
        lifespan=lifespan,
        docs_url="/api/docs",
        redoc_url="/api/redoc",
        openapi_url="/api/openapi.json"
    )

    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register API routers
    app.include_router(projects.router, prefix="/api/v1/projects", tags=["Projects"])
    app.include_router(papers.router, prefix="/api/v1/papers", tags=["Papers"])
    app.include_router(chat.router, prefix="/api/v1/chat", tags=["Chat"])
    app.include_router(
        recommendations.router,
        prefix="/api/v1/recommendations",
        tags=["Recommendations"]
    )
    app.include_router(documents.router, prefix="/api/v1/documents", tags=["Documents"])
    app.include_router(agents.router, prefix="/api/v1/agents", tags=["Agents"])
    app.include_router(settings_api.router, prefix="/api/v1/settings", tags=["Settings"])
    app.include_router(queue.router, prefix="/api/v1", tags=["Queue"])
    app.include_router(websocket.router, prefix="/api/v1")  # WebSocket endpoint

    # Health check endpoint
    @app.get("/health", tags=["Health"])
    async def health_check():
        """Check if the API is running."""
        return {"status": "healthy", "version": "2.0.0"}

    # Root endpoint
    @app.get("/", tags=["Root"])
    async def root():
        """API root endpoint."""
        return {
            "message": "PhD Research Assistant API",
            "version": "2.0.0",
            "docs": "/api/docs"
        }

    return app


# Create the application instance
app = create_app()


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.reload,
        log_level=settings.log_level.lower()
    )
