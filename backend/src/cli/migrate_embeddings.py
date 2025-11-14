"""
CLI tool to migrate project embeddings from MiniLM to OpenAI.

This script re-indexes papers with new embedding provider (OpenAI).
"""

import sys
import logging
from pathlib import Path
import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from typing import Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import get_settings, Settings
from src.services.project_service import ProjectService
from src.services.paper_service import PaperService
from src.core.database import DatabaseManager, Paper
from src.core.vector_store import VectorStoreManager
from src.core.llm import LLMManager, PromptManager
from sqlmodel import select

# Setup
app = typer.Typer()
console = Console()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def estimate_cost(num_papers: int, avg_pages: int = 20) -> dict:
    """
    Estimate migration cost.

    Args:
        num_papers: Number of papers to process
        avg_pages: Average pages per paper

    Returns:
        Dictionary with cost estimates
    """
    # Assume ~500 tokens per page, 4 pages per chunk
    tokens_per_paper = (avg_pages / 4) * 500 * 4  # chunks * tokens_per_chunk
    total_tokens = num_papers * tokens_per_paper

    # OpenAI text-embedding-3-large: $0.13 per 1M tokens
    cost_usd = (total_tokens / 1_000_000) * 0.13

    return {
        "num_papers": num_papers,
        "estimated_tokens": int(total_tokens),
        "estimated_cost_usd": round(cost_usd, 2)
    }


def validate_settings(settings: Settings) -> bool:
    """
    Validate that required settings are configured.

    Args:
        settings: Application settings

    Returns:
        True if valid, False otherwise
    """
    if settings.embedding_provider == "openai":
        if not settings.openai_api_key:
            console.print("[red]Error: OPENAI_API_KEY not set in .env file[/red]")
            return False

    return True


@app.command()
def migrate(
    project_id: Optional[str] = typer.Option(None, help="Specific project ID to migrate (or all if not specified)"),
    provider: str = typer.Option("openai", help="Embedding provider (openai, minilm)"),
    model: str = typer.Option("text-embedding-3-large", help="Embedding model name"),
    dry_run: bool = typer.Option(False, help="Show what would be done without doing it"),
    force: bool = typer.Option(False, help="Skip confirmation prompts")
):
    """
    Migrate project embeddings to a new provider.

    This will:
    1. Delete existing ChromaDB collection
    2. Recreate collection with new embedding dimensions
    3. Re-process all papers with new embeddings
    """
    console.print("[bold blue]PhD Research Assistant - Embedding Migration Tool[/bold blue]\n")

    # Load settings
    settings = get_settings()

    # Override embedding provider settings
    settings.embedding_provider = provider
    settings.embedding_model = model

    # Validate settings
    if not validate_settings(settings):
        raise typer.Exit(code=1)

    # Get projects root
    projects_root = Path.cwd().parent / "projects"
    if not projects_root.exists():
        projects_root = Path.cwd() / "projects"

    project_service = ProjectService(projects_root=projects_root)

    # Determine which projects to migrate
    if project_id:
        projects_to_migrate = [project_id]
        console.print(f"Migrating single project: [cyan]{project_id}[/cyan]\n")
    else:
        projects = project_service.list_projects()
        projects_to_migrate = [p.id for p in projects]
        console.print(f"Found [cyan]{len(projects_to_migrate)}[/cyan] projects to migrate\n")

    # Process each project
    for pid in projects_to_migrate:
        console.print(f"\n[bold]Project: {pid}[/bold]")
        console.print("=" * 60)

        try:
            # Get managers
            db_manager = project_service.get_db_manager(pid)

            # Count papers
            with db_manager.get_session() as session:
                papers = session.exec(select(Paper)).all()
                num_papers = len(papers)

            if num_papers == 0:
                console.print("[yellow]No papers found, skipping[/yellow]")
                continue

            # Estimate cost
            cost_info = estimate_cost(num_papers)
            console.print(f"Papers to process: [cyan]{cost_info['num_papers']}[/cyan]")
            console.print(f"Estimated tokens: [cyan]{cost_info['estimated_tokens']:,}[/cyan]")
            console.print(f"Estimated cost: [green]${cost_info['estimated_cost_usd']}[/green]")

            if dry_run:
                console.print("[yellow]DRY RUN - No changes made[/yellow]")
                continue

            # Confirm
            if not force:
                confirm = typer.confirm(
                    f"\nProceed with migrating {pid}?",
                    default=False
                )
                if not confirm:
                    console.print("[yellow]Skipped[/yellow]")
                    continue

            # Delete and recreate vector store
            console.print("\n[yellow]Deleting existing vector store...[/yellow]")
            vector_store_path = projects_root / pid / "data" / "vector_store"
            if vector_store_path.exists():
                import shutil
                shutil.rmtree(vector_store_path)
                console.print("[green]✓[/green] Deleted old vector store")

            # Create new vector store with new embeddings
            console.print(f"[yellow]Creating new vector store with {provider}/{model}...[/yellow]")
            vector_manager = project_service.get_vector_manager(pid)
            console.print(f"[green]✓[/green] Created new vector store (dim: {vector_manager.embedding_dimension})")

            # Initialize LLM and prompt managers
            llm_config = {}
            if settings.openai_api_key:
                llm_config["openai"] = {
                    "api_key": settings.openai_api_key,
                    "model": settings.openai_model
                }
            if settings.deepseek_api_key:
                llm_config["deepseek"] = {
                    "api_key": settings.deepseek_api_key,
                    "model": settings.deepseek_model
                }
            llm_config["default_provider"] = settings.default_llm_provider

            llm_manager = LLMManager(config=llm_config)
            prompt_manager = PromptManager()

            # Create paper service
            project_root = projects_root / pid
            paper_service = PaperService(
                db_manager=db_manager,
                vector_manager=vector_manager,
                llm_manager=llm_manager,
                prompt_manager=prompt_manager,
                project_root=project_root
            )

            # Re-process all papers
            console.print("\n[yellow]Re-indexing papers with new embeddings...[/yellow]")

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                console=console
            ) as progress:
                task = progress.add_task("Processing papers...", total=num_papers)

                for paper in papers:
                    try:
                        # Re-embed the paper
                        if paper.file_path and Path(paper.file_path).exists():
                            # Extract and embed paper content
                            from src.core.utils.pdf_processor import extract_text_from_pdf

                            text_chunks = extract_text_from_pdf(
                                paper.file_path,
                                pages_per_chunk=settings.pdf_pages_per_chunk
                            )

                            # Add to vector store
                            metadata = {
                                "title": paper.title,
                                "authors": paper.authors,
                                "year": paper.year
                            }

                            vector_manager.add_paper_chunks(
                                paper_id=paper.id,
                                chunks=text_chunks,
                                metadata=metadata
                            )

                            progress.console.print(
                                f"  [green]✓[/green] {paper.title[:50]}..."
                            )
                        else:
                            progress.console.print(
                                f"  [yellow]⚠[/yellow] Skipped (file not found): {paper.title[:50]}..."
                            )

                    except Exception as e:
                        progress.console.print(
                            f"  [red]✗[/red] Error processing {paper.title[:50]}: {e}"
                        )

                    progress.advance(task)

            console.print(f"\n[bold green]✓ Migration complete for {pid}[/bold green]")

        except Exception as e:
            console.print(f"[bold red]Error migrating {pid}: {e}[/bold red]")
            logger.exception(e)
            continue

    console.print("\n[bold green]All migrations complete![/bold green]")


if __name__ == "__main__":
    app()
