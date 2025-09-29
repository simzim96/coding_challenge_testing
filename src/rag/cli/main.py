from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel

from src.rag.core.answer import ContextDocument, OpenAIAnswerGenerator
from src.rag.core.chunker import split_text_into_chunks
from src.rag.core.embedder import SimpleTFIDFEmbedder
from src.rag.core.loader import TextLoader
from src.rag.core.retriever import Retriever

console = Console()


def build_retriever_from_file(file_path: Path) -> Retriever:
    loader = TextLoader(file_path)
    text = loader.load()
    chunks = split_text_into_chunks(text)
    embedder = SimpleTFIDFEmbedder()
    retriever = Retriever(embedder)
    retriever.index(chunks)
    return retriever


@click.group()
def cli() -> None:
    """Simple RAG CLI over a local text file."""


@cli.command()
@click.option(
    "--file",
    "file_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=Path("raw_data/ARC_Intelligence_Profile.txt"),
    show_default=True,
    help="Path to the input text file to chat with.",
)
@click.option(
    "--answer",
    "use_llm",
    is_flag=True,
    default=False,
    help="If set, and OPENAI_API_KEY is available, generate an answer using OpenAI.",
)
def chat(file_path: Path, use_llm: bool) -> None:
    """Start an interactive chat with the provided text file."""
    console.print(Panel.fit(f"Indexing file: [bold]{file_path}[/bold]"))
    retriever = build_retriever_from_file(file_path)

    console.print("Type your question. Use /exit to quit.")
    while True:
        try:
            query = console.input("[bold cyan]>[/bold cyan] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\nGoodbye!")
            break
        if not query:
            continue
        if query.lower() in {"/exit", "quit", ":q"}:
            console.print("Goodbye!")
            break

        docs = retriever.retrieve(query, k=3)
        if not docs:
            console.print("[yellow]No relevant context found.[/yellow]")
            continue

        context = "\n\n".join(d.content for d in docs)
        console.rule("Top Context")
        console.print(context)
        console.rule()

        if use_llm:
            try:
                generator = OpenAIAnswerGenerator()
                answer = generator.answer(
                    query,
                    [ContextDocument(content=d.content) for d in docs],
                )
                console.print(Panel.fit(answer, title="Answer"))
            except Exception as exc:  # noqa: BLE001
                console.print(f"[yellow]LLM unavailable or failed:[/yellow] {exc}")


if __name__ == "__main__":
    cli()
