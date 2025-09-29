#!/usr/bin/env python3
import argparse
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel

from src.rag.core.answer import ContextDocument, OpenAIAnswerGenerator
from src.rag.core.chunker import split_text_into_chunks
from src.rag.core.embedder import SimpleTFIDFEmbedder
from src.rag.core.loader import TextLoader
from src.rag.core.retriever import Retriever


def build_retriever(file_path: Path) -> Retriever:
    loader = TextLoader(file_path)
    text = loader.load()
    chunks = split_text_into_chunks(text)
    embedder = SimpleTFIDFEmbedder()
    retriever = Retriever(embedder)
    retriever.index(chunks)
    return retriever


def main() -> None:
    # Load environment variables from a local .env file if present
    load_dotenv()
    parser = argparse.ArgumentParser(description="Chat with a local text file")
    parser.add_argument(
        "--file",
        type=Path,
        default=Path("raw_data/ARC_Intelligence_Profile.txt"),
        help="Path to the input text file",
    )
    parser.add_argument(
        "--answer",
        action="store_true",
        help="If set, and OPENAI_API_KEY is available, generate an answer using OpenAI.",
    )
    args = parser.parse_args()

    console = Console()
    console.print(Panel.fit(f"Indexing file: [bold]{args.file}[/bold]"))
    retriever = build_retriever(args.file)

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

        if args.answer:
            try:
                generator = OpenAIAnswerGenerator()
                answer = generator.answer(
                    query, [ContextDocument(content=d.content) for d in docs]
                )
                console.print(Panel.fit(answer, title="Answer"))
            except Exception as exc:  # noqa: BLE001
                console.print(f"[yellow]LLM unavailable or failed:[/yellow] {exc}")


if __name__ == "__main__":
    main()
