# arc-rag-cli

A simple RAG (Retrieval-Augmented Generation) application that indexes a local text file and lets you chat with it via the terminal. Includes:

- Chunking + TF‑IDF retrieval over a text file
- Optional LLM answering with OpenAI
- Optional multi‑turn Agent using OpenAI tool‑calling with a retrieval tool
- Minimal CLI and a simple `main.py` runner
- Tests via `pytest`

## Tech Stack
- Python 3.12+
- Poetry
- Retrieval: scikit‑learn TF‑IDF + cosine similarity
- CLI: click, rich
- Optional LLM/Agent: openai

## Project Layout
- `raw_data/ARC_Intelligence_Profile.txt`: sample source text
- `src/rag/core/`: loader, chunker, embedder, retriever, answer (OpenAI), agent (tool‑calling)
- `src/rag/cli/main.py`: CLI entrypoint (module)
- `scripts/chat.py`: simple runnable script
- `main.py`: root runner for quick start
- `tests/`: pytest tests

## Setup
1) Install dependencies

```bash
poetry install --no-root
```

2) (Optional) Create a `.env` file for OpenAI

```bash
# .env
OPENAI_API_KEY=sk-...
```

`main.py` automatically loads `.env` via `python-dotenv`.

## Running
### Fastest start (root runner)
- Without LLM answers:

```bash
PYTHONPATH=. python main.py --file raw_data/ARC_Intelligence_Profile.txt
```

- With LLM answers:

```bash
PYTHONPATH=. python main.py --file raw_data/ARC_Intelligence_Profile.txt --answer
```

- With multi‑turn Agent (tool‑calling over your index):

```bash
PYTHONPATH=. python main.py --file raw_data/ARC_Intelligence_Profile.txt --agent
```

Type questions; use `/exit` to quit.

### Module CLI (no install)
```bash
poetry run python -m src.rag.cli.main chat --file raw_data/ARC_Intelligence_Profile.txt --answer
```

### Simple script
```bash
poetry run python scripts/chat.py --file raw_data/ARC_Intelligence_Profile.txt --answer
```

### Optional: install the project to get a `rag` command
If you want `poetry run rag ...`, add packages for the `src` layout in your `pyproject.toml`:

```toml
[tool.poetry]
packages = [{ include = "rag", from = "src" }]
```

Then:

```bash
echo "# arc-rag-cli" > README.md  # if missing
poetry install
poetry run rag chat --file raw_data/ARC_Intelligence_Profile.txt --answer
```

## Tests
Run all tests:

```bash
poetry run pytest -q
```

## How it works
- `loader.TextLoader`: reads the source text file
- `chunker.split_text_into_chunks`: splits text with overlap (defaults 800 chars, 100 overlap)
- `embedder.SimpleTFIDFEmbedder`: builds TF‑IDF index; cosine similarity for retrieval
- `retriever.Retriever`: indexes chunks and returns top‑k by similarity
- `answer.OpenAIAnswerGenerator`: builds a concise prompt from retrieved chunks and calls OpenAI
- `agent.RetrievalToolAgent`: maintains conversation history and uses OpenAI tool‑calling with a `search_docs` tool backed by the retriever

## Common commands
- Format/lint via pre‑commit:

```bash
poetry run pre-commit run --all-files
```

- Start chat with agent:

```bash
PYTHONPATH=. python main.py --agent --file raw_data/ARC_Intelligence_Profile.txt
```

## Security & notes
- Do not commit secrets. Place keys in `.env` or export in your shell.
- We do not log sensitive values.
- The TF‑IDF retriever is lightweight and good for an MVP; swap for embeddings as needed.
