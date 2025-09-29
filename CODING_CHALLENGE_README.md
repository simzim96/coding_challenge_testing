# Coding Challenge: LLM Answer Correctness Testing

## Goal
Add an automated evaluation framework to this repo to measure the correctness of the answers produced by the LLM in our RAG pipeline. You may use `deepeval`, `ragas`, or another well‑maintained OSS library of your choice. Prefer a minimal, focused setup that is easy to run in CI.

## Context
This project exposes a simple RAG system over a local text file and provides two LLM modes:
- Answer generation over retrieved chunks (`--answer`)
- Multi‑turn Agent using a retrieval tool (`--agent`)

We want an evaluation suite that can score answer correctness (and optionally faithfulness/groundedness) against a small, curated set of Q&A examples derived from the provided document(s).

## Requirements
- Add one evaluation framework (deepeval/ragas/other) to the project with Poetry
- Create a small evaluation dataset (5–15 Q&A items) grounded in `raw_data/ARC_Intelligence_Profile.txt`
- Implement an evaluation script, e.g., `scripts/eval_answers.py`, that:
  - Runs the RAG pipeline to produce answers for each question
  - Scores answers using the chosen framework
  - Prints a concise report (overall score + per‑question breakdown)
- Add at least one thresholded assertion so CI can fail if quality regresses (e.g., average score ≥ 0.6)
- Provide clear README notes for how to run locally

## Suggested Approach
1) Choose a framework:
   - `deepeval`: easy to start, supports metrics like faithfulness, answer relevancy, and context precision
   - `ragas`: purpose‑built for RAG; offers faithfulness, answer relevancy, context recall/precision
2) Build a tiny evaluation set in `eval_data/qa.jsonl` with fields:
   - `{"question": str, "expected_answer": str}`
3) Implement `scripts/eval_answers.py`:
   - Load `.env` if present
   - Build the retriever from the same file(s)
   - For each question, get top‑k chunks, call the LLM in a deterministic mode (low temperature), and collect answers
   - Score with your chosen metric(s)
   - Emit a summary table and an overall score
4) Add a pytest (or standalone) check that fails under a configurable threshold

## Deliverables
- Dependencies added in `pyproject.toml`
- `eval_data/qa.jsonl` with 5–15 examples
- `scripts/eval_answers.py` runnable via Poetry
- `README` (or section in this file) with run instructions
- Optional: CI note (how you would wire this in GitHub Actions)

## Evaluation Criteria
- **Correctness measurement**: Does the metric reasonably reflect answer correctness/groundedness?
- **Reproducibility**: Is it easy to run locally with clear instructions? Are seeds/temperatures controlled?
- **Simplicity**: Minimal boilerplate; easy to modify/extend in the future
- **Integration**: Uses existing retriever/agent without large refactors
- **Reporting**: Output is readable and actionable (overall score, per‑item details)
- **Thresholding**: Contains a fail‑fast mechanism for regression

## Getting Started (example)
- Install deps (example with deepeval):

```bash
poetry add deepeval
```

- Add Q&A examples under `eval_data/qa.jsonl`
- Run evaluation script:

```bash
poetry run python scripts/eval_answers.py --file raw_data/ARC_Intelligence_Profile.txt --k 3 --threshold 0.6
```

This should produce a concise report and a non‑zero exit code if the threshold is not met.

---
Feel free to propose improvements (additional metrics, dataset structure, CI wiring). Aim for a pragmatic MVP first.
