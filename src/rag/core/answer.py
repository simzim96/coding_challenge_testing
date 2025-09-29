import os
from dataclasses import dataclass
from typing import List

from openai import OpenAI


@dataclass
class ContextDocument:
    content: str


class OpenAIAnswerGenerator:
    """Generates an answer from query + retrieved context using OpenAI."""

    def __init__(self, model: str = "gpt-4o-mini") -> None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY not set")
        # The OpenAI constructor picks up API key from env by default
        self.client = OpenAI()
        self.model = model

    def build_prompt(self, query: str, context_docs: List[ContextDocument]) -> str:
        context = "\n\n".join(d.content for d in context_docs)
        return (
            "You are a concise assistant. Answer the user's question strictly using the context.\n"
            "If the answer is not in the context, say you don't know.\n\n"
            f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
        )

    def answer(self, query: str, context_docs: List[ContextDocument]) -> str:
        prompt = self.build_prompt(query, context_docs)
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )
        return completion.choices[0].message.content.strip()
