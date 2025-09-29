import json
from typing import Any, Dict, List

from openai import OpenAI

from src.rag.core.retriever import Retriever

TOOLS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_docs",
            "description": "Retrieve top-k chunks relevant to the query from the indexed corpus.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "User query to search for",
                    },
                    "k": {"type": "integer", "default": 3},
                },
                "required": ["query"],
            },
        },
    }
]


class RetrievalToolAgent:
    """Agent that uses OpenAI tool-calling with a retrieval tool and keeps chat history."""

    def __init__(self, retriever: Retriever, model: str = "gpt-4o-mini") -> None:
        self.retriever = retriever
        self.model = model
        self.client = OpenAI()
        self.messages: List[Dict[str, Any]] = [
            {
                "role": "system",
                "content": (
                    "You are a concise assistant. "
                    "Use the 'search_docs' tool to look up information in the indexed document. "
                    "Cite only from retrieved chunks. "
                    "If the answer is not present, say you don't know."
                ),
            }
        ]

    def _handle_tool_calls(self, tool_calls: List[Any], fallback_query: str) -> None:
        for call in tool_calls:
            name = call.function.name
            if name != "search_docs":
                continue
            try:
                args = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {"query": fallback_query, "k": 3}
            query = args.get("query", fallback_query)
            k = int(args.get("k", 3))
            docs = self.retriever.retrieve(query, k=k)
            tool_output = "\n\n".join(d.content for d in docs) if docs else ""
            # Append the tool result message tied to the tool_call_id
            self.messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": tool_output or "",
                }
            )

    def ask(self, user_input: str) -> str:
        # Add user turn
        self.messages.append({"role": "user", "content": user_input})

        # First pass: allow tool-calls
        first = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            tools=TOOLS,
        )
        assistant_message = first.choices[0].message
        self.messages.append(
            {
                "role": "assistant",
                "content": assistant_message.content or "",
                "tool_calls": assistant_message.tool_calls,
            }
        )

        # If tools were requested, execute and send results, then get final answer
        if assistant_message.tool_calls:
            self._handle_tool_calls(
                assistant_message.tool_calls, fallback_query=user_input
            )
            follow_up = self.client.chat.completions.create(
                model=self.model,
                messages=self.messages,
            )
            final_msg = follow_up.choices[0].message
            self.messages.append(
                {"role": "assistant", "content": final_msg.content or ""}
            )
            return (final_msg.content or "").strip()

        # If no tool calls, return the assistant content directly
        return (assistant_message.content or "").strip()
