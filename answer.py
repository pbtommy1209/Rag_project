# answer.py — call Ollama chat model with RAG prompt and citations
import argparse, json, os, textwrap
from typing import List, Dict
from ollama_client import OllamaClient

SYS_PROMPT = """You are a precise research assistant. You will be given a user question and several passages from a single source.
Answer **concisely** using only the information in the passages. Where you use a piece of information, **quote** a short span
and **cite** it with a bracket like [p1], [p2], etc. If the answer cannot be found, say "I couldn't find that in the document."."""

def build_context(passages: List[Dict]) -> str:
    """
    Format passages with ids for citation.
    """
    lines = []
    for i, p in enumerate(passages, 1):
        lines.append(f"[p{i}] {p['chunk']}")
    return "\n\n".join(lines)

def rag_answer(question: str, passages: List[Dict], chat_model: str, ollama_url: str, temperature: float = 0.2) -> str:
    client = OllamaClient(ollama_url)
    ctx = build_context(passages)
    user_prompt = f"""Question: {question}

Passages:
{ctx}

Write the answer with citations like [p#]."""
    messages = [
        {"role": "system", "content": SYS_PROMPT},
        {"role": "user", "content": user_prompt}
    ]
    out = client.chat(model=chat_model, messages=messages, temperature=temperature, stream=False)
    return out.strip()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--question", required=True)
    ap.add_argument("--passages_json", required=True, help="JSON file with a list of passages (from retrieve.py)")
    ap.add_argument("--chat-model", type=str, default="llama3.2:latest")
    ap.add_argument("--ollama-url", type=str, default="http://localhost:11434")
    args = ap.parse_args()

    with open(args.passages_json, "r", encoding="utf-8") as f:
        passages = json.load(f)
    print(rag_answer(args.question, passages, args.chat_model, args.ollama_url))

if __name__ == "__main__":
    main()
