# ollama_client.py — minimal HTTP client for Ollama /api/embeddings and /api/chat
import requests
from typing import List, Dict, Any, Optional

class OllamaClient:
    def __init__(self, base_url: str = "http://localhost:11434", timeout: int = 120):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    # -------- Embeddings --------
    def embed_batch(self, model: str, texts: List[str]) -> List[List[float]]:
        """
        Ollama /api/embeddings typically accepts a single string. We'll iterate for a batch.
        """
        vecs = []
        url = f"{self.base_url}/api/embeddings"
        for t in texts:
            payload = {"model": model, "input": t}
            r = requests.post(url, json=payload, timeout=self.timeout)
            r.raise_for_status()
            data = r.json()
            if "embedding" not in data:
                raise RuntimeError(f"Unexpected embedding response: {data}")
            vecs.append(data["embedding"])
        return vecs

    # -------- Chat / Generate --------
    def chat(self, model: str, messages: List[Dict[str, str]],
             temperature: float = 0.2, stream: bool = False, options: Optional[Dict[str, Any]] = None) -> str:
        """
        Use /api/chat (non-stream) to get a single assistant message back.
        """
        url = f"{self.base_url}/api/chat"
        payload = {"model": model, "messages": messages, "stream": stream}
        if options is not None:
            payload["options"] = options
        # temperature via options (varies by model); many models accept "temperature" in options.
        if options is None:
            payload["options"] = {"temperature": temperature}
        else:
            payload["options"].update({"temperature": temperature})

        r = requests.post(url, json=payload, timeout=self.timeout)
        r.raise_for_status()
        data = r.json()
        # Non-streamed chat returns a final 'message' field
        msg = data.get("message", {})
        return msg.get("content", "").strip()
