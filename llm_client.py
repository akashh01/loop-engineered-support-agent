"""
Shared client for both chat completion and embeddings.

Defaults to a local Ollama instance so the demo is reproducible without an
API key. Swapping the chat call to Claude is a one-line change, noted below.
Embeddings stay on a local model either way, since Chroma needs a
consistent embedding function across every insert and query.
"""

import requests

OLLAMA_BASE = "http://localhost:11434"
OLLAMA_CHAT_MODEL = "qwen2.5"
OLLAMA_EMBED_MODEL = "nomic-embed-text"


def call_llm(prompt: str) -> str:
    """Send a prompt to the local Ollama chat model and return the text response."""
    response = requests.post(
        f"{OLLAMA_BASE}/api/generate",
        json={"model": OLLAMA_CHAT_MODEL, "prompt": prompt, "stream": False},
        timeout=120,
    )
    response.raise_for_status()
    return response.json()["response"]


def embed_text(text: str) -> list[float]:
    """Embed a string using the local Ollama embedding model."""
    response = requests.post(
        f"{OLLAMA_BASE}/api/embeddings",
        json={"model": OLLAMA_EMBED_MODEL, "prompt": text},
        timeout=60,
    )
    response.raise_for_status()
    return response.json()["embedding"]


# --- Swap-in alternative: Claude via the Anthropic API for the chat call ---
# import anthropic
# client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
#
# def call_llm(prompt: str) -> str:
#     message = client.messages.create(
#         model="claude-sonnet-4-6",
#         max_tokens=1024,
#         messages=[{"role": "user", "content": prompt}],
#     )
#     return message.content[0].text
#
# Embeddings still need a local model even if chat moves to Claude, since
# Anthropic doesn't serve an embeddings endpoint -- keep embed_text() on
# Ollama (or swap to a hosted embeddings provider) regardless.
# -----------------------------------------------------------------------------
