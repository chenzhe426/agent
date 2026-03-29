from langchain_ollama import ChatOllama

from app.config import (
    AGENT_TEMPERATURE,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    OLLAMA_TIMEOUT,
)


def get_chat_llm() -> ChatOllama:
    return ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=AGENT_TEMPERATURE,
        timeout=OLLAMA_TIMEOUT,
    )