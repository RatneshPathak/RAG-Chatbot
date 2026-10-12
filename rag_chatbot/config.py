"""Configuration helpers: environment variables, default paths, service checks.

Secrets are only ever read from the environment (optionally populated from a
local, git-ignored ``.env`` file). Nothing here prints or stores a key.
"""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
VECTORSTORES_DIR = REPO_ROOT / "vectorstores"
DEFAULT_PDF = DATA_DIR / "1. dietary supplements - for whom.pdf"

DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_OLLAMA_EMBED_MODEL = "nomic-embed-text"
DEFAULT_OLLAMA_CHAT_MODEL = "gemma3:1b"
DEFAULT_OPENAI_EMBED_MODEL = "text-embedding-3-small"

_PLACEHOLDERS = {"", "your-key-here", "your_api_key_here", "changeme", "sk-..."}


class MissingConfigError(RuntimeError):
    """A required setting or credential is missing."""


class ServiceUnavailableError(RuntimeError):
    """An external service (e.g. Ollama) is unreachable or lacks a model."""


def load_env(path: str | os.PathLike | None = None) -> bool:
    """Load a local ``.env`` file if python-dotenv is installed. Returns True if one was loaded."""
    try:
        from dotenv import load_dotenv
    except ImportError:  # pragma: no cover - python-dotenv is in requirements
        return False
    return bool(load_dotenv(path or REPO_ROOT / ".env"))


def _clean(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip().strip('"').strip("'")
    return None if value.lower() in _PLACEHOLDERS else value


def get_openai_api_key(environ: dict | None = None) -> str:
    """Return the OpenAI key from ``OPENAI_API_KEY`` (legacy name ``OPENAI_KEY`` also accepted)."""
    env = os.environ if environ is None else environ
    key = _clean(env.get("OPENAI_API_KEY")) or _clean(env.get("OPENAI_KEY"))
    if not key:
        raise MissingConfigError(
            "OPENAI_API_KEY is not set. Copy .env.example to .env, put your key in it, "
            "and restart the kernel (see README, 'OpenAI configuration')."
        )
    return key


def get_openai_chat_model(environ: dict | None = None) -> str:
    """Chat model name from ``OPENAI_CHAT_MODEL``. Deliberately has no default: model
    availability changes, so you must pick a currently available model yourself."""
    env = os.environ if environ is None else environ
    model = _clean(env.get("OPENAI_CHAT_MODEL"))
    if not model:
        raise MissingConfigError(
            "OPENAI_CHAT_MODEL is not set. Choose a chat model available to your account "
            "(see https://platform.openai.com/docs/models) and set it in .env."
        )
    return model


def get_openai_embedding_model(environ: dict | None = None) -> str:
    env = os.environ if environ is None else environ
    return _clean(env.get("OPENAI_EMBEDDING_MODEL")) or DEFAULT_OPENAI_EMBED_MODEL


def get_ollama_url(environ: dict | None = None) -> str:
    env = os.environ if environ is None else environ
    return _clean(env.get("OLLAMA_BASE_URL")) or DEFAULT_OLLAMA_URL


def get_ollama_embed_model(environ: dict | None = None) -> str:
    env = os.environ if environ is None else environ
    return _clean(env.get("OLLAMA_EMBED_MODEL")) or DEFAULT_OLLAMA_EMBED_MODEL


def get_ollama_chat_model(environ: dict | None = None) -> str:
    env = os.environ if environ is None else environ
    return _clean(env.get("OLLAMA_CHAT_MODEL")) or DEFAULT_OLLAMA_CHAT_MODEL


def _model_matches(wanted: str, installed: str) -> bool:
    # "nomic-embed-text" is installed as "nomic-embed-text:latest"
    if ":" in wanted:
        return wanted == installed
    return installed == wanted or installed.startswith(wanted + ":")


def check_ollama(models: list[str], base_url: str | None = None, client=None) -> None:
    """Raise ServiceUnavailableError with an actionable message if Ollama is down or a model is missing.

    ``client`` can be injected for testing; otherwise ``ollama.Client`` is used.
    """
    base_url = base_url or get_ollama_url()
    try:
        if client is None:
            import ollama

            client = ollama.Client(host=base_url)
        listing = client.list()
    except Exception as exc:
        raise ServiceUnavailableError(
            f"Cannot reach Ollama at {base_url} ({type(exc).__name__}). "
            "Install Ollama from https://ollama.com, start it (it runs as a background service; "
            "or run `ollama serve`), then retry."
        ) from exc
    raw = listing["models"] if isinstance(listing, dict) else getattr(listing, "models", [])
    installed = []
    for m in raw:
        name = (m.get("model") or m.get("name")) if isinstance(m, dict) else getattr(m, "model", None)
        if name:
            installed.append(name)
    missing = [w for w in models if not any(_model_matches(w, i) for i in installed)]
    if missing:
        cmds = "\n".join(f"  ollama pull {m}" for m in missing)
        raise ServiceUnavailableError(f"Ollama is running but these models are not installed:\n{cmds}")


# --- model factories (lazy imports so tests/imports do not need every provider) ---------------

def make_ollama_embeddings(model: str | None = None, base_url: str | None = None):
    from langchain_ollama import OllamaEmbeddings

    return OllamaEmbeddings(model=model or get_ollama_embed_model(), base_url=base_url or get_ollama_url())


def make_ollama_llm(model: str | None = None, base_url: str | None = None, **kwargs):
    from langchain_ollama import ChatOllama

    return ChatOllama(model=model or get_ollama_chat_model(), base_url=base_url or get_ollama_url(), **kwargs)


def make_openai_embeddings(model: str | None = None):
    try:
        from langchain_openai import OpenAIEmbeddings
    except ImportError as exc:
        raise MissingConfigError("Install OpenAI extras: pip install -r requirements-openai.txt") from exc
    return OpenAIEmbeddings(model=model or get_openai_embedding_model(), api_key=get_openai_api_key())


def make_openai_llm(model: str | None = None, **kwargs):
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:
        raise MissingConfigError("Install OpenAI extras: pip install -r requirements-openai.txt") from exc
    kwargs.setdefault("temperature", 0)
    return ChatOpenAI(model=model or get_openai_chat_model(), api_key=get_openai_api_key(), **kwargs)
