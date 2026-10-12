import pytest

from rag_chatbot import config


def test_openai_key_missing_raises_without_leaking():
    with pytest.raises(config.MissingConfigError) as e:
        config.get_openai_api_key({})
    assert "OPENAI_API_KEY" in str(e.value)


@pytest.mark.parametrize("val", ["", "your-key-here", "  ", "changeme"])
def test_openai_key_placeholders_rejected(val):
    with pytest.raises(config.MissingConfigError):
        config.get_openai_api_key({"OPENAI_API_KEY": val})


def test_openai_key_read_and_legacy_name():
    assert config.get_openai_api_key({"OPENAI_API_KEY": "abc123"}) == "abc123"
    assert config.get_openai_api_key({"OPENAI_KEY": '"legacy"'}) == "legacy"


def test_chat_model_has_no_silent_default():
    with pytest.raises(config.MissingConfigError):
        config.get_openai_chat_model({})
    assert config.get_openai_chat_model({"OPENAI_CHAT_MODEL": "m"}) == "m"


def test_defaults():
    assert config.get_ollama_url({}) == "http://localhost:11434"
    assert config.get_ollama_embed_model({}) == "nomic-embed-text"
    assert config.get_ollama_chat_model({}) == "gemma3:1b"
    assert config.get_openai_embedding_model({}) == "text-embedding-3-small"


def test_paths_exist():
    assert config.DATA_DIR.is_dir() and config.DEFAULT_PDF.is_file()


class _Client:
    def __init__(self, names=None, boom=False):
        self.names, self.boom = names or [], boom

    def list(self):
        if self.boom:
            raise ConnectionError("refused")
        return {"models": [{"model": n} for n in self.names]}


def test_check_ollama_unreachable():
    with pytest.raises(config.ServiceUnavailableError, match="Cannot reach Ollama"):
        config.check_ollama(["x"], client=_Client(boom=True))


def test_check_ollama_missing_model_gives_pull_command():
    with pytest.raises(config.ServiceUnavailableError, match="ollama pull gemma3:1b"):
        config.check_ollama(["nomic-embed-text", "gemma3:1b"], client=_Client(["nomic-embed-text:latest"]))


def test_check_ollama_ok_with_latest_tag():
    config.check_ollama(["nomic-embed-text"], client=_Client(["nomic-embed-text:latest"]))


def test_check_ollama_real_client_unreachable_port():
    with pytest.raises(config.ServiceUnavailableError):
        config.check_ollama(["x"], base_url="http://127.0.0.1:9")  # nothing listens here
