import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def run(*args, env=None):
    import os

    e = {**os.environ, **(env or {})}
    return subprocess.run([sys.executable, *args], capture_output=True, text=True, cwd=REPO, env=e)


def test_validate_script_ok_and_detects_tamper(store, tmp_path):
    d = tmp_path / "idx"
    ok = run("scripts/validate_index.py", str(d))
    assert ok.returncode == 0 and '"ok": true' in ok.stdout
    (d / "index.pkl").write_bytes((d / "index.pkl").read_bytes() + b"x")
    bad = run("scripts/validate_index.py", str(d))
    assert bad.returncode == 1 and "INVALID" in bad.stderr


def test_validate_script_missing_dir(tmp_path):
    r = run("scripts/validate_index.py", str(tmp_path / "nothing"))
    assert r.returncode == 1 and "manifest" in r.stderr


def test_build_openai_without_key_fails_cleanly(tmp_path):
    r = run("scripts/build_index.py", "--provider", "openai", "--out", str(tmp_path / "o"),
            env={"OPENAI_API_KEY": "", "OPENAI_KEY": "", "OLLAMA_BASE_URL": ""})
    assert r.returncode == 1 and "OPENAI_API_KEY" in r.stderr and not (tmp_path / "o").exists()


def test_build_ollama_unreachable_fails_cleanly(tmp_path):
    r = run("scripts/build_index.py", "--provider", "ollama", "--out", str(tmp_path / "o"),
            env={"OLLAMA_BASE_URL": "http://127.0.0.1:9"})
    assert r.returncode == 1 and "Cannot reach Ollama" in r.stderr


def test_ask_without_index_fails_cleanly(tmp_path):
    r = run("scripts/ask.py", "--provider", "ollama", "--index", str(tmp_path / "none"))
    assert r.returncode == 1 and "manifest" in r.stderr
