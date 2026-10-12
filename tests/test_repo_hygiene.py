
"""Static checks on the repository itself: notebooks, secrets, requirements, ignores, authorship."""

import ast
import importlib
import json
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
NOTEBOOKS = sorted(REPO.glob("*.ipynb"))
SKIP_DIRS = {
    ".git",
    ".venv",
    "__pycache__",
    ".pytest_cache",
    ".ipynb_checkpoints",
}


def _text_files():
    """Yield repository text files while excluding generated/binary content."""
    excluded_suffixes = {".pdf", ".png", ".faiss", ".pkl"}

    for path in REPO.rglob("*"):
        if not path.is_file():
            continue

        relative_parts = set(path.relative_to(REPO).parts)

        if relative_parts & SKIP_DIRS:
            continue

        if path.suffix.lower() in excluded_suffixes:
            continue

        yield path


def test_six_notebooks_present():
    assert {notebook.name for notebook in NOTEBOOKS} == {
        "01-langchain_Basics.ipynb",
        "02-langchain_Prompts.ipynb",
        "03-langchain_chains.ipynb",
        "Embeddings_generation_withOllama.ipynb",
        "RAG_WITHOLLAMA.ipynb",
        "RAG_withOpenai.ipynb",
    }


@pytest.mark.parametrize("nb", NOTEBOOKS, ids=lambda path: path.name)
def test_notebook_valid_and_code_compiles(nb):
    import nbformat

    data = nbformat.read(nb, as_version=4)
    nbformat.validate(data)

    for index, cell in enumerate(data.cells):
        if cell.cell_type == "code":
            ast.parse(cell.source, filename=f"{nb.name}[{index}]")


@pytest.mark.parametrize("nb", NOTEBOOKS, ids=lambda path: path.name)
def test_notebook_imports_resolve(nb):
    import nbformat

    missing = []

    for cell in nbformat.read(nb, as_version=4).cells:
        if cell.cell_type != "code":
            continue

        for node in ast.walk(ast.parse(cell.source)):
            modules = []

            if (
                isinstance(node, ast.ImportFrom)
                and node.module
                and node.level == 0
            ):
                modules = [(node.module, [alias.name for alias in node.names])]

            elif isinstance(node, ast.Import):
                modules = [(alias.name, []) for alias in node.names]

            for module_name, names in modules:
                # Optional demo dependency; the notebook may guard its use.
                if module_name == "sentence_transformers":
                    continue

                if module_name.split(".")[0] == "langchain_openai":
                    pytest.importorskip("langchain_openai")

                try:
                    module = importlib.import_module(module_name)
                except ImportError as error:
                    missing.append(f"{module_name}: {error}")
                    continue

                for name in names:
                    if name == "*" or hasattr(module, name):
                        continue

                    if not _is_submodule(module_name, name):
                        missing.append(f"{module_name}.{name}")

    assert not missing, missing


def _is_submodule(module_name, name):
    try:
        importlib.import_module(f"{module_name}.{name}")
        return True
    except ImportError:
        return False


@pytest.mark.parametrize("nb", NOTEBOOKS, ids=lambda path: path.name)
def test_notebooks_have_no_local_paths_or_stderr_noise(nb):
    raw = nb.read_text(encoding="utf-8")

    forbidden_patterns = [
        r"C:\\Users",
        r"c:\\ML",
        "AppData",
        r"/Users/",
        r"/home/\w+/",
        "LU_Rag",
    ]

    for pattern in forbidden_patterns:
        assert not re.search(pattern, raw, re.IGNORECASE), (
            f"{nb.name} contains a local-path marker: {pattern}"
        )

    data = json.loads(raw)

    for cell in data["cells"]:
        for output in cell.get("outputs", []):
            assert not (
                output.get("output_type") == "stream"
                and output.get("name") == "stderr"
            ), f"{nb.name} contains captured stderr output"


def test_no_secrets_in_repo():
    patterns = [
        r"sk-[A-Za-z0-9_-]{20,}",
        r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
        r"ghp_[A-Za-z0-9]{30,}",
        r"AKIA[0-9A-Z]{16}",
        r"xox[baprs]-[A-Za-z0-9-]{10,}",
        r"""(?i)api[_-]?key\s*=\s*['"][A-Za-z0-9_\-]{16,}['"]""",
    ]

    hits = []

    for path in _text_files():
        # This test file intentionally contains detection patterns as examples.
        if path.name == "test_repo_hygiene.py":
            continue

        content = path.read_text(encoding="utf-8", errors="ignore")

        for pattern in patterns:
            if re.search(pattern, content):
                hits.append(f"{path.relative_to(REPO)} matches {pattern}")

    assert not hits, hits


def test_env_and_venv_are_gitignored():
    """Local environment files may exist, but Git ignore rules must protect them."""
    gitignore_path = REPO / ".gitignore"
    assert gitignore_path.is_file(), ".gitignore is missing"

    patterns = {
        line.strip()
        for line in gitignore_path.read_text(encoding="utf-8").splitlines()
    }

    assert ".env" in patterns, ".env must be ignored"
    assert ".env.*" in patterns, "Environment-specific .env files must be ignored"
    assert "!.env.example" in patterns, ".env.example must remain publishable"
    assert ".venv/" in patterns, "The local virtual environment must be ignored"
    assert "__pycache__/" in patterns, "Python caches must be ignored"

    # Do not fail merely because a local virtual environment or cache exists.
    # These files belong on the developer's machine, not in the published source.


def test_requirements_are_utf8_and_pinned():
    for name in [
        "requirements.txt",
        "requirements-openai.txt",
        "requirements-dev.txt",
    ]:
        path = REPO / name
        assert path.is_file(), f"{name} is missing"

        raw = path.read_bytes()
        assert not raw.startswith(
            (b"\xff\xfe", b"\xfe\xff")
        ), f"{name} must be UTF-8, not UTF-16"

        raw.decode("utf-8")

    requirements = [
        line.strip()
        for line in (REPO / "requirements.txt")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]

    assert requirements, "requirements.txt must not be empty"
    assert all("==" in line for line in requirements), (
        "Every direct requirement in requirements.txt must be version-pinned"
    )

    assert not any(
        line.lower().startswith(("pywin32", "pypdf2"))
        for line in requirements
    )


def test_gitignore_covers_essentials_but_not_source():
    gitignore = (REPO / ".gitignore").read_text(encoding="utf-8")

    for entry in [".env", ".venv/", "__pycache__/", "vectorstores/*"]:
        assert entry in gitignore, f".gitignore must contain {entry}"

    assert "!.env.example" in gitignore
    assert "*.ipynb" not in gitignore
    assert "*.pdf" not in gitignore

    assert (REPO / ".env.example").is_file()

    example_env = (REPO / ".env.example").read_text(encoding="utf-8")
    assert "sk-" not in example_env, (
        ".env.example must contain placeholders, not an API key"
    )


def test_authorship_is_ratnesh_pathak():
    license_text = (REPO / "LICENSE").read_text(encoding="utf-8")

    assert "Copyright (c) 2025 Ratnesh Pathak" in license_text

    for notebook in NOTEBOOKS:
        metadata = json.loads(
            notebook.read_text(encoding="utf-8")
        )["metadata"]

        assert metadata["authors"] == [{"name": "Ratnesh Pathak"}]

    readme = (REPO / "README.md").read_text(encoding="utf-8")
    assert "Ratnesh Pathak" in readme


def test_repo_size_reasonable():
    total_size = sum(
        path.stat().st_size
        for path in REPO.rglob("*")
        if path.is_file()
        and not (set(path.relative_to(REPO).parts) & SKIP_DIRS)
    )

    assert total_size < 5_000_000, (
        f"Repository files exceed the 5 MB limit: {total_size} bytes"
    )


def test_scripts_import_and_show_help():
    import subprocess
    import sys

    for script_name in [
        "build_index.py",
        "validate_index.py",
        "ask.py",
    ]:
        script_path = REPO / "scripts" / script_name

        assert script_path.is_file(), f"Missing script: {script_name}"

        result = subprocess.run(
            [sys.executable, str(script_path), "--help"],
            capture_output=True,
            text=True,
            cwd=REPO,
        )

        assert result.returncode == 0, (
            f"{script_name} --help failed:\n{result.stderr}"
        )
