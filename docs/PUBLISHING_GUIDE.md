
# Publishing RAG-Chatbot to GitHub

Repository: https://github.com/RatneshPathak/RAG-Chatbot

This guide documents the release-preparation workflow for publishing the RAG-Chatbot project safely. The remote repository was confirmed empty during the initial clone. Check its state again before pushing, in case it has changed.

The project has passed 65 automated tests on Python 3.11.8, and `python -m pip check` reported no broken requirements. The test run produced one non-blocking FAISS/NumPy deprecation warning.

## 1. Repository and branch

The project is prepared in the local clone:

```bash
cd ~/Downloads/RAG-Chatbot-remote
git branch --show-current
git remote -v
git status --short
```

The working branch for release preparation is:

```text
audit-and-release-prep
```

The Git remote should point to:

```text
https://github.com/RatneshPathak/RAG-Chatbot.git
```

Do not copy the original project over this clone again unless necessary. The prepared files have already been copied into this working directory.

## 2. Environment and generated-file protection

The `.gitignore` file excludes local secrets, Python environments, caches, and generated vector stores.

Important rules include:

- `.env` and `.env.*` are ignored.
- `.env.example` remains publishable as a placeholder template.
- `.venv/`, Python caches, and test caches are ignored.
- Generated files under `vectorstores/` are ignored; `vectorstores/README.md` remains publishable.
- Notebook source files are not ignored.

Verify the virtual environment is ignored:

```bash
git check-ignore -v .venv/Scripts/python.exe
```

Check for environment and private-key files:

```bash
find . -type f \( -name ".env" -o -name "*.pem" -o -name "*.key" \) \
  -not -path "./.git/*" \
  -not -path "./.venv/*"
```

Review any results before publishing. `.env.example` is expected to exist and should contain placeholders only.

If a real API key or credential was ever committed to Git history, revoke or rotate it with the provider. Removing it from the latest version alone does not make the credential safe.

## 3. Quality checks

Activate the existing virtual environment if necessary.

On Git Bash for Windows:

```bash
source .venv/Scripts/activate
```

Run the automated tests and dependency check:

```bash
python -m pytest -v
python -m pip check
```

Expected results from the verified release-preparation run:

- 65 tests passed.
- No broken Python requirements.
- One non-blocking FAISS/NumPy deprecation warning.

Check whitespace and review the files:

```bash
git diff --check
git status --short
git status --short --ignored
```

Ignored files such as `.venv/`, `.pytest_cache/`, and `__pycache__/` may appear in the ignored-file listing. They should not be staged.

Run a credential-pattern scan before publishing:

```bash
grep -RInE \
  --exclude-dir=.git \
  --exclude-dir=.venv \
  --exclude-dir=__pycache__ \
  --exclude-dir=.pytest_cache \
  --exclude='test_repo_hygiene.py' \
  --exclude='PUBLISHING_GUIDE.md' \
  'sk-[A-Za-z0-9_-]{20,}|-----BEGIN .*PRIVATE KEY-----|ghp_[A-Za-z0-9]{30,}|AKIA[0-9A-Z]{16}|xox[baprs]-[A-Za-z0-9-]{10,}' .
```

No matches is the desired result. This scan is a safeguard, not a guarantee that every possible secret has been detected.

## 4. Review and stage the project

Review the documentation, license, environment template, requirements, notebooks, sample data, and source files.

Check that:

- `.env.example` contains no real credentials.
- No local environment or generated FAISS/pickle index is staged.
- The sample PDF has its attribution and licensing information in `data/README.md`.
- The `LICENSE` covers original code and notebooks, not third-party data with separate license terms.
- The README accurately explains installation, configuration, usage, and limitations.

Stage the intended project files:

```bash
git add -A
git status --short
git diff --cached --check
git diff --cached --stat
```

Inspect the staged file list carefully:

```bash
git diff --cached --name-only
```

If any secret, `.env` file, `.venv` file, cache, or generated index appears in the staged list, stop and correct the staging before committing.

## 5. Configure Git authorship

Set the author identity for this repository:

```bash
git config user.name "Ratnesh Pathak"
git config user.email "YOUR_GITHUB_VERIFIED_OR_NOREPLY_EMAIL"
```

Replace the email placeholder with an email address associated with your GitHub account, or your GitHub-provided noreply address. Do not use the literal placeholder.

Verify the settings:

```bash
git config user.name
git config user.email
```

## 6. Commit the reviewed files

Only after reviewing the staged changes:

```bash
git commit -m "Prepare RAG chatbot for release"
```

Verify the commit:

```bash
git status --short
git log -1 --oneline
```

The working tree should be clean, apart from ignored local files.

## 7. Verify the remote and push the branch

Check the remote again:

```bash
git ls-remote origin
```

The initial clone found an empty repository. If the remote now shows refs, stop and inspect them before pushing.

If the remote is still empty, push the release-preparation branch:

```bash
git push -u origin audit-and-release-prep
```

This is a normal branch push and does not require a force-push.

## 8. Finish on GitHub

After the push succeeds:

1. Open https://github.com/RatneshPathak/RAG-Chatbot
2. Confirm that `audit-and-release-prep` contains the expected files.
3. Review the README rendering and repository contents.
4. If GitHub offers a pull request for the branch, open one and review it.
5. Merge the pull request when satisfied, or deliberately choose the repository's default-branch workflow if you prefer a different release path.

Do not force-push or rewrite history as part of this routine release process.

## GitHub publication checklist

- [ ] Automated tests pass.
- [ ] Dependencies pass `pip check`.
- [ ] No real credentials are included.
- [ ] Local virtual environments and caches are ignored.
- [ ] Generated vector stores are not staged.
- [ ] Third-party sample-data attribution is present.
- [ ] Staged changes have been reviewed.
- [ ] Git author identity is configured.
- [ ] Commit has been created.
- [ ] Remote state has been checked immediately before pushing.
- [ ] Published files and README have been verified on GitHub.
