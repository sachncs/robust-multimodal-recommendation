# todo — `morel` refactor

Single-word naming. No `_` prefixes. No suffixes. No vague names.

## Phase 0 — Scaffold, brand, CI baseline

- [x] #1 `pyproject.toml` → `name = "morel"`, `version = "0.1.0"`
- [x] #2 `pyproject.toml` → `[project.scripts] morel = "morel.cli:main"`
- [x] #3 `pyproject.toml` → `setuptools_scm`; remove hard-coded version
- [x] #4 `pyproject.toml` → declare runtime + dev + extras deps
- [x] #5 `pyproject.toml` → `[tool.coverage.*]` `fail_under = 85`, branch
- [x] #6 `pyproject.toml` → `py.typed` via package-data
- [x] #7 `pyproject.toml` → optional extras `serve`, `bench`, `text`, `vision`, `dev`
- [x] #8 `pyproject.toml` → ruff lint selection
- [x] #9 `pyproject.toml` → per-file-ignores
- [x] #10 `pyproject.toml` → pydocstyle `numpy`
- [x] #11 `pyproject.toml` → mypy strict for `morel/`
- [x] #12 `pyproject.toml` → import-linter contracts
- [x] #13 `rmr/__init__.py` → remove `__version__`
- [x] #14 `README.md` → rebrand; remove PyPI claims
- [x] #15 `README.md` → remove nonexistent-symbol references
- [x] #16 `CHANGELOG.md` → `[0.3.0]` + `[Unreleased]`
- [x] #17 `SECURITY.md` → real email
- [x] #18 `CONTRIBUTING.md` → rebrand URLs
- [x] #19 `docs/*.md` → URL replace
- [x] #20 `MANIFEST.in` → NEW
- [x] #21 `.pre-commit-config.yaml` → DELETE
- [x] #22 `.github/ISSUE_TEMPLATE/*.md` → remove pre-commit
- [x] #23 `CONTRIBUTING.md` → remove pre-commit line
- [x] #24 `README.md` → remove pre-commit line
- [x] #25 `.github/workflows/ci.yml` → REWRITE
- [x] #26 `.github/workflows/release.yml` → NEW
- [x] #27 `.github/dependabot.yml` → REWRITE
- [x] #28 `.github/CODEOWNERS` → NEW
- [x] #29 `import-linter.ini` → NEW
- [x] #30 `.codecov.yml` → NEW
