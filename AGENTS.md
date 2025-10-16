# Repository Guidelines

## Project Structure & Module Organization
PyCaret’s production code sits under `pycaret/`, with domain modules such as `classification/`, `regression/`, `anomaly/`, and shared infrastructure in `internal/` (preprocessing, display backends, experiment orchestration). Feature work for the dashboard and engine lives in `src/dashboard/`, `src/analysis/`, and `src/engine/`, which mirror the runtime package but allow faster iteration. Tests are in `tests/`, grouped by feature (`tests/dashboard/`, `tests/benchmarks/`). Supporting assets reside in `docs/`, `examples/`, and `datasets/`.

## Build, Test, and Development Commands
- `pip install -e ".[dev,test,full]"` – install editable package with linting, typing, and full optional deps.
- `black pycaret/ src/` and `isort pycaret/ src/` – auto-format code; isort is already configured to mirror Black’s style.
- `flake8 pycaret/ src/` – run lint checks with the repository ignore list.
- `mypy pycaret/ src/` – type-check the public modules.
- `pytest tests/ -v` – execute the full suite; add `--cov=pycaret --cov-report=html` when collecting coverage.

## Coding Style & Naming Conventions
Use 4-space indentation and prefer expressive, snake_case names for functions, variables, and fixtures. Public classes should be PascalCase and match the experiment or container they extend (for example, `CustomModelContainer`). Keep lines comfortably under the 300-character linting limit, even though Flake8 permits it. Favor type hints on new APIs, maintain docstring sections used elsewhere in the package, and prefer module-level constants for reusable IDs.

## Testing Guidelines
Write tests alongside the feature area: dashboard features belong in `tests/dashboard/`; core engine behavior in the closest `tests/` submodule. Use `pytest` markers already defined (`benchmark`, `plotting`, `tuning_grid`, `tuning_random`) to isolate heavy scenarios; run `pytest tests/ -m "not benchmark"` before opening a PR to keep feedback loops fast. Prefer parametrized tests and reuse fixtures from `tests/conftest.py`. Regenerate coverage reports when code paths change.

## Commit & Pull Request Guidelines
Follow the bracketed prefix convention from recent history (e.g., `[BUG]`, `[MNT]`, `CI:`) and keep summaries imperative: `[BUG] Fix duplicate index detection (#4151)`. Reference issues with `Fixes #123` when relevant and document key reproduction steps or screenshots for dashboard updates. Each PR should describe scope, testing performed, and any dependency churn. Keep commits focused; group formatting changes separately from feature logic. Request review from maintainers responsible for the touched module.
