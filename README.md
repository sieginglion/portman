# Portman

Portfolio analysis and valuation service. Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

## Quick start

```sh
uv sync
./bin/run.sh
```

Use `uv run` for project commands; do not activate `.venv` manually.

## Test

```sh
uv run --locked python -m unittest discover -v
```

## Quality checks

```sh
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked ty check
```

Fix lint and formatting issues:

```sh
uv run ruff check --fix .
uv run ruff format .
```
