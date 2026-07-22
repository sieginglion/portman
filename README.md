# Portman

Portfolio analysis and valuation service.

## Setup

The project uses [uv](https://docs.astral.sh/uv/) to manage dependencies and
the local virtual environment. Python 3.12 is selected automatically.

```sh
uv sync
```

This creates a local `.venv/`; use `uv run` rather than activating it
manually.

## Run

Start the backend with:

```sh
./bin/run.sh
```

## Tests

```sh
uv run --locked python -m unittest discover -v
```

## Code quality

Check linting, formatting, and types with:

```sh
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked ty check
```

Apply Ruff's safe lint fixes and formatter with:

```sh
uv run ruff check --fix .
uv run ruff format .
```
