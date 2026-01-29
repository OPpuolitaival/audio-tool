set -x
export PYTHONPATH=$(pwd)

uv sync

uv run ruff check --fix .
uv run ruff format
