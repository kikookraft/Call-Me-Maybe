.PHONY: install run debug clean lint lint-strict

run: install
	uv run python -m src

install:
	uv sync

debug:
	uv run python -m pdb -m src

clean:
	rm -rf __pycache__ .mypy_cache src/__pycache__
	rm -rf data/output/
	rm -rf .venv

lint:
	uv run flake8 . --exclude=.venv,__init__.py
	uv run mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	uv run flake8 . --exclude=.venv,__init__.py
	uv run mypy . --strict
