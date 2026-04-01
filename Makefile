.PHONY: install run debug clean lint lint-strict

install:
	python3 -m uv sync

run:
	python3 -m uv run python -m src

debug:
	python3 -m uv run python -m pdb -m src

clean:
	rm -rf __pycache__ .mypy_cache src/__pycache__
	rm -rf output/
	rm -rf .venv

lint:
	python3 -m uv run flake8 .
	python3 -m uv run mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	python3 -m uv run flake8 .
	python3 -m uv run mypy . --strict
