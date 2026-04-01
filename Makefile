.PHONY: install run debug clean lint lint-strict

install:
	py -m uv sync

run:
	py -m uv run python -m src

debug:
	py -m uv run python -m pdb -m src

clean:
	rm -rf __pycache__ .mypy_cache src/__pycache__
	rm -rf output/
	rm -rf .venv

lint:
	py -m uv run flake8 .
	py -m uv run mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	py -m uv run flake8 .
	py -m uv run mypy . --strict
