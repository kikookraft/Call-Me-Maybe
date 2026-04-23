.PHONY: install run debug clean lint lint-strict

export UV_CACHE_DIR=/goinfre/$(USER)/cmm_cache
export UV_PROJECT_ENVIRONMENT=/goinfre/$(USER)/cmm_venv
export HF_HOME=/goinfre/$(USER)/hf_cache
unexport VIRTUAL_ENV

run: install
	uv run python -m src

install:
	uv sync

debug:
	uv run python -m pdb -m src

clean:
	rm -rf __pycache__ .mypy_cache src/__pycache__
	rm -rf data/output/
	rm -rf $(UV_PROJECT_ENVIRONMENT)
	rm -rf $(UV_CACHE_DIR)
	rm -rf llm_sdk/__pycache__
	rm -rf llm_sdk/llm_sdk/__pycache__

lint:
	uv run flake8 . --exclude=.venv,__init__.py
	uv run mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	uv run flake8 . --exclude=.venv,__init__.py
	uv run mypy . --strict

ask: install
	uv run python -m src --interactive