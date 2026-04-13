*This project has been created as part of the 42 curriculum by tobesson.*

# Call Me Maybe - Introduction to function calling in LLMs

## Description
This project aims to implement a function calling tool that translates prompts into structured function calls using constrained decoding with a small language model (`Qwen/Qwen3-0.6B`).  
It bridges the gap between human requests and precise function execution by guiding the model token-by-token to output 100% valid JSON matching a specific schema.

## Instructions
To install and run this project, make sure you have Python 3.10+ and [`uv`](https://github.com/astral-sh/uv) installed on your system. 

### Model Installation and Setup
This project uses the `Qwen/Qwen3-0.6B` model via the provided `llm_sdk`. To ensure the model is properly downloaded and configured:
1. Ensure `uv` is installed (`pip install uv` or via system package manager).
2. Install the project dependencies including the `llm_sdk` by running:
   ```bash
   make install
   ```
3. The SDK will automatically handle fetching the required model artifacts when you run the main script. Alternatively, you might need to run the specific SDK preparation script if provided, or simply run the main application:

```bash
uv run python -m src --functions_definition data/input/functions_definition.json --input data/input/function_calling_tests.json --output data/output/function_calling_results.json
```

### Makefile Targets
- `make install`: Install project dependencies using `uv`.
- `make run`: Execute the main script of the project.
- `make debug`: Run the main script in debug mode using Python's built-in debugger.
- `make clean`: Remove temporary files, caches, and build artifacts.
- `make lint`: Execute `flake8` and `mypy` with provided configuration to ensure code quality.
- `make lint-strict`: Execute `flake8` and `mypy` natively in strict mode.

## Resources
- [Qwen models on HuggingFace](https://huggingface.co/Qwen)
- Official Python [typing documentation](https://docs.python.org/3/library/typing.html)
- [Pydantic documentation](https://docs.pydantic.dev/latest/)
- [Constrained Decoding in LLMs](https://arxiv.org/abs/2307.09702)

**AI Usage:** AI was used in this project to assist in understanding constrained decoding, readme writing,
and generating regular expressions for docstrings validation.

## Algorithm Explanation
The solution first asks the model to choose a function name from the available definitions using token-level prefix constraints, so only names that match a valid function remain possible during decoding. Once the function is selected, the program generates short value candidates for each parameter, validates them against the expected type, and assembles the final result with Pydantic before writing it with `json.dumps`. This keeps the output schema strict while still letting the model decide the function call.

## Design Decisions
- `pydantic` validates function definitions, inputs, and the final output object so schema mistakes are caught early.
- The decoder stays small on purpose: it only constrains the function-name choice, then lets the program validate and serialize the final parameters.

## Performance Analysis
Constraining the function-name search space keeps decoding focused on the valid function set and removes malformed output at the schema level. The final JSON serialization is deterministic, so the program stays reliable even when the model produces noisy intermediate text. The implementation remains lightweight enough to stay within the time budget for the provided prompt set.

## Challenges Faced
- Handling model output that is not always clean required a fallback path that still returns a valid object instead of crashing.
- Keeping the code simple while still validating nested function definitions with Pydantic took some care.

## Testing Strategy
Basic tests were carried out with the provided input files and a few malformed JSON cases. The validation helpers were checked against missing files, invalid function definitions, and empty prompts to make sure the program reports the problem cleanly instead of crashing.

## Example usage
```bash
make run
# or
uv run python -m src --input custom_prompts.json --output result.json
```
