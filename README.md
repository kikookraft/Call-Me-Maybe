*This project has been created as part of the 42 curriculum by tobesson.*

# Call Me Maybe - Introduction to function calling in LLMs

## Description
This project aims to implement a function calling tool that translates natural language prompts into structured function calls using constrained decoding with a small language model (`Qwen/Qwen3-0.6B`). It bridges the gap between natural language requests and precise function execution by guiding the model token-by-token to output 100% valid JSON matching a specific schema.

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
python3 -m uv run python -m src --functions_definition data/input/functions_definition.json --input data/input/function_calling_tests.json --output data/output/function_calls.json
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

**AI Usage:** AI was used in this project to assist in understanding constrained decoding theory, drafting boilerplate token masking structures, and generating regular expressions for docstrings validation.

## Algorithm Explanation
The solution decodes the LLM's raw logits output step-by-step. At each token generation step, it parses the previously generated text and restricts ("constrains") the vocabulary list by evaluating valid tokens based on JSON grammar rules and the target Pydantic schema structure. Invalid tokens are flagged by setting their logits to negative infinity. 

## Design Decisions
- `pydantic` dynamically models the function definitions to easily validate allowed fields and types.
- The stateful JSON parser evaluates the expected next character to compute the allowed tokens efficiently without re-evaluating the whole string from the beginning at each token step.

## Performance Analysis
The constrained generation limits the vocabulary search space, vastly improving the reliability of the small `Qwen3-0.6B` model to produce syntactically valid JSON. Accuracy in selecting the correct function and parameters is robust compared to standard unconstrained beam search, while keeping evaluation speed well within the 5-minute requirement.

## Challenges Faced
- Managing tokenization idiosyncrasies (e.g., spaces inside tokens, split variables) while matching against continuous JSON strings required a custom token-prefix matching index structure.

## Testing Strategy
Basic tests were carried out on common prompts with the provided input files. Additional stress tests covered invalid inputs and diverse argument inputs using `pytest` to verify the JSON structure remained 100% compliant under unconstrained logic faults.

## Example usage
```bash
make run
# or
python3 -m uv run python -m src --input custom_prompts.json --output result.json
```
