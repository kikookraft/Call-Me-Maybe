*This project has been created as part of the 42 curriculum by tobesson.*

# Call Me Maybe - Introduction to function calling in LLMs

## Description
This project implements a function calling tool that translates prompts into structured function calls using a small language model (`Qwen/Qwen3-0.6B`).  
It uses a hybrid pipeline: the model selects the function name with constrained decoding, then the program extracts the arguments deterministically, executes the function, and writes a strict JSON result.

The tool supports both batch mode and interactive mode, so you can process a JSON file of prompts or type prompts live at the terminal.

## Instructions
To install and run this project, make sure you have Python 3.10+ and [`uv`](https://github.com/astral-sh/uv) installed on your system. 

### Model Installation and Setup
This project uses the `Qwen/Qwen3-0.6B` model via the provided `llm_sdk`. To ensure the model is properly downloaded and configured:
1. Ensure `uv` is installed (`pip install uv` or via system package manager).
2. Install the project dependencies including the `llm_sdk` by running:
   ```bash
   make install
   ```
3. The SDK will automatically handle fetching the required model artifacts when you run the main script.

### Usage Modes

#### Batch Mode (Default)
Process a JSON file containing multiple prompts and write results to output file:
```bash
make run
# or
uv run python -m src --functions_definition data/input/functions_definition.json --input data/input/function_calling_tests.json --output data/output/function_calling_results.json
```

**Example input file** (`data/input/function_calling_tests.json`):
```json
[
  {"prompt": "What is the sum of 2 and 3?"},
  {"prompt": "What is 45 multiplied by 45?"},
  {"prompt": "Greet alice"}
]
```

**Example output file** (`data/output/function_calling_results.json`):
```json
[
  {
    "prompt": "What is the sum of 2 and 3?",
    "name": "fn_add_numbers",
    "parameters": {"a": 2.0, "b": 3.0}
  },
  {
    "prompt": "What is 45 multiplied by 45?",
    "name": "fn_multiply_numbers",
    "parameters": {"a": 45.0, "b": 45.0}
  },
  {
    "prompt": "Greet alice",
    "name": "fn_greet",
    "parameters": {"name": "alice"}
  }
]
```

#### Interactive Mode
Prompt the model in real-time with live output and save results to file:
```bash
make ask
# or
uv run python -m src --interactive
```

This mode allows you to enter prompts one at a time and immediately see the model's function selection and execution results:
```
Interactive mode enabled. Press Ctrl+C or Ctrl+D to exit.
Prompt> what is 45 * 45
Prompt: what is 45 * 45
fn_multiply_numbers
(3/32)
Output: 2025.0

Prompt> what is 100 / 5
Prompt: what is 100 / 5
fn_divide_numbers
(2/32)
Output: 20.0

Prompt> what is the capital of paris
Prompt: what is the capital of paris
fn_not_implemented
(1/32)
Output: <not implemented>

Prompt> Interactive mode closed (Ctrl+D).
Results written to data/output/function_calling_results.json
```

### Makefile Targets
- `make install`: Install project dependencies using `uv`.
- `make run`: Execute batch mode processing input file to output file.
- `make ask`: Run interactive mode with real-time prompts (press Ctrl+C or Ctrl+D to exit).
- `make debug`: Run batch mode in debug mode using Python's built-in debugger.
- `make clean`: Remove temporary files, caches, and build artifacts.
- `make lint`: Execute `flake8` and `mypy` with provided configuration to ensure code quality.
- `make lint-strict`: Execute `flake8` and `mypy` in strict mode.

## Resources
- [Qwen models on HuggingFace](https://huggingface.co/Qwen)
- Official Python [typing documentation](https://docs.python.org/3/library/typing.html)
- [Pydantic documentation](https://docs.pydantic.dev/latest/)
- [Constrained Decoding in LLMs](https://arxiv.org/abs/2307.09702)

**AI Usage:** AI was used in this project to assist in understanding constrained decoding, readme writing,
and generating regular expressions for docstrings validation.

## Algorithm Explanation
The solution first asks the model to choose a function name from the available definitions using token-level prefix constraints, so only names that match a valid function remain possible during decoding. Once the function is selected, the program generates short value candidates for each parameter, validates them against the expected type, and assembles the final result with Pydantic before writing it with `json.dumps`. This keeps the output schema strict while still letting the model decide the function call.


## Available Functions

The project includes the following functions that can be called:

| Function | Description | Parameters | Returns |
|----------|-------------|-----------|---------|
| `fn_add_numbers` | Add two numbers and return their sum | `a` (number), `b` (number) | number |
| `fn_multiply_numbers` | Multiply two numbers and return their product | `a` (number), `b` (number) | number |
| `fn_divide_numbers` | Divide first number by second and return result | `a` (number), `b` (number) | number |
| `fn_greet` | Generate a greeting message for a person | `name` (string) | string |
| `fn_reverse_string` | Reverse a string and return the result | `s` (string) | string |
| `fn_get_square_root` | Calculate the square root of a number | `a` (number) | number |
| `fn_substitute_string_with_regex` | Replace all occurrences matching a regex pattern | `source_string` (string), `regex` (string), `replacement` (string) | string |
| `fn_not_implemented` | Placeholder for unrecognized requests | (none) | string |

## How It Works

### Architecture Overview

The project follows a **hybrid processing pipeline** combining LLM-guided function selection with deterministic parameter extraction:

```
User Prompt (text)
  ↓
[1] Function Selection (Constrained Decoding)
  • LLM generates function name with prefix constraints
  • Only valid function names remain possible tokens
  • Uses TokenChoiceDecoder with token filtering
  ↓
[2] Parameter Extraction (Deterministic)
  • extraction per parameter type
  • Numbers: Extract all floats in order of appearance
  • Strings: Extract quoted strings with escape handling
  • Regex patterns: Map natural language ("replace all numbers") to regex
  ↓
[3] Validation (Pydantic)
  • Validate extracted parameters against function schema
  • Type checking and fallback to defaults if needed
  ↓
[4] Execution & Output
  • Execute function with validated parameters
  • Display result in terminal (interactive mode)
  • Save to JSON (both batch and interactive modes)
```

### Function Selection Process

**Stage 1: Constrained Decoding**

The model receives a prompt listing available functions:
```
Available functions:
- fn_add_numbers: Add two numbers (addition operator +). Use for sum, total, plus operations.
- fn_multiply_numbers: Multiply two numbers (multiplication operator *). Use for product, times, multiply operations.
- fn_divide_numbers: Divide first number by second (division operator /). Use for division, quotient, divided by operations.
...

Prompt: what is 45 multiplied by 45
Function name:
```

The `TokenChoiceDecoder` constrains token generation so only prefixes that lead to valid function names are allowed. This ensures the output is always one of the available functions.

### Parameter Extraction Process

**Stage 2: Smart Parameter Extraction**

Once the function is selected, parameters are extracted using context-aware rules:

**For numeric parameters:**
```python
# Input: "what is 45 multiplied by 45"
# Extracts: [45.0, 45.0]
numbers = re.findall(r"-?\d+(?:\.\d+)?", prompt)
```

**For string parameters:**
```python
# Input: "greet alice"
# Pattern: Extract word after "greet" keyword
name = extract_word_after_pattern("greet", text)  # → "alice"
```

**For regex patterns:**
```python
# Input: "replace all numbers in '123abc456' with 'X'"
# Maps: "numbers" → r"\d+"
# Maps: "vowels" → r"[AEIOUaeiou]"
# Maps: "spaces" → r"\s+"
```

### Validation & Fallback

**Stage 3: Pydantic Validation**

All extracted parameters are validated against the function schema. If validation fails:
- Try to apply sensible defaults (0 for numbers, empty string for text)
- Fall back to `fn_not_implemented` if function is unrecognized

### Output Pipeline

**Stage 4: Execution & Rendering**

- **Interactive mode**: Execute function and display result in green in terminal, collect for JSON output
- **Batch mode**: Execute and display each result, append to results list
- **Both modes**: Write final JSON to output file with schema: `{prompt, name, parameters}`

## Algorithm Explanation

The project uses a hybrid pipeline. First, the model selects a function name from the available definitions using token-level prefix constraints, so only valid function names can be produced. Then the program extracts the parameter values deterministically from the prompt, validates them against the expected schema, and writes the result as JSON. This keeps the output strict while making the argument extraction much more reliable for a small model.

## Design Decisions
- **Hybrid approach**: the LLM picks the function name, while deterministic code extracts the arguments.
- **Context-aware extraction**: parameter parsing uses function-specific heuristics such as numeric scanning, keyword matching, and regex mapping.
- **Fallback strategy**: when no suitable function is found, the project falls back to `fn_not_implemented` instead of crashing.
- **Shared execution path**: interactive mode and batch mode reuse the same validation and execution logic, so both modes behave consistently.

## Performance Analysis
Constraining the function-name search space keeps decoding focused on the valid function set and removes malformed output at the schema level. Deterministic parameter extraction also avoids the common failure modes of small models on arithmetic and text-heavy prompts. The implementation remains lightweight enough to stay within the time budget for the provided prompt set.

## Challenges Faced
- Handling model output that is not always clean required a fallback path that still returns a valid object instead of crashing.
- Keeping the code simple while still validating nested function definitions with Pydantic took some care.
- Supporting both batch and interactive execution without duplicating the pipeline required a shared result-building flow.

## Testing Strategy
Basic tests were carried out with:
- Batch mode with various prompts for addition, multiplication, division, and text operations
- Interactive mode with live prompts and Ctrl+C/Ctrl+D exit handling
- Malformed JSON cases and edge cases such as empty definitions and invalid types
- Fallback behavior when functions do not match, such as `"capital of paris"` mapping to `fn_not_implemented`
- Validation helpers against missing files, invalid function definitions, and empty prompts

## Example usage

**Batch mode:**
```bash
make run
# or
uv run python -m src --input custom_prompts.json --output result.json
```

**Interactive mode:**
```bash
make ask
# or
uv run python -m src --interactive
```

In interactive mode, you can enter prompts one by one, see the chosen function and computed output immediately, and the session results are written to `data/output/function_calling_results.json` when you exit with Ctrl+C or Ctrl+D.
