import os
from typing import Any, cast

from .json_formater import (
    read_json,
    validate_function_definition,
    validate_input_prompt,
)
from .output import print_colored


def execute_function(name: str, parameters: dict[str, Any]) -> Any:
    """Execute a custom function by name with provided parameters."""
    if name == "fn_add_numbers":
        return parameters.get("a", 0) + parameters.get("b", 0)
    if name == "fn_subtract_numbers":
        return parameters.get("a", 0) - parameters.get("b", 0)
    if name == "fn_multiply_numbers":
        return parameters.get("a", 0) * parameters.get("b", 0)
    if name == "fn_divide_numbers":
        b: Any = parameters.get("b", 1)
        return parameters.get("a", 0) / b if b != 0 else float("inf")
    if name == "fn_greet":
        return f"Hello, {parameters.get('name', 'User')}!"
    if name == "fn_reverse_string":
        s: Any = parameters.get("s", "")
        return str(s)[::-1]
    raise ValueError(f"Unknown function: {name}")


def check_files_exist(*file_paths: str) -> bool:
    """Verify that all required files exist."""
    missing_files: list[str] = [
        file for file in file_paths if not os.path.isfile(file)
    ]
    if missing_files:
        print_colored(
            f"Missing required files: {', '.join(missing_files)}", "red"
        )
        return False
    return True


def check_inputs_validity(func_def_path: str, input_path: str) -> bool:
    """Validate the function definitions file and the input file."""
    try:
        func_defs_raw: Any = read_json(func_def_path)
        inputs_raw: Any = read_json(input_path)
    except Exception as e:
        print_colored(f"Error reading JSON files: {e}", "red")
        return False

    if not isinstance(func_defs_raw, list):
        print_colored("Function definitions must be a list.", "red")
        return False

    if not isinstance(inputs_raw, list):
        print_colored("Input prompts must be a list.", "red")
        return False

    func_defs: list[dict[str, Any]] = cast(list[dict[str, Any]], func_defs_raw)
    inputs: list[dict[str, Any]] = cast(list[dict[str, Any]], inputs_raw)

    for func in func_defs:
        if not validate_function_definition(func):
            print_colored(f"Invalid function definition: {func}", "red")
            return False

    for item in inputs:
        if not validate_input_prompt(item):
            print_colored(f"Invalid input prompt: {item}", "red")
            return False

    print_colored("All inputs and function definitions are valid.", "green")
    return True
