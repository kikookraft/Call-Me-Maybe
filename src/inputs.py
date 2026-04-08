import sys
from typing import Any, cast
from .output import print_colored
from .json_formater import (
    read_json, validate_input_prompt, validate_function_definition
)


def execute_function(name: str, parameters: dict[str, Any]) -> Any:
    """Execute a custom function by name with provided parameters."""
    if name == "fn_add_numbers":
        return parameters.get("a", 0) + parameters.get("b", 0)
    elif name == "fn_subtract_numbers":
        return parameters.get("a", 0) - parameters.get("b", 0)
    elif name == "fn_multiply_numbers":
        return parameters.get("a", 0) * parameters.get("b", 0)
    elif name == "fn_divide_numbers":
        b: Any = parameters.get("b", 1)
        return parameters.get("a", 0) / b if b != 0 else float('inf')
    elif name == "fn_greet":
        return f"Hello, {parameters.get('name', 'User')}!"
    elif name == "fn_reverse_string":
        s: Any = parameters.get("s", "")
        return str(s)[::-1]
    else:
        raise ValueError(f"Unknown function: {name}")


def check_files_exist(*file_paths: str) -> bool:
    """Verify for each files passed as argument if they exist
    if not create the path and the file
    if this fails, open raise an error"""
    import os
    missing_files: list[str] = [
        file for file in file_paths if not os.path.isfile(file)]
    if missing_files:
        # create folder and empty files if they don't exist
        for file in missing_files:
            os.makedirs(os.path.dirname(file), exist_ok=True)
            with open(file, 'w', encoding='utf-8') as f:
                f.write("{}" if file.endswith('.json') else "")
        print_colored(
            f"Created missing files: {', '.join(missing_files)}",
            "yellow"
        )
    return True


def check_inputs_validity(func_def_path: str, input_path: str) -> bool:
    """Take the functions definition file and the input file path
    This function verify """
    try:
        func_defs: Any = read_json(func_def_path)
        inputs: Any = read_json(input_path)
    except Exception as e:
        print_colored(f"Error reading JSON files: {e}", "red")
        return False

    if not isinstance(func_defs, list):
        print_colored("Function definitions must be a list.", "red")
        return False

    if not isinstance(inputs, list):
        print_colored("Input prompts must be a list.", "red")
        return False

    for func in func_defs:
        if not isinstance(func, dict):
            print_colored(f"Invalid function definition: {func}", "red")
            sys.exit(1)
        if not validate_function_definition(cast(dict[str, Any], func)):
            print_colored(f"Invalid function definition: {func}", "red")
            sys.exit(1)

    for item in inputs:
        if not isinstance(item, dict):
            print_colored(f"Invalid input prompt: {item}", "red")
            sys.exit(1)
        if not validate_input_prompt(cast(dict[str, Any], item)):
            print_colored(f"Invalid input prompt: {item}", "red")
            sys.exit(1)

    msg = "All inputs and function definitions are valid."
    print_colored(msg, "green")
    return True
