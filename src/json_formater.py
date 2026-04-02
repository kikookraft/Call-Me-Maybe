import json
from typing import Any, cast


def read_file(file_path: str) -> str:
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()


def write_file(file_path: str, content: str) -> None:
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)


def read_json(file_path: str) -> list[dict[str, Any]]:
    return json.loads(read_file(file_path))


def write_json(file_path: str, data: list[dict[str, Any]]) -> None:
    write_file(file_path, json.dumps(data))


def append_json(file_path: str, data: dict[str, Any]) -> None:
    existing_data: list[dict[str, Any]] = read_json(file_path)
    existing_data.append(data)
    write_json(file_path, existing_data)


def validate_prompt(prompt: dict[str, Any]) -> bool:
    required_keys: set[str] = {"prompt", "description", "parameters"}
    if not all(key in prompt for key in required_keys):
        return False
    if not isinstance(prompt["parameters"], dict):
        return False
    return True


def func_result_check(funcdef: list[dict[str, Any]],
                      prompt: dict[str, Any]) -> None:
    """Search in the function definition and verify the parameters types.

    Args:
        funcdef: The list of function definitions.
        prompt: Expected to contain at least 'name' and 'parameters'.

    Raises:
        ValueError: If the input is invalid or mismatch.
    """
    if "name" not in prompt or "parameters" not in prompt:
        raise ValueError("Prompt is missing 'name' or 'parameters'.")

    func_name: str = str(prompt["name"])
    raw_parameters: Any = prompt["parameters"]

    if not isinstance(raw_parameters, dict):
        raise ValueError("Parameters must be a dictionary.")

    parameters: dict[str, Any] = cast(dict[str, Any], raw_parameters)

    func_schema: dict[str, Any] | None = None
    for func in funcdef:  # search for the base function
        if func.get("name") == func_name:
            func_schema = func
            break

    if func_schema is None:
        raise ValueError(f"Function '{func_name}' not found.")

    schema_params: dict[str, Any] = func_schema.get("parameters", {})

    # Check missing parameters and their types
    for key, spec in schema_params.items():
        if key not in parameters:
            raise ValueError(f"Missing parameter '{key}' for '{func_name}'.")

        expected_type: str = str(spec.get("type"))
        val: Any = parameters[key]

        if expected_type == "number" and not isinstance(val, (int, float)):
            raise ValueError(f"Parameter '{key}' must be a number.")
        elif expected_type == "string" and not isinstance(val, str):
            raise ValueError(f"Parameter '{key}' must be a string.")
        elif expected_type == "boolean" and not isinstance(val, bool):
            raise ValueError(f"Parameter '{key}' must be a boolean.")

    # Check for extra parameters
    for key in parameters:
        if key not in schema_params:
            raise ValueError(f"Extra param '{key}' not in '{func_name}'.")


if __name__ == "__main__":
    file: str = "data/input/functions_definition.json"
    data: list[dict[str, Any]] = read_json(file)
    print(data)
