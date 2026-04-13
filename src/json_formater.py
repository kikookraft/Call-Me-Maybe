import json
from typing import Any, cast

from pydantic import ValidationError

from .schemas import FunctionDefinition, InputPrompt


def read_file(file_path: str) -> str:
    """Read from file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def write_file(file_path: str, content: str) -> None:
    """Write to file."""
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)


def read_json(file_path: str) -> Any:
    """Read file as JSON."""
    return json.loads(read_file(file_path))


def write_json(file_path: str, data: list[dict[str, Any]]) -> None:
    """Write JSON to file."""
    write_file(file_path, json.dumps(data, indent=4))


def append_json(file_path: str, data: dict[str, Any]) -> None:
    """Add a dict to a JSON file."""
    existing_data: list[dict[str, Any]] = cast(list[dict[str, Any]], read_json(file_path))
    existing_data.append(data)
    write_json(file_path, existing_data)


def validate_input_prompt(prompt: dict[str, Any]) -> bool:
    """Validate that the input prompt has the correct structure."""
    try:
        InputPrompt.model_validate(prompt)
    except ValidationError:
        return False
    return True


def validate_function_definition(funcdef: dict[str, Any]) -> bool:
    """Validate that the function definition has the correct structure."""
    try:
        FunctionDefinition.model_validate(funcdef)
    except ValidationError:
        return False
    return True


def func_result_check(funcdef: list[dict[str, Any]], prompt: dict[str, Any]) -> None:
    """Search in the function definition and verify the parameters types."""
    if "name" not in prompt or "parameters" not in prompt:
        raise ValueError("Prompt is missing 'name' or 'parameters'.")

    func_name: str = str(prompt["name"])
    raw_parameters: Any = prompt["parameters"]

    if not isinstance(raw_parameters, dict):
        raise ValueError("Parameters must be a dictionary.")

    parameters: dict[str, Any] = cast(dict[str, Any], raw_parameters)

    func_schema: dict[str, Any] | None = None
    for func in funcdef:
        if func.get("name") == func_name:
            func_schema = func
            break

    if func_schema is None:
        raise ValueError(f"Function '{func_name}' not found.")

    schema_params: dict[str, Any] = func_schema.get("parameters", {})

    for key, spec in schema_params.items():
        if key not in parameters:
            raise ValueError(f"Missing parameter '{key}' for '{func_name}'.")

        expected_type: str = str(spec.get("type"))
        val: Any = parameters[key]

        if expected_type == "number" and not isinstance(val, (int, float)):
            raise ValueError(f"Parameter '{key}' must be a number.")
        if expected_type == "string" and not isinstance(val, str):
            raise ValueError(f"Parameter '{key}' must be a string.")
        if expected_type == "boolean" and not isinstance(val, bool):
            raise ValueError(f"Parameter '{key}' must be a boolean.")

    for key in parameters:
        if key not in schema_params:
            raise ValueError(f"Extra param '{key}' not in '{func_name}'.")


def extract_json_from_text(text: str) -> dict[str, Any] | None:
    """Extract minimal complete JSON from text using brace balancing."""
    stripped: str = text.lstrip()
    if not stripped.startswith("{"):
        return None

    brace_count: int = 0
    json_end: int = 0
    in_string: bool = False
    escape: bool = False

    for i, char in enumerate(stripped):
        if escape:
            escape = False
            continue
        if char == "\\":
            escape = True
            continue
        if char == '"' and not escape:
            in_string = not in_string
            continue
        if not in_string:
            if char == "{":
                brace_count += 1
            elif char == "}":
                brace_count -= 1
                if brace_count == 0:
                    json_end = i + 1
                    break

    if json_end == 0:
        return None

    json_str: str = stripped[:json_end]
    try:
        parsed: dict[str, Any] = cast(dict[str, Any], json.loads(json_str))
        return parsed
    except json.JSONDecodeError:
        return None


def is_json_complete(text: str) -> bool:
    """Check if the text contains complete, valid JSON."""
    return extract_json_from_text(text) is not None


if __name__ == "__main__":
    file: str = "data/input/functions_definition.json"
    data: Any = read_json(file)
    print(data)
