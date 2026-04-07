import argparse
import re
import sys
import time
import numpy as np
from typing import Any, cast
from .output import print_colored, p_col_arg, Terminal
from .json_formater import (
    read_json, validate_input_prompt, validate_function_definition
)

# because loading the model is reaaalllyyy long
before: float = time.time()
print_colored("Loading model, please wait...", "yellow")
from llm_sdk import Small_LLM_Model
print_colored(
    f"Model loaded in {(time.time() - before):.2f}",
    "green")


class LLM_Model:
    """Contain context, llm and other things"""

    def __init__(
            self,
            funcdef: list[dict[str, Any]],
            input_dict: list[dict[str, Any]]) -> None:
        self.model = Small_LLM_Model()
        self.funcdef: list[dict[str, Any]] = funcdef
        self.input_dict: list[dict[str, Any]] = input_dict
        self.last_rendered_lines: int = 0

    def generate(self, prompt: str, max_tokens: int = 75) -> str:
        """Generate response token by token."""
        # Encode prompt to token IDs
        tokenized_inputs: list[int] = self.model.encode(prompt)[0].tolist()
        print_colored("Generating response token-by-token...", "yellow")
        self.last_rendered_lines = 0

        output_tokens: list[int] = []
        second_best_tokens: list[int] = []
        third_best_tokens: list[int] = []

        # generates tokens
        for _ in range(max_tokens):
            # Ask model for logits...
            logits: list[float] = self.model.get_logits_from_input_ids(
                tokenized_inputs
            )

            # Keep 2nd and 3rd best token streams across generation steps.
            top_three: list[int] = self.get_top_k_tokens(3, logits)
            next_token_id: int = top_three[0]
            second_best_tokens.append(top_three[1])
            third_best_tokens.append(top_three[2])

            output_tokens.append(next_token_id)

            # Append to our running context
            tokenized_inputs.append(next_token_id)

            # print the generated response so far + the rest
            self.format_gen(
                generated=output_tokens,
                second_best_tokens=second_best_tokens,
                third_best_tokens=third_best_tokens,
                prompt=prompt
            )

        print()
        return self.model.decode(output_tokens)
    
    def get_top_k_tokens(self, k: int, logits: list[float]) -> list[int]:
        """Get the top k most weighted tokens from the last generation step."""
        # Return token IDs sorted from highest to lowest logit.
        top_k_indices = np.argpartition(logits, -k)[-k:]
        sorted_top_k_indices = top_k_indices[np.argsort(np.array(logits)[top_k_indices])[::-1]]
        return sorted_top_k_indices.tolist()
    
    def tttext(self, tokens: list[int]) -> str:
        """Translate a list of token IDs back to text."""
        return self.model.decode(tokens)

    def _count_rendered_lines(self, text: str) -> int:
        """Count terminal lines including automatic wrapping."""
        try:
            cols: int = max(1, Terminal.get_size()[0])
        except OSError:
            cols = 80

        ansi_re = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
        total: int = 0
        for raw_line in text.split("\n"):
            visible_len = len(ansi_re.sub("", raw_line))
            total += max(1, (visible_len + cols - 1) // cols)
        return total

    def format_gen(
            self,
            generated: list[int],
            second_best_tokens: list[int],
            third_best_tokens: list[int],
            prompt: str) -> None:
        """Format the generated response for better readability.
        clear terminal each time, this function is called each time
        a new token is generated, and reprint everything:
        - print the prompt in blue 
        - the generated response in green
        - the setence with all 2nd most weighted tokens in yellow
        - the setence with all 3rd most weighted tokens in yellow
         """
        if self.last_rendered_lines > 0:
            Terminal.up(self.last_rendered_lines)

        Terminal.clear_to_end()

        generated_text: str = self.tttext(generated)
        second_text: str = self.tttext(second_best_tokens)
        third_text: str = self.tttext(third_best_tokens)

        panel_text: str = (
            "Prompt:\n"
            f"{prompt}\n\n"
            "Generated Response:\n"
            f"{generated_text}\n\n"
            "2nd most weighted tokens:\n"
            f"{second_text}\n\n"
            "3rd most weighted tokens:\n"
            f"{third_text}"
        )

        print("Prompt:")
        print_colored(prompt, "blue")
        print("\nGenerated Response:")
        print_colored(generated_text, "green")
        print("\n2nd most weighted tokens:")
        print_colored(second_text, "second_best")
        print("\n3rd most weighted tokens:")
        print_colored(third_text, "third_best")

        self.last_rendered_lines = self._count_rendered_lines(panel_text)



def execute_function(name: str, parameters: dict[str, Any]) -> Any:
    """Execute a custom function by name with provided parameters."""
    if name == "fn_add_numbers":
        return parameters.get("a", 0) + parameters.get("b", 0)
    elif name == "fn_subtract_numbers":
        return parameters.get("a", 0) - parameters.get("b", 0)
    elif name == "fn_multiply_numbers":
        return parameters.get("a", 0) * parameters.get("b", 0)
    elif name == "fn_divide_numbers":
        b = parameters.get("b", 1)
        return parameters.get("a", 0) / b if b != 0 else float('inf')
    elif name == "fn_greet":
        return f"Hello, {parameters.get('name', 'User')}!"
    elif name == "fn_reverse_string":
        s = parameters.get("s", "")
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

    for func in cast(list[Any], func_defs):
        if not isinstance(func, dict):
            print_colored(f"Invalid function definition: {func}", "red")
            sys.exit(1)
        if not validate_function_definition(cast(dict[str, Any], func)):
            print_colored(f"Invalid function definition: {func}", "red")
            sys.exit(1)

    for item in cast(list[Any], inputs):
        if not isinstance(item, dict):
            print_colored(f"Invalid input prompt: {item}", "red")
            sys.exit(1)
        if not validate_input_prompt(cast(dict[str, Any], item)):
            print_colored(f"Invalid input prompt: {item}", "red")
            sys.exit(1)

    print_colored("All inputs and function definitions are valid.", "green")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Call Me Maybe - Function Calling with LLMs"
    )
    parser.add_argument(
        "--functions_definition",
        type=str,
        default="data/input/functions_definition.json",
        help="Path to function definitions"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/input/function_calling_tests.json",
        help="Path to input prompts"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/output/function_calls.json",
        help="Path to output file"
    )
    args: argparse.Namespace = parser.parse_args()

    try:
        check_files_exist(args.functions_definition, args.input, args.output)
        if not check_inputs_validity(args.functions_definition, args.input):
            return
        p_col_arg("Input file: ", args.input, "blue")
        p_col_arg("Function definitions: ", args.functions_definition, "blue")
        p_col_arg("Output file: ", args.output, "blue")
    except Exception as e:
        print_colored(f"Error checking/creating files: {e}", "red")
        return

    try:
        func_defs: list[dict[str, Any]] = read_json(args.functions_definition)
        inputs: list[dict[str, Any]] = read_json(args.input)
        llm = LLM_Model(func_defs, inputs)

        while True:
            print_colored("\nEnter a prompt to generate from", "magenta")
            prompt: str = input("> ")
            llm.generate(prompt)

    except KeyboardInterrupt:
        print_colored("\nGeneration interrupted by user.", "red")
    except Exception as e:
        print_colored(f"An error occurred: {e}", "red")
    print_colored("Exiting program. Please wait...", "yellow")


if __name__ == "__main__":
    main()
