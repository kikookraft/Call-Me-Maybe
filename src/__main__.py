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
from .inputs import (
    execute_function, check_files_exist, check_inputs_validity
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
        self.model: Any = Small_LLM_Model()
        self.funcdef: list[dict[str, Any]] = funcdef
        self.input_dict: list[dict[str, Any]] = input_dict
        self.last_rendered_lines: int = 0
        self.pre_prompt: str = self.build_pre_prompt()

    def build_pre_prompt(self) -> str:
        """Build the pre-prompt with the function definitions."""
        pre_prompt: str = "You are a helpful assistant that can call functions to answer questions.\n\n"
        pre_prompt += "Reply in short and concise answers, and if you need to call a function, use the following format:\n\n"
        pre_prompt += "```\n"
        pre_prompt += "{\n"
        pre_prompt += '  "function_call": {\n'
        pre_prompt += '    "name": "function_name",\n'
        pre_prompt += '    "parameters": {\n'
        pre_prompt += '      "param1": "value1",\n'
        pre_prompt += '      "param2": "value2"\n'
        pre_prompt += '    }\n'
        pre_prompt += '  }\n'
        pre_prompt += "}\n"
        pre_prompt += "```\n\n"
        pre_prompt += "Here are the available functions:\n"
        for func in self.funcdef:
            func_name: str = str(func.get("name", "unknown_function"))
            func_desc: str = str(func.get("description", "No description provided."))
            pre_prompt += f"- {func_name}: {func_desc}\n"

            parameters: Any = func.get("parameters", {})
            pre_prompt += "  Parameters:\n"
            if isinstance(parameters, dict) and parameters:
                for param_name, param_spec in cast(dict[str, Any], parameters).items():
                    if isinstance(param_spec, dict):
                        param_spec_dict: dict[str, Any] = cast(dict[str, Any], param_spec)
                        param_type: str = str(param_spec_dict.get("type", "any"))
                        param_desc: str = str(param_spec_dict.get("description", "No description provided."))
                    else:
                        param_type = "any"
                        param_desc = "No description provided."
                    pre_prompt += f"    - {param_name} ({param_type}): {param_desc}\n"
            else:
                pre_prompt += "    - none\n"

            returns: Any = func.get("returns", {})
            if isinstance(returns, dict):
                returns_dict: dict[str, Any] = cast(dict[str, Any], returns)
                return_type: str = str(returns_dict.get("type", "any"))
                return_desc: str = str(returns_dict.get("description", ""))
            else:
                return_type = "any"
                return_desc = ""
            pre_prompt += f"  Returns: {return_type} - {return_desc}\n\n"
        return pre_prompt

    def generate(self, prompt: str, max_tokens: int = 10) -> str:
        """Generate response token by token."""
        # Encode prompt to token IDs
        tokenized_inputs: list[int] = self.model.encode(self.pre_prompt + prompt)[0].tolist()
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
        sorted_top_k_indices: Any = top_k_indices[np.argsort(np.array(logits)[top_k_indices])[::-1]]
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

        ansi_re: re.Pattern[str] = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
        total: int = 0
        for raw_line in text.split("\n"):
            visible_len: int = len(ansi_re.sub("", raw_line))
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
        print_colored(second_text, "gray")
        print("\n3rd most weighted tokens:")
        print_colored(third_text, "gray")

        self.last_rendered_lines = self._count_rendered_lines(panel_text)


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

        for entry in inputs:
            prompt: str = str(entry.get("prompt", ""))
            # print_colored("\nEnter a prompt to generate from", "magenta")
            # prompt: str = input("> ")
            llm.generate(prompt, max_tokens=100)
            print("\n" + "-" * 50 + "\n")

    except KeyboardInterrupt:
        print_colored("\nGeneration interrupted by user.", "red")
    except Exception as e:
        print_colored(f"An error occurred: {e}", "red")
    print_colored("Exiting program. Please wait...", "yellow")


if __name__ == "__main__":
    main()
