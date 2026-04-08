import argparse
import re
import time
import json as json_module
import numpy as np
from typing import Any, cast
from llm_sdk import Small_LLM_Model
from .output import print_colored, p_col_arg, Terminal
from .json_formater import (
    read_json, write_json, extract_json_from_text, is_json_complete
)
from .inputs import (
    check_files_exist, check_inputs_validity
)

# because loading the model is reaaalllyyy long
before: float = time.time()
print_colored("Loading model, please wait...", "yellow")
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
        pre_prompt: str = "Functions:\n"
        for func in self.funcdef:
            func_name: str = str(func.get("name", "unknown_function"))
            parameters: Any = func.get("parameters", {})
            param_list: list[str] = []
            if isinstance(parameters, dict):
                for param_name in cast(dict[str, Any], parameters).keys():
                    param_list.append(str(param_name))
            params_str = ', '.join(param_list) if param_list else ''
            pre_prompt += f"{func_name}({params_str})\n"
        return pre_prompt

    def generate(self, prompt: str, max_tokens: int = 120) -> str:
        """Generate response token by token.

        Early stopping on complete JSON.
        """
        # Prime the model to output JSON immediately after the prompt.
        full_input: str = (
            self.pre_prompt
            + "\nReturn ONLY a valid JSON object using this exact format:\n"
            + '{"function_call":{"name":"<function_name>","arguments":{}}}\n'
            + "Do not output explanations or extra text.\n"
            + f"Prompt: {prompt}\n"
            + "JSON:\n"
            + '{"function_call":{"name":"'
        )
        # Encode prompt to token IDs
        tokenized_inputs: list[int] = self.model.encode(full_input)[0].tolist()
        self.last_rendered_lines = 0

        output_tokens: list[int] = []

        # generates tokens
        for _ in range(max_tokens):
            # Ask model for logits...
            logits: list[float] = self.model.get_logits_from_input_ids(
                tokenized_inputs
            )

            next_token_id: int = self.get_top_k_tokens(1, logits)[0]

            output_tokens.append(next_token_id)

            # Append to our running context
            tokenized_inputs.append(next_token_id)

            # Build full JSON with prefix for display and validation
            prefix = '{"function_call":{"name":"'
            generated_text: str = prefix + self.model.decode(output_tokens)

            # Render live output with prompt and token progression
            # (show complete JSON).
            self.format_gen(prompt=prompt, full_json=generated_text,
                            max_tokens=max_tokens,
                            token_count=len(output_tokens))

            # Check for complete JSON and stop early.
            if is_json_complete(generated_text):
                # Extract the clean JSON to skip any trailing garbage
                clean_json_obj: dict[str, Any] | None = (
                    extract_json_from_text(generated_text))
                if clean_json_obj:
                    print()
                    return json_module.dumps(clean_json_obj)

        prefix = '{"function_call":{"name":"'
        fallback_generated_text: str = (
            prefix + self.model.decode(output_tokens)
        )
        fallback_clean_json_obj: dict[str, Any] | None = (
            extract_json_from_text(fallback_generated_text))
        if fallback_clean_json_obj:
            print()
            return json_module.dumps(fallback_clean_json_obj)
        print()
        return fallback_generated_text

    def get_top_k_tokens(self, k: int, logits: list[float]) -> list[int]:
        """Get the top k most weighted tokens from the last generation step."""
        # Return token IDs sorted from highest to lowest logit.
        top_k_indices = np.argpartition(logits, -k)[-k:]
        sorted_indices = np.argsort(np.array(logits)[top_k_indices])[::-1]
        sorted_top_k_indices: Any = top_k_indices[sorted_indices]
        return cast(list[int], sorted_top_k_indices.tolist())

    def tttext(self, tokens: list[int]) -> str:
        """Translate a list of token IDs back to text."""
        return cast(str, self.model.decode(tokens))

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

    def format_gen(self, prompt: str, full_json: str, max_tokens: int,
                   token_count: int) -> None:
        """Render prompt, token progression, and complete JSON response."""
        """in-place."""
        if self.last_rendered_lines > 0:
            Terminal.up(self.last_rendered_lines)

        Terminal.clear_to_end()

        completion: float = (token_count / max_tokens) * 100 if (
            max_tokens > 0) else 0.0

        print_colored(f"Prompt: {prompt}", "magenta")
        pct: str = f"({token_count}/{max_tokens} - "
        pct += f"{completion:.1f}% completion)"
        print_colored(pct, "blue")
        print()
        print_colored(full_json, "green")

        pct_panel: str = f"({token_count}/{max_tokens} - "
        pct_panel += f"{completion:.1f}% completion)"
        panel_text: str = (
            f"Prompt: {prompt}\n"
            f"{pct_panel} completion\n\n"
            f"{full_json}"
        )
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
        func_defs: list[dict[str, Any]] = read_json(
            args.functions_definition)
        inputs: list[dict[str, Any]] = read_json(args.input)
        llm = LLM_Model(func_defs, inputs)

        results: list[dict[str, Any]] = []
        for entry in inputs:
            prompt: str = str(entry.get("prompt", ""))
            json_result: dict[str, Any] | None = None
            generated: str = ""

            # Retry with larger generation budget before giving up.
            for token_budget in (40, 80, 120):
                generated = llm.generate(prompt, max_tokens=token_budget)
                json_result = extract_json_from_text(generated)
                if json_result:
                    break
                else:
                    print_colored(
                        f"Generated output is not valid JSON. Retrying with "
                        f"larger token budget ({token_budget} tokens)...",
                        "yellow"
                    )

            if json_result:
                results.append({
                    "prompt": prompt,
                    "response": json_result
                })
            else:
                # Keep the prompt in output to avoid silently missing entries.
                results.append({
                    "prompt": prompt,
                    "response": {
                        "error": "Unable to extract valid function_call JSON",
                        "raw": generated
                    }
                })

            print("\n" + "-" * 50 + "\n")

        # Write all results to output file
        try:
            write_json(args.output, results)
            success_msg: str = f"Results written to {args.output}"
            print_colored(success_msg, "green")
        except Exception as e:
            error_msg: str = f"Error writing results to output file: {e}"
            print_colored(error_msg, "red")

    except KeyboardInterrupt:
        print_colored("\nGeneration interrupted by user.", "red")
    except Exception as e:
        print_colored(f"An error occurred: {e}", "red")
    print_colored("Exiting program. Please wait...", "yellow")


if __name__ == "__main__":
    main()
