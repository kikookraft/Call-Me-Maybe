import argparse
import os
import re
import time
from typing import Any, Callable, cast

from llm_sdk import Small_LLM_Model

from .inputs import check_files_exist, check_inputs_validity
from .json_formater import read_json, write_json
from .output import Terminal, p_col_arg, print_colored
from .schemas import FunctionCallResult, FunctionDefinition, InputPrompt


class TokenChoiceDecoder:
    """Select one allowed value with token-level prefix constraints."""

    def __init__(self, model: Small_LLM_Model) -> None:
        self.model: Small_LLM_Model = model
        self._token_text_cache: dict[int, str] = {}
        sample_logits: list[float] = self.model.get_logits_from_input_ids(
            self.model.encode(" ")[0].tolist()
        )
        self._vocab_size: int = len(sample_logits)

    def _token_text(self, token_id: int) -> str:
        token_text: str | None = self._token_text_cache.get(token_id)
        if token_text is None:
            token_text = self.model.decode([token_id])
            self._token_text_cache[token_id] = token_text
        return token_text

    def choose(
        self,
        prompt: str,
        options: list[str],
        max_tokens: int = 32,
        render_step: Callable[[str, int, int], None] | None = None,
    ) -> str:
        """Generate one option by keeping only tokens compatible
        with the prefix."""
        if not options:
            raise ValueError("At least one option is required.")

        prompt_ids: list[int] = self.model.encode(prompt)[0].tolist()
        generated_ids: list[int] = []
        generated_text: str = ""

        for _ in range(max_tokens):
            logits: list[float] = self.model.get_logits_from_input_ids(
                prompt_ids + generated_ids
            )

            valid_token_ids: list[int] = []
            for token_id in range(self._vocab_size):
                token_text: str = self._token_text(token_id)
                if not token_text:
                    continue
                candidate_text: str = generated_text + token_text
                if any(option.startswith(candidate_text)
                       for option in options):
                    valid_token_ids.append(token_id)

            if not valid_token_ids:
                break

            next_token_id: int = max(
                valid_token_ids, key=lambda token_id: logits[token_id])
            generated_ids.append(next_token_id)
            generated_text += self._token_text(next_token_id)

            if render_step is not None:
                render_step(generated_text, len(generated_ids), max_tokens)

            if generated_text in options:
                return generated_text

        matching_options: list[str] = [
            option for option in options if option.startswith(generated_text)
        ]
        if matching_options:
            return max(matching_options, key=len)
        return options[0]


class LLM_Model:
    """Contain context, llm and other things."""

    def __init__(
        self,
        funcdef: list[dict[str, Any]],
        input_dict: list[dict[str, Any]],
    ) -> None:
        before: float = time.time()
        print_colored("Loading model, please wait...", "yellow")
        self.model: Small_LLM_Model = Small_LLM_Model()
        print_colored(
            f"Model loaded in {(time.time() - before):.2f}", "green")
        self.funcdef: list[FunctionDefinition] = [
            FunctionDefinition.model_validate(item) for item in funcdef
        ]
        self.input_dict: list[InputPrompt] = [
            InputPrompt.model_validate(item) for item in input_dict
        ]
        self.decoder: TokenChoiceDecoder = TokenChoiceDecoder(self.model)
        self.panel_header_printed: bool = False
        self.last_dynamic_lines: int = 0

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

    def _render_live_panel(
        self,
        prompt: str,
        stage: str,
        current_text: str,
        token_count: int,
        max_tokens: int,
    ) -> None:
        """Render generation progress in place."""
        if not self.panel_header_printed:
            print_colored(f"Prompt: {prompt}", "magenta")
            self.panel_header_printed = True

        if self.last_dynamic_lines > 0:
            Terminal.up(self.last_dynamic_lines)
            Terminal.clear_to_end()

        completion: float = (
            token_count / max_tokens) * 100 if max_tokens > 0 else 0.0
        print_colored(f"Stage: {stage}", "cyan")
        print_colored(
            f"({token_count}/{max_tokens} - {completion:.1f}% completion)",
            "blue",
        )
        print()
        print_colored(current_text if current_text else "<empty>", "green")

        dynamic_text: str = (
            f"Stage: {stage}\n"
            f"({token_count}/{max_tokens} - {completion:.1f}% completion)\n\n"
            f"{current_text if current_text else '<empty>'}"
        )
        self.last_dynamic_lines = self._count_rendered_lines(dynamic_text)

    def _finish_live_panel(self) -> None:
        """Finish one live rendering block."""
        if self.panel_header_printed:
            print()
        self.panel_header_printed = False
        self.last_dynamic_lines = 0

    def _function_prompt(self, prompt: str) -> str:
        """Build the instruction used to select the function name."""
        lines: list[str] = ["Available functions:"]
        for func in self.funcdef:
            lines.append(f"- {func.name}: {func.description}")
        lines.append(f"Prompt: {prompt}")
        lines.append("Function name:")
        return "\n".join(lines)

    def _generate_text(
        self,
        prompt: str,
        max_tokens: int = 32,
        display_prompt: str = "",
        stage: str = "generation",
    ) -> str:
        """Generate a short text completion with greedy decoding."""
        prompt_ids: list[int] = self.model.encode(prompt)[0].tolist()
        generated_ids: list[int] = []
        shown_prompt: str = display_prompt if display_prompt else prompt

        for _ in range(max_tokens):
            logits: list[float] = self.model.get_logits_from_input_ids(
                prompt_ids + generated_ids
            )
            next_token_id: int = max(
                range(len(logits)), key=logits.__getitem__)
            generated_ids.append(next_token_id)
            token_text: str = self.model.decode([next_token_id])
            current_text: str = self.model.decode(generated_ids).strip()
            self._render_live_panel(
                prompt=shown_prompt,
                stage=stage,
                current_text=current_text,
                token_count=len(generated_ids),
                max_tokens=max_tokens,
            )
            if "\n" in token_text:
                break

        decoded_text: str = cast(str, self.model.decode(generated_ids))
        return decoded_text.strip()

    def _choose_function(self, prompt: str) -> FunctionDefinition:
        """Pick the function name with constrained decoding."""
        function_names: list[str] = [func.name for func in self.funcdef]

        def render_step(
            current: str, token_count: int, max_tokens: int
        ) -> None:
            self._render_live_panel(
                prompt=prompt,
                stage="function selection",
                current_text=current,
                token_count=token_count,
                max_tokens=max_tokens,
            )

        chosen_name: str = self.decoder.choose(
            self._function_prompt(prompt),
            function_names,
            render_step=render_step,
        )
        for func in self.funcdef:
            if func.name == chosen_name:
                return func
        return self.funcdef[0]

    def _extract_number(self, prompt: str, raw_text: str) -> float:
        """Extract a numeric value from model output or from the prompt."""
        match = re.search(r"-?\d+(?:\.\d+)?", raw_text)
        if match is None:
            match = re.search(r"-?\d+(?:\.\d+)?", prompt)
        if match is None:
            return 0.0
        return float(match.group(0))

    def _extract_boolean(self, raw_text: str) -> bool:
        """Extract a boolean value from the model output."""
        lowered: str = raw_text.lower()
        if "true" in lowered:
            return True
        if "false" in lowered:
            return False
        return False

    def _extract_string(self, prompt: str, raw_text: str) -> str:
        """Extract a string value from model output or the original prompt."""
        quoted = re.search(r'"([^"]+)"', raw_text)
        if quoted is None:
            quoted = re.search(r"'([^']+)'", raw_text)
        if quoted is not None:
            return quoted.group(1).strip()

        prompt_quoted = re.search(r'"([^"]+)"', prompt)
        if prompt_quoted is None:
            prompt_quoted = re.search(r"'([^']+)'", prompt)
        if prompt_quoted is not None:
            return prompt_quoted.group(1).strip()

        return raw_text.strip().strip('"').strip("'")

    def _extract_value(
        self, prompt: str, raw_text: str, type_name: str
    ) -> Any:
        """Convert model output to the expected JSON type."""
        if type_name == "number":
            return self._extract_number(prompt, raw_text)
        if type_name == "boolean":
            return self._extract_boolean(raw_text)
        return self._extract_string(prompt, raw_text)

    def build_result(self, prompt: str) -> dict[str, Any]:
        """Create one function call result for a single prompt."""
        self.panel_header_printed = False
        self.last_dynamic_lines = 0
        try:
            function_def: FunctionDefinition = self._choose_function(prompt)
            parameters: dict[str, Any] = {}

            for param_name, param_spec in function_def.parameters.items():
                value_prompt: str = (
                    f"Prompt: {prompt}\n"
                    f"Function: {function_def.name}\n"
                    f"Parameter: {param_name}\n"
                    f"Type: {param_spec.type}\n"
                    "Return only the value."
                )
                raw_text: str = self._generate_text(
                    prompt=value_prompt,
                    display_prompt=prompt,
                    stage=f"parameter {param_name}",
                )
                parameters[param_name] = self._extract_value(
                    prompt, raw_text, param_spec.type
                )

            result = FunctionCallResult(
                prompt=prompt,
                name=function_def.name,
                parameters=parameters,
            )
            return result.model_dump(mode="json")
        finally:
            self._finish_live_panel()


def _default_parameters(function_def: FunctionDefinition) -> dict[str, Any]:
    """Build a safe fallback parameter set."""
    defaults: dict[str, Any] = {}
    for param_name, param_spec in function_def.parameters.items():
        if param_spec.type == "number":
            defaults[param_name] = 0.0
        elif param_spec.type == "boolean":
            defaults[param_name] = False
        else:
            defaults[param_name] = ""
    return defaults


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Call Me Maybe - Function Calling with LLMs"
    )
    parser.add_argument(
        "--functions_definition",
        type=str,
        default="data/input/functions_definition.json",
        help="Path to function definitions",
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/input/function_calling_tests.json",
        help="Path to input prompts",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/output/function_calling_results.json",
        help="Path to output file",
    )
    args: argparse.Namespace = parser.parse_args()

    try:
        if not check_files_exist(args.functions_definition, args.input):
            return
        if not check_inputs_validity(args.functions_definition, args.input):
            return
        p_col_arg("Input file", args.input, "blue")
        p_col_arg("Function definitions", args.functions_definition, "blue")
        p_col_arg("Output file", args.output, "blue")
    except Exception as e:
        print_colored(f"Error checking files: {e}", "red")
        return

    try:
        func_defs: list[dict[str, Any]] = read_json(args.functions_definition)
        inputs: list[dict[str, Any]] = read_json(args.input)
        llm = LLM_Model(func_defs, inputs)

        output_dir: str = os.path.dirname(args.output)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        results: list[dict[str, Any]] = []
        for entry in inputs:
            prompt: str = str(entry.get("prompt", ""))
            try:
                result = llm.build_result(prompt)
            except Exception as e:
                print_colored(
                    f"Failed to build a result for prompt '{prompt}': {e}",
                    "red",
                )
                fallback_function: FunctionDefinition = llm.funcdef[0]
                result = FunctionCallResult(
                    prompt=prompt,
                    name=fallback_function.name,
                    parameters=_default_parameters(fallback_function),
                ).model_dump(mode="json")
            results.append(result)

        try:
            write_json(args.output, results)
            print_colored(f"Results written to {args.output}", "green")
        except Exception as e:
            print_colored(f"Error writing results to output file: {e}", "red")

    except KeyboardInterrupt:
        print_colored("Generation interrupted by user.", "red")
    except Exception as e:
        print_colored(f"An error occurred: {e}", "red")

    print_colored("Exiting program. Please wait...", "yellow")


if __name__ == "__main__":
    main()
