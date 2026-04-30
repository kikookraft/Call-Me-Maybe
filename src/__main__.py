import argparse
import os
import re
import time
from typing import Any, Callable

from llm_sdk import Small_LLM_Model

from .inputs import check_files_exist, check_inputs_validity, execute_function
from .json_formater import (
    func_result_check,
    read_json,
    validate_function_definition,
    write_json,
)
from .output import Terminal, p_col_arg, print_colored
from .schemas import (
    FunctionCallResult,
    FunctionDefinition,
    InputPrompt,
    ParameterSpec,
)


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

            import heapq

            valid_token_ids: list[int] = []
            # sort logits output and pick the first valid one to optimize
            # Extract top 1024 tokens to avoid sorting/evaluating 150k elements
            top_k_indices = heapq.nlargest(
                1024, range(self._vocab_size), key=logits.__getitem__
            )

            for token_id in top_k_indices:
                token_text: str = self._token_text(token_id)
                if not token_text:
                    continue
                candidate_text: str = generated_text + token_text
                if any(
                    option.startswith(candidate_text) for option in options
                ):
                    valid_token_ids.append(token_id)
                    break

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
        raise ValueError("Unable to decode a valid option.")

    def choose_with_constraints(
        self,
        prompt: str,
        is_valid_prefix: Callable[[str], bool],
        is_complete: Callable[[str], bool],
        max_tokens: int = 32,
        render_step: Callable[[str, int, int], None] | None = None,
    ) -> str:
        """Generate text while enforcing prefix/complete constraints."""
        prompt_ids: list[int] = self.model.encode(prompt)[0].tolist()
        generated_ids: list[int] = []
        generated_text: str = ""

        for _ in range(max_tokens):
            logits: list[float] = self.model.get_logits_from_input_ids(
                prompt_ids + generated_ids
            )

            import heapq

            valid_token_ids: list[int] = []

            # Extract top 1024 tokens to avoid sorting/evaluating 150k elements
            top_k_indices = heapq.nlargest(
                1024, range(self._vocab_size), key=logits.__getitem__
            )

            for token_id in top_k_indices:
                token_text: str = self._token_text(token_id)
                if not token_text:
                    continue
                candidate_text: str = generated_text + token_text
                if is_valid_prefix(candidate_text):
                    valid_token_ids.append(token_id)
                    break
                    break

            if not valid_token_ids:
                break

            next_token_id: int = max(
                valid_token_ids, key=lambda token_id: logits[token_id]
            )
            generated_ids.append(next_token_id)
            generated_text += self._token_text(next_token_id)

            if render_step is not None:
                render_step(generated_text, len(generated_ids), max_tokens)

            if is_complete(generated_text):
                return generated_text

        if is_complete(generated_text):
            return generated_text
        raise ValueError("Unable to decode a value that matches constraints.")


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

        current_color: str = (
            "gray" if stage == "function selection" else "green"
        )
        print_colored(
            current_text if current_text else "<empty>",
            current_color,
        )
        print_colored(
            f"({token_count}/{max_tokens})",
            "yellow",
        )

        dynamic_text: str = (
            f"{current_text if current_text else '<empty>'}\n"
            f"({token_count}/{max_tokens})"
        )
        self.last_dynamic_lines = self._count_rendered_lines(dynamic_text)

    def _finish_live_panel(self) -> None:
        """Finish one live rendering block."""
        self.panel_header_printed = False
        self.last_dynamic_lines = 0

    def _function_prompt(self, prompt: str) -> str:
        """Build the instruction used to select the function name."""
        lines: list[str] = [
            "You are a function router.",
            "Return exactly one function name from the list.",
            "Do not explain.",
            "",
            "Routing hints:",
            "- square root, sqrt -> fn_get_square_root",
            "- replace, substitute, regex pattern matching ->"
            " fn_substitute_string_with_regex",
            "- reverse string -> fn_reverse_string",
            "- greet someone -> fn_greet",
            "- add/sum/plus/total -> fn_add_numbers",
            "- multiply/product/times -> fn_multiply_numbers",
            "- divide/quotient -> fn_divide_numbers",
            "",
            "Few-shot examples:",
            "Prompt: What is the square root of 16?",
            "Function name: fn_get_square_root",
            "Prompt: Replace all vowels in 'Programming is fun' with *",
            "Function name: fn_substitute_string_with_regex",
            "Prompt: Reverse the string 'hello'",
            "Function name: fn_reverse_string",
            "",
            "Available functions:",
        ]
        for func in self.funcdef:
            lines.append(f"- {func.name}: {func.description}")

        if any(func.name == "fn_not_implemented" for func in self.funcdef):
            lines.append(
                (
                    "If prompt does not match any function, choose "
                    "fn_not_implemented."
                )
            )

        lines.append(f"Prompt: {prompt}")
        lines.append("Function name:")
        return "\n".join(lines)

    def _generate_text(
        self,
        prompt: str,
        is_valid_prefix: Callable[[str], bool],
        is_complete: Callable[[str], bool],
        max_tokens: int,
        display_prompt: str = "",
        stage: str = "generation",
    ) -> str:
        """Generate constrained text completion for one stage."""
        shown_prompt: str = display_prompt if display_prompt else prompt

        def render_step(current: str, token_count: int, max_tok: int) -> None:
            self._render_live_panel(
                prompt=shown_prompt,
                stage=stage,
                current_text=current.strip(),
                token_count=token_count,
                max_tokens=max_tok,
            )

        return self.decoder.choose_with_constraints(
            prompt=prompt,
            is_valid_prefix=is_valid_prefix,
            is_complete=is_complete,
            max_tokens=max_tokens,
            render_step=render_step,
        )

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

    def _extract_numbers_from_text(self, text: str) -> list[float]:
        """Extract all numbers from text in order of appearance."""
        matches: list[str] = re.findall(
            r"-?\d+(?:\.\d+)?", text
        )
        return [float(m) for m in matches]

    def _extract_quoted_strings(self, text: str) -> list[str]:
        """Extract all quoted strings from text in order."""
        matches: list[tuple[int, str]] = []
        i: int = 0
        while i < len(text):
            if text[i] in ('"', "'"):
                quote_char: str = text[i]
                j: int = i + 1
                while j < len(text) and text[j] != quote_char:
                    if text[j] == "\\" and j + 1 < len(text):
                        j += 2
                    else:
                        j += 1
                if j < len(text) and text[j] == quote_char:
                    extracted: str = text[i + 1:j]
                    matches.append((i, extracted))
                    i = j + 1
                else:
                    i += 1
            else:
                i += 1
        return [s for _, s in matches]

    def _extract_word_after_pattern(
        self, text: str, pattern: str
    ) -> str:
        """Extract first word-like token after a pattern."""
        match: re.Match[str] | None = re.search(
            rf"(?:{pattern})\s+([\w]+)", text, re.IGNORECASE
        )
        if match:
            return match.group(1)
        return ""

    def _strip_wrapping_quotes(self, text: str) -> str:
        """Remove surrounding single/double quotes when present."""
        cleaned: str = text.strip()
        if len(cleaned) >= 2 and cleaned[0] == cleaned[-1]:
            if cleaned[0] in ('"', "'"):
                return cleaned[1:-1]
        return cleaned

    def _regex_from_replace_target(self, target: str) -> str:
        """Map natural-language replace targets to practical regex."""
        lowered: str = target.strip().lower()
        if lowered in ("number", "numbers", "digit", "digits"):
            return r"\d+"
        if lowered in ("vowel", "vowels"):
            return r"[AEIOUaeiou]"
        if lowered in ("space", "spaces", "whitespace"):
            return r"\s+"
        if lowered.startswith("word "):
            return re.escape(lowered[5:])
        return re.escape(target.strip())

    def _extract_replace_target(self, prompt: str) -> str:
        """Extract text between 'replace all' and 'in'."""
        match: re.Match[str] | None = re.search(
            r"replace\s+all\s+(.+?)\s+in\s+",
            prompt,
            re.IGNORECASE,
        )
        if match is None:
            return ""
        return self._strip_wrapping_quotes(match.group(1).strip())

    def _extract_replacement_text(self, prompt: str) -> str:
        """Extract text after 'with' in replace/substitute prompts."""
        match: re.Match[str] | None = re.search(
            r"\swith\s+(.+)$",
            prompt,
            re.IGNORECASE,
        )
        if match is None:
            return ""
        candidate: str = match.group(1).strip().rstrip(".?!")
        return self._strip_wrapping_quotes(candidate)

    def _extract_value(
        self,
        prompt: str,
        function_name: str,
        param_name: str,
        param_spec: ParameterSpec,
        param_index: int,
        param_count: int,
    ) -> Any:
        """Extract a parameter value using regex and context."""
        if param_spec.type == "number":
            numbers: list[float] = self._extract_numbers_from_text(prompt)
            if numbers and param_index < len(numbers):
                return numbers[param_index]
            return 0.0
        if param_spec.type == "boolean":
            lowered: str = prompt.lower()
            if "true" in lowered or "yes" in lowered:
                return True
            if "false" in lowered or "no" in lowered:
                return False
            return False
        if param_spec.type == "string":
            quoted_strings: list[str] = self._extract_quoted_strings(
                prompt
            )
            lowered_prompt: str = prompt.lower()
            if param_name == "name":
                extracted_word: str = self._extract_word_after_pattern(
                    prompt, r"greet"
                )
                if extracted_word:
                    return extracted_word

            # Handle: "Replace all <target> in <source> with <replacement>"
            replace_all_cond: bool = "replace all" in lowered_prompt
            if param_name == "regex" and replace_all_cond:
                target: str = self._extract_replace_target(prompt)
                if target:
                    return self._regex_from_replace_target(target)
            if param_name == "replacement" and replace_all_cond:
                replacement: str = self._extract_replacement_text(prompt)
                if replacement:
                    return replacement

            src_cond: bool = (
                param_name == "source_string"
                and ("substitute" in lowered_prompt
                     or "replace" in lowered_prompt)
            )
            if src_cond:
                if quoted_strings:
                    if len(quoted_strings) > 1:
                        return quoted_strings[-1]
                    return quoted_strings[0]
            if param_name == "regex" and (
                "substitute" in lowered_prompt
                or "replace" in lowered_prompt
            ):
                if len(quoted_strings) > 1:
                    return quoted_strings[0]
                words_regex: list[str] = re.findall(r"\w+", prompt)
                if len(words_regex) >= 4:
                    return words_regex[3]
            subst_cond: bool = (
                param_name == "replacement"
                and ("substitute" in lowered_prompt
                     or "replace" in lowered_prompt)
            )
            if subst_cond:
                if len(quoted_strings) > 1:
                    return quoted_strings[1]
                words_replace: list[str] = re.findall(r"\w+", prompt)
                if words_replace:
                    return words_replace[-1]
            if quoted_strings and param_index < len(quoted_strings):
                return quoted_strings[param_index]
            match: re.Match[str] | None = re.search(
                rf"{re.escape(param_name)}[\s:=]+([^\s,;.!?]+)",
                prompt,
                re.IGNORECASE,
            )
            if match:
                return match.group(1)
            return ""
        raise ValueError(f"Unsupported parameter type: {param_spec.type}")

    def build_result(self, prompt: str) -> dict[str, Any]:
        """Create one function call result for a single prompt."""
        self.panel_header_printed = False
        self.last_dynamic_lines = 0
        try:
            function_def: FunctionDefinition = self._choose_function(prompt)
            parameters: dict[str, Any] = {}
            param_list: list[tuple[str, ParameterSpec]] = list(
                function_def.parameters.items()
            )

            for param_index, (param_name, param_spec) in enumerate(
                param_list
            ):
                parameters[param_name] = self._extract_value(
                    prompt=prompt,
                    function_name=function_def.name,
                    param_name=param_name,
                    param_spec=param_spec,
                    param_index=param_index,
                    param_count=len(param_list),
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


def _print_result_preview(result: dict[str, Any]) -> None:
    """Print computed execution output with clear spacing."""
    name_value: Any = result.get("name")
    raw_params: Any = result.get("parameters")
    function_name: str = str(name_value)

    if not isinstance(raw_params, dict):
        print_colored("Output: <invalid parameters>", "red")
        print()
        return

    try:
        executed_output: Any = execute_function(function_name, raw_params)
        p_col_arg("Output", str(executed_output), "green")
    except Exception as e:
        print_colored(f"Output execution failed: {e}", "red")
    print()


def _build_result_with_fallback(
    llm: LLM_Model,
    func_defs: list[dict[str, Any]],
    prompt: str,
) -> dict[str, Any]:
    """Build one result and apply the same fallback policy if needed."""
    try:
        result: dict[str, Any] = llm.build_result(prompt)
        func_result_check(func_defs, result)
        return result
    except Exception as e:
        print_colored(
            f"Failed to build a result for prompt '{prompt}': {e}",
            "red",
        )
        fallback_function: FunctionDefinition = llm.funcdef[0]
        fallback_result: dict[str, Any] = FunctionCallResult(
            prompt=prompt,
            name=fallback_function.name,
            parameters=_default_parameters(fallback_function),
        ).model_dump(mode="json")
        print_colored(
            (
                "Using explicit fallback result with default parameter"
                " values."
            ),
            "yellow",
        )
        func_result_check(func_defs, fallback_result)
        return fallback_result


def _validate_functions_only(functions_definition_path: str) -> bool:
    """Validate function definitions for interactive mode."""
    try:
        func_defs_raw: Any = read_json(functions_definition_path)
    except Exception as e:
        print_colored(f"Error reading function definitions: {e}", "red")
        return False

    if not isinstance(func_defs_raw, list):
        print_colored("Function definitions must be a list.", "red")
        return False
    if len(func_defs_raw) == 0:
        print_colored("Function definitions list cannot be empty.", "red")
        return False

    for func in func_defs_raw:
        if (
            not isinstance(func, dict)
            or not validate_function_definition(func)
        ):
            print_colored(f"Invalid function definition: {func}", "red")
            return False
    return True


def _interactive_loop(
    llm: LLM_Model,
    func_defs: list[dict[str, Any]],
    output_path: str,
) -> None:
    """Run interactive prompt loop until Ctrl+C/Ctrl+D."""
    print_colored(
        "Interactive mode enabled. Press Ctrl+C or Ctrl+D to exit.",
        "cyan",
    )

    results: list[dict[str, Any]] = []

    while True:
        try:
            user_prompt: str = input("Prompt> ").strip()
        except EOFError:
            print_colored("Interactive mode closed (Ctrl+D).", "yellow")
            break
        except KeyboardInterrupt:
            print_colored("\nInteractive mode interrupted (Ctrl+C).", "yellow")
            break

        if user_prompt == "":
            continue

        result: dict[str, Any] = _build_result_with_fallback(
            llm=llm,
            func_defs=func_defs,
            prompt=user_prompt,
        )
        _print_result_preview(result)
        results.append(result)

    output_dir: str = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    try:
        write_json(output_path, results)
        print_colored(f"Results written to {output_path}", "green")
    except Exception as e:
        print_colored(f"Error writing results to output file: {e}", "red")


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
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run interactive prompt mode instead of reading --input file",
    )
    args: argparse.Namespace = parser.parse_args()

    try:
        if args.interactive:
            if not check_files_exist(args.functions_definition):
                return
            if not _validate_functions_only(args.functions_definition):
                return
        else:
            if not check_files_exist(args.functions_definition, args.input):
                return
            if not check_inputs_validity(
                args.functions_definition,
                args.input,
            ):
                return
            p_col_arg("Input file", args.input, "blue")
        p_col_arg("Function definitions", args.functions_definition, "blue")
        if args.interactive:
            p_col_arg("Output file", args.output, "blue")
        else:
            p_col_arg("Output file", args.output, "blue")
    except Exception as e:
        print_colored(f"Error checking files: {e}", "red")
        return

    try:
        func_defs: list[dict[str, Any]] = read_json(args.functions_definition)
        inputs: list[dict[str, Any]] = []
        if not args.interactive:
            inputs = read_json(args.input)
        if not func_defs:
            print_colored(
                (
                    "Function definitions are empty. At least one "
                    "function is required."
                ),
                "red",
            )
            return
        llm = LLM_Model(func_defs, inputs)

        if args.interactive:
            _interactive_loop(llm, func_defs, args.output)
            print_colored("Exiting program. Please wait...", "yellow")
            return

        output_dir: str = os.path.dirname(args.output)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        results: list[dict[str, Any]] = []
        for entry in inputs:
            prompt: str = str(entry.get("prompt", ""))
            result = _build_result_with_fallback(
                llm=llm,
                func_defs=func_defs,
                prompt=prompt,
            )

            _print_result_preview(result)
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
