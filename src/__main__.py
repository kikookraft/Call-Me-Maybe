import argparse
import numpy as np
from llm_sdk import Small_LLM_Model
from color import print_colored, p_col_arg


def test_model() -> None:
    print_colored(
        "Initializing model (saving/loading to HF hub might take a moment)...",
        "blue"
    )
    model = Small_LLM_Model()

    while True:
        prompt = input("Enter a prompt to generate from: ")

        # Encode prompt to token IDs
        tokenized_inputs = model.encode(prompt)[0].tolist()
        print("Generating response token-by-token...")

        # generates tokens
        for _ in range(75):
            # Ask model for logits of the next token, given the current context
            logits = model.get_logits_from_input_ids(tokenized_inputs)

            # Pick the token with the highest probability (greedy decoding)
            next_token_id = int(np.argmax(logits))

            # Decode just the new token and print it immediately
            print(model.decode([next_token_id]), end="", flush=True)

            # Append to our running context
            tokenized_inputs.append(next_token_id)

        print()  # Add a newline after generation finishes


def check_files_exist(*file_paths: str) -> bool:
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
        p_col_arg("Input file: ", args.input, "blue")
        p_col_arg("Function definitions: ", args.functions_definition, "blue")
        p_col_arg("Output file: ", args.output, "blue")
    except Exception as e:
        print_colored(f"Error checking/creating files: {e}", "red")
        return
    # try:
    #     test_model()
    # except KeyboardInterrupt:
    #     print_colored("\nGeneration interrupted by user.", "red")
    # except Exception as e:
    #     print_colored(f"An error occurred: {e}", "red")


if __name__ == "__main__":
    main()
