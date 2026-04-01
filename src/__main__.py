import argparse
import numpy as np
from llm_sdk import Small_LLM_Model


def test_model() -> None:
    print(
        "Initializing model (saving/loading to HF hub might take a moment)...")
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
    args = parser.parse_args()

    # Process args...
    print(f"Reading from {args.input}")
    print(f"Function definitions from {args.functions_definition}")
    print(f"Writing to {args.output}")
    try:
        test_model()
    except KeyboardInterrupt:
        print("\nGeneration interrupted by user.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
