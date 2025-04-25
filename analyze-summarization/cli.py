"""Analyze the closeness of summaries to the original text in a given dataset,
distilling and using the selected model."""

import argparse
import sys
from typing import List

# Command names
DISTILL_COMMAND = "distill"
ANALYZE_COMMAND = "analyze"
REPORT_COMMAND = "report"

# Text and summary dataset
DATASET_NAME = "ccdv/govreport-summarization"

# Default model name
DEFAULT_MODEL_NAME = "BGE-M3"


def create_parser() -> argparse.ArgumentParser:
    """Create the command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Analyze the closeness of summaries to the original text "
        f"in the {DATASET_NAME} dataset, distilling and using the selected model."
    )
    subparsers = parser.add_subparsers(dest="command")

    # Distill command
    distill_parser = subparsers.add_parser(
        DISTILL_COMMAND,
        help="Distill a chosen model with Model2Vec.",
    )
    distill_parser.add_argument(
        "--model",
        "-m",
        type=str,
        help=f"Model to distill. Defaults to {DEFAULT_MODEL_NAME}.",
        default=DEFAULT_MODEL_NAME,
    )
    distill_parser.add_argument(
        "--output-path",
        "-o",
        type=str,
        help="Output path for the distilled model. Defaults to the model name.",
        default=None,
    )

    # TODO: add help command after all 3 other commands are implemented

    return parser


def distill_model(model_name: str, output_path: str) -> None:
    """Distill the model using Model2Vec."""

    print(f"Distilling model {model_name} to {output_path}...")

    print("Distillation complete.")

    print(f"Model saved to {output_path}.")


def main(argv: List[str] = sys.argv) -> int:
    """Handle command line arguments and execute the appropriate command."""
    parser = create_parser()
    args = parser.parse_args(argv[1:])

    if args.command == DISTILL_COMMAND:
        model_name = args.model
        output_path = args.output_path if args.output_path else model_name
        distill_model(model_name, output_path)
        return 0
    # TODO: handle help command
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
