"""Analyze the closeness of summaries to the original text in a given dataset,
distilling and using the selected model."""

import argparse
import sys
from typing import List

from commands.analyze import analyze_dataset
from commands.distill import distill_model

# Command names
DISTILL_COMMAND = "distill"
ANALYZE_COMMAND = "analyze"
REPORT_COMMAND = "report"

# Text and summary dataset
DATASET_NAME = "ccdv/govreport-summarization"

# Default model
DEFAULT_MODEL_IDENTIFIER = "BAAI/bge-m3"


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
        help=f"Model to distill. Defaults to {DEFAULT_MODEL_IDENTIFIER}.",
        default=DEFAULT_MODEL_IDENTIFIER,
    )
    distill_parser.add_argument(
        "--output-path",
        "-o",
        type=str,
        help="Output path for the distilled model. Defaults to the model name.",
        default=None,
    )

    # Analyze command
    analyze_parser = subparsers.add_parser(
        ANALYZE_COMMAND,
        help="Analyze the closeness of summaries to the original text in the "
        "dataset, using the specified distilled model, and outputs a CSV file "
        "with the distance results.",
    )
    analyze_parser.add_argument(
        "--model",
        "-m",
        type=str,
        help="Model to use for analysis.",
        required=True,
    )
    analyze_parser.add_argument(
        "--output-path",
        "-o",
        type=str,
        help="Output path for the analysis results.",
        required=True,
    )

    # TODO: add help command after all 3 other commands are implemented

    return parser


def main(argv: List[str] = sys.argv) -> int:
    """Handle command line arguments and execute the appropriate command."""
    parser = create_parser()
    args = parser.parse_args(argv[1:])

    if args.command == DISTILL_COMMAND:
        distill_model(args.model, args.output_path)
        return 0
    elif args.command == ANALYZE_COMMAND:
        analyze_dataset(args.model, args.output_path)
        return 0
    # TODO: handle help command
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
