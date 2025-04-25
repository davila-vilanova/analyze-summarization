"""Analyze the closeness of summaries to the original text in a given dataset,
distilling and using the selected model."""

import argparse
import sys
from typing import List

from analyze import DATASET_NAME, analyze_dataset
from distill import distill_model
from report import generate_report

# Command names
DISTILL_COMMAND = "distill"
ANALYZE_COMMAND = "analyze"
REPORT_COMMAND = "report"


# Defaults
DEFAULT_MODEL_IDENTIFIER = "BAAI/bge-m3"
DEFAULT_PCA_DIMS = 256  # Number of PCA dimensions for distillation
DEFAULT_DATA_SPLIT = "validation"  # Default dataset split for analysis
DEFAULT_SPLIT_INTO_SENTENCES = False  # Whether to split the text into sentences


def main(argv: List[str] = sys.argv) -> int:
    """Handle command line arguments and execute the appropriate command."""
    # TODO: defensive input validation
    parser = create_parser()
    args = parser.parse_args(argv[1:])

    if args.command == DISTILL_COMMAND:
        distill_model(args.model, args.output_path, args.pca_dims)
        return 0
    elif args.command == ANALYZE_COMMAND:
        analyze_dataset(
            args.model,
            args.output_path,
            args.split,
            args.split_into_sentences,
            args.skip,
            args.take,
        )
        return 0
    elif args.command == REPORT_COMMAND:
        generate_report(args.input_path, args.output_path)
        return 0
    else:
        parser.print_help()
        return 1


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
    distill_parser.add_argument(
        "--pca-dims",
        "-d",
        type=int,
        help="Number of PCA dimensions for distillation. Defaults to "
        f"{DEFAULT_PCA_DIMS}.",
        default=DEFAULT_PCA_DIMS,
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
    analyze_parser.add_argument(
        "--split",
        "-s",
        type=str,
        help="Dataset split to analyze. Use 'train', 'validation', or 'test'. "
        f"Defaults to '{DEFAULT_DATA_SPLIT}'.",
        default=DEFAULT_DATA_SPLIT,
    )
    analyze_parser.add_argument(
        "--split-into-sentences",
        action="store_true",
        help="Whether to split the text into sentences before analysis. "
        f"Defaults to '{DEFAULT_SPLIT_INTO_SENTENCES}'.",
        default=DEFAULT_SPLIT_INTO_SENTENCES,
    )
    analyze_parser.add_argument(
        "--skip",
        type=int,
        help="Number of samples to skip in the dataset.",
        default=None,
    )
    analyze_parser.add_argument(
        "--take",
        type=int,
        help="Number of samples to take from the dataset.",
        default=None,
    )

    # Report command
    report_parser = subparsers.add_parser(
        REPORT_COMMAND,
        help="Generate a report from the analysis results.",
    )
    report_parser.add_argument(
        "--input-path",
        "-i",
        type=str,
        help="Input path for the analysis results.",
        required=True,
    )
    report_parser.add_argument(
        "--output-path",
        "-o",
        type=str,
        help="Output path for the report.",
        required=True,
    )

    return parser


if __name__ == "__main__":
    sys.exit(main())
