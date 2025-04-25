import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import pandas as pd


@dataclass
class AnalysisMetadata:
    date: str
    model: str
    dataset: str
    split: str
    skip: Optional[int]
    take: Optional[int]
    split_into_sentences: bool


def save_analysis(
    similarities: List[float],
    metadata: AnalysisMetadata,
    output: str,
) -> None:
    """Serialize the similarities to a CSV file."""

    metadata_df = pd.DataFrame(metadata.__dict__, index=["metadata"])
    similarities_df = pd.DataFrame({"similarity": similarities})
    combined_df = pd.concat([metadata_df, similarities_df], axis=1)
    combined_df.to_csv(
        output, index=False
    )  # set index=False to avoid saving DataFrame index


# TODO: Avoid turning the similarities into a list of floats, as they may be
# turned into a dataframe later. And do this while satisfying the type checker.
# Perhaps combine similarities dataframe and metadata into a single dataclass
def load_analysis(input: str) -> Tuple[List[float], AnalysisMetadata]:
    """Load the analysis from a CSV file."""
    df = pd.read_csv(input)
    metadata = AnalysisMetadata(
        **df.iloc[0][0:-1].to_dict()
    )  # All columns except the last one, which is the similarity
    similarities = df.iloc[1:, -1].astype(float).tolist()
    return similarities, metadata


@dataclass
class Report:
    similarity_count: int
    average_similarity: float
    min_similarity: float
    max_similarity: float
    distribution: pd.Series
    metadata: AnalysisMetadata


def save_report(
    report: Report,
    output: str,
) -> None:
    """Serialize the report to a folder with three CSV files, containing
    analysis metadata, various summary statistics and the similarity distribution."""
    report_dict = report.__dict__.copy()

    # Extract the distribution from the report dictionary, it will be saved
    # in a separate CSV file
    distribution = report_dict.pop("distribution")

    # Extract the metadata from the report dictionary, as it will also be saved
    # in a separate CSV file
    metadata = pd.DataFrame(report_dict.pop("metadata").__dict__, index=["metadata"])

    # Make dir at the output path if it doesn't exist
    os.makedirs(output, exist_ok=True)
    report_df = pd.DataFrame(report_dict, index=["report"])

    # Save all CSV files in the output folder
    report_df.to_csv(os.path.join(output, "summary_statistics.csv"), index=False)
    metadata.to_csv(os.path.join(output, "metadata.csv"), index=False)
    distribution.to_csv(os.path.join(output, "distribution.csv"), index=True)
