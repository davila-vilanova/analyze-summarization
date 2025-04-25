import pandas as pd

from serialization import Report, load_analysis, save_report


def generate_report(input: str, output: str) -> None:
    """
    Generate and save a report from the analysis results.

    Args:
        input (str): Input path for the analysis results.
        output (str): Output path for the report.
    """
    similarities, metadata = load_analysis(input)
    similarities_df = pd.DataFrame({"similarity": similarities})

    report = Report(
        similarity_count=len(similarities),
        average_similarity=similarities_df["similarity"].mean(),
        min_similarity=similarities_df["similarity"].min(),
        max_similarity=similarities_df["similarity"].max(),
        distribution=_calculate_distribution(similarities_df["similarity"]),
        metadata=metadata,
    )

    save_report(report, output)

    print(f"Report saved to {output}.")


def _calculate_distribution(similarities: pd.Series, bins: int = 10) -> pd.Series:
    """
    Calculate the distribution of similarities.

    Args:
        similarities (pd.Series): Series of similarity values.
        bins (int): Number of bins for the distribution.

    Returns:
        pd.Series: Distribution of similarities.
    """
    return (
        pd.cut(similarities, bins=bins).value_counts(normalize=True, sort=False) * 100
    )
