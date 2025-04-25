import pandas as pd

from serialization import Report, load_analysis, save_report


def generate_report(input: str, output: str) -> None:
    """
    Generate a report from the analysis results.

    Args:
        input (str): Input path for the analysis results.
        output (str): Output path for the report.
    """
    similarities, metadata = load_analysis(input)
    similarities_df = pd.DataFrame({"similarity": similarities})

    # Build the report
    distribution = (
        pd.cut(similarities_df["similarity"], bins=10).value_counts(
            normalize=True, sort=False
        )
        * 100
    )

    save_report(
        Report(
            similarity_count=len(similarities),
            average_similarity=similarities_df["similarity"].mean(),
            min_similarity=similarities_df["similarity"].min(),
            max_similarity=similarities_df["similarity"].max(),
            distribution=distribution,
            metadata=metadata,
        ),
        output=output,
    )
    print(f"Report saved to {output}.")
