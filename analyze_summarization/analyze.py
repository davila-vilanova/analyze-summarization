from typing import Optional

import numpy as np
from datasets import load_dataset
from model2vec import StaticModel
from serialization import AnalysisMetadata, save_analysis

DATASET_NAME = "ccdv/govreport-summarization"


REPORT_KEY = "report"
SUMMARY_KEY = "summary"


def analyze_dataset(
    model_path: str,
    output: str,
    split: str,
    skip: Optional[int] = None,
    take: Optional[int] = None,
) -> None:
    """Analyze the closeness of summaries to the original text in the dataset,
    using the specified distilled model, and outputting a CSV file with the
    distance results.

    Args:
        model_path (str): Path to the distilled model.
        output (str): Output path for the analysis results.
        split (str): Dataset split to analyze.
        skip (Optional[int]): Number of samples to skip in the dataset.
        take (Optional[int]): Number of samples to take from the dataset.
    """

    print(f"Analyzing with {model_path}...")

    model = StaticModel.from_pretrained(model_path)
    similarities = []

    print(f"Loading dataset {DATASET_NAME} for streaming.")

    dataset = load_dataset(DATASET_NAME, split=split, streaming=True)
    subset = dataset if skip is None else dataset.skip(skip)
    subset = subset if take is None else subset.take(take)
    for sample in subset:
        text1 = sample[REPORT_KEY]
        text2 = sample[SUMMARY_KEY]
        similarity = _text_similarity(text1, text2, model)
        similarities.append(similarity)

    print(f"Outputting results to {output}...")

    save_analysis(
        similarities,
        metadata=AnalysisMetadata(
            date=np.datetime_as_string(np.datetime64("now"), unit="s"),
            model=model_path,
            dataset=DATASET_NAME,
            split=split,
            skip=skip,
            take=take,
        ),
        output=output,
    )

    print(
        f"Analysis complete. Calculated {len(similarities)} similarities. Results saved to {output}."
    )


def _cosine_similarity(vec1, vec2) -> float:
    """Calculate the cosine similarity between two vectors."""
    # Calculate dot product
    dot_product = np.dot(vec1, vec2)

    # Calculate magnitudes
    magnitude1 = np.linalg.norm(vec1)
    magnitude2 = np.linalg.norm(vec2)

    # Calculate cosine similarity
    similarity = (
        dot_product / (magnitude1 * magnitude2) if magnitude1 and magnitude2 else 0.0
    )

    return similarity


def _text_similarity(
    text1: str,
    text2: str,
    model: StaticModel,
) -> float:
    """Calculate the similarity between two texts using the provided model.
    Args:
        text1 (str): First text to compare.
        text2 (str): Second text to compare.
        model (StaticModel): Model to use to calculate the embeddings.
    """

    vec1, vec2 = model.encode(
        [text1, text2],
        show_progress_bar=False,
        max_length=None,  # reports can be quite long
    )

    return _cosine_similarity(vec1, vec2)
