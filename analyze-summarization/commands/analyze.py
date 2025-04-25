from typing import Optional
import numpy as np
import pandas as pd
from datasets import load_dataset
from model2vec import StaticModel

DATASET_NAME = "ccdv/govreport-summarization"

# Whether to split reports and summaries into sentences before encoding.
# Open question: how do the results vary when passing the whole text vs.
# splitting it into sentences? Which is a more precise indicator of similarity?
SPLIT_INTO_SENTENCES = False

REPORT_KEY = "report"
SUMMARY_KEY = "summary"


# encode() takes a list of sentences to encode. Open question: how do the results
# vary when passing the whole text vs. splitting it into sentences? Which is a
# more precise indicator of similarity?
def _split_into_sentences(text: str) -> list[str]:
    # quick and dirty, perhaps naive sentence splitting
    sentences = text.split(". ")
    # Restore the period at the end of each sentence
    sentences = [s + "." for s in sentences]
    return sentences


def _cosine_similarity(vec1, vec2) -> float:
    # Calculate dot product
    dot_product = np.dot(vec1, vec2)

    # Calculate magnitudes
    magnitude1 = np.linalg.norm(vec1)
    magnitude2 = np.linalg.norm(vec2)

    # Calculate cosine similarity
    similarity = dot_product / (magnitude1 * magnitude2)

    return similarity


def _text_similarity(
    text1: str,
    text2: str,
    model: StaticModel,
    split_into_sentences: bool,
) -> float:
    if split_into_sentences:
        text1 = _split_into_sentences(text1)
        text2 = _split_into_sentences(text2)

    vec1, vec2 = model.encode(
        sentences=[text1, text2],
        show_progress_bar=False,
        max_length=None,  # reports can be quite long
    )

    return _cosine_similarity(vec1, vec2)


def analyze_dataset(
    model_path: str,
    output: str,
    split: str,
    skip: Optional[int] = None,
    take: Optional[int] = None,
) -> None:
    """Analyze the closeness of summaries to the original text in the dataset."""
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
        similarity = _text_similarity(text1, text2, model, SPLIT_INTO_SENTENCES)
        similarities.append(similarity)

    print(f"Outputting results to {output}...")
    # save similarities to CSV along with metadata
    metadata = {
        "date": np.datetime_as_string(np.datetime64("now"), unit="s"),
        "model": model_path,
        "dataset": DATASET_NAME,
        "split": split,
        "skip": skip,
        "take": take,
        "split_into_sentences": SPLIT_INTO_SENTENCES,
    }

    df = pd.DataFrame({"similarity": similarities})
    metadata_df = pd.DataFrame(metadata, index=["metadata"])
    df = pd.concat([metadata_df, df])
    df.to_csv(output, index=False)  # set index=False to avoid saving DataFrame index

    print(
        f"Analysis complete. Calculated {len(similarities)} similarities. Results saved to {output}."
    )
