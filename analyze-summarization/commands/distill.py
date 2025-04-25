from typing import Optional
from model2vec import StaticModel
from model2vec.distill import distill

_PCA_DIMS = 256  # Number of PCA dimensions for distillation


def _default_output_path(model_id: str) -> str:
    """Extract the model name from the model ID."""
    return model_id.split("/")[-1]  # Extract part after the slash


def distill_model(model_id: str, output_path: Optional[str]) -> None:
    """Distill the model using Model2Vec."""
    if output_path is None:
        output_path = _default_output_path(model_id)

    print(f"Distilling model {model_id} to {output_path}...")
    # Load the sentence transformer model from the Hugging Face hub and distill it
    distilled = distill(model_name="BAAI/bge-base-en-v1.5", pca_dims=_PCA_DIMS)
    print("Distillation complete.")

    # Save the distilled model to the specified output path
    distilled.save_pretrained(output_path)

    print(f"Model saved to {output_path}.")
