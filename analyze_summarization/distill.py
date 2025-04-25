from typing import Optional
from model2vec.distill import distill


def _default_output_path(model_id: str) -> str:
    """Extract the model name from the model ID."""
    return model_id.split("/")[-1]  # Extract part after the slash


def distill_model(model_id: str, output_path: Optional[str], pca_dims: int) -> None:
    """Distill the model using Model2Vec.

    Args:
        model_id (str): Model ID to distill.
        output_path (Optional[str]): Output path for the distilled model.
            Defaults to the model name.
        pca_dims (int): Number of PCA dimensions for distillation.
    """
    if output_path is None:
        output_path = _default_output_path(model_id)

    print(f"Distilling model {model_id} to {output_path}...")
    # Load the sentence transformer model from the Hugging Face hub and distill it
    distilled = distill(model_id, pca_dims=pca_dims)
    print("Distillation complete.")

    # Save the distilled model to the specified output path
    distilled.save_pretrained(output_path)

    print(f"Model saved to {output_path}.")
