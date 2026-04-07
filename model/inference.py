# inference.py
# Standalone inference helper used by the FastAPI container.
# Loads the current Production model + preprocessors from the MLflow registry
# and exposes a simple predict() function.
#
# Usage (inside the inference container):
#   from inference import FacePredictor
#   predictor = FacePredictor()          # loads Production model automatically
#   label, confidence = predictor.predict(image_array)

import pickle
import tempfile
from pathlib import Path
from typing import Optional

import mlflow
import mlflow.pytorch
import numpy as np
import torch
from mlflow import MlflowClient

from config import cfg

# Specify device for inference (GPU if available, else CPU)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FacePredictor:
    """
    Wraps the MLflow-registered Production model for inference.

    Parameters
    ----------
    model_name : str   — Model Registry name (default from cfg)
    stage      : str   — Registry stage to load from (default "Production")
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        stage: str = "Production",
    ):
        self.model_name = model_name or cfg.model_name
        self.stage = stage

        mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
        self._load()

    # Loading model + preprocessors

    def _load(self):
        """Pull model + preprocessors from MLflow and initialise for inference."""
        
        # Get latest Production model version from registry
        client = MlflowClient()
        versions = client.get_latest_versions(self.model_name, stages=[self.stage])
        if not versions:
            raise RuntimeError(
                f"No model in stage '{self.stage}' for '{self.model_name}'. "
                "Train and promote a model first."
            )

        version = versions[0]
        self.model_version = version.version
        run_id = version.run_id

        print(f"Loading {self.model_name} v{self.model_version} "
              f"from run {run_id} ({self.stage})")

        # Load PyTorch model
        model_uri = f"models:/{self.model_name}@production"
        self.model = mlflow.pytorch.load_model(model_uri, map_location=DEVICE)
        self.model.eval()

        # Load class names from model metadata
        model_info = client.get_model_version(self.model_name, self.model_version)
        # model_info.description or tags — metadata stored at log time
        run = client.get_run(run_id)
        n_classes = int(run.data.params.get("n_classes", self.model.n_classes))

        # Download preprocessors artifact (pca.pkl, scaler.pkl)
        with tempfile.TemporaryDirectory() as tmpdir:
            local_path = client.download_artifacts(
                run_id, "preprocessors", tmpdir
            )
            pca_file = Path(local_path) / "pca.pkl"
            scaler_file = Path(local_path) / "scaler.pkl"
            with open(pca_file, "rb") as f:
                self.pca = pickle.load(f)
            with open(scaler_file, "rb") as f:
                self.scaler = pickle.load(f)

        print(f"Ready — {n_classes} identities, "
              f"{self.pca.n_components_} PCA components.")

    def reload(self):
        """Hot-swap to the latest Production model (call from a background thread)."""
        print("Reloading model from registry…")
        self._load()

    # Preprocessing

    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """
        Transform a raw face image array into a model-ready tensor.

        Parameters
        ----------
        image : np.ndarray of shape (H, W) or (H, W, C)
            Greyscale or colour face image. Values in [0, 255] or [0, 1].

        Returns
        -------
        torch.Tensor of shape (1, n_pca_components)
        """
        # Flatten to 1-D
        flat = image.flatten().reshape(1, -1).astype(np.float32)
        # Scale, then PCA-project
        scaled = self.scaler.transform(flat)
        pca_feat = self.pca.transform(scaled).astype(np.float32)
        return torch.tensor(pca_feat, dtype=torch.float32).to(DEVICE)

    # ── Prediction ────────────────────────────────────────────────────────────

    @torch.no_grad()    # no gradients needed for inference
    def predict(self, image: np.ndarray):
        """
        Predict the identity of a face.

        Returns
        -------
        class_idx  : int   — predicted class index
        confidence : float — softmax probability of the top prediction
        """
        x = self.preprocess(image)
        logits = self.model(x)
        probs = torch.softmax(logits, dim=1)[0]
        class_idx = int(probs.argmax().item())
        confidence = float(probs[class_idx].item())
        return class_idx, confidence

    @torch.no_grad()
    def predict_top_k(self, image: np.ndarray, k: int = 5):
        """
        Returns the top-k predicted identities with their probabilities.

        Returns
        -------
        list of (class_idx, confidence) tuples, sorted descending by confidence
        """
        x = self.preprocess(image)
        logits = self.model(x)
        probs = torch.softmax(logits, dim=1)[0]
        top_k = probs.topk(min(k, len(probs)))
        return [
            (int(idx.item()), float(prob.item()))
            for idx, prob in zip(top_k.indices, top_k.values)
        ]
