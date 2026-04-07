# config.py
# Central configuration for the face recognition pipeline.
# Edit these values to tune the model without touching training logic.

from dataclasses import dataclass


@dataclass
class Config:
    # Data
    
    # LFW filter: only keep identities with at least this many images (keeps classes large enough to learn from)
    min_faces_per_person: int = 30
    image_resize: float = 0.4        # Downsample factor passed to fetch_lfw_people
    test_size: float = 0.2           # portion of data held out for testing and evaluation
    random_state: int = 42

    # PCA
    n_components: int = 128          # Number of principal components to keep

    # Model arch
    hidden_dims: tuple = (512, 256)  # hidden layer widths
    dropout: float = 0.3

    # Training hyperparameters
    epochs: int = 50
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4       # L2 regularisation
    patience: int = 8                # Early-stopping patience (epochs)

    # MLflow
    mlflow_tracking_uri: str = "http://localhost:5000"
    experiment_name: str = "face_recognition_lfw"
    model_name: str = "face_recognition_model"   # Name in the Model Registry

    # Promotion threshold: new model must beat the current production model's
    # weighted F1-score by at least this margin to be promoted.
    promotion_f1_threshold: float = 0.0        # 0.0 = any improvement promotes


cfg = Config()
