# dataset.py
# Handles everything between raw LFW images and model-ready tensors:
#   1. Download / load LFW via scikit-learn
#   2. Stratified train/test split
#   3. Fit PCA on TRAIN split only, transform both splits
#   4. Wrap in a PyTorch Dataset for use with DataLoader

import pickle   # for saving PCA and scaler artifacts
from pathlib import Path

import numpy as np
import torch
from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

from config import cfg


# PyTorch Dataset

class FaceDataset(Dataset):
    """Thin wrapper so DataLoader can iterate over (feature, label) pairs."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        # X: (N, n_components) float32 array
        # y: (N,) int64 label array
        # Converting to torch tensors
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# Data loading + preprocessing

def load_and_preprocess():
    """
    Returns
    -------
    train_ds : FaceDataset
    test_ds  : FaceDataset
    pca      : fitted sklearn PCA (will be saved as an MLflow artifact)
    scaler   : fitted StandardScaler (save alongside PCA)
    class_names : list[str]  — human-readable identity labels
    """
    print("Fetching LFW dataset (may download on first run)…")
    lfw = fetch_lfw_people(
        min_faces_per_person=cfg.min_faces_per_person,
        resize=cfg.image_resize,
        color=False,          # greyscale keeps dimensionality manageable
    )

    X_raw = lfw.data          # shape (N, H*W), already flattened by sklearn
    y = lfw.target            # integer class labels
    class_names = list(lfw.target_names)
    n_classes = len(class_names)

    print(f"  {X_raw.shape[0]} images | {n_classes} identities")
    print(f"  Image vector length: {X_raw.shape[1]}")

    # Stratified train / test split
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y,
        test_size=cfg.test_size,
        stratify=y,
        random_state=cfg.random_state,
    )

    # StandardScaler (fit on train only)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    # PCA (fit on train only)
    n_components = min(cfg.n_components, X_train_scaled.shape[0],
                       X_train_scaled.shape[1])
    print(f"Fitting PCA with {n_components} components…")
    pca = PCA(n_components=n_components, whiten=True, random_state=cfg.random_state)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    variance_explained = pca.explained_variance_ratio_.sum()
    print(f"  Variance explained: {variance_explained:.1%}")

    train_ds = FaceDataset(X_train_pca.astype(np.float32), y_train)
    test_ds = FaceDataset(X_test_pca.astype(np.float32), y_test)

    return train_ds, test_ds, pca, scaler, class_names


def save_preprocessors(pca: PCA, scaler: StandardScaler, output_dir: str = "."):
    """Pickle PCA + scaler so the inference container can reuse them."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "pca.pkl", "wb") as f:
        pickle.dump(pca, f)
    with open(out / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    print(f"Saved pca.pkl and scaler.pkl to {out}/")
    return str(out / "pca.pkl"), str(out / "scaler.pkl")


def load_preprocessors(directory: str = "."):
    """Load pickled PCA + scaler at inference time."""
    d = Path(directory)
    with open(d / "pca.pkl", "rb") as f:
        pca = pickle.load(f)
    with open(d / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return pca, scaler
