# train.py
# End-to-end training entry point:
#   1. Load & preprocess LFW data
#   2. Train the MLP with early stopping
#   3. Evaluate on the test set (accuracy, per-class F1, ROC-AUC)
#   4. Log everything to MLflow (params, metrics, artifacts, model)
#   5. Conditionally promote the model to "Production" in the Model Registry
#      if it beats the current production model's weighted F1 score.

import os
import tempfile
from pathlib import Path

import mlflow
import mlflow.pytorch
import numpy as np
import torch
import torch.nn as nn
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader

from config import cfg
from dataset import load_and_preprocess, save_preprocessors
from model import FaceRecognitionMLP


# Device setup
# GPU if available, else CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# Training loop

def train_one_epoch(model, loader, criterion, optimizer):
    model.train()   # set model to training mode (enables dropout, batchnorm updates)
    total_loss, correct, total = 0.0, 0, 0
    # Iterate over batches
    for X_batch, y_batch in loader:
        # Move batch to device  
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
        optimizer.zero_grad()   # zero gradients from previous step
        logits = model(X_batch) # forward pass to get logits
        loss = criterion(logits, y_batch)   # compute loss
        loss.backward()          # backprop to compute gradients
        optimizer.step()     # Update metrics and counters

        # Update metrics
        total_loss += loss.item() * len(y_batch)
        correct += (logits.argmax(1) == y_batch).sum().item()
        total += len(y_batch)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion):
    """Evaluates the model on the test set."""
    model.eval()    # set model to evaluation mode (disables dropout, batchnorm updates)
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels, all_probs = [], [], []

    # Iterate over batches
    for X_batch, y_batch in loader:
        # Move batch to device
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
        logits = model(X_batch)     # forward pass to get logits
        loss = criterion(logits, y_batch)   # compute loss

        probs = torch.softmax(logits, dim=1)    # convert logits to probabilities
        preds = logits.argmax(dim=1)            # predicted class indices

        # Update metrics
        total_loss += loss.item() * len(y_batch)
        correct += (preds == y_batch).sum().item()
        total += len(y_batch)

        # Store predictions, labels, and probabilities for overall metrics
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y_batch.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    # Compute overall metrics
    avg_loss = total_loss / total
    accuracy = correct / total
    all_probs_np = np.array(all_probs)
    all_labels_np = np.array(all_labels)
    all_preds_np = np.array(all_preds)

    f1 = f1_score(all_labels_np, all_preds_np, average="weighted", zero_division=0)

    # ROC-AUC: one-vs-rest, only if every class has at least one sample
    try:
        roc_auc = roc_auc_score(
            all_labels_np, all_probs_np,
            multi_class="ovr", average="weighted",
        )
    except ValueError:
        roc_auc = float("nan")

    return avg_loss, accuracy, f1, roc_auc, all_labels_np, all_preds_np


# ── MLflow helpers ────────────────────────────────────────────────────────────

def get_production_f1(client: MlflowClient, model_name: str) -> float:
    """
    Returns the weighted F1-score of the current Production model in the
    registry, or -1.0 if no Production model exists yet.
    """
    try:
        version = client.get_model_version_by_alias(model_name, "production")
        run = client.get_run(version.run_id)
        return float(run.data.metrics.get("test_f1_weighted", -1.0))
    except MlflowException:
        return -1.0


def promote_model(client: MlflowClient, model_name: str, version: str):
    """Archive any current Production versions, then promote the new one."""
    try:
        client.delete_registered_model_alias(model_name, "production")
    except MlflowException:
        pass  # No existing production alias
    client.set_registered_model_alias(model_name, "production", version)
    print(f"Model '{model_name}' version {version} promoted to Production.")


# Main training pipeline

def main():
    # Data
    train_ds, test_ds, pca, scaler, class_names = load_and_preprocess()
    n_classes = len(class_names)
    input_dim = train_ds.X.shape[1]

    # loading data into PyTorch DataLoaders for batching and shuffling
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=0, pin_memory=(DEVICE.type == "cuda"),
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=0,
    )

    print(f"\nClasses ({n_classes}): {class_names}")
    print(f"Input dim (PCA components): {input_dim}\n")

    # MLflow setup
    mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
    mlflow.set_experiment(cfg.experiment_name)
    client = MlflowClient()

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        print(f"MLflow run ID: {run_id}\n")

        # Log hyperparameters
        mlflow.log_params({
            "min_faces_per_person": cfg.min_faces_per_person,
            "image_resize": cfg.image_resize,
            "n_pca_components": input_dim,
            "hidden_dims": str(cfg.hidden_dims),
            "dropout": cfg.dropout,
            "epochs": cfg.epochs,
            "batch_size": cfg.batch_size,
            "learning_rate": cfg.learning_rate,
            "weight_decay": cfg.weight_decay,
            "n_classes": n_classes,
            "train_size": len(train_ds),
            "test_size": len(test_ds),
        })

        # Model, loss, optimiser
        model = FaceRecognitionMLP(
            input_dim=input_dim,
            n_classes=n_classes,
            hidden_dims=cfg.hidden_dims,
            dropout=cfg.dropout,
        ).to(DEVICE)

        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )
        # Cosine annealing LR schedule — decays smoothly, no hyperparam tuning
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.epochs,
        )

        # Training with early stopping
        best_val_f1 = -1.0
        best_state = None
        patience_counter = 0

        for epoch in range(1, cfg.epochs + 1):
            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer
            )
            val_loss, val_acc, val_f1, val_roc_auc, _, _ = evaluate(
                model, test_loader, criterion
            )
            scheduler.step()

            # Log per-epoch metrics
            mlflow.log_metrics({
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "val_f1_weighted": val_f1,
                "val_roc_auc": val_roc_auc,
                "lr": scheduler.get_last_lr()[0],
            }, step=epoch)

            print(
                f"Epoch {epoch:3d}/{cfg.epochs} | "
                f"Train loss {train_loss:.4f} acc {train_acc:.3f} | "
                f"Val loss {val_loss:.4f} acc {val_acc:.3f} "
                f"F1 {val_f1:.3f} AUC {val_roc_auc:.3f}"
            )

            # Early stopping — track best model weights
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= cfg.patience:
                    print(f"\nEarly stopping at epoch {epoch} (patience={cfg.patience})")
                    break

        # Final evaluation using best weights
        model.load_state_dict(best_state)
        _, test_acc, test_f1, test_roc_auc, y_true, y_pred = evaluate(
            model, test_loader, criterion
        )

        print(f"\n" + "="*50 + "\nTest results\n" + "="*50)
        print(f"Accuracy : {test_acc:.4f}")
        print(f"F1 (wtd) : {test_f1:.4f}")
        print(f"ROC-AUC  : {test_roc_auc:.4f}")

        report = classification_report(
            y_true, y_pred, target_names=class_names, zero_division=0
        )
        print("\nPer-class report:")
        print(report)

        # Log final test metrics
        mlflow.log_metrics({
            "test_accuracy": test_acc,
            "test_f1_weighted": test_f1,
            "test_roc_auc": test_roc_auc,
        })

        # Save and log preprocessors as artifacts
        with tempfile.TemporaryDirectory() as tmpdir:
            pca_path, scaler_path = save_preprocessors(pca, scaler, tmpdir)
            mlflow.log_artifact(pca_path, artifact_path="preprocessors")
            mlflow.log_artifact(scaler_path, artifact_path="preprocessors")

            # Save classification report as text artifact
            report_path = Path(tmpdir) / "classification_report.txt"
            report_path.write_text(report)
            mlflow.log_artifact(str(report_path))

        # Log the PyTorch model
        # Log class names as model metadata so inference knows label-->name mapping
        model_info = mlflow.pytorch.log_model(
            pytorch_model=model,
            name="model",
            registered_model_name=cfg.model_name,
            metadata={
                "class_names": [str(c) for c in class_names],
                "input_dim": int(input_dim),
                "n_classes": int(n_classes),
                "hidden_dims": [int (h) for h in cfg.hidden_dims],
                "dropout": float(cfg.dropout),
            },
        )
        new_version = model_info.registered_model_version
        print(f"\nRegistered model version: {new_version}")

        # Threshold-based promotion
        prod_f1 = get_production_f1(client, cfg.model_name)
        improvement = test_f1 - prod_f1

        print(f"\nProduction F1 (current): {prod_f1:.4f}")
        print(f"New model F1           : {test_f1:.4f}")
        print(f"Improvement            : {improvement:+.4f} "
              f"(threshold: {cfg.promotion_f1_threshold:+.4f})")

        if improvement > cfg.promotion_f1_threshold:
            promote_model(client, cfg.model_name, str(new_version))
            mlflow.set_tag("promoted", "true")
        else:
            print("Model did NOT meet the promotion threshold — staying in Staging.")
            client.set_registered_model_alias(
                cfg.model_name, f"staging-v{new_version}", str(new_version)
            )
            mlflow.set_tag("promoted", "false")

    print(f"\nDone. View run at {cfg.mlflow_tracking_uri}/#/experiments/")


if __name__ == "__main__":
    main()
