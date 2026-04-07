# model.py
# MLP classifier that operates on PCA-reduced face feature vectors.
#
# Architecture rationale:
#   Raw LFW pixels → PCA (128 components) → MLP → softmax over identities
#
#   A CNN would be the natural choice for raw images, but since PCA already extracts the
#   principal axes of facial variance, an MLP works just fine and is much
#   lighter (important for low-resource containers).

import torch
import torch.nn as nn


class FaceRecognitionMLP(nn.Module):
    """
    Multi-layer perceptron for closed-set face identification.

    Parameters
    ----------
    input_dim  : int   — number of PCA components (e.g. 128)
    n_classes  : int   — number of unique identities
    hidden_dims: tuple — widths of hidden layers, e.g. (512, 256)
    dropout    : float — dropout probability applied after each hidden layer
    """

    def __init__(
        self,
        input_dim: int,
        n_classes: int,
        hidden_dims: tuple = (512, 256),
        dropout: float = 0.3,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.n_classes = n_classes

        # Build hidden layers dynamically from hidden_dims
        layers = []
        in_dim = input_dim
        for out_dim in hidden_dims:
            # Linear, BatchNorm, ReLU, Dropout
            layers += [
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
            ]
            in_dim = out_dim

        # Final classification head (no activation — CrossEntropyLoss expects logits)
        layers.append(nn.Linear(in_dim, n_classes))

        self.network = nn.Sequential(*layers)

        # Weight initialisation
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Kaiming normal initialisation for ReLU activations
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (batch, input_dim) float32 tensor

        Returns
        -------
        logits : (batch, n_classes) float32 tensor
        """
        return self.network(x)  # forward pass to get logits

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Returns predicted class indices (argmax over logits)."""
        with torch.no_grad():
            logits = self.forward(x)
        return logits.argmax(dim=1) # dim=1 is the batch dimension

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Returns softmax probabilities over all classes."""
        with torch.no_grad():
            logits = self.forward(x)
        return torch.softmax(logits, dim=1)  # dim=1 is the batch dimension
