"""The PyTorch MLP that is being explained.

Origin: ``detect_claude.py`` l.930-986 (``SmallNet``, ``train_neural_net``,
``predict_neural_net``), unchanged. Requires ``torch``.

Architecture: 5 -> 128 -> 128 -> 1 with Softplus activations (smooth, so second
derivatives exist everywhere). Inputs and target are standardised with
scikit-learn ``StandardScaler``; training uses Adam (lr 1e-3), MSE loss,
batch size 128, 300 epochs and ``ReduceLROnPlateau`` on the epoch training loss.
There is no early stopping and no validation split.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from .scm import N_FEATURES


class SmallNet(nn.Module):
    def __init__(self, n_in: int = N_FEATURES, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, hidden),
            nn.Softplus(),
            nn.Linear(hidden, hidden),
            nn.Softplus(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def train_neural_net(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_epochs: int = 300,
    lr: float = 1e-3,
    batch_size: int = 128,
    seed: int = 42,
):
    torch.manual_seed(seed)
    scaler_X = StandardScaler().fit(X_train)
    scaler_y = StandardScaler().fit(y_train.reshape(-1, 1))

    Xt = torch.tensor(scaler_X.transform(X_train), dtype=torch.float32)
    yt = torch.tensor(scaler_y.transform(y_train.reshape(-1, 1)).ravel(), dtype=torch.float32)

    dataset = TensorDataset(Xt, yt)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = SmallNet(n_in=X_train.shape[1])
    opt = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, "min", patience=20, factor=0.5)
    loss_fn = nn.MSELoss()

    for epoch in range(n_epochs):
        epoch_loss = 0.0
        for batch_X, batch_y in dataloader:
            opt.zero_grad()
            loss = loss_fn(model(batch_X), batch_y)
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
        scheduler.step(epoch_loss / len(dataloader))

    model.eval()
    return model, scaler_X, scaler_y


def predict_neural_net(model: nn.Module, scaler_X: StandardScaler, scaler_y: StandardScaler, X: np.ndarray):
    with torch.no_grad():
        X_sc = torch.tensor(scaler_X.transform(X), dtype=torch.float32)
        pred_sc = model(X_sc).numpy().reshape(-1, 1)
        return scaler_y.inverse_transform(pred_sc).ravel()
