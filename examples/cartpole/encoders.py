"""Pre-built encoder architectures for adaptive perception.

Each encoder maps raw observations to a fixed-size latent tensor.
The LLM (or heuristic fallback) selects which encoder to instantiate
based on the discovered observation type.
"""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Encoder protocol
# ---------------------------------------------------------------------------


class Encoder(Protocol):
    """Common interface for all observation encoders."""

    def encode(self, observation: Any) -> torch.Tensor:
        """Encode a single raw observation into a latent tensor."""
        ...

    def train_step(self, observation: Any) -> float:
        """Run one encoder training step (e.g. reconstruction loss).

        Returns the loss value.  Implementations that are not trainable
        should return `0.0`.
        """
        ...

    @property
    def latent_dim(self) -> int:
        """Dimensionality of the latent representation."""
        ...


# ---------------------------------------------------------------------------
# MLP encoder  (for numeric vector observations)
# ---------------------------------------------------------------------------


class MLPEncoder(nn.Module):
    """Encodes numeric vectors through a trainable MLP + optional decoder.

    When `trainable=True` the encoder is trained via a reconstruction
    (autoencoder) loss each time `train_step` is called.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        latent_dim: int = 64,
        *,
        trainable: bool = False,
        lr: float = 1e-3,
    ) -> None:
        super().__init__()
        self._latent_dim = latent_dim
        self._trainable = trainable
        self._input_dim = input_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU(),
        )

        self.decoder: nn.Module | None = None
        self._optimizer: torch.optim.Optimizer | None = None
        if trainable:
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, input_dim),
            )
            self._optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    @property
    def latent_dim(self) -> int:
        return self._latent_dim

    # -- public API ----------------------------------------------------------

    def encode(self, observation: Any) -> torch.Tensor:
        x = self._to_tensor(observation)
        with torch.no_grad() if not self._trainable else torch.enable_grad():
            return self.encoder(x)

    def train_step(self, observation: Any) -> float:
        if not self._trainable or self.decoder is None or self._optimizer is None:
            return 0.0

        x = self._to_tensor(observation)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = nn.functional.mse_loss(x_hat, x)

        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()
        return loss.item()

    # -- helpers -------------------------------------------------------------

    def _to_tensor(self, obs: Any) -> torch.Tensor:
        if isinstance(obs, torch.Tensor):
            return obs.float()
        if isinstance(obs, np.ndarray):
            return torch.from_numpy(obs).float()
        return torch.tensor(obs, dtype=torch.float32)


# ---------------------------------------------------------------------------
# CNN encoder  (placeholder for image observations)
# ---------------------------------------------------------------------------


class CNNEncoder(nn.Module):
    """Placeholder CNN encoder for image observations."""

    def __init__(
        self,
        input_channels: int = 3,
        latent_dim: int = 64,
        *,
        trainable: bool = False,
        lr: float = 1e-3,
    ) -> None:
        super().__init__()
        self._latent_dim = latent_dim
        self._trainable = trainable

        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        # Final projection set lazily on first call
        self._proj: nn.Linear | None = None
        self._optimizer: torch.optim.Optimizer | None = None
        self._lr = lr

    @property
    def latent_dim(self) -> int:
        return self._latent_dim

    def encode(self, observation: Any) -> torch.Tensor:
        x = self._to_image_tensor(observation)
        features = self.conv(x).flatten()
        if self._proj is None:
            self._proj = nn.Linear(features.shape[0], self._latent_dim)
            if self._trainable:
                self._optimizer = torch.optim.Adam(self.parameters(), lr=self._lr)
        return self._proj(features)

    def train_step(self, observation: Any) -> float:
        return 0.0  # placeholder, no training logic yet

    def _to_image_tensor(self, obs: Any) -> torch.Tensor:
        if isinstance(obs, torch.Tensor):
            t = obs.float()
        elif isinstance(obs, np.ndarray):
            t = torch.from_numpy(obs).float()
        else:
            t = torch.tensor(obs, dtype=torch.float32)
        # Expect (H, W, C) → (C, H, W)
        if t.dim() == 3 and t.shape[-1] in (1, 3, 4):
            t = t.permute(2, 0, 1)
        return t


# ---------------------------------------------------------------------------
# Identity encoder  (fallback)
# ---------------------------------------------------------------------------


class IdentityEncoder:
    """No-op encoder: flatten and cast to float tensor."""

    def __init__(self, latent_dim: int | None = None) -> None:
        self._latent_dim = latent_dim  # set on first observation if None

    @property
    def latent_dim(self) -> int:
        if self._latent_dim is None:
            raise RuntimeError("latent_dim unknown, call encode() first")
        return self._latent_dim

    def encode(self, observation: Any) -> torch.Tensor:
        if isinstance(observation, torch.Tensor):
            t = observation.float().flatten()
        elif isinstance(observation, np.ndarray):
            t = torch.from_numpy(observation).float().flatten()
        else:
            t = torch.tensor(observation, dtype=torch.float32).flatten()
        if self._latent_dim is None:
            self._latent_dim = t.shape[0]
        return t

    def train_step(self, observation: Any) -> float:
        return 0.0


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_encoder(
    encoder_type: str,
    input_dim: int,
    latent_dim: int = 64,
    *,
    trainable: bool = False,
) -> MLPEncoder | CNNEncoder | IdentityEncoder:
    """Instantiate an encoder by name."""
    if encoder_type == "mlp":
        return MLPEncoder(input_dim, latent_dim=latent_dim, trainable=trainable)
    if encoder_type == "cnn":
        return CNNEncoder(latent_dim=latent_dim, trainable=trainable)
    return IdentityEncoder(latent_dim=input_dim)
