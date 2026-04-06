"""Dyna-Q transition model: learned world model + imagined rollouts.

Learns a neural network that predicts (Δstate, reward) from (state, action).
During planning, generates synthetic transitions and feeds them back to
the value function for accelerated learning.
"""

from __future__ import annotations

import random
from collections import deque
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from oak.interfaces import TransitionModel, ValueFunction
from oak.types import (
    OptionId,
    PlanningUpdate,
    Transition,
)


class _WorldModelNetwork(nn.Module):
    """Predicts (next_state - state, reward) from (state, one_hot_action)."""

    def __init__(self, state_dim: int, num_actions: int, hidden: int = 128) -> None:
        super().__init__()
        self.num_actions = num_actions
        self.trunk = nn.Sequential(
            nn.Linear(state_dim + num_actions, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.delta_head = nn.Linear(hidden, state_dim)
        self.reward_head = nn.Linear(hidden, 1)

    def forward(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        one_hot = F.one_hot(action.long(), self.num_actions).float()
        if state.dim() == 1:
            state = state.unsqueeze(0)
            one_hot = one_hot.unsqueeze(0)
        x = torch.cat([state, one_hot], dim=-1)
        h = self.trunk(x)
        delta = self.delta_head(h)
        reward = self.reward_head(h).squeeze(-1)
        return delta, reward


class DynaTransitionModel(TransitionModel[torch.Tensor, Any, dict[str, Any]]):
    """Dyna-Q: learn a world model, plan with imagined transitions.

    Parameters
    ----------
    state_dim:
        Dimensionality of the subjective state.
    num_actions:
        Number of discrete actions (discovered via trial-and-error).
    """

    def __init__(
        self,
        state_dim: int,
        num_actions: int,
        *,
        lr: float = 1e-3,
        buffer_capacity: int = 5_000,
        model_train_batch: int = 32,
        device: torch.device | None = None,
    ) -> None:
        self._state_dim = state_dim
        self._num_actions = num_actions
        self._model_train_batch = model_train_batch
        self._device = device or torch.device("cpu")

        self._model = _WorldModelNetwork(state_dim, num_actions).to(self._device)
        self._optimizer = torch.optim.Adam(self._model.parameters(), lr=lr)

        # Store real transitions for model training and planning sampling
        self._buffer: deque[tuple[torch.Tensor, int, float, torch.Tensor, bool]] = (
            deque(maxlen=buffer_capacity)
        )
        self._model_loss: float = 0.0
        self._step_count = 0

    # ------------------------------------------------------------------
    # TransitionModel interface
    # ------------------------------------------------------------------

    def update(
        self,
        transition: Transition[Any, torch.Tensor, dict[str, Any]],
    ) -> None:
        state = transition.subjective_state.detach()
        next_state = transition.next_subjective_state.detach()
        action = int(transition.action)
        reward = transition.reward
        done = transition.terminated

        self._buffer.append((state, action, reward, next_state, done))
        self._step_count += 1

        # Train world model on mini-batch from buffer
        if len(self._buffer) >= self._model_train_batch:
            self._train_model()

    def integrate_option_models(self) -> None:
        pass  # no option models in this implementation

    def plan(
        self,
        subjective_state: torch.Tensor,
        value_function: ValueFunction[torch.Tensor, Any, dict[str, Any]],
        budget: int,
    ) -> PlanningUpdate[Any]:
        """Dyna-Q planning: generate imagined transitions and update values."""
        # Don't plan until the world model has enough training data
        # to produce useful predictions (avoids corrupting Q-values early)
        min_warmup = max(self._model_train_batch, 500)
        if self._step_count < min_warmup:
            return PlanningUpdate(
                search_statistics={"planning_steps": 0, "model_loss": self._model_loss}
            )

        planning_steps = 0
        for _ in range(budget):
            # Sample a past state and action from experience
            idx = random.randrange(len(self._buffer))
            state, action, _, _, _ = self._buffer[idx]

            # Predict next state and reward using the world model
            with torch.no_grad():
                s = state.to(self._device)
                action_t = torch.tensor(action, device=self._device)
                delta, reward_pred = self._model(s, action_t)
                next_state_pred = s + delta.squeeze(0)

            # Create synthetic transition and feed to value function
            synthetic: Transition[Any, torch.Tensor, dict[str, Any]] = Transition(
                subjective_state=state,
                action=action,
                reward=reward_pred.item(),
                next_subjective_state=next_state_pred,
                terminated=False,  # imagined transitions are not terminal
            )
            value_function.update(synthetic, planning=True)
            planning_steps += 1

        return PlanningUpdate(
            search_statistics={
                "planning_steps": planning_steps,
                "model_loss": self._model_loss,
            }
        )

    def remove_option_models(self, option_ids: Sequence[OptionId]) -> None:
        pass

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _train_model(self) -> None:
        batch = random.sample(list(self._buffer), self._model_train_batch)
        states = torch.stack([b[0] for b in batch]).to(self._device)
        actions = torch.tensor([b[1] for b in batch], device=self._device)
        rewards = torch.tensor([b[2] for b in batch], dtype=torch.float32, device=self._device)
        next_states = torch.stack([b[3] for b in batch]).to(self._device)

        delta_target = next_states - states
        delta_pred, reward_pred = self._model(states, actions)

        loss = F.mse_loss(delta_pred, delta_target) + F.mse_loss(reward_pred, rewards)

        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

        self._model_loss = loss.item()
