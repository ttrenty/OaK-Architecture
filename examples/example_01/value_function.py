"""Option-level value function with GVFs, utility assessment, and curation.

Learns Q_Omega(s, o), the value of executing option *o* from subjective
state *s*, plus auxiliary General Value Function (GVF) predictions tied
to discovered features.

Maintains a replay buffer and target network for stable DQN-style learning.
"""

from __future__ import annotations

import random
from collections import deque
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn

from oak.interfaces import ValueFunction
from oak.types import (
    ComponentKind,
    CurationDecision,
    GeneralValueFunctionId,
    OptionId,
    Transition,
    UsageRecord,
    UtilityRecord,
)


class _QOmegaNetwork(nn.Module):
    """MLP that outputs Q-values for each option slot."""

    def __init__(self, state_dim: int, max_options: int, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, max_options),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReplayBuffer:
    """Simple circular replay buffer storing tensors."""

    def __init__(self, capacity: int = 10_000) -> None:
        self._buf: deque[tuple[torch.Tensor, int, float, torch.Tensor, bool]] = deque(
            maxlen=capacity
        )

    def push(
        self,
        state: torch.Tensor,
        option_idx: int,
        reward: float,
        next_state: torch.Tensor,
        done: bool,
    ) -> None:
        self._buf.append(
            (state.detach(), option_idx, reward, next_state.detach(), done)
        )

    def sample(
        self, batch_size: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = random.sample(list(self._buf), batch_size)
        states = torch.stack([b[0] for b in batch])
        options = torch.tensor([b[1] for b in batch], dtype=torch.long)
        rewards = torch.tensor([b[2] for b in batch], dtype=torch.float32)
        next_states = torch.stack([b[3] for b in batch])
        dones = torch.tensor([b[4] for b in batch], dtype=torch.float32)
        return states, options, rewards, next_states, dones

    def __len__(self) -> int:
        return len(self._buf)


class OptionValueFunction(ValueFunction[torch.Tensor, Any, dict[str, Any]]):
    """DQN-style value function over option slots with auxiliary GVF heads.

    Parameters
    ----------
    state_dim:
        Dimensionality of the subjective state (encoder latent dim).
    max_options:
        Maximum number of option slots.  Inactive slots are masked.
    """

    def __init__(
        self,
        state_dim: int,
        max_options: int = 8,
        *,
        lr: float = 1e-3,
        gamma: float = 0.99,
        buffer_capacity: int = 5_000,
        batch_size: int = 64,
        target_sync_freq: int = 200,
        device: torch.device | None = None,
    ) -> None:
        self._state_dim = state_dim
        self._max_options = max_options
        self._gamma = gamma
        self._batch_size = batch_size
        self._target_sync_freq = target_sync_freq
        self._device = device or torch.device("cpu")

        # Q_Omega networks
        self._q_net = _QOmegaNetwork(state_dim, max_options).to(self._device)
        self._target_net = _QOmegaNetwork(state_dim, max_options).to(self._device)
        self._target_net.load_state_dict(self._q_net.state_dict())
        self._optimizer = torch.optim.Adam(self._q_net.parameters(), lr=lr)

        # Option mask: True for slots that have an active option
        self._option_mask = [False] * max_options
        self._option_names: dict[int, str] = {}  # slot → option_id

        # Replay
        self._buffer = ReplayBuffer(buffer_capacity)
        self._step_count = 0

        # GVF auxiliary heads (added dynamically)
        self._gvf_heads: nn.ModuleDict = nn.ModuleDict()
        self._gvf_optim: torch.optim.Optimizer | None = None

        # Utility tracking
        self._usage_counts: dict[tuple[str, str], float] = {}
        self._gvf_error_accum: dict[str, float] = {}
        self._gvf_error_count: dict[str, int] = {}

        # Track which option was active for the last transition
        self._last_option_idx: int = 0

    # ------------------------------------------------------------------
    # Option slot management (called by ReactivePolicy)
    # ------------------------------------------------------------------

    def register_option(self, option_id: str) -> int:
        """Activate an option slot.  Returns the slot index."""
        for idx in range(self._max_options):
            if not self._option_mask[idx]:
                self._option_mask[idx] = True
                self._option_names[idx] = option_id
                return idx
        raise RuntimeError("No free option slots")

    def deregister_option(self, option_id: str) -> None:
        for idx, oid in list(self._option_names.items()):
            if oid == option_id:
                self._option_mask[idx] = False
                del self._option_names[idx]
                return

    def option_idx_for(self, option_id: str) -> int | None:
        for idx, oid in self._option_names.items():
            if oid == option_id:
                return idx
        return None

    @property
    def active_option_ids(self) -> list[str]:
        return [self._option_names[i] for i in sorted(self._option_names)]

    def set_last_option_idx(self, idx: int) -> None:
        """Record which option was active so `update` can store it."""
        self._last_option_idx = idx

    # ------------------------------------------------------------------
    # GVF management
    # ------------------------------------------------------------------

    def add_gvf(self, gvf_id: str) -> None:
        if gvf_id not in self._gvf_heads:
            self._gvf_heads[gvf_id] = nn.Linear(self._state_dim, 1).to(self._device)
            self._gvf_error_accum[gvf_id] = 0.0
            self._gvf_error_count[gvf_id] = 0
            # Rebuild GVF optimizer
            if self._gvf_heads:
                self._gvf_optim = torch.optim.Adam(
                    self._gvf_heads.parameters(), lr=1e-3
                )

    # ------------------------------------------------------------------
    # ValueFunction interface
    # ------------------------------------------------------------------

    def update(
        self,
        transition: Transition[Any, torch.Tensor, dict[str, Any]],
        *,
        planning: bool = False,
    ) -> Mapping[GeneralValueFunctionId, float]:
        state = transition.subjective_state
        next_state = transition.next_subjective_state
        done = transition.terminated
        option_idx = self._resolve_option_idx(transition.option_id)

        if option_idx is None:
            return {}

        # Only store real transitions in replay buffer (not synthetic from planning)
        if not planning:
            self._buffer.push(
                state, option_idx, transition.reward, next_state, done
            )

        td_errors: dict[str, float] = {}

        # Compute per-transition advantage for the active option
        with torch.no_grad():
            s = state.to(self._device)
            ns = next_state.to(self._device)
            q_current = self._q_net(s)
            q_next = self._q_net(ns)
            mask = torch.tensor(
                self._option_mask, dtype=torch.bool, device=self._device
            )
            q_next_masked = q_next.clone()
            q_next_masked[~mask] = -float("inf")
            if mask.any():
                q_next_max = q_next_masked[mask].max().item()
            else:
                q_next_max = 0.0
            q_active = q_current[option_idx].item()
            advantage = (
                transition.reward
                + self._gamma * q_next_max * (1.0 - float(done))
                - q_active
            )
        td_errors["advantage"] = advantage

        if planning:
            # Planning: do a single online TD update from the synthetic transition
            # (don't trigger full buffer-based learning to avoid overtraining)
            self._online_td_update(
                state,
                next_state,
                done,
                transition.reward,
                option_idx,
            )
        else:
            # Q_Omega learning from replay
            if len(self._buffer) >= self._batch_size:
                td_errors.update(self._learn_q_omega())

            # GVF learning (online, single transition)
            td_errors.update(self._learn_gvfs(transition))

            # Target network sync
            self._step_count += 1
            if self._step_count % self._target_sync_freq == 0:
                self._target_net.load_state_dict(self._q_net.state_dict())

        return td_errors

    def predict(
        self,
        subjective_state: torch.Tensor,
    ) -> Mapping[GeneralValueFunctionId, float]:
        s = subjective_state.to(self._device)
        with torch.no_grad():
            q_values = self._q_net(s)

        result: dict[str, float] = {}
        for idx, oid in self._option_names.items():
            result[f"q_{oid}"] = q_values[idx].item()

        # GVF predictions
        for gvf_id, head in self._gvf_heads.items():
            with torch.no_grad():
                result[f"gvf_{gvf_id}"] = head(s).item()

        return result

    def observe_usage(self, usage_records: Sequence[UsageRecord]) -> None:
        for rec in usage_records:
            key = (rec.kind.value, rec.component_id)
            self._usage_counts[key] = self._usage_counts.get(key, 0.0) + rec.amount

    def utility_scores(self) -> Sequence[UtilityRecord]:
        records: list[UtilityRecord] = []
        for (kind_val, comp_id), amount in self._usage_counts.items():
            kind = ComponentKind(kind_val)
            # Combine usage with GVF error magnitude for richer utility
            gvf_utility = 0.0
            gvf_key = comp_id
            if (
                gvf_key in self._gvf_error_accum
                and self._gvf_error_count.get(gvf_key, 0) > 0
            ):
                gvf_utility = (
                    abs(self._gvf_error_accum[gvf_key]) / self._gvf_error_count[gvf_key]
                )
            records.append(
                UtilityRecord(
                    kind=kind,
                    component_id=comp_id,
                    utility=amount + gvf_utility,
                )
            )
        return tuple(records)

    def curate(self) -> CurationDecision:
        # Conservative: only drop GVFs with near-zero utility after many steps
        if self._step_count < 500:
            return CurationDecision()

        drop_gvfs: list[str] = []
        for gvf_id in list(self._gvf_heads.keys()):
            count = self._gvf_error_count.get(gvf_id, 0)
            if count > 200:
                avg_err = abs(self._gvf_error_accum.get(gvf_id, 0.0)) / max(count, 1)
                if avg_err < 1e-5:
                    drop_gvfs.append(gvf_id)

        return CurationDecision(drop_general_value_functions=tuple(drop_gvfs))

    def remove(
        self,
        general_value_function_ids: Sequence[GeneralValueFunctionId],
    ) -> None:
        for gvf_id in general_value_function_ids:
            if gvf_id in self._gvf_heads:
                del self._gvf_heads[gvf_id]
                self._gvf_error_accum.pop(gvf_id, None)
                self._gvf_error_count.pop(gvf_id, None)

    # ------------------------------------------------------------------
    # Internal learning methods
    # ------------------------------------------------------------------

    def _online_td_update(
        self,
        state: torch.Tensor,
        next_state: torch.Tensor,
        done: bool,
        reward: float,
        option_idx: int,
    ) -> None:
        """Single online TD update for Dyna-Q planning (no replay buffer)."""
        s = state.to(self._device)
        ns = next_state.to(self._device)

        q_all = self._q_net(s)
        q_current = q_all[option_idx]

        with torch.no_grad():
            q_next = self._target_net(ns)
            mask = torch.tensor(
                self._option_mask, dtype=torch.bool, device=self._device
            )
            q_next[~mask] = -float("inf")
            q_next_max = (
                q_next[mask].max()
                if mask.any()
                else torch.tensor(0.0, device=self._device)
            )
            target = reward + self._gamma * q_next_max * (1.0 - float(done))

        loss = nn.functional.smooth_l1_loss(q_current, target)
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

    def _learn_q_omega(self) -> dict[str, float]:
        """Sample a batch and do one DQN gradient step on Q_Omega."""
        states, options, rewards, next_states, dones = self._buffer.sample(
            self._batch_size
        )
        states = states.to(self._device)
        options = options.to(self._device)
        rewards = rewards.to(self._device)
        next_states = next_states.to(self._device)
        dones = dones.to(self._device)

        # Current Q-values for the options that were actually taken
        q_all = self._q_net(states)
        q_taken = q_all.gather(1, options.unsqueeze(1)).squeeze(1)

        # Target: r + γ * max_o' Q_target(s', o') * (1 - done)
        with torch.no_grad():
            q_next = self._target_net(next_states)
            mask = torch.tensor(
                self._option_mask, dtype=torch.bool, device=self._device
            )
            q_next[:, ~mask] = -float("inf")
            if mask.any():
                q_next_max = q_next.max(dim=1).values
            else:
                q_next_max = torch.zeros(self._batch_size, device=self._device)
            targets = rewards + self._gamma * q_next_max * (1.0 - dones)

        loss = nn.functional.smooth_l1_loss(q_taken, targets)
        self._optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self._q_net.parameters(), 10.0)
        self._optimizer.step()

        # Return per-option TD errors for the batch (mean)
        td = (targets - q_taken).detach()
        return {"q_omega_td": td.mean().item()}

    def _resolve_option_idx(self, option_id: OptionId | None) -> int | None:
        """Resolve which option slot should receive credit for this transition."""
        if option_id is not None:
            return self.option_idx_for(option_id)

        if (
            0 <= self._last_option_idx < self._max_options
            and self._option_mask[self._last_option_idx]
        ):
            return self._last_option_idx

        return None

    def _learn_gvfs(
        self, transition: Transition[Any, torch.Tensor, dict[str, Any]]
    ) -> dict[str, float]:
        """Online GVF updates (one transition)."""
        if not self._gvf_heads or self._gvf_optim is None:
            return {}

        state = transition.subjective_state.to(self._device)
        next_state = transition.next_subjective_state.to(self._device)
        done = transition.terminated

        errors: dict[str, float] = {}
        total_loss = torch.tensor(0.0)

        for gvf_id, head in self._gvf_heads.items():
            pred = head(state).squeeze()
            with torch.no_grad():
                next_pred = head(next_state).squeeze()
            # Simple GVF: cumulant = reward, continuation = gamma * (1 - done)
            target = transition.reward + self._gamma * next_pred * (1.0 - float(done))
            td_err = (target - pred).item()
            total_loss = total_loss + nn.functional.smooth_l1_loss(pred, target)

            errors[f"gvf_{gvf_id}"] = td_err
            self._gvf_error_accum[gvf_id] = (
                self._gvf_error_accum.get(gvf_id, 0.0) + td_err
            )
            self._gvf_error_count[gvf_id] = self._gvf_error_count.get(gvf_id, 0) + 1

        if total_loss.requires_grad:
            self._gvf_optim.zero_grad()
            total_loss.backward()
            self._gvf_optim.step()

        return errors
