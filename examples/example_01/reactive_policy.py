"""Option-Critic reactive policy for OaK agents.

Each option maintains its own action-value Q-network Q_o(s, a) trained
via DQN.  A meta-policy selects which option to execute using Q_Omega
from the value function.  Termination conditions β_o are learned
alongside.

Options are created dynamically from ingested subtasks.
"""

from __future__ import annotations

import random
from collections import deque
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from oak.interfaces import ReactivePolicy
from oak.types import (
    GeneralValueFunctionId,
    OptionId,
    PlanningUpdate,
    SubtaskId,
    SubtaskSpec,
    Transition,
)

from .value_function import OptionValueFunction
from .schema import ExampleSubjectiveState, StateTensorAdapter


class _OptionNetworks(nn.Module):
    """Per-option action-value Q(s,a) network and termination β(s)."""

    def __init__(self, state_dim: int, num_actions: int, hidden: int = 64) -> None:
        super().__init__()
        self.q_net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_actions),
        )
        # Target Q-network for stable DQN learning
        self.q_target = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_actions),
        )
        self.q_target.load_state_dict(self.q_net.state_dict())

        termination_output = nn.Linear(hidden, 1)
        self.termination = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            termination_output,
            nn.Sigmoid(),
        )
        # Initialize termination bias low so options persist longer initially
        # sigmoid(-2) ≈ 0.12 → options rarely terminate at first
        with torch.no_grad():
            termination_output.bias.fill_(-2.0)

    def q_values(self, state: torch.Tensor) -> torch.Tensor:
        return self.q_net(state)

    def q_target_values(self, state: torch.Tensor) -> torch.Tensor:
        return self.q_target(state)

    def sync_target(self) -> None:
        self.q_target.load_state_dict(self.q_net.state_dict())

    def soft_update_target(self, tau: float) -> None:
        for tp, op in zip(self.q_target.parameters(), self.q_net.parameters()):
            tp.data.mul_(1.0 - tau).add_(op.data, alpha=tau)

    def stop_prob(self, state: torch.Tensor) -> float:
        with torch.no_grad():
            return self.termination(state).item()


class OptionCriticPolicy(ReactivePolicy[ExampleSubjectiveState, Any, dict[str, Any]]):
    """Option-Critic with DQN-style intra-option action learning.

    Each option has its own Q(s,a) network trained via DQN, providing
    stable action learning.  The meta-policy Q_Omega (in the value
    function) selects which option to execute.
    """

    def __init__(
        self,
        value_function: OptionValueFunction,
        num_actions: int,
        state_dim: int,
        *,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay_steps: int = 5_000,
        lr: float = 1e-3,
        gamma: float = 0.99,
        buffer_capacity: int = 5_000,
        batch_size: int = 64,
        device: torch.device | None = None,
        state_adapter: StateTensorAdapter | None = None,
    ) -> None:
        self._value_function = value_function
        self._num_actions = num_actions
        self._state_dim = state_dim
        self._device = device or torch.device("cpu")
        self._gamma = gamma
        self._batch_size = batch_size
        self._state_adapter = state_adapter or StateTensorAdapter()

        self._epsilon_start = epsilon_start
        self._epsilon_end = epsilon_end
        self._epsilon_decay_steps = epsilon_decay_steps

        # Option storage: option_id → (networks, slot_index, subtask_id)
        self._options: dict[OptionId, tuple[_OptionNetworks, int, SubtaskId]] = {}
        self._subtasks: dict[SubtaskId, SubtaskSpec] = {}

        self._active_option_id: OptionId | None = None

        # Shared replay buffer for all options
        # (state, action, reward, next_state, done, option_id)
        self._buffer: deque[
            tuple[torch.Tensor, int, float, torch.Tensor, bool, OptionId]
        ] = deque(maxlen=buffer_capacity)

        # Separate optimizers for Q-networks and termination networks
        # to prevent Adam momentum drift between the two
        self._lr = lr
        self._q_optimizer: torch.optim.Optimizer | None = None
        self._term_optimizer: torch.optim.Optimizer | None = None

        # Target network sync
        self._step_count = 0
        self._target_sync_freq = 200
        self._max_grad_norm = 10.0

    # ------------------------------------------------------------------
    # ReactivePolicy interface
    # ------------------------------------------------------------------

    def update(
        self,
        transition: Transition[Any, ExampleSubjectiveState, dict[str, Any]],
        td_errors: Mapping[GeneralValueFunctionId, float],
    ) -> None:
        if not self._options or self._q_optimizer is None:
            return
        if self._active_option_id is None:
            return

        state = self._state_adapter.tensor(transition.subjective_state)
        action = (
            int(transition.action) if isinstance(transition.action, (int, float)) else 0
        )
        next_state = self._state_adapter.tensor(transition.next_subjective_state)
        done = transition.terminated
        reward = transition.reward

        # Store in option replay buffer
        self._buffer.append(
            (
                state.detach(),
                action,
                reward,
                next_state.detach(),
                done,
                self._active_option_id,
            )
        )

        # DQN learning for each option from replay
        if len(self._buffer) >= self._batch_size:
            self._learn_option_q()

        # Termination gradient for the active option
        self._learn_termination(transition)

        # Periodically sync target networks
        self._step_count += 1
        if self._step_count % self._target_sync_freq == 0:
            for _, (nets, _, _) in self._options.items():
                nets.sync_target()

    def apply_planning_update(self, update: PlanningUpdate[Any]) -> None:
        pass

    def ingest_subtasks(self, subtasks: Sequence[SubtaskSpec]) -> None:
        for subtask in subtasks:
            if subtask.subtask_id in self._subtasks:
                continue
            self._subtasks[subtask.subtask_id] = subtask

            option_id = f"option:{subtask.subtask_id}"
            nets = _OptionNetworks(self._state_dim, self._num_actions).to(self._device)
            slot_idx = self._value_function.register_option(option_id)

            # Also add a GVF for this feature
            self._value_function.add_gvf(subtask.feature_id)

            self._options[option_id] = (nets, slot_idx, subtask.subtask_id)
            self._rebuild_optimizer()

    def integrate_options(self) -> None:
        pass  # options are registered immediately in ingest_subtasks

    def select_action(
        self,
        subjective_state: ExampleSubjectiveState,
        option_stop_threshold: float,
    ) -> tuple[Any, OptionId | None]:
        # If no options exist, return random action
        if not self._options:
            return random.randrange(self._num_actions), None

        s = self._state_adapter.tensor(subjective_state).to(self._device)

        # Check if active option should continue
        if self._active_option_id is not None:
            if self._active_option_id in self._options:
                nets, _, _ = self._options[self._active_option_id]
                stop_p = nets.stop_prob(s)
                if stop_p < option_stop_threshold:
                    # Continue option, select action via ε-greedy on Q_o
                    action = self._select_action_for_option(nets, s)
                    slot_idx = self._options[self._active_option_id][1]
                    self._value_function.set_last_option_idx(slot_idx)
                    return action, self._active_option_id
            # Terminate
            self._active_option_id = None

        # Select new option via Q_Omega (ε-greedy)
        option_id = self._select_option(subjective_state)
        self._active_option_id = option_id

        nets, slot_idx, _ = self._options[option_id]
        self._value_function.set_last_option_idx(slot_idx)

        action = self._select_action_for_option(nets, s)
        return action, option_id

    def clear_active_option(self) -> None:
        self._active_option_id = None

    def remove_options(self, option_ids: Sequence[OptionId]) -> None:
        for oid in option_ids:
            if oid in self._options:
                self._value_function.deregister_option(oid)
                del self._options[oid]
            if self._active_option_id == oid:
                self._active_option_id = None
        if self._options:
            self._rebuild_optimizer()
        else:
            self._q_optimizer = None
            self._term_optimizer = None

    def remove_subtasks(self, subtask_ids: Sequence[SubtaskId]) -> None:
        for sid in subtask_ids:
            self._subtasks.pop(sid, None)
            option_id = f"option:{sid}"
            if option_id in self._options:
                self.remove_options([option_id])

    # ------------------------------------------------------------------
    # Epsilon management
    # ------------------------------------------------------------------

    def decay_epsilon(self) -> None:
        pass  # epsilon is now step-based, computed from _step_count

    @property
    def epsilon(self) -> float:
        # Linear decay from start to end over decay_steps
        return max(
            self._epsilon_end,
            self._epsilon_start - self._step_count / self._epsilon_decay_steps,
        )

    # ------------------------------------------------------------------
    # Internal: action selection
    # ------------------------------------------------------------------

    def _select_option(self, state: ExampleSubjectiveState) -> OptionId:
        """Epsilon-greedy option selection using Q_Omega."""
        option_ids = list(self._options.keys())

        if random.random() < self.epsilon:
            return random.choice(option_ids)

        q_preds = self._value_function.predict(state)
        best_oid = option_ids[0]
        best_q = -float("inf")
        for oid in option_ids:
            q = q_preds.get(f"q_{oid}", 0.0)
            if q > best_q:
                best_q = q
                best_oid = oid
        return best_oid

    def _select_action_for_option(
        self, nets: _OptionNetworks, state: torch.Tensor
    ) -> int:
        """Epsilon-greedy action selection using the option's Q(s,a)."""
        if random.random() < self.epsilon:
            return random.randrange(self._num_actions)

        with torch.no_grad():
            q = nets.q_values(state)
        return int(q.argmax().item())

    # ------------------------------------------------------------------
    # Internal: learning
    # ------------------------------------------------------------------

    def _learn_option_q(self) -> None:
        """DQN update for all option Q-networks from shared replay.

        All options learn from the full batch (not grouped by which option
        generated the transition).  This is valid because DQN is off-policy:
        Q_o(s,a) converges to Q*(s,a) regardless of the behavior policy.
        """
        batch = random.sample(list(self._buffer), self._batch_size)

        states = torch.stack([it[0] for it in batch]).to(self._device)
        actions = torch.tensor(
            [it[1] for it in batch], dtype=torch.long, device=self._device
        )
        rewards = torch.tensor(
            [it[2] for it in batch], dtype=torch.float32, device=self._device
        )
        next_states = torch.stack([it[3] for it in batch]).to(self._device)
        dones = torch.tensor(
            [it[4] for it in batch], dtype=torch.float32, device=self._device
        )

        total_loss = torch.tensor(0.0, device=self._device)

        for oid in self._options:
            nets, _, _ = self._options[oid]

            # Current Q-values
            q_all = nets.q_values(states)
            q_taken = q_all.gather(1, actions.unsqueeze(1)).squeeze(1)

            # Target using target network for stability
            with torch.no_grad():
                q_next = nets.q_target_values(next_states)
                q_next_max = q_next.max(dim=1).values
                targets = rewards + self._gamma * q_next_max * (1.0 - dones)

            loss = F.smooth_l1_loss(q_taken, targets)
            total_loss = total_loss + loss

        if total_loss.requires_grad and self._q_optimizer is not None:
            self._q_optimizer.zero_grad()
            total_loss.backward()
            for oid in self._options:
                nets, _, _ = self._options[oid]
                nn.utils.clip_grad_norm_(nets.q_net.parameters(), self._max_grad_norm)
            self._q_optimizer.step()

    def _learn_termination(
        self, transition: Transition[Any, ExampleSubjectiveState, dict[str, Any]]
    ) -> None:
        """Update termination condition for the active option."""
        if (
            self._active_option_id is None
            or self._active_option_id not in self._options
        ):
            return
        if self._term_optimizer is None:
            return

        nets, _, _ = self._options[self._active_option_id]
        next_state = self._state_adapter.tensor(transition.next_subjective_state).to(self._device)

        # Advantage of terminating: V(s') - Q_Omega(s', current_option)
        q_preds = self._value_function.predict(transition.next_subjective_state)
        q_vals = [q_preds.get(f"q_{oid}", 0.0) for oid in self._options]
        q_current = q_preds.get(f"q_{self._active_option_id}", 0.0)
        v_next = max(q_vals) if q_vals else 0.0

        # If V(s') > Q(s', current_option), termination is beneficial
        term_advantage = v_next - q_current

        beta = nets.termination(next_state)
        term_loss = beta.squeeze() * term_advantage * 0.01

        if term_loss.requires_grad:
            self._term_optimizer.zero_grad()
            term_loss.backward()
            for _, (n, _, _) in self._options.items():
                nn.utils.clip_grad_norm_(n.termination.parameters(), 0.5)
            self._term_optimizer.step()

    # ------------------------------------------------------------------
    # Internal: parameter management
    # ------------------------------------------------------------------

    def _all_q_params(self) -> list[nn.Parameter]:
        params: list[nn.Parameter] = []
        for _, (nets, _, _) in self._options.items():
            params.extend(nets.q_net.parameters())
        return params

    def _all_term_params(self) -> list[nn.Parameter]:
        params: list[nn.Parameter] = []
        for _, (nets, _, _) in self._options.items():
            params.extend(nets.termination.parameters())
        return params

    def _rebuild_optimizer(self) -> None:
        q_params = self._all_q_params()
        term_params = self._all_term_params()
        self._q_optimizer = (
            torch.optim.Adam(q_params, lr=self._lr) if q_params else None
        )
        self._term_optimizer = (
            torch.optim.Adam(term_params, lr=self._lr) if term_params else None
        )
