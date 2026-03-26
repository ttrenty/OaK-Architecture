"""Targeted debugging of OaK components on CartPole.

Tests each component in isolation to find where learning fails.
"""

from __future__ import annotations

import random
import sys
from collections import deque
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.cartpole.reactive_policy import _OptionNetworks
from examples.cartpole.value_function import OptionValueFunction
from examples.cartpole.runner import build_agent

from oak_architecture import OaKAgent
from oak_architecture.types import Transition

# DQN hyperparameters (tuned for CartPole-v1)
HIDDEN = 64
LR = 1e-3
BATCH_SIZE = 64
BUFFER_SIZE = 5_000
GAMMA = 0.99
TARGET_SYNC_STEPS = 200
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY_STEPS = 5_000


def test_build_agent() -> bool:
    """Test that the agent builds successfully with the expected config."""
    print("=" * 60)
    print("TEST: build_agent() returns valid OaKAgent")
    print("=" * 60)

    config = {
        "obs_type": "numeric_vector",
        "obs_shape": (4,),
        "obs_dtype": "float32",
        "action_type": "discrete",
        "action_n": 2,
        "encoder_type": "mlp",
        "latent_dim": 64,
    }
    agent = build_agent(config)
    if not isinstance(agent, OaKAgent):
        print("FAIL: build_agent() did not return an OaKAgent")
        return False
    print("PASS: Agent build check OK")
    return True


# ── Test 1: Vanilla DQN baseline ──────────────────────────────────────
def test_vanilla_dqn(num_episodes: int = 500) -> bool:
    """Vanilla DQN on CartPole, establishes that DQN can solve it."""
    print("=" * 60)
    print("TEST: Vanilla DQN baseline (no OaK)")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = gym.make("CartPole-v1")

    q_net = nn.Sequential(
        nn.Linear(4, HIDDEN),
        nn.ReLU(),
        nn.Linear(HIDDEN, HIDDEN),
        nn.ReLU(),
        nn.Linear(HIDDEN, 2),
    ).to(device)
    target_net = nn.Sequential(
        nn.Linear(4, HIDDEN),
        nn.ReLU(),
        nn.Linear(HIDDEN, HIDDEN),
        nn.ReLU(),
        nn.Linear(HIDDEN, 2),
    ).to(device)
    target_net.load_state_dict(q_net.state_dict())
    optimizer = torch.optim.Adam(q_net.parameters(), lr=LR)

    buffer: deque = deque(maxlen=BUFFER_SIZE)
    recent: deque[float] = deque(maxlen=100)
    total_steps = 0

    for ep in range(num_episodes):
        obs, _ = env.reset()
        ep_reward = 0.0

        while True:
            # Step-based epsilon decay
            epsilon = max(EPS_END, EPS_START - total_steps / EPS_DECAY_STEPS)

            state = torch.tensor(obs, dtype=torch.float32, device=device)
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    action = q_net(state).argmax().item()

            next_obs, reward, term, trunc, _ = env.step(action)
            done = term or trunc
            buffer.append((obs, action, reward, next_obs, done))
            obs = next_obs
            ep_reward += reward
            total_steps += 1

            # DQN update
            if len(buffer) >= BATCH_SIZE:
                batch = random.sample(list(buffer), BATCH_SIZE)
                s = torch.tensor(
                    np.array([b[0] for b in batch]), dtype=torch.float32, device=device
                )
                a = torch.tensor([b[1] for b in batch], dtype=torch.long, device=device)
                r = torch.tensor(
                    [b[2] for b in batch], dtype=torch.float32, device=device
                )
                ns = torch.tensor(
                    np.array([b[3] for b in batch]), dtype=torch.float32, device=device
                )
                d = torch.tensor(
                    [b[4] for b in batch], dtype=torch.float32, device=device
                )

                q_vals = q_net(s).gather(1, a.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    q_next = target_net(ns).max(dim=1).values
                    targets = r + GAMMA * q_next * (1.0 - d)
                loss = F.smooth_l1_loss(q_vals, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Target sync
            if total_steps % TARGET_SYNC_STEPS == 0:
                target_net.load_state_dict(q_net.state_dict())

            if done:
                break

        recent.append(ep_reward)
        avg = sum(recent) / len(recent)

        if ep % 50 == 0:
            print(
                f"  Ep {ep:4d} | Reward: {ep_reward:6.1f} | Avg(100): {avg:6.1f} | Eps: {epsilon:.3f} | Steps: {total_steps}"
            )

        if len(recent) >= 100 and avg >= 475.0:
            print(f"  SOLVED at episode {ep}! Avg={avg:.1f}")
            env.close()
            return True

    env.close()
    final_avg = sum(list(recent)) / max(len(recent), 1)
    print(f"  Final avg: {final_avg:.1f}")
    ok = final_avg >= 200
    print(
        f"  {'PASS' if ok else 'FAIL'}: DQN baseline {'learns well' if ok else 'needs more tuning'}"
    )
    return ok


# ── Test 2: Single _OptionNetworks DQN ─────────────────────────────────
def test_single_option_q() -> bool:
    """Test _OptionNetworks Q-network (same hyperparameters as vanilla DQN)."""
    print("\n" + "=" * 60)
    print("TEST: _OptionNetworks DQN")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = gym.make("CartPole-v1")

    # Use larger hidden size to match baseline
    nets = _OptionNetworks(state_dim=4, num_actions=2, hidden=HIDDEN).to(device)
    optimizer = torch.optim.Adam(nets.q_net.parameters(), lr=LR)

    buffer: deque = deque(maxlen=BUFFER_SIZE)
    recent: deque[float] = deque(maxlen=100)
    total_steps = 0

    for ep in range(500):
        obs, _ = env.reset()
        state = torch.tensor(obs, dtype=torch.float32, device=device)
        ep_reward = 0.0

        while True:
            epsilon = max(EPS_END, EPS_START - total_steps / EPS_DECAY_STEPS)

            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    action = nets.q_values(state).argmax().item()

            next_obs, reward, term, trunc, _ = env.step(action)
            done = term or trunc
            next_state = torch.tensor(next_obs, dtype=torch.float32, device=device)
            buffer.append((state.cpu(), action, reward, next_state.cpu(), done))
            state = next_state
            ep_reward += reward
            total_steps += 1

            if len(buffer) >= BATCH_SIZE:
                batch = random.sample(list(buffer), BATCH_SIZE)
                s = torch.stack([b[0] for b in batch]).to(device)
                a = torch.tensor([b[1] for b in batch], dtype=torch.long, device=device)
                r = torch.tensor(
                    [b[2] for b in batch], dtype=torch.float32, device=device
                )
                ns = torch.stack([b[3] for b in batch]).to(device)
                d = torch.tensor(
                    [b[4] for b in batch], dtype=torch.float32, device=device
                )

                q_vals = nets.q_values(s).gather(1, a.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    q_next = nets.q_target_values(ns).max(dim=1).values
                    targets = r + GAMMA * q_next * (1.0 - d)
                loss = F.smooth_l1_loss(q_vals, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if total_steps % TARGET_SYNC_STEPS == 0:
                nets.sync_target()

            if done:
                break

        recent.append(ep_reward)
        avg = sum(recent) / len(recent)

        if ep % 50 == 0:
            print(
                f"  Ep {ep:4d} | Reward: {ep_reward:6.1f} | Avg(100): {avg:6.1f} | Eps: {epsilon:.3f}"
            )

        if len(recent) >= 100 and avg >= 475.0:
            print(f"  SOLVED at episode {ep}! Avg={avg:.1f}")
            env.close()
            return True

    env.close()
    final_avg = sum(list(recent)) / max(len(recent), 1)
    print(f"  Final avg: {final_avg:.1f}")
    ok = final_avg >= 200
    print(f"  {'PASS' if ok else 'FAIL'}: _OptionNetworks DQN")
    return ok


# ── Test 3: Q-value and termination sanity ─────────────────────────────
def test_sanity() -> bool:
    """Check initial Q-values, stop probability, and target network."""
    print("\n" + "=" * 60)
    print("TEST: Q-value and termination sanity")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nets = _OptionNetworks(state_dim=4, num_actions=2, hidden=HIDDEN).to(device)

    test_state = torch.randn(4, device=device)
    q_init = nets.q_values(test_state)
    print(f"  Initial Q-values: {q_init.detach().cpu().numpy()}")
    assert abs(q_init[0].item()) < 10, f"Q-values too large"

    stop_p = nets.stop_prob(test_state)
    print(f"  Initial stop_prob: {stop_p:.4f} (expect ~0.12)")
    assert stop_p < 0.3, f"stop_prob too high"

    q_target = nets.q_target_values(test_state)
    diff = (q_init - q_target).abs().max().item()
    print(f"  Q/target diff: {diff:.6f} (expect ~0)")
    assert diff < 1e-5

    print("  PASS")
    return True


# ── Test 4: ValueFunction update ───────────────────────────────────────
def test_value_function() -> bool:
    """Check ValueFunction produces valid TD errors and predictions."""
    print("\n" + "=" * 60)
    print("TEST 4: ValueFunction update + predict")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vf = OptionValueFunction(state_dim=4, max_options=8, device=device)

    idx = vf.register_option("opt_0")
    vf.set_last_option_idx(idx)
    vf.add_gvf("gvf_0")

    for i in range(200):
        tr = Transition(
            subjective_state=torch.randn(4),
            action=random.randint(0, 1),
            reward=1.0,
            next_subjective_state=torch.randn(4),
            terminated=(i == 199),
        )
        td = vf.update(tr)

    assert "advantage" in td
    preds = vf.predict(torch.randn(4))
    assert "q_opt_0" in preds
    q_val = preds["q_opt_0"]
    print(f"  Q_Omega value: {q_val:.4f}")
    print(f"  TD errors: {td}")
    print("  PASS")
    return True


# ── Test 5: Q-learning equivalence ────────────────────────────────────
def test_q_equivalence() -> bool:
    """Train vanilla DQN and _OptionNetworks on identical data, compare Q-values."""
    print("\n" + "=" * 60)
    print("TEST: Q-learning equivalence (same data, same init)")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build both networks with the same initial weights
    torch.manual_seed(42)
    vanilla_q = nn.Sequential(
        nn.Linear(4, HIDDEN),
        nn.ReLU(),
        nn.Linear(HIDDEN, HIDDEN),
        nn.ReLU(),
        nn.Linear(HIDDEN, 2),
    ).to(device)
    vanilla_target = nn.Sequential(
        nn.Linear(4, HIDDEN),
        nn.ReLU(),
        nn.Linear(HIDDEN, HIDDEN),
        nn.ReLU(),
        nn.Linear(HIDDEN, 2),
    ).to(device)
    vanilla_target.load_state_dict(vanilla_q.state_dict())
    vanilla_opt = torch.optim.Adam(vanilla_q.parameters(), lr=LR)

    torch.manual_seed(42)
    option_nets = _OptionNetworks(state_dim=4, num_actions=2, hidden=HIDDEN).to(device)
    option_opt = torch.optim.Adam(option_nets.q_net.parameters(), lr=LR)

    # Verify initial Q-values match
    test_state = torch.tensor([0.1, -0.2, 0.05, 0.3], device=device)
    v_q0 = vanilla_q(test_state).detach()
    o_q0 = option_nets.q_values(test_state).detach()
    init_diff = (v_q0 - o_q0).abs().max().item()
    print(f"  Initial Q-value diff: {init_diff:.6f}")
    assert init_diff < 1e-5, f"Initial Q-values differ: {init_diff}"

    # Generate fixed dataset of transitions
    random.seed(123)
    np.random.seed(123)
    env = gym.make("CartPole-v1")
    env.reset(seed=123)

    transitions = []
    obs, _ = env.reset()
    for _ in range(500):
        action = env.action_space.sample()
        next_obs, reward, term, trunc, _ = env.step(action)
        transitions.append((obs, action, reward, next_obs, term or trunc))
        if term or trunc:
            obs, _ = env.reset()
        else:
            obs = next_obs
    env.close()

    # Train both on the same batches
    buffer = list(transitions)
    for step in range(200):
        batch = random.sample(buffer, min(BATCH_SIZE, len(buffer)))
        s = torch.tensor(
            np.array([b[0] for b in batch]), dtype=torch.float32, device=device
        )
        a = torch.tensor([b[1] for b in batch], dtype=torch.long, device=device)
        r = torch.tensor([b[2] for b in batch], dtype=torch.float32, device=device)
        ns = torch.tensor(
            np.array([b[3] for b in batch]), dtype=torch.float32, device=device
        )
        d = torch.tensor([b[4] for b in batch], dtype=torch.float32, device=device)

        # Vanilla DQN update
        v_qvals = vanilla_q(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            v_qnext = vanilla_target(ns).max(dim=1).values
            v_targets = r + GAMMA * v_qnext * (1.0 - d)
        v_loss = F.smooth_l1_loss(v_qvals, v_targets)
        vanilla_opt.zero_grad()
        v_loss.backward()
        vanilla_opt.step()

        # _OptionNetworks update
        o_qvals = option_nets.q_values(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            o_qnext = option_nets.q_target_values(ns).max(dim=1).values
            o_targets = r + GAMMA * o_qnext * (1.0 - d)
        o_loss = F.smooth_l1_loss(o_qvals, o_targets)
        option_opt.zero_grad()
        o_loss.backward()
        option_opt.step()

        # Sync targets at the same schedule
        if (step + 1) % 50 == 0:
            vanilla_target.load_state_dict(vanilla_q.state_dict())
            option_nets.sync_target()

    # Compare final Q-values
    v_qf = vanilla_q(test_state).detach()
    o_qf = option_nets.q_values(test_state).detach()
    final_diff = (v_qf - o_qf).abs().max().item()
    print(f"  Final Q-value diff after 200 updates: {final_diff:.6f}")
    print(f"  Vanilla Q: {v_qf.cpu().numpy()}")
    print(f"  Option  Q: {o_qf.cpu().numpy()}")

    ok = final_diff < 0.1  # Should be very close if architectures are equivalent
    print(f"  {'PASS' if ok else 'FAIL'}: Q-values {'match' if ok else 'diverge'}")
    return ok


if __name__ == "__main__":
    results = {}
    results["sanity"] = test_sanity()
    results["value_function"] = test_value_function()
    results["q_equivalence"] = test_q_equivalence()
    results["vanilla_dqn"] = test_vanilla_dqn()
    if results["vanilla_dqn"]:
        results["option_q"] = test_single_option_q()
    else:
        print("\n  Skipping Test 2: vanilla DQN failed, fix baseline first")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, ok in results.items():
        print(f"  {name}: {'PASS' if ok else 'FAIL'}")
