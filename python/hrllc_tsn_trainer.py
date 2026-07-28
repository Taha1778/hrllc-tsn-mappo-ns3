import argparse
import csv
import filecmp
import json
import math
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "default_config.json"
WEIGHTS_DIR = PROJECT_ROOT / "weights"
RUNS_DIR = PROJECT_ROOT / "runs"
EXTERNAL_DIR = PROJECT_ROOT / "external"
NS3_DIR = EXTERNAL_DIR / "ns-3.44"
CURRENT_NPZ = WEIGHTS_DIR / "current_model.npz"
CURRENT_WEIGHTS = WEIGHTS_DIR / "current_model.weights"
SUMMARY_CSV = RUNS_DIR / "summary.csv"
VALIDATION_SUMMARY_CSV = RUNS_DIR / "validation_summary.csv"
FINAL_TEST_SUMMARY_CSV = RUNS_DIR / "final_test_summary.csv"
WORKFLOW_STATUS_JSON = RUNS_DIR / "workflow_status.json"


# Windows-side workflow utilities: status reporting, WSL path conversion, and
# checked subprocess execution. C++ simulation runs in WSL; training remains here.
def write_workflow_status(status, **details):
    """Persist a concise, machine-readable state without audible alerts."""
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": status,
        "updated_unix": time.time(),
        **details,
    }
    WORKFLOW_STATUS_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def to_wsl_path(path: Path) -> str:
    resolved = str(path.resolve())
    try:
        result = subprocess.run(
            ["wsl", "wslpath", "-a", resolved],
            text=True,
            capture_output=True,
            check=True,
        )
        converted = result.stdout.strip()
        if converted:
            return converted
    except Exception:
        pass
    drive = resolved[0].lower()
    rest = resolved[2:].replace("\\", "/")
    return f"/mnt/{drive}{rest}"


def run_checked(cmd, cwd=None):
    result = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(
            "Command failed:\n"
            + " ".join(cmd)
            + "\n\nSTDOUT:\n"
            + result.stdout
            + "\n\nSTDERR:\n"
            + result.stderr
        )
    return result


def load_config():
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


# Reproduction guard: paper-stated values are checked before a paper-aligned run.
def verify_paper_configuration(config):
    """Reject accidental drift in values explicitly stated by the paper."""
    if not config.get("enforce_paper_configuration", True):
        return
    expected = config["paper_fixed_values"]
    mismatches = [
        f"{key}={config.get(key)!r} (expected {value!r})"
        for key, value in expected.items()
        if config.get(key) != value
    ]
    if mismatches:
        raise ValueError("Paper-configuration drift detected: " + "; ".join(mismatches))


class ActorNetwork(nn.Module):
    """Decentralized policy with categorical power and HARQ action heads."""

    def __init__(self, obs_dim, hidden_dim, q_max, n_action_dim):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.power_head = nn.Linear(hidden_dim, q_max)
        self.retransmission_head = nn.Linear(hidden_dim, n_action_dim)
        for layer in (self.fc1, self.fc2, self.power_head, self.retransmission_head):
            nn.init.orthogonal_(layer.weight, gain=math.sqrt(2.0))
            nn.init.constant_(layer.bias, 0.0)

    def forward(self, obs, valid_n_count):
        hidden = F.relu(self.fc1(obs))
        hidden = F.relu(self.fc2(hidden))
        power_logits = self.power_head(hidden)
        retransmission_logits = self.retransmission_head(hidden)
        if valid_n_count < retransmission_logits.shape[-1]:
            retransmission_logits = retransmission_logits.clone()
            retransmission_logits[:, valid_n_count:] = -1e9
        return power_logits, retransmission_logits


class CriticNetwork(nn.Module):
    """Centralized value network used only during training."""

    def __init__(self, state_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.value_head = nn.Linear(hidden_dim, 1)
        for layer in (self.fc1, self.fc2, self.value_head):
            nn.init.orthogonal_(layer.weight, gain=math.sqrt(2.0))
            nn.init.constant_(layer.bias, 0.0)

    def forward(self, state):
        hidden = F.relu(self.fc1(state))
        hidden = F.relu(self.fc2(hidden))
        return self.value_head(hidden).squeeze(-1)


# Atomic publication prevents brief OneDrive locks from corrupting checkpoints.
def replace_with_retry(temporary_path, destination_path, attempts=120, delay_seconds=0.5):
    """Atomically publish an artifact, tolerating short OneDrive/indexer locks."""
    last_error = None
    for attempt in range(attempts):
        try:
            os.replace(temporary_path, destination_path)
            return
        except PermissionError as exc:
            last_error = exc
            if attempt + 1 < attempts:
                time.sleep(delay_seconds)
    raise PermissionError(
        f"Could not replace {destination_path} after {attempts * delay_seconds:.0f} seconds; "
        "close any application holding the file and retry."
    ) from last_error


def temporary_artifact_path(destination_path):
    return destination_path.with_name(
        f".{destination_path.stem}.{os.getpid()}.{time.time_ns()}.tmp{destination_path.suffix}"
    )


# Build a fresh model from configuration. Actors receive local observations;
# the critic receives the joint state collected from every device.
def create_model(config):
    torch.manual_seed(int(config["seed"]))
    k = int(config["k"])
    obs_dim = k + 2
    state_dim = 3 * k
    hidden_dim = int(config["hidden_dim"])
    q_max = int(config["q_max"])
    actor_n_action_dims = []
    for agent in range(k):
        latency_ms = float(
            config["flow_type_1_latency_ms"] if agent < k // 2 else config["flow_type_2_latency_ms"]
        )
        actor_n_action_dims.append(int(math.floor(latency_ms / config["retransmission_ms"])) + 1)
    n_action_dim = max(actor_n_action_dims)
    actors = [
        ActorNetwork(obs_dim, hidden_dim, q_max, actor_n_action_dims[agent]) for agent in range(k)
    ]
    critic = CriticNetwork(state_dim, hidden_dim)
    model = {
        "k": k,
        "obs_dim": obs_dim,
        "state_dim": state_dim,
        "hidden_dim": hidden_dim,
        "q_max": q_max,
        "n_action_dim": n_action_dim,
        "actor_n_action_dims": actor_n_action_dims,
        "actors": actors,
        "critic": critic,
    }
    model["actor_optimizers"] = [
        torch.optim.Adam(actor.parameters(), lr=float(config["actor_learning_rate"])) for actor in actors
    ]
    model["critic_optimizer"] = torch.optim.Adam(
        critic.parameters(), lr=float(config["critic_learning_rate"])
    )
    return model


def apply_preset_policy(model, config, preset):
    raise ValueError("Preset policies are retired for the Python-compatibility campaign.")


# The checkpoint retains PyTorch parameters and Adam state. The text export
# contains actor parameters only, which C++ ns-3 loads for policy inference.
def save_model_npz(model, path):
    temporary_path = temporary_artifact_path(path)
    try:
        torch.save(
            {
                "format": "torch_mappo_v2",
                "meta": [model[key] for key in ("k", "obs_dim", "state_dim", "hidden_dim", "q_max", "n_action_dim")],
                "actor_n_action_dims": model["actor_n_action_dims"],
                "actors": [actor.state_dict() for actor in model["actors"]],
                "critic": model["critic"].state_dict(),
                "actor_optimizers": [optimizer.state_dict() for optimizer in model["actor_optimizers"]],
                "critic_optimizer": model["critic_optimizer"].state_dict(),
            },
            temporary_path,
        )
        replace_with_retry(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink(missing_ok=True)


def load_model_npz(path):
    data = torch.load(path, map_location="cpu", weights_only=True)
    if data.get("format") != "torch_mappo_v2":
        raise ValueError("Checkpoint predates actor-specific retransmission heads; start a fresh campaign.")
    meta = data["meta"]
    model = {
        "k": meta[0],
        "obs_dim": meta[1],
        "state_dim": meta[2],
        "hidden_dim": meta[3],
        "q_max": meta[4],
        "n_action_dim": meta[5],
        "actor_n_action_dims": list(data["actor_n_action_dims"]),
        "actors": [
            ActorNetwork(meta[1], meta[3], meta[4], n_action_dim)
            for n_action_dim in data["actor_n_action_dims"]
        ],
        "critic": CriticNetwork(meta[2], meta[3]),
    }
    for actor, state_dict in zip(model["actors"], data["actors"]):
        actor.load_state_dict(state_dict)
    model["critic"].load_state_dict(data["critic"])
    model["actor_optimizers"] = [torch.optim.Adam(actor.parameters()) for actor in model["actors"]]
    for optimizer, state_dict in zip(model["actor_optimizers"], data["actor_optimizers"]):
        optimizer.load_state_dict(state_dict)
    model["critic_optimizer"] = torch.optim.Adam(model["critic"].parameters())
    model["critic_optimizer"].load_state_dict(data["critic_optimizer"])
    return model


def write_vector(f, label, values):
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    f.write(f"{label} {flat.size}\n")
    f.write(" ".join(f"{x:.17g}" for x in flat))
    f.write("\n")


def write_matrix(f, label, matrix):
    matrix = np.asarray(matrix, dtype=np.float64)
    f.write(f"{label} {matrix.shape[0]} {matrix.shape[1]}\n")
    f.write(" ".join(f"{x:.17g}" for x in matrix.reshape(-1)))
    f.write("\n")


def write_cpp_weights(model, path):
    temporary_path = temporary_artifact_path(path)
    try:
        with temporary_path.open("w", encoding="utf-8", newline="\n") as f:
            f.write("HRLLC_TSN_MAPPO_WEIGHTS_V1\n")
            f.write(f"K {model['k']}\n")
            f.write(f"OBS_DIM {model['obs_dim']}\n")
            f.write(f"STATE_DIM {model['state_dim']}\n")
            f.write(f"HIDDEN_DIM {model['hidden_dim']}\n")
            f.write(f"Q_MAX {model['q_max']}\n")
            f.write(f"N_ACTION_DIM {model['n_action_dim']}\n")
            for i, actor in enumerate(model["actors"]):
                f.write(f"ACTOR {i}\n")
                write_matrix(f, "W1", actor.fc1.weight.detach().cpu().numpy())
                write_vector(f, "B1", actor.fc1.bias.detach().cpu().numpy())
                write_matrix(f, "W2", actor.fc2.weight.detach().cpu().numpy())
                write_vector(f, "B2", actor.fc2.bias.detach().cpu().numpy())
                write_matrix(f, "WQ", actor.power_head.weight.detach().cpu().numpy())
                write_vector(f, "BQ", actor.power_head.bias.detach().cpu().numpy())
                write_matrix(f, "WN", actor.retransmission_head.weight.detach().cpu().numpy())
                write_vector(f, "BN", actor.retransmission_head.bias.detach().cpu().numpy())
        replace_with_retry(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink(missing_ok=True)


def ensure_model(config, reset=False):
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    recreated = False
    if reset or not CURRENT_NPZ.exists():
        model = create_model(config)
        save_model_npz(model, CURRENT_NPZ)
        recreated = True
    else:
        try:
            model = load_model_npz(CURRENT_NPZ)
        except ValueError:
            model = create_model(config)
            save_model_npz(model, CURRENT_NPZ)
            recreated = True
        expected = (int(config["k"]), int(config["k"]) + 2, 3 * int(config["k"]))
        actual = (model["k"], model["obs_dim"], model["state_dim"])
        if actual != expected:
            model = create_model(config)
            save_model_npz(model, CURRENT_NPZ)
            recreated = True
    write_cpp_weights(model, CURRENT_WEIGHTS)
    return model, recreated


# Forward-pass helpers mirror the C++ actor. The retry head masks actions that
# cannot meet the selected latency-feasibility rule before applying softmax.
def softmax(logits):
    logits = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / np.sum(exp, axis=1, keepdims=True)


def valid_retransmission_count(agent, config, n_action_dim):
    type_one = agent < int(config["k"]) // 2
    latency = float(config["flow_type_1_latency_ms"] if type_one else config["flow_type_2_latency_ms"])
    payload_bytes = float(config["flow_type_1_bytes"] if type_one else config["flow_type_2_bytes"])
    budget = max(0.0, latency - float(config.get("retransmission_guard_ms", 0.0)))
    # Equation (25) is the paper's explicit action-space definition.
    max_n = max(0, int(math.floor(budget / float(config["retransmission_ms"]) + float(config["probability_floor"]))))
    if config.get("conservative_retry_feasibility_mask", False):
        tsn_bits_per_ms = float(config["tsn_rate_mbps"]) * 1000.0
        type1_count = (int(config["k"]) + 1) // 2
        type2_count = int(config["k"]) // 2
        worst_tsn_arrival_offset = (
            type1_count * float(config["flow_type_1_bytes"]) * 8.0
            + type2_count * float(config["flow_type_2_bytes"]) * 8.0
        ) / tsn_bits_per_ms
        feasible_budget = max(
            0.0,
            latency - payload_bytes * 8.0 / tsn_bits_per_ms
            - worst_tsn_arrival_offset - float(config["mini_slot_ms"]),
        )
        max_n = min(max_n, int(math.floor(feasible_budget / float(config["retransmission_ms"]) + float(config["probability_floor"]))))
    return max(1, min(max_n + 1, n_action_dim))


def masked_softmax(logits, valid_count, mask_logit):
    masked = logits.copy()
    if valid_count < masked.shape[1]:
        masked[:, valid_count:] = mask_logit
    return softmax(masked)


def actor_forward(actor, obs, valid_n_count=None, mask_logit=-1e100):
    z1 = obs @ actor["w1"].T + actor["b1"]
    h1 = np.tanh(z1)
    z2 = h1 @ actor["w2"].T + actor["b2"]
    h2 = np.tanh(z2)
    q_probs = softmax(h2 @ actor["wq"].T + actor["bq"])
    n_logits = h2 @ actor["wn"].T + actor["bn"]
    n_probs = masked_softmax(n_logits, valid_n_count, mask_logit) if valid_n_count is not None else softmax(n_logits)
    cache = {"obs": obs, "h1": h1, "h2": h2, "q_probs": q_probs, "n_probs": n_probs}
    return q_probs, n_probs, cache


def critic_forward(critic, state):
    z1 = state @ critic["w1"].T + critic["b1"]
    h1 = np.tanh(z1)
    z2 = h1 @ critic["w2"].T + critic["b2"]
    h2 = np.tanh(z2)
    value = (h2 @ critic["wv"].T + critic["bv"]).reshape(-1)
    cache = {"state": state, "h1": h1, "h2": h2}
    return value, cache


def log_prob_from_probs(probs, actions, probability_floor):
    return np.log(np.clip(probs[np.arange(actions.size), actions], probability_floor, 1.0))


# Translate the per-agent CSV rows emitted by ns-3 into centralized states,
# decentralized observations, sampled actions, and common rewards for PPO.
def reconstruct_training_arrays(df, k):
    lambda_cols = [f"lambda_norm_{i}" for i in range(k)]
    power_cols = [f"prev_power_norm_{i}" for i in range(k)]
    n_cols = [f"prev_n_norm_{i}" for i in range(k)]
    first_by_step = df.sort_values(["step", "agent"]).groupby("step", sort=True).first().reset_index()
    states = first_by_step[lambda_cols + power_cols + n_cols].to_numpy(dtype=np.float64)
    rewards = first_by_step["reward"].to_numpy(dtype=np.float64)

    per_agent = []
    for agent in range(k):
        part = df[df["agent"] == agent].sort_values("step").copy()
        obs = np.concatenate(
            [
                part[lambda_cols].to_numpy(dtype=np.float64),
                part[[f"prev_power_norm_{agent}", f"prev_n_norm_{agent}"]].to_numpy(dtype=np.float64),
            ],
            axis=1,
        )
        per_agent.append(
            {
                "obs": obs,
                "q_actions": part["action_q_index"].to_numpy(dtype=np.int64),
                "n_actions": part["action_n_index"].to_numpy(dtype=np.int64),
                "q_logp": part["action_q_logp"].to_numpy(dtype=np.float64),
                "n_logp": part["action_n_logp"].to_numpy(dtype=np.float64),
            }
        )
    return states, rewards, per_agent


def compute_gae(rewards, values, gamma, gae_lambda, advantage_std_floor):
    next_values = np.concatenate([values[1:], np.array([0.0], dtype=np.float64)])
    deltas = rewards + gamma * next_values - values
    adv = np.zeros_like(rewards)
    running = 0.0
    for t in reversed(range(rewards.size)):
        running = deltas[t] + gamma * gae_lambda * running
        adv[t] = running
    returns = adv + values
    if adv.std() > advantage_std_floor:
        norm_adv = (adv - adv.mean()) / (adv.std() + advantage_std_floor)
    else:
        norm_adv = adv
    return norm_adv, returns


# The critic learns state values with Huber loss; each actor then learns its
# power and retransmission policy using the clipped PPO objective.
def update_critic(critic, optimizer, states, returns, lr, huber_delta, epochs, config):
    sample_count = states.shape[0]
    minibatch_size = min(max(1, int(config["ppo_minibatch_size"])), sample_count)
    rng = np.random.default_rng(int(config["seed"]) + optimizer["step"])
    for _ in range(epochs):
        for indices in np.array_split(rng.permutation(sample_count), math.ceil(sample_count / minibatch_size)):
            batch = max(1, indices.size)
            values, cache = critic_forward(critic, states[indices])
            error = returns[indices] - values
            grad_error = np.where(np.abs(error) <= huber_delta, error, huber_delta * np.sign(error))
            dvalue = -grad_error[:, None] / batch

            grad_wv = dvalue.T @ cache["h2"]
            grad_bv = dvalue.sum(axis=0)
            dh2 = dvalue @ critic["wv"]
            dz2 = dh2 * (1.0 - cache["h2"] ** 2)
            grad_w2 = dz2.T @ cache["h1"]
            grad_b2 = dz2.sum(axis=0)
            dh1 = dz2 @ critic["w2"]
            dz1 = dh1 * (1.0 - cache["h1"] ** 2)
            grad_w1 = dz1.T @ cache["state"]
            grad_b1 = dz1.sum(axis=0)

            apply_adam(
                critic,
                {
                    "wv": grad_wv,
                    "bv": grad_bv,
                    "w2": grad_w2,
                    "b2": grad_b2,
                    "w1": grad_w1,
                    "b1": grad_b1,
                },
                optimizer,
                lr,
                float(config["adam_beta1"]),
                float(config["adam_beta2"]),
                float(config["adam_epsilon"]),
                float(config["max_grad_norm"]),
            )


def ppo_active_mask(ratio, advantage, clip_eps):
    return np.where(
        advantage >= 0.0,
        ratio <= 1.0 + clip_eps,
        ratio >= 1.0 - clip_eps,
    )


def update_actor(
    actor,
    optimizer,
    obs,
    q_actions,
    n_actions,
    advantages,
    old_q_logp,
    old_n_logp,
    lr,
    clip_eps,
    epochs,
    valid_n_count,
    config,
):
    sample_count = obs.shape[0]
    minibatch_size = min(max(1, int(config["ppo_minibatch_size"])), sample_count)
    rng = np.random.default_rng(int(config["seed"]) + optimizer["step"])
    for _ in range(epochs):
        for indices in np.array_split(rng.permutation(sample_count), math.ceil(sample_count / minibatch_size)):
            batch = max(1, indices.size)
            q_probs, n_probs, cache = actor_forward(
                actor, obs[indices], valid_n_count, float(config["softmax_mask_logit"])
            )
            q_logp = log_prob_from_probs(q_probs, q_actions[indices], float(config["probability_floor"]))
            n_logp = log_prob_from_probs(n_probs, n_actions[indices], float(config["probability_floor"]))
            q_ratio = np.exp(q_logp - old_q_logp[indices])
            n_ratio = np.exp(n_logp - old_n_logp[indices])

            q_factor = np.where(
                ppo_active_mask(q_ratio, advantages[indices], clip_eps), advantages[indices] * q_ratio, 0.0
            )
            n_factor = np.where(
                ppo_active_mask(n_ratio, advantages[indices], clip_eps), advantages[indices] * n_ratio, 0.0
            )

            dq_logits = q_probs.copy()
            dq_logits[np.arange(batch), q_actions[indices]] -= 1.0
            dq_logits *= q_factor[:, None] / batch

            dn_logits = n_probs.copy()
            dn_logits[np.arange(batch), n_actions[indices]] -= 1.0
            dn_logits *= n_factor[:, None] / batch
            if valid_n_count < dn_logits.shape[1]:
                dn_logits[:, valid_n_count:] = 0.0

            grad_wq = dq_logits.T @ cache["h2"]
            grad_bq = dq_logits.sum(axis=0)
            grad_wn = dn_logits.T @ cache["h2"]
            grad_bn = dn_logits.sum(axis=0)
            dh2 = dq_logits @ actor["wq"] + dn_logits @ actor["wn"]
            dz2 = dh2 * (1.0 - cache["h2"] ** 2)
            grad_w2 = dz2.T @ cache["h1"]
            grad_b2 = dz2.sum(axis=0)
            dh1 = dz2 @ actor["w2"]
            dz1 = dh1 * (1.0 - cache["h1"] ** 2)
            grad_w1 = dz1.T @ cache["obs"]
            grad_b1 = dz1.sum(axis=0)

            apply_adam(
                actor,
                {
                    "wq": grad_wq,
                    "bq": grad_bq,
                    "wn": grad_wn,
                    "bn": grad_bn,
                    "w2": grad_w2,
                    "b2": grad_b2,
                    "w1": grad_w1,
                    "b1": grad_b1,
                },
                optimizer,
                lr,
                float(config["adam_beta1"]),
                float(config["adam_beta2"]),
                float(config["adam_epsilon"]),
                float(config["max_grad_norm"]),
            )


# Read result metrics from a simulation trajectory, then update the complete
# MAPPO model using that trajectory before the next simulation round starts.
def metrics_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    return {
        "reward_mean": float(df.groupby("step")["reward"].first().mean()),
        "failure_rate": 1.0 - float(df["successes"].sum() / max(1, df["frames"].sum())),
        "latency_violation_rate": float(df["latency_violations"].sum() / max(1, df["frames"].sum())),
        "reliability_violation_rate": float(df["reliability_violations"].sum() / max(1, df["frames"].sum())),
        "avg_e2e_ms": float(np.average(df["avg_e2e_ms"], weights=np.maximum(df["frames"], 1))),
        "avg_lw_ms": float(np.average(df["avg_lw_ms"], weights=np.maximum(df["frames"], 1))),
        "avg_failure_probability": float(np.average(df["avg_failure_probability"], weights=np.maximum(df["frames"], 1))),
    }


def update_model_from_csv(model, csv_path, config):
    df = pd.read_csv(csv_path)
    states, rewards, per_agent = reconstruct_training_arrays(df, model["k"])
    old_values, _ = critic_forward(model["critic"], states)
    advantages, _ = compute_gae(
        rewards,
        old_values,
        float(config["gamma"]),
        float(config["gae_lambda"]),
        float(config["advantage_std_floor"]),
    )
    td_targets = rewards + float(config["gamma"]) * np.concatenate(
        [old_values[1:], np.array([0.0], dtype=np.float64)]
    )

    old_logps = []
    for agent, item in enumerate(per_agent):
        valid_n = valid_retransmission_count(agent, config, model["n_action_dim"])
        q_probs, n_probs, _ = actor_forward(
            model["actors"][agent], item["obs"], valid_n, float(config["softmax_mask_logit"])
        )
        old_logps.append(
            (
                log_prob_from_probs(q_probs, item["q_actions"], float(config["probability_floor"])),
                log_prob_from_probs(n_probs, item["n_actions"], float(config["probability_floor"])),
            )
        )

    epochs = int(config["updates_per_round"])
    update_critic(
        model["critic"],
        model["optimizer"]["critic"],
        states,
        td_targets,
        float(config["critic_learning_rate"]),
        float(config["huber_delta"]),
        epochs,
        config,
    )
    for agent, item in enumerate(per_agent):
        valid_n = valid_retransmission_count(agent, config, model["n_action_dim"])
        update_actor(
            model["actors"][agent],
            model["optimizer"]["actors"][agent],
            item["obs"],
            item["q_actions"],
            item["n_actions"],
            advantages,
            old_logps[agent][0],
            old_logps[agent][1],
            float(config["actor_learning_rate"]),
            float(config["ppo_clip"]),
            epochs,
            valid_n,
            config,
        )

    return metrics_from_csv(csv_path)


# PyTorch PPO replacement for the earlier hand-written NumPy gradient code.
# The simulator remains external: it supplies one 256-step trajectory as CSV.
def update_model_from_csv(model, csv_path, config):
    df = pd.read_csv(csv_path)
    states_np, rewards_np, per_agent = reconstruct_training_arrays(df, model["k"])
    states = torch.as_tensor(states_np, dtype=torch.float32)
    rewards = torch.as_tensor(rewards_np, dtype=torch.float32)

    with torch.no_grad():
        old_values = model["critic"](states)
    advantages_np, returns_np = compute_gae(
        rewards_np,
        old_values.cpu().numpy(),
        float(config["gamma"]),
        float(config["gae_lambda"]),
        float(config["advantage_std_floor"]),
    )
    advantages = torch.as_tensor(advantages_np, dtype=torch.float32)
    returns = torch.as_tensor(returns_np, dtype=torch.float32)

    actor_batches = []
    with torch.no_grad():
        for agent, item in enumerate(per_agent):
            obs = torch.as_tensor(item["obs"], dtype=torch.float32)
            q_actions = torch.as_tensor(item["q_actions"], dtype=torch.long)
            n_actions = torch.as_tensor(item["n_actions"], dtype=torch.long)
            actor_batches.append(
                (
                    obs,
                    q_actions,
                    n_actions,
                    torch.as_tensor(item["q_logp"], dtype=torch.float32),
                    torch.as_tensor(item["n_logp"], dtype=torch.float32),
                    model["actor_n_action_dims"][agent],
                )
            )

    clip_eps = float(config["ppo_clip"])
    max_grad_norm = float(config["max_grad_norm"])
    for _ in range(int(config["updates_per_round"])):
        for actor, optimizer, batch in zip(model["actors"], model["actor_optimizers"], actor_batches):
            obs, q_actions, n_actions, old_q_logp, old_n_logp, valid_n = batch
            q_logits, n_logits = actor(obs, valid_n)
            q_ratio = torch.exp(Categorical(logits=q_logits).log_prob(q_actions) - old_q_logp)
            n_ratio = torch.exp(Categorical(logits=n_logits).log_prob(n_actions) - old_n_logp)
            q_loss = -torch.minimum(q_ratio * advantages, torch.clamp(q_ratio, 1 - clip_eps, 1 + clip_eps) * advantages).mean()
            n_loss = -torch.minimum(n_ratio * advantages, torch.clamp(n_ratio, 1 - clip_eps, 1 + clip_eps) * advantages).mean()
            optimizer.zero_grad()
            (q_loss + n_loss).backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), max_grad_norm)
            optimizer.step()

        critic_optimizer = model["critic_optimizer"]
        critic_optimizer.zero_grad()
        critic_loss = F.huber_loss(model["critic"](states), returns, delta=float(config["huber_delta"]))
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(model["critic"].parameters(), max_grad_norm)
        critic_optimizer.step()

    return metrics_from_csv(csv_path)


# Result lifecycle: persist round metrics, choose the best validation checkpoint,
# and record the exact configuration and seed ranges used by a campaign.
def append_summary(round_index, csv_path, metrics):
    global SUMMARY_CSV
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "round",
        "csv",
        "reward_mean",
        "failure_rate",
        "latency_violation_rate",
        "reliability_violation_rate",
        "avg_e2e_ms",
        "avg_lw_ms",
        "avg_failure_probability",
    ]
    row = {"round": round_index, "csv": str(csv_path)}
    row.update(metrics)
    for attempt in range(2):
        exists = SUMMARY_CSV.exists()
        try:
            with SUMMARY_CSV.open("a", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if not exists:
                    writer.writeheader()
                writer.writerow(row)
            return
        except PermissionError:
            if attempt == 0:
                SUMMARY_CSV = RUNS_DIR / f"summary_{int(time.time())}_{os.getpid()}.csv"
            else:
                raise


def save_best_if_needed(model, metrics):
    best_meta = WEIGHTS_DIR / "best_metrics.json"
    current = float(metrics["failure_rate"])
    previous = None
    if best_meta.exists():
        try:
            previous = json.loads(best_meta.read_text(encoding="utf-8")).get("failure_rate")
        except Exception:
            previous = None
    if previous is None or current < float(previous):
        save_model_npz(model, WEIGHTS_DIR / "best_model.npz")
        write_cpp_weights(model, WEIGHTS_DIR / "best_model.weights")
        best_meta.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def mean_metrics(metrics_list):
    if not metrics_list:
        raise ValueError("At least one validation result is required")
    keys = metrics_list[0].keys()
    return {key: float(np.mean([metrics[key] for metrics in metrics_list])) for key in keys}


def append_evaluation_summary(path, round_index, metrics, seed_count):
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        fieldnames = ["round", "seed_count", *metrics.keys()]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow({"round": round_index, "seed_count": seed_count, **metrics})


def append_validation_summary(round_index, metrics, seed_count):
    append_evaluation_summary(VALIDATION_SUMMARY_CSV, round_index, metrics, seed_count)


def write_campaign_manifest(config, args):
    paper_enforced = bool(config.get("enforce_paper_configuration", True))
    manifest = {
        "purpose": (
            "Fresh MAPPO-H training campaign aligned with the implementation paper"
            if paper_enforced
            else "Extension campaign with paper configuration enforcement disabled"
        ),
        "training": {
            "episodes_requested": args.rounds,
            "trajectory_steps": args.iterations,
            "updates_per_trajectory": config["updates_per_round"],
            "training_seed_start": args.seed,
        },
        "model_selection": {
            "deterministic": config["evaluation_action_mode"] == "argmax",
            "validation_every_rounds": args.validation_every,
            "validation_seed_start": args.validation_seed_start,
            "validation_seed_count": args.validation_seeds,
        },
        "final_test": {
            "deterministic": config["evaluation_action_mode"] == "argmax",
            "seed_start": args.final_test_seed_start,
            "seed_count": args.final_test_seeds,
            "separate_from_training_and_validation": True,
        },
        "config": config,
        "paper_alignment": {
            "paper_configuration_enforced": paper_enforced,
            "actor_learning_rate": "1e-4",
            "critic_learning_rate": "5e-3",
            "reward": "Equations (27)-(29): penalties only for latency and reliability violations",
            "initial_action_prior": "Random network weights with balanced-power, latency-valid action biases only",
            "shadow_fading": config["shadow_fading_mode"],
            "evaluation_action_mode": config["evaluation_action_mode"],
            "retransmission_limit": "Equation (25) action set, with the documented conservative L_W feasibility mask",
        },
    }
    (RUNS_DIR / "campaign_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


# Fresh starts remove only workflow-generated artifacts, never source files.
def remove_generated_artifacts():
    """Delete only files produced by this training workflow, never source files."""
    run_patterns = [
        "round_*.csv",
        "validation_round_*.csv",
        "summary*.csv",
        "validation_summary*.csv",
        "final_test_summary*.csv",
        "final_test_round_*.csv",
        "campaign_manifest.json",
        "best_result_summary.*",
        "paper_comparison_easy.csv",
    ]
    weight_patterns = [
        "current_model.npz",
        "current_model.weights",
        "best_model.npz",
        "best_model.weights",
        "best_metrics.json",
        "model_after_round_*.npz",
    ]
    for pattern in run_patterns:
        for path in RUNS_DIR.glob(pattern):
            if path.is_file():
                path.unlink()
    for pattern in weight_patterns:
        for path in WEIGHTS_DIR.glob(pattern):
            if path.is_file():
                path.unlink()


def reset_summary_file():
    global SUMMARY_CSV
    if not SUMMARY_CSV.exists():
        return
    for _ in range(12):
        try:
            SUMMARY_CSV.unlink()
            return
        except PermissionError:
            time.sleep(0.5)
    SUMMARY_CSV = RUNS_DIR / f"summary_{int(time.time())}_{os.getpid()}.csv"


def available_csv_path(round_index, prefix="round", seed=None):
    name = f"{prefix}_{round_index:04d}"
    if seed is not None:
        name += f"_seed_{seed:05d}"
    base = RUNS_DIR / f"{name}.csv"
    if not base.exists():
        return base
    for _ in range(8):
        try:
            base.unlink()
            return base
        except PermissionError:
            time.sleep(0.25)
    stamp = int(time.time())
    return RUNS_DIR / f"{name}_{stamp}_{os.getpid()}.csv"


# WSL/ns-3 bridge: synchronize changed C++ source, build it if necessary, and
# pass weights plus every simulation parameter to one C++ trajectory execution.
def ensure_ns3_ready():
    ns3_runner = NS3_DIR / "ns3"
    scratch_file = NS3_DIR / "scratch" / "hrllc_tsn_mappo.cc"
    if not ns3_runner.exists() or not scratch_file.exists():
        raise RuntimeError(
            "ns-3 is not ready. Run scripts/setup_ns3_wsl.ps1 first from PowerShell."
        )
    project_source = PROJECT_ROOT / "ns3" / "scratch" / "hrllc_tsn_mappo.cc"
    if not filecmp.cmp(project_source, scratch_file, shallow=False):
        shutil.copy2(project_source, scratch_file)
        cmake_candidates = sorted(EXTERNAL_DIR.glob("cmake-*-linux-x86_64/bin"))
        if cmake_candidates:
            cmake_bin_wsl = to_wsl_path(cmake_candidates[-1])
            path_prefix = f"export PATH={shlex.quote(cmake_bin_wsl + ':/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin')}; "
        else:
            path_prefix = ""
        ns3_dir_wsl = to_wsl_path(NS3_DIR)
        command = f"{path_prefix}cd {shlex.quote(ns3_dir_wsl)} && ./ns3 build hrllc_tsn_mappo"
        run_checked(["wsl", "bash", "-lc", command])


def next_round_start():
    if not SUMMARY_CSV.exists():
        return 0
    try:
        df = pd.read_csv(SUMMARY_CSV)
        if df.empty or "round" not in df:
            return 0
        return int(df["round"].max()) + 1
    except Exception:
        return 0


def python_reference_gain_csv(seed, k, max_attempts):
    """Serialize the supplied Python environment's deterministic Gamma draws."""
    gains = []
    for agent in range(k):
        for attempt in range(1, max_attempts + 1):
            rng = np.random.RandomState((seed + agent * 10000 + attempt * 100) % (2**32))
            gains.append(rng.gamma(shape=attempt, scale=1.0))
    return ",".join(f"{gain:.17g}" for gain in gains)


def verify_cpp_action_log_probabilities(model, csv_path, config):
    """Check C++ inference against the exported PyTorch policy on recorded states."""
    df = pd.read_csv(csv_path)
    _, _, per_agent = reconstruct_training_arrays(df, model["k"])
    for agent, item in enumerate(per_agent):
        obs = torch.as_tensor(item["obs"], dtype=torch.float32)
        q_actions = torch.as_tensor(item["q_actions"], dtype=torch.long)
        n_actions = torch.as_tensor(item["n_actions"], dtype=torch.long)
        with torch.no_grad():
            q_logits, n_logits = model["actors"][agent](obs, model["actor_n_action_dims"][agent])
            q_logp = Categorical(logits=q_logits).log_prob(q_actions).cpu().numpy()
            n_logp = Categorical(logits=n_logits).log_prob(n_actions).cpu().numpy()
        if not np.allclose(q_logp, item["q_logp"], rtol=2e-5, atol=2e-5):
            raise RuntimeError(f"C++/PyTorch power log-probability mismatch for actor {agent}")
        if not np.allclose(n_logp, item["n_logp"], rtol=2e-5, atol=2e-5):
            raise RuntimeError(f"C++/PyTorch retransmission log-probability mismatch for actor {agent}")


def run_ns3_round(round_index, model, config, args, *, seed=None, deterministic=None, prefix="round"):
    write_cpp_weights(model, CURRENT_WEIGHTS)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    seed = args.seed + round_index if seed is None else seed
    deterministic = args.deterministic if deterministic is None else deterministic
    csv_path = available_csv_path(round_index, prefix=prefix, seed=seed if prefix != "round" else None)
    reference_gains = python_reference_gain_csv(seed, model["k"], model["n_action_dim"])

    project_wsl_real = to_wsl_path(PROJECT_ROOT)
    project_wsl = f"/tmp/hrllc_tsn_mappo_project_{os.getpid()}"
    ns3_dir_wsl = f"{project_wsl}/external/ns-3.44"
    weights_wsl = f"{project_wsl}/weights/current_model.weights"
    csv_wsl = f"{project_wsl}/runs/{csv_path.name}"
    cmake_candidates = sorted(EXTERNAL_DIR.glob("cmake-*-linux-x86_64/bin"))
    if cmake_candidates:
        cmake_bin_wsl = to_wsl_path(cmake_candidates[-1])
        path_prefix = f"export PATH={shlex.quote(cmake_bin_wsl + ':/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin')}; "
    else:
        path_prefix = ""
    program_args = (
        f"--weights={shlex.quote(weights_wsl)} "
        f"--csv={shlex.quote(csv_wsl)} "
        f"--iterations={args.iterations} "
        f"--K={args.k} "
        f"--seed={seed} "
        f"--radius={args.radius} "
        f"--speedMin={config['speed_min_mps']} "
        f"--speedMax={config['speed_max_mps']} "
        f"--hyperperiodMs={config['hyperperiod_ms']} "
        f"--tsnRateMbps={config['tsn_rate_mbps']} "
        f"--hrllcBandwidthMbps={config['hrllc_bandwidth_mbps']} "
        f"--carrierFrequencyGhz={config['carrier_frequency_ghz']} "
        f"--miniSlotMs={config['mini_slot_ms']} "
        f"--retransmissionMs={config['retransmission_ms']} "
        f"--retransmissionGuardMs={config.get('retransmission_guard_ms', 0.0)} "
        f"--hopCount={config['hop_count']} "
        f"--flow1PeriodMs={config['flow_type_1_period_ms']} "
        f"--flow1LatencyMs={config['flow_type_1_latency_ms']} "
        f"--flow1Bytes={config['flow_type_1_bytes']} "
        f"--flow1Epsilon={config['flow_type_1_epsilon']} "
        f"--flow2PeriodMs={config['flow_type_2_period_ms']} "
        f"--flow2LatencyMs={config['flow_type_2_latency_ms']} "
        f"--flow2Bytes={config['flow_type_2_bytes']} "
        f"--flow2Epsilon={config['flow_type_2_epsilon']} "
        f"--pH={args.power} "
        f"--gammaDb={args.gamma_db} "
        f"--shadowFadingStdDb={config['shadow_fading_std_db']} "
        f"--smallScaleFadingRate={config['small_scale_fading_exponential_rate']} "
        f"--thermalNoiseDensityDbmHz={config['thermal_noise_density_dbm_hz']} "
        f"--pathLossC={config['path_loss_c_db']} "
        f"--pathLossD0={config['path_loss_d0_m']} "
        f"--pathLossD1={config['path_loss_d1_m']} "
        f"--boundaryTurnMinRad={config['mobility_boundary_turn_min_rad']} "
        f"--boundaryTurnMaxRad={config['mobility_boundary_turn_max_rad']} "
        f"--minimumDistanceM={config['minimum_distance_m']} "
        f"--lambdaNormalizationOffsetDb={config['lambda_normalization_offset_db']} "
        f"--lambdaNormalizationScaleDb={config['lambda_normalization_scale_db']} "
        f"--lambdaNormalizationMin={config['lambda_normalization_min']} "
        f"--lambdaNormalizationMax={config['lambda_normalization_max']} "
        f"--softmaxMaskLogit={config['softmax_mask_logit']} "
        f"--probabilityFloor={config['probability_floor']} "
        f"--denominatorFloor={config['denominator_floor']} "
        f"--reliabilityPenaltyWeight={config['reliability_penalty_weight']} "
        f"--shadowFadingPerStep={str(config['shadow_fading_mode'] == 'iid_per_step').lower()} "
        f"--conservativeRetryFeasibility={str(config.get('conservative_retry_feasibility_mask', False)).lower()} "
        f"--referenceGains={reference_gains} "
        f"--deterministic={str(deterministic).lower()} "
        f"--runIndex={round_index}"
    )
    command = (
        f"{path_prefix}"
        f"ln -sfnT {shlex.quote(project_wsl_real)} {shlex.quote(project_wsl)} && "
        f"{shlex.quote(ns3_dir_wsl + '/build/scratch/ns3.44-hrllc_tsn_mappo-default')} "
        f"{program_args}"
    )
    run_checked(["wsl", "bash", "-lc", command])
    verify_cpp_action_log_probabilities(model, csv_path, config)
    return csv_path


# Evaluate a fixed collection of scenarios and average their metrics. Validation
# and final testing call this with separate seed ranges to avoid data leakage.
def evaluate_model(round_index, model, config, args, *, seed_start, seed_count, prefix, summary_path):
    results = []
    for offset in range(seed_count):
        seed = seed_start + offset
        csv_path = run_ns3_round(
            round_index,
            model,
            config,
            args,
            seed=seed,
            deterministic=config["evaluation_action_mode"] == "argmax",
            prefix=prefix,
        )
        results.append(metrics_from_csv(csv_path))
    averaged = mean_metrics(results)
    append_evaluation_summary(summary_path, round_index, averaged, seed_count)
    return averaged


def validate_model(round_index, model, config, args):
    return evaluate_model(
        round_index,
        model,
        config,
        args,
        seed_start=args.validation_seed_start,
        seed_count=args.validation_seeds,
        prefix="validation_round",
        summary_path=VALIDATION_SUMMARY_CSV,
    )


# Program entry point: resolve JSON defaults, support preflight and final-test
# modes, then execute the simulation -> CSV -> MAPPO-update training loop.
def main():
    parser = argparse.ArgumentParser(description="Windows orchestrator for HRLLC-TSN MAPPO ns-3 workflow")
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--radius", type=float, default=None)
    parser.add_argument("--power", type=float, default=None)
    parser.add_argument("--gamma-db", type=float, default=None)
    parser.add_argument("--reset-model", action="store_true")
    parser.add_argument(
        "--fresh-start",
        action="store_true",
        help="Delete generated run CSVs and model checkpoints before starting a new campaign.",
    )
    parser.add_argument(
        "--validation-every",
        type=int,
        default=None,
        help="Validate the updated model every N training rounds using fixed deterministic scenarios.",
    )
    parser.add_argument(
        "--validation-seeds",
        type=int,
        default=None,
        help="Number of fixed scenarios used to rank training checkpoints.",
    )
    parser.add_argument(
        "--validation-seed-start",
        type=int,
        default=None,
        help="First fixed seed reserved for validation; it never overlaps training seeds.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=None,
        help="Save a versioned checkpoint every N training rounds.",
    )
    parser.add_argument("--final-test-seed-start", type=int, default=None)
    parser.add_argument("--final-test-seeds", type=int, default=None)
    parser.add_argument(
        "--final-evaluate-best",
        action="store_true",
        help="Evaluate best_model.npz on the held-out final test seeds without further training.",
    )
    parser.add_argument(
        "--preset-policy",
        choices=["none", "n1", "n2", "low-latency", "channel-n1", "channel-mixed"],
        default="none",
    )
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Check paper-fixed configuration, Python dependencies, and the ns-3 build without training.",
    )
    args = parser.parse_args()

    config = load_config()
    if int(config["hidden_layer_count"]) != 2:
        raise ValueError("This weight-file format implements exactly two hidden layers.")
    if len(config["initial_action_power_bias"]) < int(config["q_max"]):
        raise ValueError("initial_action_power_bias must contain at least q_max values.")
    if config.get("shadow_fading_mode") not in {"iid_per_step", "static_per_scenario"}:
        raise ValueError("shadow_fading_mode must be 'iid_per_step' or 'static_per_scenario'.")
    if config.get("evaluation_action_mode") not in {"sample", "argmax"}:
        raise ValueError("evaluation_action_mode must be 'sample' or 'argmax'.")
    if config.get("initialization_method", "orthogonal") not in {"orthogonal", "normal_fixed", "xavier_uniform"}:
        raise ValueError("initialization_method must be 'orthogonal', 'normal_fixed', or 'xavier_uniform'.")
    if config.get("initial_action_prior") not in {"balanced_safe", "none"}:
        raise ValueError("initial_action_prior must be 'balanced_safe' or 'none'.")
    if args.iterations is None:
        args.iterations = int(config["iterations"])
    if args.k is None:
        args.k = int(config["k"])
    if args.seed is None:
        args.seed = int(config["seed"])
    if args.radius is None:
        args.radius = float(config["radius_m"])
    if args.power is None:
        args.power = float(config["total_power_w"])
    if args.gamma_db is None:
        args.gamma_db = float(config["gamma_threshold_db"])
    if args.rounds is None:
        args.rounds = int(config["training_episodes"])
    if args.validation_every is None:
        args.validation_every = int(config["validation_every_rounds"])
    if args.validation_seeds is None:
        args.validation_seeds = int(config["validation_seed_count"])
    if args.validation_seed_start is None:
        args.validation_seed_start = int(config["validation_seed_start"])
    if args.checkpoint_every is None:
        args.checkpoint_every = int(config["checkpoint_every_rounds"])
    if args.final_test_seed_start is None:
        args.final_test_seed_start = int(config["final_test_seed_start"])
    if args.final_test_seeds is None:
        args.final_test_seeds = int(config["final_test_seed_count"])
    config["k"] = args.k

    if (
        args.validation_every < 1
        or args.validation_seeds < 1
        or args.checkpoint_every < 1
        or args.final_test_seeds < 1
    ):
        raise ValueError("validation and checkpoint intervals must be positive")

    try:
        write_workflow_status("running", mode="preflight" if args.preflight else "training")
        verify_paper_configuration(config)
        ensure_ns3_ready()
        if args.preflight:
            model, _ = ensure_model(config)
            original_iterations = args.iterations
            args.iterations = 1
            smoke_csv = run_ns3_round(-1, model, config, args, seed=args.seed, prefix="preflight_smoke")
            args.iterations = original_iterations
            smoke_metrics = metrics_from_csv(smoke_csv)
            write_workflow_status(
                "preflight_passed",
                mode="preflight",
                paper_configuration="verified",
                ns3="ready",
                smoke_failure_rate=smoke_metrics["failure_rate"],
            )
            print(f"preflight=PASS status_file={WORKFLOW_STATUS_JSON}")
            return 0
        if args.fresh_start:
            remove_generated_artifacts()
            write_campaign_manifest(config, args)
        if args.reset_model or args.fresh_start:
            reset_summary_file()
        if args.final_evaluate_best:
            best_path = WEIGHTS_DIR / "best_model.npz"
            if not best_path.exists():
                raise RuntimeError("No best model exists. Train and validate a campaign first.")
            model = load_model_npz(best_path)
            final_metrics = evaluate_model(
                round_index=0,
                model=model,
                config=config,
                args=args,
                seed_start=args.final_test_seed_start,
                seed_count=args.final_test_seeds,
                prefix="final_test_round",
                summary_path=FINAL_TEST_SUMMARY_CSV,
            )
            (RUNS_DIR / "final_test_metrics.json").write_text(
                json.dumps(final_metrics, indent=2), encoding="utf-8"
            )
            print(
                f"final_test_failure_rate={final_metrics['failure_rate']:.6g} "
                f"seeds={args.final_test_seeds}"
            )
            write_workflow_status("completed", mode="final_evaluation", summary=str(FINAL_TEST_SUMMARY_CSV))
            return 0
        model, recreated = ensure_model(config, reset=args.reset_model or args.fresh_start)
        if args.preset_policy != "none":
            apply_preset_policy(model, config, args.preset_policy)
            save_model_npz(model, CURRENT_NPZ)
            write_cpp_weights(model, CURRENT_WEIGHTS)
        if (args.fresh_start or args.reset_model) and not args.eval_only:
            initial_metrics = validate_model(-1, model, config, args)
            initial_metrics["training_round"] = -1
            initial_metrics["validation_seed_count"] = args.validation_seeds
            save_best_if_needed(model, initial_metrics)
            print(
                f"initial_validation_failure_rate={initial_metrics['failure_rate']:.6g} "
                f"seeds={args.validation_seeds}"
            )
        if recreated and not args.reset_model and SUMMARY_CSV.exists():
            SUMMARY_CSV.unlink()
        start_round = 0 if args.reset_model or args.fresh_start or recreated else next_round_start()
        for offset in range(args.rounds):
            round_index = start_round + offset
            csv_path = run_ns3_round(round_index, model, config, args)
            if args.eval_only:
                metrics = metrics_from_csv(csv_path)
            else:
                metrics = update_model_from_csv(model, csv_path, config)
                save_model_npz(model, CURRENT_NPZ)
                write_cpp_weights(model, CURRENT_WEIGHTS)
                if (round_index + 1) % args.checkpoint_every == 0:
                    versioned = WEIGHTS_DIR / f"model_after_round_{round_index:04d}.npz"
                    save_model_npz(model, versioned)
                if (round_index + 1) % args.validation_every == 0 or offset == args.rounds - 1:
                    validation_metrics = validate_model(round_index, model, config, args)
                    validation_metrics["training_round"] = round_index
                    validation_metrics["validation_seed_count"] = args.validation_seeds
                    save_best_if_needed(model, validation_metrics)
                    print(
                        f"validation_round={round_index} "
                        f"mean_failure_rate={validation_metrics['failure_rate']:.6g} "
                        f"seeds={args.validation_seeds}"
                    )
            append_summary(round_index, csv_path, metrics)
            print(
                f"round={round_index} csv={csv_path} "
                f"failure_rate={metrics['failure_rate']:.6g} "
                f"reward_mean={metrics['reward_mean']:.6g}"
            )
        print(f"summary_csv={SUMMARY_CSV}")
        write_workflow_status("completed", mode="training", summary=str(SUMMARY_CSV))
    except Exception as exc:
        write_workflow_status("failed", mode="preflight" if args.preflight else "training", error=str(exc))
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
