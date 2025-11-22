"""
DTC 3.0 Base Configuration

Frozen dataclass containing all static hyperparameters for XLA compilation.
All parameters must be compile-time constants to ensure static shapes.
"""

from dataclasses import dataclass
from typing import Tuple
import chex


@chex.dataclass(frozen=True)
class DTCConfig:
    """Static configuration for DTC 3.0 JAX/TPU implementation."""

    # ===== Environment Settings =====
    # JAX-native grid world: 16×16 grid + 8 inventory + 2 position = 266
    obs_dim: int = 266  # Observation dimension (JAX GridWorld)
    action_dim: int = 5  # Action dimension (up, down, left, right, collect)

    # ===== Training Settings =====
    global_batch_size: int = 256  # Must be divisible by TPU core count (e.g., 8)
    num_tpu_cores: int = 8  # Standard v3-8 TPU pod
    local_batch_size: int = 32  # global_batch_size / num_tpu_cores

    # ===== Sequence Settings =====
    sequence_length: int = 50  # Training sequence length
    max_dream_horizon: int = 64  # Maximum imagination rollout (static)

    # ===== Replay Buffer Settings =====
    replay_capacity: int = 200_000  # Must fit in TPU HBM (32GB per core)
    frontier_size: int = 1_000  # Recent data window for frontier sampling
    frontier_fraction: float = 0.5  # Mix ratio: 50% frontier, 50% episodic

    # ===== World Model Architecture =====
    ensemble_size: int = 5  # Number of ensemble models (vmap axis)
    latent_dim_deterministic: int = 1024  # Deterministic state (h_t)
    latent_dim_stochastic: int = 32  # Stochastic state (z_t)
    hidden_dim: int = 512  # MLP hidden dimension

    # GRU Cell Settings
    gru_hidden_dim: int = 1024  # GRU recurrent hidden state

    # ===== Actor-Critic Architecture =====
    actor_hidden_dims: Tuple[int, ...] = (512, 512)
    critic_hidden_dims: Tuple[int, ...] = (512, 512)

    # ===== Optimization Settings =====
    learning_rate_world_model: float = 3e-4
    learning_rate_actor_critic: float = 1e-4
    weight_decay: float = 1e-6
    grad_clip_norm: float = 100.0

    # Adam/AdamW optimizer settings
    adam_b1: float = 0.9
    adam_b2: float = 0.999
    adam_eps: float = 1e-8

    # ===== Intrinsic Motivation Settings =====
    alpha_fast: float = 0.1  # Fast EMA coefficient
    alpha_slow: float = 0.005  # Slow EMA coefficient

    # Boredom scaling
    boredom_threshold: float = 1e-3  # When r_lp < threshold, agent is "bored"
    boredom_scale: float = 0.5  # Scales down horizon when bored
    min_horizon: int = 8  # Minimum dream horizon (never go below this)

    # ===== Loss Weights =====
    # World Model
    kl_weight: float = 0.1  # KL divergence weight in ELBO
    reconstruction_weight: float = 1.0  # Observation reconstruction weight

    # Actor-Critic
    value_loss_weight: float = 0.5
    entropy_weight: float = 1e-3  # Policy entropy regularization

    # ===== Numerical Stability =====
    log_std_min: float = -10.0  # Minimum log std (prevents exp underflow)
    log_std_max: float = 2.0  # Maximum log std (prevents exploding variance)
    epsilon: float = 1e-8  # Small constant for numerical stability

    # ===== Precision Settings =====
    # Networks use bfloat16 for throughput
    # Intrinsic calculations use float32 for precision
    use_bfloat16: bool = True

    # ===== Discount Factors =====
    gamma: float = 0.99  # Standard RL discount
    lambda_gae: float = 0.95  # GAE lambda for advantage estimation

    # ===== Training Loop =====
    num_train_steps: int = 1_000_000  # Total training steps
    log_interval: int = 100  # Steps between logging
    checkpoint_interval: int = 10_000  # Steps between checkpoints

    # ===== Exploration =====
    initial_exploration_steps: int = 5_000  # Random action steps at start
    action_noise_std: float = 0.3  # Gaussian exploration noise

    def __post_init__(self):
        """Validate configuration constraints."""
        # Ensure batch size is divisible by TPU cores
        assert self.global_batch_size % self.num_tpu_cores == 0, \
            f"global_batch_size ({self.global_batch_size}) must be divisible by num_tpu_cores ({self.num_tpu_cores})"

        # Ensure local batch size is correct
        assert self.local_batch_size == self.global_batch_size // self.num_tpu_cores, \
            f"local_batch_size must equal global_batch_size / num_tpu_cores"

        # Ensure frontier size fits in replay capacity
        assert self.frontier_size < self.replay_capacity, \
            "frontier_size must be less than replay_capacity"

        # Ensure min_horizon <= max_dream_horizon
        assert self.min_horizon <= self.max_dream_horizon, \
            "min_horizon must be <= max_dream_horizon"


# Default configuration instance
DEFAULT_CONFIG = DTCConfig()
