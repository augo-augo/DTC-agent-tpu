"""
Base configuration for DTC 3.0 JAX/TPU implementation.

All hyperparameters are defined as frozen dataclasses to ensure they are
compatible with JAX's pytree system and prevent accidental mutations.
"""

from typing import Tuple
import chex
from flax import struct


@struct.dataclass
class DTCConfig:
    """Frozen configuration for DTC 3.0.

    All parameters are static and known at compile time to enable XLA optimization.
    """

    # ========== Global Training Parameters ==========
    global_batch_size: int = 256  # Must be divisible by TPU core count (e.g., 8)
    num_tpu_cores: int = 8
    seed: int = 42

    # ========== World Model Architecture ==========
    ensemble_size: int = 5
    latent_dim_deterministic: int = 1024  # GRU hidden state size
    latent_dim_stochastic: int = 32  # Stochastic latent dimension
    hidden_dim: int = 512  # Hidden layer size for MLPs

    # ========== Sequence & Replay Parameters ==========
    sequence_length: int = 50  # Training sequence length
    dream_horizon_max: int = 64  # Maximum imagination horizon (static for XLA)
    replay_capacity: int = 200_000  # Must fit in TPU HBM

    # ========== Learning Rates & Optimization ==========
    learning_rate_world_model: float = 3e-4
    learning_rate_actor: float = 3e-4
    learning_rate_critic: float = 3e-4
    adam_eps: float = 1e-8
    grad_clip_norm: float = 100.0
    weight_decay: float = 1e-6

    # ========== Intrinsic Motivation Parameters ==========
    alpha_fast: float = 0.1  # Fast EMA coefficient for competence
    alpha_slow: float = 0.005  # Slow EMA coefficient for competence
    novelty_scale: float = 1.0  # Scaling factor for clean novelty
    boredom_threshold: float = 0.01  # Threshold for detecting boredom

    # ========== RSSM Stability Parameters ==========
    log_std_min: float = -10.0  # Minimum log std to prevent collapse
    log_std_max: float = 2.0  # Maximum log std to prevent explosion
    kl_weight: float = 1.0  # KL divergence weight in ELBO
    kl_balance: float = 0.8  # Balance between reconstruction and KL

    # ========== Actor-Critic Parameters ==========
    discount_factor: float = 0.99  # Gamma for RL
    lambda_gae: float = 0.95  # Lambda for GAE
    actor_entropy_scale: float = 1e-3  # Entropy regularization

    # ========== Replay Buffer Sampling ==========
    frontier_size: int = 1000  # Number of recent steps to prioritize
    frontier_prob: float = 0.5  # Probability of sampling from frontier vs episodic

    # ========== Precision Configuration ==========
    # bfloat16 for network forward passes (throughput)
    compute_dtype: str = "bfloat16"
    # float32 for intrinsic rewards, EMAs, optimizer states (stability)
    intrinsic_dtype: str = "float32"

    # ========== Environment Parameters (Placeholder) ==========
    obs_dim: int = 64  # Observation dimension (will be overridden by env)
    action_dim: int = 6  # Action dimension (will be overridden by env)

    def __post_init__(self):
        """Validate configuration constraints."""
        # Ensure batch size is divisible by TPU cores
        assert self.global_batch_size % self.num_tpu_cores == 0, \
            f"Batch size {self.global_batch_size} must be divisible by {self.num_tpu_cores} TPU cores"

        # Ensure replay capacity can hold full sequences
        assert self.replay_capacity >= self.sequence_length * self.global_batch_size, \
            "Replay capacity must be larger than sequence_length * batch_size"

        # Validate EMA coefficients
        assert 0 < self.alpha_fast < 1, "alpha_fast must be in (0, 1)"
        assert 0 < self.alpha_slow < 1, "alpha_slow must be in (0, 1)"
        assert self.alpha_slow < self.alpha_fast, "alpha_slow must be < alpha_fast for dual timescale"

    @property
    def batch_per_core(self) -> int:
        """Batch size per TPU core."""
        return self.global_batch_size // self.num_tpu_cores


# Default configuration instance
DEFAULT_CONFIG = DTCConfig()
