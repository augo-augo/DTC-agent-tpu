"""
Common type aliases and data structures for DTC 3.0.

All state containers must be Flax PyTreeNodes for proper JAX tracing.
"""

from typing import Any, Tuple, NamedTuple
import jax.numpy as jnp
from jax import Array
from flax import struct
import chex


# ===== Basic Type Aliases =====
PRNGKey = chex.PRNGKey
Params = chex.ArrayTree  # PyTree of parameters
Array = chex.Array
Scalar = chex.Scalar


# ===== RSSM State =====
@struct.dataclass
class RSSMState:
    """Recurrent State-Space Model state.

    Contains both deterministic (h_t) and stochastic (z_t) components.
    All arrays must have leading batch dimension.
    """
    deterministic: Array  # Shape: [batch, latent_dim_deterministic]
    stochastic: Array  # Shape: [batch, latent_dim_stochastic]

    @classmethod
    def zeros(cls, batch_size: int, det_dim: int, stoch_dim: int) -> 'RSSMState':
        """Create zero-initialized state."""
        return cls(
            deterministic=jnp.zeros((batch_size, det_dim), dtype=jnp.float32),
            stochastic=jnp.zeros((batch_size, stoch_dim), dtype=jnp.float32),
        )


# ===== Ensemble Output =====
@struct.dataclass
class EnsembleOutput:
    """Output from world model ensemble.

    Contains predictions from all ensemble members for uncertainty estimation.
    First dimension is ensemble_size.
    """
    means: Array  # Shape: [ensemble_size, batch, feature_dim]
    stds: Array  # Shape: [ensemble_size, batch, feature_dim]
    states: RSSMState  # Batched state (may have ensemble dimension)

    @property
    def ensemble_size(self) -> int:
        """Get ensemble size from means shape."""
        return self.means.shape[0]


# ===== Intrinsic Motivation State =====
@struct.dataclass
class IntrinsicState:
    """State for intrinsic motivation tracking.

    Tracks dual-timescale EMAs for competence-based reward.
    All values must be float32 for numerical precision.
    """
    c_slow: Scalar  # Slow EMA of prediction error (float32)
    c_fast: Scalar  # Fast EMA of prediction error (float32)
    boredom_accumulator: Scalar  # Tracks consecutive low-competence steps (float32)
    step_count: Scalar  # Total steps taken (int32)

    @classmethod
    def init(cls) -> 'IntrinsicState':
        """Initialize intrinsic state with zeros."""
        return cls(
            c_slow=jnp.array(0.0, dtype=jnp.float32),
            c_fast=jnp.array(0.0, dtype=jnp.float32),
            boredom_accumulator=jnp.array(0.0, dtype=jnp.float32),
            step_count=jnp.array(0, dtype=jnp.int32),
        )


# ===== Replay Buffer State =====
@struct.dataclass
class ReplayBuffer:
    """Static ring buffer for on-chip storage.

    Pre-allocated arrays that live on TPU HBM. No dynamic resizing.
    Uses modulo arithmetic for ring buffer indexing.
    """
    # Data arrays (all have leading capacity dimension)
    observations: Array  # Shape: [capacity, obs_dim]
    actions: Array  # Shape: [capacity, action_dim]
    rewards: Array  # Shape: [capacity]
    dones: Array  # Shape: [capacity] (bool stored as int32 for XLA)

    # Ring buffer pointers
    ptr: Scalar  # Current write position (int32)
    count: Scalar  # Total items added, saturates at capacity (int32)
    capacity: Scalar  # Maximum capacity (int32, static)

    @classmethod
    def create(cls, capacity: int, obs_dim: int, action_dim: int) -> 'ReplayBuffer':
        """Create empty buffer with pre-allocated arrays."""
        return cls(
            observations=jnp.zeros((capacity, obs_dim), dtype=jnp.float32),
            actions=jnp.zeros((capacity, action_dim), dtype=jnp.float32),
            rewards=jnp.zeros((capacity,), dtype=jnp.float32),
            dones=jnp.zeros((capacity,), dtype=jnp.int32),
            ptr=jnp.array(0, dtype=jnp.int32),
            count=jnp.array(0, dtype=jnp.int32),
            capacity=jnp.array(capacity, dtype=jnp.int32),
        )


# ===== Transition Data =====
@struct.dataclass
class Transition:
    """Single environment transition.

    Used for batched environment interaction and buffer insertion.
    """
    observation: Array  # Shape: [batch, obs_dim]
    action: Array  # Shape: [batch, action_dim]
    reward: Array  # Shape: [batch]
    next_observation: Array  # Shape: [batch, obs_dim]
    done: Array  # Shape: [batch] (bool as int32)


# ===== Training Batch =====
@struct.dataclass
class TrainingBatch:
    """Batch of sequences for world model training.

    All arrays have shape [batch, sequence_length, ...].
    """
    observations: Array  # Shape: [batch, seq_len, obs_dim]
    actions: Array  # Shape: [batch, seq_len, action_dim]
    rewards: Array  # Shape: [batch, seq_len]
    dones: Array  # Shape: [batch, seq_len]


# ===== Agent Parameters =====
@struct.dataclass
class AgentParams:
    """Container for all trainable parameters.

    Separates world model and actor-critic for independent optimization.
    """
    world_model: Params  # Ensemble RSSM parameters
    actor: Params  # Policy network parameters
    critic: Params  # Value network parameters


# ===== Optimizer States =====
@struct.dataclass
class OptimizerStates:
    """Container for all optimizer states (e.g., Adam momentum).

    Must match structure of AgentParams.
    """
    world_model: Any  # Optax optimizer state
    actor_critic: Any  # Optax optimizer state


# ===== Training Metrics =====
class TrainingMetrics(NamedTuple):
    """Metrics logged during training.

    All values are scalars. Using NamedTuple for easy unpacking.
    """
    # World Model Losses
    world_model_loss: Scalar
    reconstruction_loss: Scalar
    kl_loss: Scalar

    # Actor-Critic Losses
    actor_loss: Scalar
    critic_loss: Scalar
    entropy: Scalar

    # Intrinsic Motivation
    clean_novelty: Scalar
    competence_reward: Scalar
    boredom: Scalar
    dream_horizon: Scalar

    # Environment Interaction
    episode_return: Scalar
    episode_length: Scalar


# ===== Dream Rollout State =====
@struct.dataclass
class DreamState:
    """State carried through imagination rollout.

    Used in jax.lax.scan for dreaming.
    """
    rssm_state: RSSMState
    cumulative_reward: Array  # Shape: [batch]
    value_target: Array  # Shape: [batch] (for critic target)
    step_index: Scalar  # Current dream step (int32)


# ===== Type Annotations for Common Shapes =====
# These are for documentation purposes
BatchedObservation = Array  # Shape: [batch, obs_dim]
BatchedAction = Array  # Shape: [batch, action_dim]
BatchedReward = Array  # Shape: [batch]
SequenceObservation = Array  # Shape: [batch, seq_len, obs_dim]
SequenceAction = Array  # Shape: [batch, seq_len, action_dim]
