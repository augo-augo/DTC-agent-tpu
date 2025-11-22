"""
Type definitions for DTC 3.0 JAX implementation.

Centralizes all type aliases to ensure consistency across the codebase
and improve code readability.
"""

from typing import Any, Callable, Dict, Tuple, NamedTuple
import jax.numpy as jnp
import chex
from flax import struct


# ========== JAX Core Types ==========
PRNGKey = chex.PRNGKey  # JAX random key type
Array = chex.Array  # JAX array type
Params = chex.ArrayTree  # Parameters as pytree (dict of arrays)
OptState = Any  # Optimizer state (opaque type from optax)


# ========== RSSM State Types ==========
@struct.dataclass
class RSSMState:
    """Recurrent State-Space Model state.

    Combines deterministic (GRU hidden state) and stochastic (latent sample) components.
    """
    deterministic: Array  # Shape: [batch, latent_dim_deterministic]
    stochastic: Array  # Shape: [batch, latent_dim_stochastic]

    @property
    def feature(self) -> Array:
        """Concatenated feature vector for actor/critic."""
        return jnp.concatenate([self.deterministic, self.stochastic], axis=-1)


@struct.dataclass
class RSSMOutput:
    """Output from RSSM forward pass."""
    state: RSSMState  # New state after transition
    prior_mean: Array  # Prior distribution mean
    prior_std: Array  # Prior distribution std
    posterior_mean: Array  # Posterior distribution mean (when conditioned on obs)
    posterior_std: Array  # Posterior distribution std


# ========== Ensemble Types ==========
@struct.dataclass
class EnsembleOutput:
    """Output from ensemble of RSSM models.

    All arrays have shape [ensemble_size, batch, ...].
    """
    states: RSSMState  # Ensemble of states
    prior_means: Array  # Shape: [ensemble_size, batch, stoch_dim]
    prior_stds: Array  # Shape: [ensemble_size, batch, stoch_dim]
    posterior_means: Array  # Shape: [ensemble_size, batch, stoch_dim]
    posterior_stds: Array  # Shape: [ensemble_size, batch, stoch_dim]


# ========== Intrinsic Motivation State ==========
@struct.dataclass
class IntrinsicState:
    """State for dual-timescale competence tracking.

    All values must be in float32 to prevent numerical underflow.
    """
    c_slow: Array  # Slow EMA of prediction error (float32 scalar)
    c_fast: Array  # Fast EMA of prediction error (float32 scalar)
    boredom: Array  # Accumulated boredom signal (float32 scalar)
    step_count: Array  # Total steps taken (int32 scalar)

    @classmethod
    def init(cls) -> "IntrinsicState":
        """Initialize intrinsic state with zeros."""
        return cls(
            c_slow=jnp.array(0.0, dtype=jnp.float32),
            c_fast=jnp.array(0.0, dtype=jnp.float32),
            boredom=jnp.array(0.0, dtype=jnp.float32),
            step_count=jnp.array(0, dtype=jnp.int32),
        )


# ========== Replay Buffer Types ==========
@struct.dataclass
class Transition:
    """Single environment transition."""
    observation: Array  # Shape: [obs_dim]
    action: Array  # Shape: [action_dim]
    reward: Array  # Shape: [] (scalar)
    done: Array  # Shape: [] (scalar, 0 or 1)


@struct.dataclass
class ReplayBuffer:
    """Static ring buffer for on-chip storage.

    All arrays are pre-allocated with fixed capacity.
    """
    observations: Array  # Shape: [capacity, obs_dim]
    actions: Array  # Shape: [capacity, action_dim]
    rewards: Array  # Shape: [capacity]
    dones: Array  # Shape: [capacity]
    ptr: Array  # Scalar int32: current write position
    count: Array  # Scalar int32: total transitions stored (saturates at capacity)
    capacity: int  # Static capacity (not a JAX array)

    @classmethod
    def create(cls, capacity: int, obs_dim: int, action_dim: int) -> "ReplayBuffer":
        """Create empty replay buffer with pre-allocated arrays."""
        return cls(
            observations=jnp.zeros((capacity, obs_dim), dtype=jnp.float32),
            actions=jnp.zeros((capacity, action_dim), dtype=jnp.float32),
            rewards=jnp.zeros(capacity, dtype=jnp.float32),
            dones=jnp.zeros(capacity, dtype=jnp.float32),
            ptr=jnp.array(0, dtype=jnp.int32),
            count=jnp.array(0, dtype=jnp.int32),
            capacity=capacity,
        )


# ========== Agent Parameter Container ==========
@struct.dataclass
class AgentParams:
    """Container for all trainable network parameters."""
    world_model: Params  # RSSM ensemble parameters (stacked)
    actor: Params  # Policy network parameters
    critic: Params  # Value network parameters


# ========== Training Carrier State ==========
@struct.dataclass
class TrainingState:
    """Complete state threaded through the training scan loop.

    This is the 'carrier' that gets updated at each training step.
    """
    rng: PRNGKey
    agent_params: AgentParams
    world_model_opt_state: OptState
    actor_opt_state: OptState
    critic_opt_state: OptState
    replay_buffer: ReplayBuffer
    intrinsic_state: IntrinsicState
    step: Array  # Global training step counter (int32)


# ========== Dream Rollout Types ==========
@struct.dataclass
class DreamState:
    """State during imagination rollout."""
    rssm_state: RSSMState  # Current latent state
    cumulative_reward: Array  # Accumulated intrinsic reward
    cumulative_value: Array  # Accumulated value estimate


# ========== Logging & Metrics ==========
Metrics = Dict[str, Array]  # Dictionary of scalar metrics


# ========== Function Signatures ==========
ActorFn = Callable[[Params, Array], Tuple[Array, Array]]  # (params, state) -> (action_mean, action_std)
CriticFn = Callable[[Params, Array], Array]  # (params, state) -> value
StepFn = Callable[[TrainingState, Any], Tuple[TrainingState, Metrics]]  # Training step function
