"""
Replay Buffer with Stochastic Stratified Sampling.

Implements the "Stochastic Teleportation" design from Hephaestus spec:
- O(1) sampling complexity (no cosine similarity search)
- 50% Frontier batch (recent experiences)
- 50% Episodic batch (global history)
- Ring buffer in HBM (per-device sharded)
"""

import jax
import jax.numpy as jnp
from jax import random
import chex
from typing import Tuple

from dtc_jax.configs.base_config import DTCConfig
from dtc_jax.dtc.dtc_types import ReplayBuffer, TrainingBatch


def create_replay_buffer(config: DTCConfig, batch_size: int) -> ReplayBuffer:
    """
    Initialize empty replay buffer with static capacity.

    Args:
        config: DTCConfig with capacity and dimensions
        batch_size: Per-device batch size (for sharded pmap setup)

    Returns:
        ReplayBuffer PyTreeNode with allocated arrays
    """
    capacity = config.replay_capacity
    obs_dim = config.obs_dim
    action_dim = config.action_dim
    latent_det_dim = config.latent_dim_deterministic
    latent_stoch_dim = config.latent_dim_stochastic

    return ReplayBuffer(
        observations=jnp.zeros((capacity, obs_dim), dtype=jnp.float32),
        actions=jnp.zeros((capacity, action_dim), dtype=jnp.float32),
        rewards=jnp.zeros((capacity,), dtype=jnp.float32),
        dones=jnp.zeros((capacity,), dtype=jnp.bool_),
        # Store RSSM states for continuing rollouts
        latent_h=jnp.zeros((capacity, latent_det_dim), dtype=jnp.float32),
        latent_z=jnp.zeros((capacity, latent_stoch_dim), dtype=jnp.float32),
        # Intrinsic rewards for cognitive tracking
        intrinsic_rewards=jnp.zeros((capacity,), dtype=jnp.float32),
        # Buffer management
        ptr=0,  # Next write position
        count=0  # Total items in buffer (saturates at capacity)
    )


def add_transition(
    buffer: ReplayBuffer,
    observation: chex.Array,  # [obs_dim]
    action: chex.Array,  # [action_dim]
    reward: float,
    done: bool,
    latent_h: chex.Array,  # [latent_det_dim]
    latent_z: chex.Array,  # [latent_stoch_dim]
    intrinsic_reward: float
) -> ReplayBuffer:
    """
    Add a single transition to the ring buffer.

    This function is pure and returns a new buffer state.
    Use jax.lax.scan to accumulate multiple transitions.

    Args:
        buffer: Current buffer state
        observation: Observation at time t
        action: Action taken at time t
        reward: Extrinsic reward received
        done: Terminal flag
        latent_h: Deterministic RSSM state
        latent_z: Stochastic RSSM state
        intrinsic_reward: Computed intrinsic reward

    Returns:
        Updated buffer with new transition at ptr position
    """
    ptr = buffer.ptr
    capacity = buffer.observations.shape[0]

    # Write to current position (ring buffer)
    new_buffer = ReplayBuffer(
        observations=buffer.observations.at[ptr].set(observation),
        actions=buffer.actions.at[ptr].set(action),
        rewards=buffer.rewards.at[ptr].set(reward),
        dones=buffer.dones.at[ptr].set(done),
        latent_h=buffer.latent_h.at[ptr].set(latent_h),
        latent_z=buffer.latent_z.at[ptr].set(latent_z),
        intrinsic_rewards=buffer.intrinsic_rewards.at[ptr].set(intrinsic_reward),
        # Update pointers
        ptr=(ptr + 1) % capacity,  # Wrap around
        count=jnp.minimum(buffer.count + 1, capacity)  # Saturate
    )

    return new_buffer


def add_batch(
    buffer: ReplayBuffer,
    observations: chex.Array,  # [batch, obs_dim]
    actions: chex.Array,  # [batch, action_dim]
    rewards: chex.Array,  # [batch]
    dones: chex.Array,  # [batch]
    latent_h: chex.Array,  # [batch, latent_det_dim]
    latent_z: chex.Array,  # [batch, latent_stoch_dim]
    intrinsic_rewards: chex.Array  # [batch]
) -> ReplayBuffer:
    """
    Add a batch of transitions using scan for efficiency.

    Args:
        buffer: Current buffer state
        observations: Batch of observations [batch, obs_dim]
        actions: Batch of actions [batch, action_dim]
        rewards: Batch of rewards [batch]
        dones: Batch of terminal flags [batch]
        latent_h: Batch of deterministic states [batch, latent_det_dim]
        latent_z: Batch of stochastic states [batch, latent_stoch_dim]
        intrinsic_rewards: Batch of intrinsic rewards [batch]

    Returns:
        Updated buffer with all transitions added
    """
    batch_size = observations.shape[0]

    def add_single(buf, idx):
        return add_transition(
            buf,
            observations[idx],
            actions[idx],
            rewards[idx],
            dones[idx],
            latent_h[idx],
            latent_z[idx],
            intrinsic_rewards[idx]
        ), None

    final_buffer, _ = jax.lax.scan(add_single, buffer, jnp.arange(batch_size))
    return final_buffer


def sample_stochastic_stratified(
    buffer: ReplayBuffer,
    config: DTCConfig,
    key: chex.PRNGKey
) -> Tuple[TrainingBatch, chex.PRNGKey]:
    """
    Sample sequences using Stochastic Stratified Sampling (Hephaestus Spec).

    Strategy:
    - 50% Frontier Batch: Recent experiences (ptr - frontier_size, ptr)
    - 50% Episodic Batch: Random from entire history (0, count)

    This provides O(1) sampling with coverage of both recent and historical data,
    preventing catastrophic forgetting without expensive similarity search.

    Args:
        buffer: Current buffer state
        config: Config with batch_size, sequence_length, frontier_size
        key: PRNG key

    Returns:
        TrainingBatch with sequences [batch, seq_len, ...]
        Updated PRNG key
    """
    batch_size = config.local_batch_size  # Per-device batch
    seq_len = config.sequence_length
    frontier_window = config.frontier_size

    # Split batch: 50% frontier, 50% episodic
    frontier_size = batch_size // 2
    episodic_size = batch_size - frontier_size

    key, frontier_key, episodic_key = random.split(key, 3)

    # Frontier sampling: Recent experiences
    # Sample starting indices from (ptr - frontier_window) to (ptr - seq_len)
    # Ensure we have enough data for a sequence
    frontier_start = jnp.maximum(buffer.ptr - frontier_window, seq_len)
    frontier_end = jnp.maximum(buffer.ptr - seq_len, seq_len)
    frontier_end = jnp.maximum(frontier_end, frontier_start + 1)  # Ensure valid range

    frontier_indices = random.randint(
        frontier_key,
        (frontier_size,),
        minval=frontier_start,
        maxval=frontier_end
    )

    # Episodic sampling: Random from entire history
    # Sample starting indices from 0 to (count - seq_len)
    episodic_max = jnp.maximum(buffer.count - seq_len, 1)
    episodic_indices = random.randint(
        episodic_key,
        (episodic_size,),
        minval=0,
        maxval=episodic_max
    )

    # Concatenate frontier and episodic indices
    start_indices = jnp.concatenate([frontier_indices, episodic_indices], axis=0)

    # Extract sequences (vectorized)
    # For each start_idx, extract [start_idx : start_idx + seq_len]
    def extract_sequence(start_idx):
        # Generate sequence indices with modulo for ring buffer wrapping
        seq_indices = (start_idx + jnp.arange(seq_len)) % buffer.observations.shape[0]

        return TrainingBatch(
            observations=buffer.observations[seq_indices],  # [seq_len, obs_dim]
            actions=buffer.actions[seq_indices],  # [seq_len, action_dim]
            rewards=buffer.rewards[seq_indices],  # [seq_len]
            dones=buffer.dones[seq_indices],  # [seq_len]
            latent_h=buffer.latent_h[seq_indices],  # [seq_len, latent_det_dim]
            latent_z=buffer.latent_z[seq_indices],  # [seq_len, latent_stoch_dim]
            intrinsic_rewards=buffer.intrinsic_rewards[seq_indices]  # [seq_len]
        )

    # Vectorize over batch dimension
    batch = jax.vmap(extract_sequence)(start_indices)

    return batch, key


def is_ready_to_sample(buffer: ReplayBuffer, config: DTCConfig) -> bool:
    """
    Check if buffer has enough data to sample sequences.

    Args:
        buffer: Current buffer state
        config: Config with sequence_length and minimum data requirements

    Returns:
        True if buffer has at least sequence_length + frontier_size transitions
    """
    min_data = config.sequence_length + config.frontier_size
    return buffer.count >= min_data


# ============================================================================
# Utility functions for analysis (not used in training loop)
# ============================================================================

def get_buffer_stats(buffer: ReplayBuffer) -> dict:
    """Get buffer statistics for logging/debugging."""
    return {
        'count': int(buffer.count),
        'ptr': int(buffer.ptr),
        'capacity': buffer.observations.shape[0],
        'utilization': float(buffer.count) / buffer.observations.shape[0],
        'mean_reward': float(jnp.mean(buffer.rewards[:buffer.count])),
        'mean_intrinsic': float(jnp.mean(buffer.intrinsic_rewards[:buffer.count])),
    }


def reset_buffer(buffer: ReplayBuffer) -> ReplayBuffer:
    """Reset buffer to empty state (keep allocated arrays)."""
    return ReplayBuffer(
        observations=jnp.zeros_like(buffer.observations),
        actions=jnp.zeros_like(buffer.actions),
        rewards=jnp.zeros_like(buffer.rewards),
        dones=jnp.zeros_like(buffer.dones),
        latent_h=jnp.zeros_like(buffer.latent_h),
        latent_z=jnp.zeros_like(buffer.latent_z),
        intrinsic_rewards=jnp.zeros_like(buffer.intrinsic_rewards),
        ptr=0,
        count=0
    )
