"""
Experience Collector for Environment Interaction.

This module handles collecting experiences from the environment and
adding them to the replay buffer. It integrates with the RSSM for
latent state tracking and intrinsic motivation for novelty computation.

Note: This is a template. For maximum TPU performance, implement your
environment in JAX (e.g., using Jumanji or Brax).
"""

import jax
import jax.numpy as jnp
from jax import random
import chex
from typing import Tuple, Callable, Any

from dtc_jax.configs.base_config import DTCConfig
from dtc_jax.dtc.dtc_types import RSSMState, IntrinsicState, ReplayBuffer
from dtc_jax.dtc import rssm as rssm_module
from dtc_jax.dtc import intrinsic as intrinsic_module
from dtc_jax.dtc import memory as memory_module
from dtc_jax.dtc import actor_critic as ac_module


def collect_experience_step(
    carrier_state: Any,  # From trainer.CarrierState
    env_state: Any,
    env_step_fn: Callable,
    config: DTCConfig,
    deterministic: bool = False
) -> Tuple[Any, Any, dict]:
    """
    Collect a single step of experience from the environment.

    This function:
    1. Samples action from actor-critic
    2. Steps the environment
    3. Encodes observation through RSSM
    4. Computes intrinsic reward
    5. Adds transition to replay buffer

    Args:
        carrier_state: Current carrier state (from trainer)
        env_state: Current environment state
        env_step_fn: Environment step function (state, action) -> (new_state, obs, reward, done)
        config: DTCConfig
        deterministic: If True, use mean action (for evaluation)

    Returns:
        new_carrier_state: Updated carrier with new buffer entry
        new_env_state: Updated environment state
        info: Dict with step information (reward, done, etc.)
    """
    # ===== 1. Sample Action from Policy =====
    # Get current latent state (combine h and z)
    latent_concat = jnp.concatenate([
        carrier_state.current_rssm_state.deterministic,
        carrier_state.current_rssm_state.stochastic
    ], axis=-1)

    # Add sequence dimension (policy expects [batch, seq_len, latent_dim])
    latent_seq = latent_concat[:, None, :]  # [batch, 1, latent_dim]

    # Get action from actor-critic
    actor_critic = ac_module.ActorCriticWithSalience(config=config)
    carrier_state_key, action_key = random.split(carrier_state.rng_key)

    action, action_key = actor_critic.apply(
        carrier_state.actor_critic_params,
        latent_seq,
        action_key,
        deterministic,
        method=actor_critic.get_action
    )

    # Add exploration noise if not deterministic
    if not deterministic:
        carrier_state_key, noise_key = random.split(carrier_state_key)
        noise = random.normal(noise_key, action.shape) * config.action_noise_std
        action = jnp.clip(action + noise, -1.0, 1.0)

    # ===== 2. Step Environment =====
    new_env_state, observation, reward, done = env_step_fn(env_state, action)

    # ===== 3. Encode Observation through RSSM =====
    carrier_state_key, rssm_key = random.split(carrier_state_key)

    # Forward pass through RSSM ensemble (with observation - posterior mode)
    ensemble_states, ensemble_info = rssm_module.ensemble_forward(
        carrier_state.rssm_params,
        config,
        carrier_state.current_rssm_state,
        action,
        observation=observation,
        key=rssm_key
    )

    # Average ensemble predictions for next state
    new_rssm_state = RSSMState(
        deterministic=jnp.mean(ensemble_states.deterministic, axis=0),
        stochastic=jnp.mean(ensemble_states.stochastic, axis=0)
    )

    # ===== 4. Compute Intrinsic Reward =====
    # Use ensemble disagreement as novelty signal
    clean_novelty = intrinsic_module.calculate_clean_novelty(
        ensemble_states.stochastic,
        config
    )

    # Update intrinsic state (cognitive wave)
    new_intrinsic_state, intrinsic_reward = intrinsic_module.update_cognitive_wave(
        carrier_state.intrinsic_state,
        clean_novelty.mean(),  # Average over batch
        config
    )

    # ===== 5. Add Transition to Replay Buffer =====
    # Add each transition in the batch
    new_buffer = memory_module.add_batch(
        carrier_state.buffer,
        observation,
        action,
        reward,
        done,
        new_rssm_state.deterministic,
        new_rssm_state.stochastic,
        intrinsic_reward * jnp.ones_like(reward)  # Broadcast to batch
    )

    # ===== 6. Update Carrier State =====
    new_carrier_state = carrier_state.replace(
        rng_key=carrier_state_key,
        buffer=new_buffer,
        intrinsic_state=new_intrinsic_state,
        current_rssm_state=new_rssm_state
    )

    # ===== 7. Collect Step Info =====
    info = {
        'extrinsic_reward': reward,
        'intrinsic_reward': intrinsic_reward,
        'done': done,
        'clean_novelty': clean_novelty.mean(),
        'boredom': new_intrinsic_state.boredom
    }

    return new_carrier_state, new_env_state, info


def collect_episode(
    carrier_state: Any,
    env_reset_fn: Callable,
    env_step_fn: Callable,
    config: DTCConfig,
    max_steps: int = 1000,
    deterministic: bool = False
) -> Tuple[Any, dict]:
    """
    Collect a full episode of experience.

    Args:
        carrier_state: Current carrier state
        env_reset_fn: Environment reset function () -> (state, obs)
        env_step_fn: Environment step function (state, action) -> (new_state, obs, reward, done)
        config: DTCConfig
        max_steps: Maximum episode length
        deterministic: If True, use deterministic actions

    Returns:
        new_carrier_state: Updated carrier with collected experiences
        episode_info: Dict with episode statistics
    """
    # Reset environment
    env_state, initial_obs = env_reset_fn()

    # Initialize RSSM state from initial observation
    carrier_state_key, reset_key = random.split(carrier_state.rng_key)

    # Encode initial observation
    ensemble_states, _ = rssm_module.ensemble_forward(
        carrier_state.rssm_params,
        config,
        carrier_state.current_rssm_state,
        jnp.zeros((config.local_batch_size, config.action_dim)),  # Dummy action
        observation=initial_obs,
        key=reset_key
    )

    # Average ensemble
    initial_rssm_state = RSSMState(
        deterministic=jnp.mean(ensemble_states.deterministic, axis=0),
        stochastic=jnp.mean(ensemble_states.stochastic, axis=0)
    )

    carrier_state = carrier_state.replace(
        current_rssm_state=initial_rssm_state,
        rng_key=carrier_state_key
    )

    # Collect steps
    total_reward = 0.0
    total_intrinsic = 0.0
    step_count = 0

    for _ in range(max_steps):
        carrier_state, env_state, step_info = collect_experience_step(
            carrier_state,
            env_state,
            env_step_fn,
            config,
            deterministic
        )

        total_reward += step_info['extrinsic_reward'].sum()
        total_intrinsic += step_info['intrinsic_reward']
        step_count += 1

        if step_info['done'].any():
            break

    episode_info = {
        'episode_reward': total_reward,
        'episode_intrinsic': total_intrinsic,
        'episode_length': step_count
    }

    return carrier_state, episode_info


# ============================================================================
# Dummy Environment (for testing only)
# ============================================================================

class DummyEnv:
    """
    Simple dummy environment for testing the pipeline.

    Replace this with a real JAX environment (Jumanji, Brax, etc.) or
    implement your environment in JAX for maximum TPU performance.
    """

    def __init__(self, obs_dim: int = 64, action_dim: int = 8):
        self.obs_dim = obs_dim
        self.action_dim = action_dim

    def reset(self, key: chex.PRNGKey) -> Tuple[chex.PRNGKey, chex.Array]:
        """Reset environment and return initial observation."""
        key, obs_key = random.split(key)
        obs = random.normal(obs_key, (self.obs_dim,))
        return key, obs

    def step(
        self,
        state: chex.PRNGKey,
        action: chex.Array
    ) -> Tuple[chex.PRNGKey, chex.Array, float, bool]:
        """
        Step environment (dummy implementation).

        Args:
            state: Environment state (just RNG key for dummy env)
            action: Action [action_dim]

        Returns:
            new_state: Updated state
            observation: New observation [obs_dim]
            reward: Scalar reward
            done: Terminal flag
        """
        state, obs_key, reward_key, done_key = random.split(state, 4)

        # Random observation
        obs = random.normal(obs_key, (self.obs_dim,))

        # Reward based on action magnitude (dummy)
        reward = -jnp.sum(action ** 2) * 0.1

        # Random termination (10% chance)
        done = random.uniform(done_key) < 0.1

        return state, obs, reward, done


def create_dummy_env_fns(config: DTCConfig):
    """
    Create dummy environment functions for testing.

    Returns:
        reset_fn: Function () -> (state, obs)
        step_fn: Function (state, action) -> (new_state, obs, reward, done)
    """
    env = DummyEnv(config.obs_dim, config.action_dim)

    def reset_fn(key: chex.PRNGKey):
        return env.reset(key)

    def step_fn(state, action):
        # Handle batched actions (take first action for dummy env)
        action_single = action[0] if action.ndim > 1 else action
        new_state, obs, reward, done = env.step(state, action_single)

        # Broadcast to batch size
        obs_batch = jnp.stack([obs] * config.local_batch_size)
        reward_batch = jnp.array([reward] * config.local_batch_size)
        done_batch = jnp.array([done] * config.local_batch_size)

        return new_state, obs_batch, reward_batch, done_batch

    return reset_fn, step_fn
