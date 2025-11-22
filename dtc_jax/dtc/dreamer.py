"""
Dream Rollout Engine (Imagination Loop).

Implements model-based imagination using the RSSM prior:
- Rollout trajectories in latent space (no environment needed)
- Use actor-critic to estimate values during imagination
- Compute advantages via GAE for policy gradient
- Adaptive horizon based on cognitive state (boredom)
"""

import jax
import jax.numpy as jnp
from jax import random
import chex
from typing import Tuple, NamedTuple
from flax import struct

from dtc_jax.configs.base_config import DTCConfig
from dtc_jax.dtc.dtc_types import RSSMState, DreamState, IntrinsicState
from dtc_jax.dtc import rssm as rssm_module


class DreamRollout(NamedTuple):
    """Container for imagination rollout data."""
    # Trajectories in latent space
    latent_h: chex.Array  # [horizon, batch, det_dim]
    latent_z: chex.Array  # [horizon, batch, stoch_dim]
    actions: chex.Array  # [horizon, batch, action_dim]

    # Predictions
    rewards: chex.Array  # [horizon, batch] - Predicted extrinsic rewards
    intrinsic_rewards: chex.Array  # [horizon, batch] - Predicted intrinsic rewards
    values: chex.Array  # [horizon, batch] - Value estimates
    log_probs: chex.Array  # [horizon, batch] - Action log probabilities

    # For advantage computation
    dones: chex.Array  # [horizon, batch] - Predicted terminal flags (usually False in imagination)


def dream_rollout(
    initial_state: RSSMState,
    rssm_params: dict,
    actor_critic_params: dict,
    config: DTCConfig,
    key: chex.PRNGKey,
    horizon: int,
    intrinsic_state: IntrinsicState
) -> Tuple[DreamRollout, chex.PRNGKey]:
    """
    Perform imagination rollout using RSSM prior and actor-critic.

    This is the core of model-based RL: we simulate trajectories entirely
    in latent space without interacting with the environment.

    Args:
        initial_state: Starting RSSM state [batch, det_dim] and [batch, stoch_dim]
        rssm_params: RSSM model parameters (ensemble)
        actor_critic_params: Actor-critic parameters
        config: DTCConfig
        key: PRNG key
        horizon: Rollout length (adaptive based on boredom)
        intrinsic_state: For computing intrinsic rewards during rollout

    Returns:
        rollout: DreamRollout containing trajectories and predictions
        key: Updated PRNG key
    """
    from dtc_jax.dtc.actor_critic import ActorCriticWithSalience
    from dtc_jax.dtc.intrinsic import calculate_clean_novelty

    batch_size = initial_state.deterministic.shape[0]

    # Initialize storage (pre-allocate for static shapes)
    latent_h_traj = []
    latent_z_traj = []
    actions_traj = []
    rewards_traj = []
    intrinsic_rewards_traj = []
    values_traj = []
    log_probs_traj = []

    # Initialize actor-critic
    actor_critic = ActorCriticWithSalience(config=config)

    # Current state
    current_state = initial_state

    # Rollout loop
    for t in range(horizon):
        # Get current latent state (concatenate h and z)
        latent_concat = jnp.concatenate([
            current_state.deterministic,  # [batch, det_dim]
            current_state.stochastic  # [batch, stoch_dim]
        ], axis=-1)

        # Add sequence dimension for salience pooling (we only have one timestep)
        latent_seq = latent_concat[:, None, :]  # [batch, 1, total_latent_dim]

        # Get action from actor-critic
        key, action_key = random.split(key)
        action_dist, value = actor_critic.apply(
            actor_critic_params,
            latent_seq
        )

        # Sample action
        key, sample_key = random.split(key)
        action_unbounded = action_dist.sample(seed=sample_key)
        action = jnp.tanh(action_unbounded)  # Squash to [-1, 1]

        # Compute log probability (with tanh correction)
        log_prob = action_dist.log_prob(action_unbounded)
        log_det_jacobian = jnp.sum(
            jnp.log(1 - action**2 + config.epsilon),
            axis=-1
        )
        log_prob = log_prob - log_det_jacobian

        # Predict next state using RSSM prior (imagination, no observation)
        # Use ensemble and average predictions
        key, step_key = random.split(key)

        # Forward with RSSM ensemble (using prior only)
        ensemble_next_states, ensemble_info = rssm_module.ensemble_forward(
            rssm_params,
            config,
            current_state,
            action,
            observation=None,  # No observation! This triggers prior-only mode in RSSM
            key=step_key
        )

        # Average ensemble predictions for next state
        next_state = RSSMState(
            deterministic=jnp.mean(
                ensemble_next_states.deterministic, axis=0
            ),  # [batch, det_dim]
            stochastic=jnp.mean(
                ensemble_next_states.stochastic, axis=0
            )  # [batch, stoch_dim]
        )

        # Predict reward (using RSSM's reward predictor)
        # Average across ensemble
        predicted_reward = jnp.mean(ensemble_info['predicted_rewards'], axis=0)  # [batch]

        # Compute intrinsic reward from ensemble disagreement
        intrinsic_reward = calculate_clean_novelty(
            ensemble_next_states.stochastic,  # [ensemble, batch, stoch_dim]
            config
        )  # [batch]

        # Store trajectory
        latent_h_traj.append(current_state.deterministic)
        latent_z_traj.append(current_state.stochastic)
        actions_traj.append(action)
        rewards_traj.append(predicted_reward)
        intrinsic_rewards_traj.append(intrinsic_reward)
        values_traj.append(value.squeeze(-1))  # [batch, 1] → [batch]
        log_probs_traj.append(log_prob)

        # Update state
        current_state = next_state

    # Stack trajectories: [horizon, batch, ...]
    rollout = DreamRollout(
        latent_h=jnp.stack(latent_h_traj, axis=0),
        latent_z=jnp.stack(latent_z_traj, axis=0),
        actions=jnp.stack(actions_traj, axis=0),
        rewards=jnp.stack(rewards_traj, axis=0),
        intrinsic_rewards=jnp.stack(intrinsic_rewards_traj, axis=0),
        values=jnp.stack(values_traj, axis=0),
        log_probs=jnp.stack(log_probs_traj, axis=0),
        dones=jnp.zeros((horizon, batch_size), dtype=jnp.bool_)  # No termination in imagination
    )

    return rollout, key


def dream_rollout_static(
    initial_state: RSSMState,
    rssm_params: dict,
    actor_critic_params: dict,
    config: DTCConfig,
    key: chex.PRNGKey,
    intrinsic_state: IntrinsicState
) -> Tuple[DreamRollout, chex.PRNGKey]:
    """
    Static-horizon dream rollout using lax.scan for efficiency.

    This version uses a fixed maximum horizon and masks out steps
    beyond the adaptive horizon. This ensures static shapes for XLA compilation.

    Args:
        initial_state: Starting RSSM state
        rssm_params: RSSM model parameters
        actor_critic_params: Actor-critic parameters
        config: DTCConfig
        key: PRNG key
        intrinsic_state: For computing adaptive horizon

    Returns:
        rollout: DreamRollout with static max_horizon shape
        key: Updated PRNG key
    """
    from dtc_jax.dtc.actor_critic import ActorCriticWithSalience
    from dtc_jax.dtc.intrinsic import calculate_clean_novelty, compute_adaptive_horizon

    # Compute adaptive horizon based on boredom
    adaptive_horizon = compute_adaptive_horizon(intrinsic_state, config)
    max_horizon = config.max_dream_horizon

    batch_size = initial_state.deterministic.shape[0]

    # Initialize actor-critic
    actor_critic = ActorCriticWithSalience(config=config)

    def step_fn(carry, t):
        """Single rollout step."""
        state, step_key = carry

        # Get current latent state
        latent_concat = jnp.concatenate([
            state.deterministic,
            state.stochastic
        ], axis=-1)
        latent_seq = latent_concat[:, None, :]  # [batch, 1, total_latent_dim]

        # Get action and value
        step_key, action_key = random.split(step_key)
        action_dist, value = actor_critic.apply(
            actor_critic_params,
            latent_seq
        )

        # Sample action
        step_key, sample_key = random.split(step_key)
        action_unbounded = action_dist.sample(seed=sample_key)
        action = jnp.tanh(action_unbounded)

        # Log probability
        log_prob = action_dist.log_prob(action_unbounded)
        log_det_jacobian = jnp.sum(
            jnp.log(1 - action**2 + config.epsilon),
            axis=-1
        )
        log_prob = log_prob - log_det_jacobian

        # RSSM forward (prior only)
        step_key, rssm_key = random.split(step_key)
        ensemble_next_states, ensemble_info = rssm_module.ensemble_forward(
            rssm_params,
            config,
            state,
            action,
            observation=None,  # Prior-only mode
            key=rssm_key
        )

        # Average ensemble
        next_state = RSSMState(
            deterministic=jnp.mean(ensemble_next_states.deterministic, axis=0),
            stochastic=jnp.mean(ensemble_next_states.stochastic, axis=0)
        )

        # Predictions
        predicted_reward = jnp.mean(ensemble_info['predicted_rewards'], axis=0)
        intrinsic_reward = calculate_clean_novelty(
            ensemble_next_states.stochastic,
            config
        )

        # Collect step data
        step_data = {
            'latent_h': state.deterministic,
            'latent_z': state.stochastic,
            'action': action,
            'reward': predicted_reward,
            'intrinsic_reward': intrinsic_reward,
            'value': value.squeeze(-1),
            'log_prob': log_prob
        }

        # Mask: only use this step if t < adaptive_horizon
        mask = (t < adaptive_horizon).astype(jnp.float32)
        step_data = jax.tree_map(lambda x: x * mask, step_data)

        return (next_state, step_key), step_data

    # Run scan
    key, scan_key = random.split(key)
    _, trajectory = jax.lax.scan(
        step_fn,
        (initial_state, scan_key),
        jnp.arange(max_horizon)
    )

    # Construct rollout
    rollout = DreamRollout(
        latent_h=trajectory['latent_h'],  # [max_horizon, batch, det_dim]
        latent_z=trajectory['latent_z'],
        actions=trajectory['action'],
        rewards=trajectory['reward'],
        intrinsic_rewards=trajectory['intrinsic_reward'],
        values=trajectory['value'],
        log_probs=trajectory['log_prob'],
        dones=jnp.zeros((max_horizon, batch_size), dtype=jnp.bool_)
    )

    return rollout, key


def compute_dream_advantages(
    rollout: DreamRollout,
    config: DTCConfig,
    intrinsic_weight: float = 1.0
) -> Tuple[chex.Array, chex.Array]:
    """
    Compute advantages and returns for dream rollout using GAE.

    Combines extrinsic and intrinsic rewards for value learning.

    Args:
        rollout: DreamRollout from imagination
        config: DTCConfig with gamma and lambda_gae
        intrinsic_weight: Weight for intrinsic rewards (typically 1.0)

    Returns:
        advantages: [horizon, batch] - GAE advantages
        returns: [horizon, batch] - Discounted returns
    """
    # Combine rewards
    total_rewards = rollout.rewards + intrinsic_weight * rollout.intrinsic_rewards

    # Transpose to [batch, horizon] for GAE computation
    rewards_T = total_rewards.T  # [batch, horizon]
    values_T = rollout.values.T
    dones_T = rollout.dones.T

    # Compute GAE
    from dtc_jax.dtc.actor_critic import compute_gae_advantages
    advantages, returns = compute_gae_advantages(
        rewards_T,
        values_T,
        dones_T,
        config.gamma,
        config.lambda_gae
    )

    # Transpose back to [horizon, batch]
    advantages = advantages.T
    returns = returns.T

    return advantages, returns


def compute_actor_critic_loss(
    rollout: DreamRollout,
    advantages: chex.Array,
    returns: chex.Array,
    config: DTCConfig
) -> Tuple[chex.Array, dict]:
    """
    Compute actor-critic loss from dream rollout.

    Loss = Policy Loss + Value Loss - Entropy Bonus

    Args:
        rollout: DreamRollout from imagination
        advantages: [horizon, batch] - GAE advantages
        returns: [horizon, batch] - Discounted returns
        config: DTCConfig with loss weights

    Returns:
        total_loss: Scalar loss
        metrics: Dictionary of loss components
    """
    # Policy loss (negative log likelihood weighted by advantages)
    # L_policy = -E[log π(a|s) * A(s,a)]
    policy_loss = -jnp.mean(rollout.log_probs * jax.lax.stop_gradient(advantages))

    # Value loss (MSE between predicted and actual returns)
    # L_value = E[(V(s) - R)^2]
    value_loss = jnp.mean((rollout.values - jax.lax.stop_gradient(returns))**2)

    # Entropy bonus (for exploration)
    # Higher entropy = more exploration
    # Note: We don't have entropy in rollout, so we approximate it
    # For Gaussian policy: entropy ≈ log(std)
    # We'll skip this for now and add it in the full training loop
    entropy_bonus = 0.0

    # Total loss
    total_loss = policy_loss + config.value_loss_weight * value_loss - config.entropy_weight * entropy_bonus

    metrics = {
        'dream_policy_loss': policy_loss,
        'dream_value_loss': value_loss,
        'dream_total_loss': total_loss,
        'dream_mean_advantage': jnp.mean(advantages),
        'dream_mean_return': jnp.mean(returns),
        'dream_mean_value': jnp.mean(rollout.values),
    }

    return total_loss, metrics
