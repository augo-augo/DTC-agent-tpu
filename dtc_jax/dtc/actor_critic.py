"""
Actor-Critic Networks with Salience Pooling.

Implements the "Soft Attention Workspace" from Hephaestus spec:
- Salience pooling to compress latent states into global context
- Static shapes (no dynamic attention)
- Actor: Continuous action policy (Gaussian)
- Critic: Value estimation for RL
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax import struct
from typing import Tuple
import chex
import distrax

from dtc_jax.configs.base_config import DTCConfig


class SaliencePooling(nn.Module):
    """
    Soft attention mechanism that compresses a sequence into a single vector.

    The agent learns what is "salient" (interesting) based on novelty signals.
    This creates a bottleneck that forces compression of scene information.

    Architecture:
        Z ∈ R^[batch, seq_len, dim] → Salience Net → s ∈ R^[batch, seq_len, 1]
        W = softmax(s)
        z_global = Σ(Z ⊙ W)
    """
    hidden_dim: int = 256

    @nn.compact
    def __call__(self, sequence: chex.Array) -> chex.Array:
        """
        Compress sequence into single context vector via salience weighting.

        Args:
            sequence: [batch, seq_len, dim] - Sequence of latent states

        Returns:
            context: [batch, dim] - Compressed global context vector
        """
        batch_size, seq_len, dim = sequence.shape

        # Salience network: learns what's "interesting"
        # [batch, seq_len, dim] → [batch, seq_len, 1]
        salience = nn.Dense(self.hidden_dim, name='salience_1')(sequence)
        salience = nn.relu(salience)
        salience = nn.Dense(1, name='salience_2')(salience)  # [batch, seq_len, 1]

        # Compute attention weights
        weights = nn.softmax(salience, axis=1)  # [batch, seq_len, 1]

        # Weighted sum: compress sequence into single vector
        context = jnp.sum(sequence * weights, axis=1)  # [batch, dim]

        return context


class Actor(nn.Module):
    """
    Continuous action policy network.

    Outputs mean and log_std for a multivariate Gaussian distribution.
    Uses tanh squashing for bounded actions.
    """
    config: DTCConfig

    @nn.compact
    def __call__(self, state: chex.Array) -> distrax.Distribution:
        """
        Compute action distribution given state.

        Args:
            state: [batch, state_dim] - Compressed latent state

        Returns:
            action_dist: Multivariate Gaussian distribution over actions
        """
        # MLP backbone
        x = state
        for i, hidden_dim in enumerate(self.config.actor_hidden_dims):
            x = nn.Dense(hidden_dim, name=f'dense_{i}')(x)
            x = nn.relu(x)

        # Separate heads for mean and log_std
        mean = nn.Dense(self.config.action_dim, name='mean')(x)
        log_std = nn.Dense(self.config.action_dim, name='log_std')(x)

        # Clamp log_std for numerical stability (prevents exploding/vanishing variance)
        log_std = jnp.clip(
            log_std,
            self.config.log_std_min,
            self.config.log_std_max
        )

        std = jnp.exp(log_std)

        # Multivariate Normal distribution
        action_dist = distrax.MultivariateNormalDiag(loc=mean, scale_diag=std)

        # Note: Tanh squashing happens in the action selection, not here
        # This keeps the distribution differentiable for policy gradient

        return action_dist

    def sample_action(
        self,
        state: chex.Array,
        key: chex.PRNGKey,
        deterministic: bool = False
    ) -> Tuple[chex.Array, chex.PRNGKey]:
        """
        Sample action from policy.

        Args:
            state: [batch, state_dim] or [state_dim]
            key: PRNG key
            deterministic: If True, return mean action (for evaluation)

        Returns:
            action: [batch, action_dim] or [action_dim] - Sampled action in [-1, 1]
            key: Updated PRNG key
        """
        action_dist = self(state)

        if deterministic:
            # Use mean for deterministic evaluation
            action_unbounded = action_dist.mode()
        else:
            # Sample from distribution
            key, sample_key = jax.random.split(key)
            action_unbounded = action_dist.sample(seed=sample_key)

        # Squash to [-1, 1] using tanh
        action = jnp.tanh(action_unbounded)

        return action, key


class Critic(nn.Module):
    """
    Value function network.

    Estimates V(s) - the expected return from state s.
    Used for advantage estimation and policy gradient baseline.
    """
    config: DTCConfig

    @nn.compact
    def __call__(self, state: chex.Array) -> chex.Array:
        """
        Compute value estimate given state.

        Args:
            state: [batch, state_dim] - Compressed latent state

        Returns:
            value: [batch, 1] - State value estimate
        """
        # MLP backbone
        x = state
        for i, hidden_dim in enumerate(self.config.critic_hidden_dims):
            x = nn.Dense(hidden_dim, name=f'dense_{i}')(x)
            x = nn.relu(x)

        # Value head (single output)
        value = nn.Dense(1, name='value')(x)

        return value


class ActorCriticWithSalience(nn.Module):
    """
    Combined Actor-Critic with Salience Pooling.

    This module integrates:
    1. Salience pooling to compress latent sequences
    2. Actor network for policy
    3. Critic network for value estimation

    The salience mechanism allows the agent to focus on relevant
    parts of its experience based on learned importance.
    """
    config: DTCConfig

    def setup(self):
        """Initialize sub-modules."""
        self.salience_pool = SaliencePooling(hidden_dim=self.config.hidden_dim)
        self.actor = Actor(config=self.config)
        self.critic = Critic(config=self.config)

    def __call__(
        self,
        latent_sequence: chex.Array,
        return_pooled: bool = False
    ) -> Tuple[distrax.Distribution, chex.Array]:
        """
        Forward pass through salience pooling and actor-critic.

        Args:
            latent_sequence: [batch, seq_len, latent_dim] - Sequence of RSSM states
            return_pooled: If True, also return the pooled context vector

        Returns:
            action_dist: Multivariate Gaussian over actions
            value: [batch, 1] - Value estimate
            pooled_context (optional): [batch, latent_dim] - Compressed context
        """
        # Compress sequence into global context via salience pooling
        # [batch, seq_len, latent_dim] → [batch, latent_dim]
        pooled_context = self.salience_pool(latent_sequence)

        # Actor and critic operate on compressed representation
        action_dist = self.actor(pooled_context)
        value = self.critic(pooled_context)

        if return_pooled:
            return action_dist, value, pooled_context
        else:
            return action_dist, value

    def get_action(
        self,
        latent_sequence: chex.Array,
        key: chex.PRNGKey,
        deterministic: bool = False
    ) -> Tuple[chex.Array, chex.PRNGKey]:
        """
        Sample action from policy given latent sequence.

        Args:
            latent_sequence: [batch, seq_len, latent_dim] or [seq_len, latent_dim]
            key: PRNG key
            deterministic: If True, return mean action

        Returns:
            action: [batch, action_dim] - Sampled action in [-1, 1]
            key: Updated PRNG key
        """
        # Add batch dim if needed
        needs_unbatch = False
        if latent_sequence.ndim == 2:
            latent_sequence = latent_sequence[None, ...]  # [1, seq_len, latent_dim]
            needs_unbatch = True

        # Compress via salience pooling
        pooled_context = self.salience_pool(latent_sequence)

        # Sample action
        action, key = self.actor.sample_action(pooled_context, key, deterministic)

        # Remove batch dim if we added it
        if needs_unbatch:
            action = action[0]

        return action, key

    def evaluate_actions(
        self,
        latent_sequence: chex.Array,
        actions: chex.Array
    ) -> Tuple[chex.Array, chex.Array, chex.Array]:
        """
        Evaluate actions under current policy.

        Used for computing policy gradients and advantages.

        Args:
            latent_sequence: [batch, seq_len, latent_dim]
            actions: [batch, action_dim] - Actions to evaluate

        Returns:
            log_probs: [batch] - Log probability of actions
            values: [batch, 1] - Value estimates
            entropy: [batch] - Policy entropy (for exploration bonus)
        """
        # Get policy and value
        action_dist, values = self(latent_sequence)

        # Compute log probability
        # Note: Actions are in [-1, 1] (tanh squashed)
        # We need to compute log_prob of the unbounded action
        # atanh transformation: unbounded = atanh(bounded)
        actions_unbounded = jnp.arctanh(jnp.clip(actions, -0.999, 0.999))
        log_probs = action_dist.log_prob(actions_unbounded)

        # Correction for tanh squashing (change of variables)
        # log π(a|s) = log π_unbounded(f^-1(a)) - log|det(df/da)|
        # where f(x) = tanh(x), so df/dx = 1 - tanh^2(x)
        log_det_jacobian = jnp.sum(
            jnp.log(1 - actions**2 + self.config.epsilon),
            axis=-1
        )
        log_probs = log_probs - log_det_jacobian

        # Compute entropy for exploration bonus
        entropy = action_dist.entropy()

        return log_probs, values, entropy


# ============================================================================
# Utility functions for creating and managing actor-critic
# ============================================================================

def create_actor_critic_params(
    config: DTCConfig,
    key: chex.PRNGKey,
    batch_size: int = 1,
    seq_len: int = 1
) -> dict:
    """
    Initialize actor-critic parameters.

    Args:
        config: DTCConfig
        key: PRNG key
        batch_size: Batch size for dummy input
        seq_len: Sequence length for dummy input

    Returns:
        params: Pytree of network parameters
    """
    # Create dummy input for initialization
    latent_dim = config.latent_dim_deterministic + config.latent_dim_stochastic
    dummy_latent = jnp.zeros((batch_size, seq_len, latent_dim))

    # Initialize model
    model = ActorCriticWithSalience(config=config)
    params = model.init(key, dummy_latent)

    return params


def compute_gae_advantages(
    rewards: chex.Array,
    values: chex.Array,
    dones: chex.Array,
    gamma: float,
    lambda_gae: float
) -> Tuple[chex.Array, chex.Array]:
    """
    Compute Generalized Advantage Estimation (GAE).

    GAE(γ, λ) balances bias-variance tradeoff in advantage estimation.

    Args:
        rewards: [batch, seq_len] - Rewards
        values: [batch, seq_len] - Value estimates
        dones: [batch, seq_len] - Terminal flags
        gamma: Discount factor
        lambda_gae: GAE lambda parameter

    Returns:
        advantages: [batch, seq_len] - Advantage estimates
        returns: [batch, seq_len] - Discounted returns (targets for value function)
    """
    batch_size, seq_len = rewards.shape

    # Compute TD residuals: δ_t = r_t + γV(s_{t+1}) - V(s_t)
    # Note: V(s_{T+1}) = 0 (terminal state)
    next_values = jnp.concatenate([values[:, 1:], jnp.zeros((batch_size, 1))], axis=1)
    next_values = next_values * (1.0 - dones)  # Zero out terminal states

    td_residuals = rewards + gamma * next_values - values

    # Compute GAE using reverse scan
    # A_t = Σ_{l=0}^∞ (γλ)^l δ_{t+l}
    def gae_step(next_advantage, inputs):
        td_residual, done = inputs
        advantage = td_residual + gamma * lambda_gae * next_advantage * (1.0 - done)
        return advantage, advantage

    # Reverse scan (from T-1 to 0)
    _, advantages = jax.lax.scan(
        gae_step,
        jnp.zeros(batch_size),  # Initial "next advantage" (at T+1)
        (td_residuals, dones),
        reverse=True,
        unroll=1
    )

    # Transpose back to [batch, seq_len]
    advantages = advantages.T

    # Compute returns: R_t = A_t + V(s_t)
    returns = advantages + values

    return advantages, returns
