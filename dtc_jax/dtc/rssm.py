"""
Recurrent State-Space Model (RSSM) with Vectorized Ensemble.

This module implements the core world model for DTC 3.0. The ensemble is implemented
via jax.vmap over stacked parameters, NOT as a Python list of separate modules.

Key Features:
- Single RSSM module definition, vmapped across ensemble dimension
- GRU-based deterministic state with stochastic latent
- Numerical stability via log_std clipping
- Distrax distributions for proper gradient flow
"""

from typing import Tuple
import jax
import jax.numpy as jnp
import flax.linen as nn
import distrax
import chex

from dtc_jax.dtc.types import RSSMState, RSSMOutput, EnsembleOutput, PRNGKey, Params
from dtc_jax.configs.base_config import DTCConfig


class RSSMCell(nn.Module):
    """Single RSSM cell with deterministic GRU and stochastic latent.

    The deterministic state is updated via GRU recurrence.
    The stochastic state is sampled from a learned distribution.
    """

    config: DTCConfig

    @nn.compact
    def __call__(
        self,
        prev_state: RSSMState,
        action: chex.Array,
        observation: chex.Array = None,
        sample_key: PRNGKey = None,
        training: bool = True,
    ) -> RSSMOutput:
        """
        Forward pass through RSSM.

        Args:
            prev_state: Previous RSSM state
            action: Action taken (shape: [batch, action_dim])
            observation: Optional observation for posterior (shape: [batch, obs_dim])
            sample_key: PRNG key for sampling (can be None during init)
            training: If True, sample from distribution; else use mean

        Returns:
            RSSMOutput containing new state and distribution parameters
        """
        # ========== Deterministic State Update (GRU) ==========
        # Concatenate previous stochastic state with action as input to GRU
        gru_input = jnp.concatenate([prev_state.stochastic, action], axis=-1)

        # GRU recurrence: h_t = GRU(h_{t-1}, [z_{t-1}, a_t])
        # Note: Flax GRUCell returns (new_carry, new_carry) tuple
        deterministic, _ = nn.GRUCell(
            features=self.config.latent_dim_deterministic,
            name="gru_cell"
        )(prev_state.deterministic, gru_input)

        # ========== Prior Distribution p(z_t | h_t) ==========
        # The prior only depends on the deterministic state
        prior_mean, prior_std = self._build_distribution(
            deterministic,
            prefix="prior"
        )

        # ========== Posterior Distribution q(z_t | h_t, o_t) ==========
        if observation is not None:
            # When we have observations, compute posterior that conditions on them
            posterior_input = jnp.concatenate([deterministic, observation], axis=-1)
            posterior_mean, posterior_std = self._build_distribution(
                posterior_input,
                prefix="posterior"
            )
            # During training, sample from posterior (for representation learning)
            sample_mean = posterior_mean
            sample_std = posterior_std
        else:
            # When dreaming (no observations), use the prior
            posterior_mean = prior_mean
            posterior_std = prior_std
            sample_mean = prior_mean
            sample_std = prior_std

        # ========== Sample Stochastic State ==========
        if training and sample_key is not None:
            # Sample from the distribution using reparameterization trick
            eps = jax.random.normal(sample_key, shape=sample_mean.shape)
            stochastic = sample_mean + sample_std * eps
        else:
            # Use mean for deterministic inference (or during init when key is None)
            stochastic = sample_mean

        # ========== Construct New State ==========
        new_state = RSSMState(
            deterministic=deterministic,
            stochastic=stochastic,
        )

        return RSSMOutput(
            state=new_state,
            prior_mean=prior_mean,
            prior_std=prior_std,
            posterior_mean=posterior_mean,
            posterior_std=posterior_std,
        )

    def _build_distribution(
        self,
        features: chex.Array,
        prefix: str,
    ) -> Tuple[chex.Array, chex.Array]:
        """
        Build distribution parameters with stability clipping.

        Args:
            features: Input features
            prefix: Name prefix for layers ("prior" or "posterior")

        Returns:
            Tuple of (mean, std) with clipped std for numerical stability
        """
        # Hidden layer
        hidden = nn.Dense(
            features=self.config.hidden_dim,
            name=f"{prefix}_hidden"
        )(features)
        hidden = nn.relu(hidden)

        # Output mean
        mean = nn.Dense(
            features=self.config.latent_dim_stochastic,
            name=f"{prefix}_mean"
        )(hidden)

        # Output log_std with CRITICAL clipping for stability
        log_std = nn.Dense(
            features=self.config.latent_dim_stochastic,
            name=f"{prefix}_log_std"
        )(hidden)

        # STABILITY HACK: Prevent exploding/collapsing gradients during dreaming
        log_std = jnp.clip(
            log_std,
            self.config.log_std_min,
            self.config.log_std_max
        )

        std = jnp.exp(log_std)

        return mean, std

    def init_state(self, batch_size: int) -> RSSMState:
        """Initialize RSSM state with zeros."""
        return RSSMState(
            deterministic=jnp.zeros((batch_size, self.config.latent_dim_deterministic)),
            stochastic=jnp.zeros((batch_size, self.config.latent_dim_stochastic)),
        )


class RSSMEnsemble:
    """Vectorized ensemble of RSSM models using vmap.

    This class manages the ensemble by stacking parameters and using vmap
    for parallel execution, rather than maintaining separate model instances.
    """

    def __init__(self, config: DTCConfig):
        self.config = config
        self.rssm_module = RSSMCell(config)

    def init(self, key: PRNGKey, batch_size: int = 1) -> Params:
        """
        Initialize ensemble parameters by stacking single model params.

        Args:
            key: PRNG key for initialization
            batch_size: Batch size for initialization

        Returns:
            Stacked parameters with shape [ensemble_size, ...] on axis 0
        """
        # Create dummy inputs for initialization
        dummy_state = self.rssm_module.init_state(batch_size)
        dummy_action = jnp.zeros((batch_size, self.config.action_dim))
        dummy_obs = jnp.zeros((batch_size, self.config.obs_dim))

        # Initialize single model parameters
        key, init_key = jax.random.split(key)
        single_params = self.rssm_module.init(
            init_key,
            prev_state=dummy_state,
            action=dummy_action,
            observation=dummy_obs,
            sample_key=None,  # No sampling during initialization
            training=False,  # Use mean during init
        )

        # Stack parameters ensemble_size times along axis 0
        # Each ensemble member gets identical initialization (will diverge during training)
        ensemble_params = jax.tree.map(
            lambda x: jnp.stack([x] * self.config.ensemble_size, axis=0),
            single_params
        )

        return ensemble_params

    def apply_ensemble(
        self,
        params: Params,
        prev_states: RSSMState,
        actions: chex.Array,
        observations: chex.Array = None,
        keys: PRNGKey = None,
        training: bool = True,
    ) -> EnsembleOutput:
        """
        Apply RSSM across ensemble using vmap.

        Args:
            params: Stacked parameters [ensemble_size, ...]
            prev_states: Previous states [ensemble_size, batch, ...]
            actions: Actions [batch, action_dim] (broadcast across ensemble)
            observations: Observations [batch, obs_dim] (broadcast across ensemble)
            keys: PRNG keys [ensemble_size, 2] for sampling
            training: Training mode flag

        Returns:
            EnsembleOutput with all outputs stacked along ensemble dimension
        """
        # If keys not provided, create dummy keys
        if keys is None:
            keys = jax.random.split(jax.random.PRNGKey(0), self.config.ensemble_size)

        # Vmap over ensemble dimension (axis 0 for params, states, and keys)
        # Actions and observations are broadcast (None means replicate)
        vmapped_apply = jax.vmap(
            lambda p, s, k: self.rssm_module.apply(
                p,
                prev_state=s,
                action=actions,  # Broadcast across ensemble
                observation=observations,  # Broadcast across ensemble
                sample_key=k,
                training=training,
            ),
            in_axes=(0, 0, 0),  # Vmap over params, states, keys
            out_axes=0,  # Stack outputs along axis 0
        )

        # Execute vmapped function
        outputs = vmapped_apply(params, prev_states, keys)

        # Package into EnsembleOutput
        return EnsembleOutput(
            states=outputs.state,
            prior_means=outputs.prior_mean,
            prior_stds=outputs.prior_std,
            posterior_means=outputs.posterior_mean,
            posterior_stds=outputs.posterior_std,
        )

    def init_ensemble_states(self, batch_size: int) -> RSSMState:
        """
        Initialize ensemble of RSSM states.

        Args:
            batch_size: Batch size

        Returns:
            RSSMState with shape [ensemble_size, batch, ...]
        """
        single_state = self.rssm_module.init_state(batch_size)

        # Stack states for ensemble
        ensemble_states = RSSMState(
            deterministic=jnp.stack(
                [single_state.deterministic] * self.config.ensemble_size,
                axis=0
            ),
            stochastic=jnp.stack(
                [single_state.stochastic] * self.config.ensemble_size,
                axis=0
            ),
        )

        return ensemble_states


def compute_rssm_loss(
    prior_means: chex.Array,
    prior_stds: chex.Array,
    posterior_means: chex.Array,
    posterior_stds: chex.Array,
    config: DTCConfig,
) -> Tuple[chex.Array, dict]:
    """
    Compute ELBO loss for RSSM training.

    The loss encourages:
    1. Accurate posterior predictions (reconstruction via KL)
    2. Regularization via KL divergence to prior

    Args:
        prior_means: Prior means [ensemble, batch, stoch_dim]
        prior_stds: Prior stds [ensemble, batch, stoch_dim]
        posterior_means: Posterior means [ensemble, batch, stoch_dim]
        posterior_stds: Posterior stds [ensemble, batch, stoch_dim]
        config: Configuration

    Returns:
        Tuple of (loss, info_dict)
    """
    # Build distributions using distrax
    prior_dist = distrax.MultivariateNormalDiag(
        loc=prior_means,
        scale_diag=prior_stds
    )

    posterior_dist = distrax.MultivariateNormalDiag(
        loc=posterior_means,
        scale_diag=posterior_stds
    )

    # KL divergence: KL(posterior || prior)
    # This encourages the posterior to not deviate too far from the prior
    kl_div = posterior_dist.kl_divergence(prior_dist)

    # Average over ensemble and batch
    kl_loss = jnp.mean(kl_div)

    # Total loss with KL weighting
    total_loss = config.kl_weight * kl_loss

    # Info dict for logging
    info = {
        "rssm/kl_loss": kl_loss,
        "rssm/total_loss": total_loss,
        "rssm/prior_std_mean": jnp.mean(prior_stds),
        "rssm/posterior_std_mean": jnp.mean(posterior_stds),
    }

    return total_loss, info
