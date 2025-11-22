"""
Recurrent State-Space Model (RSSM) with Vectorized Ensemble.

The RSSM is the core world model that learns:
  h_t = f(h_{t-1}, z_{t-1}, a_{t-1})  [Deterministic state via GRU]
  z_t ~ p(z_t | h_t)                   [Stochastic state via prior]
  z_t ~ q(z_t | h_t, o_t)              [Posterior for training]

Critical Implementation Details:
  1. Single module definition, vmapped over ensemble dimension
  2. Log-std clipping for numerical stability
  3. Separate prior and posterior networks
  4. All forward passes support bfloat16, but distributions use float32
"""

import jax
import jax.numpy as jnp
from flax import linen as nn
import distrax
from typing import Tuple

from dtc_jax.dtc.dtc_types import RSSMState, Array, PRNGKey
from dtc_jax.configs.base_config import DTCConfig


class RSSMCell(nn.Module):
    """Single RSSM recurrent cell.

    This module will be vmapped across ensemble members.
    """
    config: DTCConfig

    def setup(self):
        """Initialize networks."""
        # GRU cell for deterministic state transition
        # Input: [z_{t-1}, a_{t-1}], Output: h_t
        self.gru_cell = nn.GRUCell(
            features=self.config.gru_hidden_dim,
            name='gru'
        )

        # Prior network: p(z_t | h_t)
        # Predicts next stochastic state from deterministic state
        self.prior_network = self._make_mlp(
            hidden_dims=[self.config.hidden_dim],
            output_dim=2 * self.config.latent_dim_stochastic,  # mean + log_std
            name='prior'
        )

        # Posterior network: q(z_t | h_t, o_t)
        # Infers stochastic state from deterministic state and observation
        self.posterior_network = self._make_mlp(
            hidden_dims=[self.config.hidden_dim],
            output_dim=2 * self.config.latent_dim_stochastic,  # mean + log_std
            name='posterior'
        )

        # Observation decoder: p(o_t | h_t, z_t)
        # Reconstructs observation from full state
        self.decoder = self._make_mlp(
            hidden_dims=[self.config.hidden_dim, self.config.hidden_dim],
            output_dim=self.config.obs_dim,
            name='decoder'
        )

        # Observation encoder: embed observation for posterior
        self.obs_encoder = self._make_mlp(
            hidden_dims=[self.config.hidden_dim],
            output_dim=self.config.hidden_dim,
            name='obs_encoder'
        )

        # Reward predictor: p(r_t | h_t, z_t)
        # Predicts extrinsic reward from full state
        self.reward_predictor = self._make_mlp(
            hidden_dims=[self.config.hidden_dim],
            output_dim=1,  # Scalar reward
            name='reward_predictor'
        )

        # Continue predictor: p(continue_t | h_t, z_t)
        # Predicts whether episode continues (1 - done)
        self.continue_predictor = self._make_mlp(
            hidden_dims=[self.config.hidden_dim],
            output_dim=1,  # Scalar logit (sigmoid for probability)
            name='continue_predictor'
        )

    def _make_mlp(self, hidden_dims: list, output_dim: int, name: str) -> nn.Sequential:
        """Create MLP with ELU activations."""
        layers = []
        for i, dim in enumerate(hidden_dims):
            layers.append(nn.Dense(dim, name=f'{name}_layer{i}'))
            layers.append(nn.elu)
        layers.append(nn.Dense(output_dim, name=f'{name}_output'))
        return nn.Sequential(layers)

    def _split_dist_params(self, params: Array) -> Tuple[Array, Array]:
        """Split concatenated mean and log_std, apply stability clipping.

        Args:
            params: Array of shape [..., 2 * dim]

        Returns:
            mean: Array of shape [..., dim]
            std: Array of shape [..., dim] (already exp'd, ready for distribution)
        """
        dim = params.shape[-1] // 2
        mean = params[..., :dim]
        log_std = params[..., dim:]

        # CRITICAL: Clip log_std to prevent numerical instability
        log_std = jnp.clip(
            log_std,
            self.config.log_std_min,
            self.config.log_std_max
        )

        std = jnp.exp(log_std)
        return mean, std

    def init_state(self, batch_size: int) -> RSSMState:
        """Create zero-initialized RSSM state."""
        return RSSMState.zeros(
            batch_size=batch_size,
            det_dim=self.config.gru_hidden_dim,
            stoch_dim=self.config.latent_dim_stochastic
        )

    def __call__(
        self,
        prev_state: RSSMState,
        action: Array,
        observation: Array = None,
        key: PRNGKey = None,
        use_sample: bool = True,
    ) -> Tuple[RSSMState, dict]:
        """Single step of RSSM.

        Args:
            prev_state: Previous RSSM state (h_{t-1}, z_{t-1})
            action: Action taken at t-1, shape [batch, action_dim]
            observation: Current observation o_t, shape [batch, obs_dim] (optional)
            key: PRNG key for sampling (required if use_sample=True)
            use_sample: If True, sample from distribution; else use mean

        Returns:
            next_state: New RSSM state (h_t, z_t)
            info: Dictionary with intermediate values for loss computation
        """
        batch_size = action.shape[0]

        # ===== Deterministic State Update (GRU) =====
        # Combine previous stochastic state and action as input
        gru_input = jnp.concatenate([prev_state.stochastic, action], axis=-1)

        # Update deterministic state: h_t = GRU(h_{t-1}, [z_{t-1}, a_{t-1}])
        new_deterministic, _ = self.gru_cell(prev_state.deterministic, gru_input)

        # ===== Prior: p(z_t | h_t) =====
        prior_params = self.prior_network(new_deterministic)
        prior_mean, prior_std = self._split_dist_params(prior_params)
        prior_dist = distrax.MultivariateNormalDiag(
            loc=prior_mean.astype(jnp.float32),
            scale_diag=prior_std.astype(jnp.float32)
        )

        # ===== Posterior: q(z_t | h_t, o_t) =====
        if observation is not None:
            # Encode observation
            obs_embed = self.obs_encoder(observation)

            # Combine with deterministic state
            posterior_input = jnp.concatenate([new_deterministic, obs_embed], axis=-1)
            posterior_params = self.posterior_network(posterior_input)
            posterior_mean, posterior_std = self._split_dist_params(posterior_params)
            posterior_dist = distrax.MultivariateNormalDiag(
                loc=posterior_mean.astype(jnp.float32),
                scale_diag=posterior_std.astype(jnp.float32)
            )

            # During training, sample from posterior
            if use_sample:
                assert key is not None, "Must provide PRNG key for sampling"
                new_stochastic = posterior_dist.sample(seed=key)
            else:
                new_stochastic = posterior_mean
        else:
            # During imagination (no observation), sample from prior
            posterior_dist = None
            if use_sample:
                assert key is not None, "Must provide PRNG key for sampling"
                new_stochastic = prior_dist.sample(seed=key)
            else:
                new_stochastic = prior_mean

        # ===== Create New State =====
        new_state = RSSMState(
            deterministic=new_deterministic,
            stochastic=new_stochastic
        )

        # ===== Decode Observation =====
        # Reconstruct observation from full state
        state_concat = jnp.concatenate([new_deterministic, new_stochastic], axis=-1)
        reconstructed_obs = self.decoder(state_concat)

        # ===== Predict Reward and Continue =====
        # Reward prediction: scalar extrinsic reward
        predicted_reward = self.reward_predictor(state_concat).squeeze(-1)  # [batch]

        # Continue prediction: probability that episode continues (1 - done)
        continue_logit = self.continue_predictor(state_concat).squeeze(-1)  # [batch]
        continue_prob = jax.nn.sigmoid(continue_logit)

        # ===== Collect Info for Loss Computation =====
        info = {
            'prior_mean': prior_mean,
            'prior_std': prior_std,
            'prior_dist': prior_dist,
            'posterior_mean': posterior_mean if observation is not None else None,
            'posterior_std': posterior_std if observation is not None else None,
            'posterior_dist': posterior_dist,
            'reconstructed_obs': reconstructed_obs,
            'predicted_rewards': predicted_reward,
            'continue_prob': continue_prob,
        }

        return new_state, info


# ===== Ensemble Creation and Management =====

def create_ensemble_params(
    config: DTCConfig,
    key: PRNGKey,
    batch_size: int = 1,
) -> Tuple[dict, RSSMState]:
    """Initialize ensemble RSSM parameters via stacking and vmap.

    This creates ONE module definition, initializes it once, then stacks
    the parameters N times along axis 0 for vectorized ensemble execution.

    Args:
        config: DTC configuration
        key: PRNG key for initialization
        batch_size: Batch size for dummy initialization input

    Returns:
        ensemble_params: Stacked parameters, shape [ensemble_size, ...]
        init_state: Initial RSSM state template
    """
    # Create single RSSM module
    rssm = RSSMCell(config)

    # Create dummy inputs for initialization
    init_state = RSSMState.zeros(
        batch_size=batch_size,
        det_dim=config.gru_hidden_dim,
        stoch_dim=config.latent_dim_stochastic
    )
    dummy_action = jnp.zeros((batch_size, config.action_dim))
    dummy_obs = jnp.zeros((batch_size, config.obs_dim))

    # Initialize single set of parameters
    init_key, sample_key = jax.random.split(key)
    single_params = rssm.init(
        init_key,
        prev_state=init_state,
        action=dummy_action,
        observation=dummy_obs,
        key=sample_key,
        use_sample=True,
    )

    # Stack parameters N times for ensemble
    # Each leaf in the pytree gets stacked along a new axis 0
    ensemble_params = jax.tree.map(
        lambda x: jnp.stack([x] * config.ensemble_size, axis=0),
        single_params
    )

    return ensemble_params, init_state


def ensemble_forward(
    params: dict,
    config: DTCConfig,
    prev_state: RSSMState,
    action: Array,
    observation: Array = None,
    key: PRNGKey = None,
    use_sample: bool = True,
) -> Tuple[RSSMState, dict]:
    """Forward pass for entire ensemble using vmap.

    This vmaps the RSSM.apply function over the ensemble dimension.

    Args:
        params: Ensemble parameters, shape [ensemble_size, ...]
        config: DTC configuration
        prev_state: Previous RSSM state (NOT ensemble-batched)
        action: Action, shape [batch, action_dim]
        observation: Observation, shape [batch, obs_dim] (optional)
        key: PRNG key (will be split for each ensemble member)
        use_sample: Whether to sample from distributions

    Returns:
        ensemble_states: New states for all ensemble members
        ensemble_info: Info dict with ensemble dimension
    """
    rssm = RSSMCell(config)

    # Split key for each ensemble member
    if key is not None:
        ensemble_keys = jax.random.split(key, config.ensemble_size)
    else:
        ensemble_keys = None

    # Vmap over ensemble dimension (axis 0 of params)
    # prev_state, action, observation are broadcasted (None = not vmapped)
    # ensemble_keys are vmapped (0 = vmap over axis 0)
    vmapped_apply = jax.vmap(
        lambda p, k: rssm.apply(
            p,
            prev_state=prev_state,
            action=action,
            observation=observation,
            key=k,
            use_sample=use_sample,
        ),
        in_axes=(0, 0 if ensemble_keys is not None else None),
        out_axes=(0, 0)  # Both outputs get ensemble dimension
    )

    ensemble_states, ensemble_info = vmapped_apply(params, ensemble_keys)

    return ensemble_states, ensemble_info


def compute_ensemble_predictions(
    ensemble_info: dict,
) -> Tuple[Array, Array]:
    """Extract mean and std predictions from ensemble.

    Used for uncertainty quantification and intrinsic reward.

    Args:
        ensemble_info: Info dict from ensemble_forward, with ensemble dimension

    Returns:
        ensemble_means: Means from each ensemble member, shape [ensemble_size, batch, dim]
        ensemble_stds: Stds from each ensemble member, shape [ensemble_size, batch, dim]
    """
    # Extract prior means and stds (these have ensemble dimension from vmap)
    ensemble_means = ensemble_info['prior_mean']  # [ensemble_size, batch, stoch_dim]
    ensemble_stds = ensemble_info['prior_std']  # [ensemble_size, batch, stoch_dim]

    return ensemble_means, ensemble_stds


# ===== Training Utilities =====

def compute_rssm_loss(
    params: dict,
    config: DTCConfig,
    observations: Array,
    actions: Array,
    key: PRNGKey,
) -> Tuple[Array, dict]:
    """Compute ELBO loss for RSSM training on a sequence.

    Args:
        params: Ensemble parameters
        config: DTC configuration
        observations: Sequence of observations, shape [batch, seq_len, obs_dim]
        actions: Sequence of actions, shape [batch, seq_len, action_dim]
        key: PRNG key

    Returns:
        loss: Scalar loss value
        metrics: Dictionary of loss components
    """
    batch_size, seq_len, _ = observations.shape

    # Initialize state
    init_state = RSSMState.zeros(
        batch_size=batch_size,
        det_dim=config.gru_hidden_dim,
        stoch_dim=config.latent_dim_stochastic
    )

    def step_fn(carry, inputs):
        """Single step for scan."""
        state, step_key = carry
        obs, act = inputs

        # Split key for this step
        step_key, forward_key = jax.random.split(step_key)

        # Ensemble forward with observation (posterior sampling)
        ensemble_states, ensemble_info = ensemble_forward(
            params=params,
            config=config,
            prev_state=state,
            action=act,
            observation=obs,
            key=forward_key,
            use_sample=True,
        )

        # Take mean across ensemble for next state
        # (During training, we average ensemble predictions for state propagation)
        next_state = jax.tree.map(
            lambda x: jnp.mean(x, axis=0),
            ensemble_states
        )

        return (next_state, step_key), ensemble_info

    # Scan over sequence
    _, ensemble_infos = jax.lax.scan(
        step_fn,
        init=(init_state, key),
        xs=(observations.transpose(1, 0, 2), actions.transpose(1, 0, 2)),  # [seq_len, batch, ...]
    )

    # ===== Reconstruction Loss =====
    # Shape: [seq_len, ensemble_size, batch, obs_dim]
    reconstructed = ensemble_infos['reconstructed_obs']

    # Mean over ensemble, then MSE over batch and features
    reconstructed_mean = jnp.mean(reconstructed, axis=1)  # [seq_len, batch, obs_dim]
    reconstruction_loss = jnp.mean(
        jnp.square(reconstructed_mean - observations.transpose(1, 0, 2))
    )

    # ===== KL Divergence Loss =====
    # KL(q(z|h,o) || p(z|h))
    # Compute KL divergence analytically for diagonal Gaussians
    # KL(N(mu1, sigma1) || N(mu0, sigma0)) =
    #   log(sigma0/sigma1) + (sigma1^2 + (mu1-mu0)^2) / (2*sigma0^2) - 1/2

    prior_mean = ensemble_infos['prior_mean']  # [seq_len, ensemble_size, batch, dim]
    prior_std = ensemble_infos['prior_std']
    posterior_mean = ensemble_infos['posterior_mean']
    posterior_std = ensemble_infos['posterior_std']

    # Compute KL analytically (more stable than using distributions)
    kl_divergence = (
        jnp.log(prior_std + 1e-8) - jnp.log(posterior_std + 1e-8) +
        (jnp.square(posterior_std) + jnp.square(posterior_mean - prior_mean)) /
        (2 * jnp.square(prior_std) + 1e-8) - 0.5
    )

    # Sum over latent dimension, mean over batch and sequence
    kl_loss = jnp.mean(jnp.sum(kl_divergence, axis=-1))

    # ===== Total Loss =====
    total_loss = (
        config.reconstruction_weight * reconstruction_loss +
        config.kl_weight * kl_loss
    )

    metrics = {
        'total_loss': total_loss,
        'reconstruction_loss': reconstruction_loss,
        'kl_loss': kl_loss,
    }

    return total_loss, metrics
