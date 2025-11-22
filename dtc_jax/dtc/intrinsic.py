"""
Intrinsic Motivation Module: Clean Novelty & Cognitive Wave.

This module implements the DTC 3.0 intrinsic reward system based on:
1. Clean Novelty: Epistemic uncertainty minus aleatoric noise
2. Cognitive Wave: Dual-timescale competence tracking (fast vs slow EMA)

CRITICAL: All calculations MUST be in float32 to prevent numerical underflow
when variance/uncertainty values approach zero.
"""

import jax
import jax.numpy as jnp
import chex

from dtc_jax.dtc.types import IntrinsicState
from dtc_jax.configs.base_config import DTCConfig


def calculate_clean_novelty(
    ensemble_means: chex.Array,
    ensemble_stds: chex.Array,
) -> chex.Array:
    """
    Calculate clean novelty from ensemble predictions.

    Clean novelty is epistemic uncertainty (disagreement) minus aleatoric
    uncertainty (expected noise), ensuring we reward true knowledge gaps
    rather than inherent stochasticity.

    Args:
        ensemble_means: Ensemble prediction means [ensemble_size, batch, dim]
        ensemble_stds: Ensemble prediction stds [ensemble_size, batch, dim]

    Returns:
        Clean novelty [batch] in float32
    """
    # CRITICAL: Cast to float32 for numerical stability
    ensemble_means = ensemble_means.astype(jnp.float32)
    ensemble_stds = ensemble_stds.astype(jnp.float32)

    # ========== Epistemic Uncertainty (Disagreement) ==========
    # Variance of the ensemble means across models
    # High when models disagree (epistemic uncertainty)
    mean_var = jnp.var(ensemble_means, axis=0)  # [batch, dim]
    epistemic_uncertainty = jnp.sum(mean_var, axis=-1)  # [batch]

    # ========== Aleatoric Uncertainty (Expected Noise) ==========
    # Mean of the predicted variances (squared stds)
    # Represents inherent stochasticity in the environment
    variance_estimates = jnp.square(ensemble_stds)  # [ensemble, batch, dim]
    mean_variance = jnp.mean(variance_estimates, axis=0)  # [batch, dim]
    aleatoric_uncertainty = jnp.sum(mean_variance, axis=-1)  # [batch]

    # ========== Clean Novelty ==========
    # Subtract aleatoric noise from epistemic uncertainty
    # Use ReLU to ensure non-negative (can't have negative uncertainty)
    clean_novelty = jax.nn.relu(epistemic_uncertainty - aleatoric_uncertainty)

    # Ensure output is float32
    return clean_novelty.astype(jnp.float32)


def update_cognitive_wave(
    intrinsic_state: IntrinsicState,
    current_error: chex.Array,
    config: DTCConfig,
) -> tuple[IntrinsicState, chex.Array, chex.Array]:
    """
    Update dual-timescale competence tracking (cognitive wave).

    The cognitive wave tracks prediction error at two timescales:
    - Fast EMA (c_fast): Quickly adapts to current performance
    - Slow EMA (c_slow): Represents long-term average performance

    When c_slow > c_fast, the agent is "in the zone" (competent).
    When c_fast > c_slow, the agent is struggling (incompetent).
    When both are equal and near zero, the agent is bored.

    Args:
        intrinsic_state: Current intrinsic motivation state
        current_error: Current prediction error [batch] or scalar
        config: Configuration with alpha_fast, alpha_slow

    Returns:
        Tuple of (new_state, competence_reward, boredom_scalar)
    """
    # CRITICAL: Ensure float32 throughout
    current_error = jnp.mean(current_error).astype(jnp.float32)  # Reduce to scalar
    current_error = jax.lax.stop_gradient(current_error)  # Don't backprop through this

    # ========== Update EMAs ==========
    # Fast EMA: c_fast = (1 - α_fast) * c_fast + α_fast * error
    new_c_fast = (
        (1.0 - config.alpha_fast) * intrinsic_state.c_fast +
        config.alpha_fast * current_error
    )

    # Slow EMA: c_slow = (1 - α_slow) * c_slow + α_slow * error
    new_c_slow = (
        (1.0 - config.alpha_slow) * intrinsic_state.c_slow +
        config.alpha_slow * current_error
    )

    # ========== Competence Reward (Learning Progress) ==========
    # Positive when slow > fast (improving faster than long-term average)
    # This rewards the "zone of proximal development"
    learning_progress = new_c_slow - new_c_fast
    competence_reward = jax.nn.relu(learning_progress)  # Only positive progress

    # ========== Boredom Detection ==========
    # Boredom occurs when:
    # 1. Both EMAs are very low (mastery achieved)
    # 2. No learning progress (competence reward near zero)

    # Check if both EMAs are below threshold (mastery)
    is_mastered = jnp.logical_and(
        new_c_fast < config.boredom_threshold,
        new_c_slow < config.boredom_threshold
    ).astype(jnp.float32)

    # Check if learning progress is near zero
    is_stagnant = (competence_reward < config.boredom_threshold).astype(jnp.float32)

    # Boredom accumulates when both conditions are met
    boredom_increment = is_mastered * is_stagnant * 0.1
    new_boredom = intrinsic_state.boredom + boredom_increment

    # Decay boredom when actively learning
    new_boredom = new_boredom * (1.0 - competence_reward * 0.1)

    # Clamp boredom to [0, 1]
    new_boredom = jnp.clip(new_boredom, 0.0, 1.0)

    # ========== Boredom-Driven Horizon Scaling ==========
    # When bored, reduce dream horizon to encourage exploration
    # boredom_scalar ∈ [0, 1]: 0 = not bored (full horizon), 1 = very bored (short horizon)
    boredom_scalar = new_boredom

    # ========== Update State ==========
    new_state = IntrinsicState(
        c_slow=new_c_slow.astype(jnp.float32),
        c_fast=new_c_fast.astype(jnp.float32),
        boredom=new_boredom.astype(jnp.float32),
        step_count=intrinsic_state.step_count + 1,
    )

    return new_state, competence_reward.astype(jnp.float32), boredom_scalar.astype(jnp.float32)


def compute_intrinsic_reward(
    ensemble_means: chex.Array,
    ensemble_stds: chex.Array,
    intrinsic_state: IntrinsicState,
    config: DTCConfig,
) -> tuple[chex.Array, IntrinsicState, dict]:
    """
    Compute complete intrinsic reward signal.

    Combines clean novelty (exploration) with competence reward (learning progress).

    Args:
        ensemble_means: Ensemble predictions [ensemble, batch, dim]
        ensemble_stds: Ensemble uncertainties [ensemble, batch, dim]
        intrinsic_state: Current intrinsic state
        config: Configuration

    Returns:
        Tuple of (intrinsic_reward [batch], new_state, info_dict)
    """
    # ========== Clean Novelty (Exploration) ==========
    novelty = calculate_clean_novelty(ensemble_means, ensemble_stds)

    # ========== Competence (Learning Progress) ==========
    # Use mean novelty as proxy for prediction error
    current_error = jnp.mean(novelty)

    new_state, competence, boredom = update_cognitive_wave(
        intrinsic_state,
        current_error,
        config
    )

    # ========== Combined Intrinsic Reward ==========
    # Weight novelty by scaling factor
    novelty_reward = config.novelty_scale * novelty

    # Combine novelty and competence
    # Both encourage learning but in different ways:
    # - Novelty: seek unknown states
    # - Competence: seek learnable challenges
    intrinsic_reward = novelty_reward + competence

    # ========== Logging Info ==========
    info = {
        "intrinsic/novelty_mean": jnp.mean(novelty),
        "intrinsic/novelty_max": jnp.max(novelty),
        "intrinsic/competence": competence,
        "intrinsic/boredom": boredom,
        "intrinsic/c_fast": new_state.c_fast,
        "intrinsic/c_slow": new_state.c_slow,
        "intrinsic/total_reward_mean": jnp.mean(intrinsic_reward),
    }

    return intrinsic_reward, new_state, info


def calculate_dynamic_horizon(
    boredom_scalar: chex.Array,
    config: DTCConfig,
) -> chex.Array:
    """
    Calculate dynamic imagination horizon based on boredom.

    When bored (mastered environment), use shorter horizons to save compute
    and encourage exploration of new areas.

    Args:
        boredom_scalar: Boredom level [0, 1]
        config: Configuration with max horizon

    Returns:
        Horizon length as integer (will be used for masking)
    """
    # Boredom = 0: Full horizon (max_horizon)
    # Boredom = 1: Minimum horizon (max_horizon // 4)
    min_horizon = config.dream_horizon_max // 4
    max_horizon = config.dream_horizon_max

    # Linear interpolation based on boredom
    horizon = max_horizon - boredom_scalar * (max_horizon - min_horizon)

    # Convert to integer
    horizon_int = jnp.round(horizon).astype(jnp.int32)

    # Clamp to valid range
    horizon_int = jnp.clip(horizon_int, min_horizon, max_horizon)

    return horizon_int
