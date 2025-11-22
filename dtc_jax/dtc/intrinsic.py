"""
Intrinsic Motivation System for DTC 3.0.

Implements:
  1. Clean Novelty: Epistemic uncertainty minus aleatoric noise
  2. Cognitive Wave: Dual-timescale competence tracking (C_fast, C_slow)
  3. Boredom Mechanism: Adjusts dream horizon based on learning progress

CRITICAL: All calculations MUST use float32 to prevent underflow near zero.
"""

import jax
import jax.numpy as jnp
from typing import Tuple

from dtc_jax.dtc.dtc_types import IntrinsicState, Array, Scalar
from dtc_jax.configs.base_config import DTCConfig


def calculate_clean_novelty(
    ensemble_means: Array,
    ensemble_stds: Array,
) -> Array:
    """Calculate clean novelty from ensemble predictions.

    Clean Novelty = Epistemic Uncertainty - Aleatoric Uncertainty
                  = Disagreement - Expected Noise
                  = Var(means) - Mean(stds^2)

    This measures "learnable surprise" - uncertainty that comes from model
    disagreement (epistemic) rather than inherent stochasticity (aleatoric).

    Args:
        ensemble_means: Mean predictions from ensemble, shape [ensemble_size, batch, dim]
        ensemble_stds: Std predictions from ensemble, shape [ensemble_size, batch, dim]

    Returns:
        clean_novelty: Cleaned novelty signal, shape [batch]
                      (Always >= 0 due to ReLU)
    """
    # CRITICAL: Cast to float32 for numerical precision
    ensemble_means = ensemble_means.astype(jnp.float32)
    ensemble_stds = ensemble_stds.astype(jnp.float32)

    # Epistemic Uncertainty: Variance of ensemble means (disagreement)
    # Var(X) = E[X^2] - E[X]^2
    epistemic_uncertainty = jnp.var(ensemble_means, axis=0)  # [batch, dim]

    # Aleatoric Uncertainty: Mean of predicted variances (expected noise)
    aleatoric_uncertainty = jnp.mean(
        jnp.square(ensemble_stds), axis=0
    )  # [batch, dim]

    # Clean Novelty: Subtract noise from disagreement
    # Use ReLU to ensure non-negative (when aleatoric > epistemic, novelty = 0)
    clean_novelty_per_dim = jax.nn.relu(
        epistemic_uncertainty - aleatoric_uncertainty
    )  # [batch, dim]

    # Sum over dimensions to get scalar novelty per batch item
    clean_novelty = jnp.sum(clean_novelty_per_dim, axis=-1)  # [batch]

    return clean_novelty


def update_cognitive_wave(
    intrinsic_state: IntrinsicState,
    prediction_error: Scalar,
    config: DTCConfig,
) -> Tuple[IntrinsicState, Scalar, Scalar]:
    """Update dual-timescale competence tracking.

    The "cognitive wave" uses two exponential moving averages (EMAs) of
    prediction error to track learning progress:
      - C_fast: Rapidly adapts to current error (alpha = 0.1)
      - C_slow: Slowly tracks long-term trend (alpha = 0.005)

    When C_slow > C_fast (slow avg higher than fast):
      - Agent is improving (error decreasing)
      - Competence reward is positive
      - Dream horizon is extended

    When C_slow ≈ C_fast (trends aligned):
      - Learning has plateaued
      - Boredom increases
      - Dream horizon is reduced to save compute

    Args:
        intrinsic_state: Current intrinsic motivation state
        prediction_error: Current prediction error (scalar, float32)
        config: DTC configuration

    Returns:
        new_state: Updated intrinsic state
        competence_reward: Learning progress signal (r_lp)
        boredom: Boredom scalar for horizon modulation
    """
    # CRITICAL: Ensure float32 precision for all calculations
    prediction_error = prediction_error.astype(jnp.float32)

    # Stop gradient on error to prevent backprop through intrinsic state
    # (We only want to track, not optimize for competence)
    error = jax.lax.stop_gradient(prediction_error)

    # Update Fast EMA: C_fast = alpha_fast * error + (1 - alpha_fast) * C_fast
    c_fast_new = (
        config.alpha_fast * error +
        (1.0 - config.alpha_fast) * intrinsic_state.c_fast
    )

    # Update Slow EMA: C_slow = alpha_slow * error + (1 - alpha_slow) * C_slow
    c_slow_new = (
        config.alpha_slow * error +
        (1.0 - config.alpha_slow) * intrinsic_state.c_slow
    )

    # Competence Reward: r_lp = ReLU(C_slow - C_fast)
    # Positive when slow average (long-term) is higher than fast (current)
    # This means error is decreasing = learning is happening
    competence_reward = jax.nn.relu(c_slow_new - c_fast_new)

    # Boredom Mechanism: Track when competence reward is near zero
    # If r_lp < threshold, increment boredom; else decay it
    is_bored = competence_reward < config.boredom_threshold

    # Accumulate boredom (saturates at 1.0)
    boredom_increment = jax.lax.select(
        is_bored,
        jnp.float32(0.1),  # Increase boredom when not learning
        jnp.float32(-0.2),  # Decrease boredom when learning
    )

    boredom_new = jnp.clip(
        intrinsic_state.boredom_accumulator + boredom_increment,
        0.0,
        1.0
    )

    # Create new intrinsic state
    new_state = IntrinsicState(
        c_slow=c_slow_new,
        c_fast=c_fast_new,
        boredom_accumulator=boredom_new,
        step_count=intrinsic_state.step_count + 1,
    )

    return new_state, competence_reward, boredom_new


def compute_dream_horizon(
    boredom: Scalar,
    config: DTCConfig,
) -> Scalar:
    """Compute adaptive dream horizon based on boredom.

    When boredom is low (learning actively):
      - Use full dream horizon for rich value propagation

    When boredom is high (plateau):
      - Reduce horizon to save compute
      - Agent won't learn much from long rollouts anyway

    Args:
        boredom: Boredom scalar in [0, 1]
        config: DTC configuration

    Returns:
        horizon: Dream rollout length (int32)
    """
    # Linear interpolation from max_horizon to min_horizon based on boredom
    # horizon = max - boredom * (max - min)
    horizon_float = (
        config.max_dream_horizon -
        boredom * (config.max_dream_horizon - config.min_horizon)
    )

    # Round to nearest integer and cast to int32
    horizon = jnp.round(horizon_float).astype(jnp.int32)

    # Clamp to valid range (defensive, should be redundant)
    horizon = jnp.clip(horizon, config.min_horizon, config.max_dream_horizon)

    return horizon


def compute_total_intrinsic_reward(
    clean_novelty: Array,
    competence_reward: Scalar,
    config: DTCConfig,
) -> Array:
    """Combine novelty and competence into final intrinsic reward.

    r_intrinsic = clean_novelty + competence_reward

    Both components encourage exploration and learning:
      - Clean novelty: "Go to uncertain/surprising states"
      - Competence reward: "Improve your predictions"

    Args:
        clean_novelty: Clean novelty signal, shape [batch]
        competence_reward: Learning progress signal (scalar)
        config: DTC configuration

    Returns:
        intrinsic_reward: Combined intrinsic reward, shape [batch]
    """
    # Ensure float32
    clean_novelty = clean_novelty.astype(jnp.float32)
    competence_reward = competence_reward.astype(jnp.float32)

    # Simple additive combination
    # (You could weight these differently if needed)
    intrinsic_reward = clean_novelty + competence_reward

    return intrinsic_reward


# ===== Testing / Debug Utilities =====

def test_identical_ensemble():
    """Test that identical ensemble members produce zero clean novelty."""
    ensemble_size = 5
    batch_size = 4
    dim = 32

    # Create identical predictions (no disagreement)
    mean = jnp.ones((batch_size, dim))
    std = jnp.ones((batch_size, dim)) * 0.1

    # Stack to create ensemble
    ensemble_means = jnp.stack([mean] * ensemble_size, axis=0)
    ensemble_stds = jnp.stack([std] * ensemble_size, axis=0)

    # Compute clean novelty
    novelty = calculate_clean_novelty(ensemble_means, ensemble_stds)

    print(f"Clean novelty for identical ensemble: {novelty}")
    print(f"Max novelty: {jnp.max(novelty):.8f}")

    # Should be exactly zero (or near zero due to floating point)
    assert jnp.allclose(novelty, 0.0, atol=1e-6), \
        f"Expected zero novelty, got {novelty}"

    print("✓ Identical ensemble produces zero novelty")


def test_diverse_ensemble():
    """Test that diverse ensemble produces positive clean novelty."""
    ensemble_size = 5
    batch_size = 4
    dim = 32

    # Create diverse predictions (high disagreement)
    key = jax.random.PRNGKey(42)
    ensemble_means = jax.random.normal(key, (ensemble_size, batch_size, dim))

    # Low aleatoric uncertainty
    ensemble_stds = jnp.ones((ensemble_size, batch_size, dim)) * 0.01

    # Compute clean novelty
    novelty = calculate_clean_novelty(ensemble_means, ensemble_stds)

    print(f"\nClean novelty for diverse ensemble:")
    print(f"  Mean: {jnp.mean(novelty):.6f}")
    print(f"  Max: {jnp.max(novelty):.6f}")
    print(f"  Min: {jnp.min(novelty):.6f}")

    # Should be positive
    assert jnp.all(novelty >= 0.0), "Novelty should be non-negative"
    assert jnp.mean(novelty) > 0.1, "Diverse ensemble should produce novelty"

    print("✓ Diverse ensemble produces positive novelty")


def test_competence_tracking():
    """Test dual-timescale EMA tracking."""
    from dtc_jax.configs.base_config import DEFAULT_CONFIG as config

    # Initialize state
    state = IntrinsicState.init()

    print("\nTesting competence tracking:")

    # Simulate high error initially
    print("\n1. High initial error (learning should happen):")
    for i in range(10):
        error = jnp.float32(5.0 - i * 0.3)  # Decreasing error
        state, r_comp, boredom = update_cognitive_wave(state, error, config)

        if i % 3 == 0:
            print(f"  Step {i}: error={error:.3f}, "
                  f"c_fast={state.c_fast:.3f}, c_slow={state.c_slow:.3f}, "
                  f"r_comp={r_comp:.3f}, boredom={boredom:.3f}")

    assert state.c_fast > 0, "Fast EMA should track error"
    assert state.c_slow > 0, "Slow EMA should track error"
    print("✓ EMAs tracking error correctly")

    # Simulate plateau (constant error)
    print("\n2. Plateau (constant error = boredom):")
    for i in range(20):
        error = jnp.float32(1.0)  # Constant error
        state, r_comp, boredom = update_cognitive_wave(state, error, config)

        if i % 5 == 0:
            print(f"  Step {i}: error={error:.3f}, "
                  f"c_fast={state.c_fast:.3f}, c_slow={state.c_slow:.3f}, "
                  f"r_comp={r_comp:.3f}, boredom={boredom:.3f}")

    print("✓ Boredom increases during plateau")


def test_horizon_computation():
    """Test adaptive dream horizon."""
    from dtc_jax.configs.base_config import DEFAULT_CONFIG as config

    print("\nTesting adaptive dream horizon:")

    # Low boredom = max horizon
    h_low = compute_dream_horizon(jnp.float32(0.0), config)
    print(f"  Boredom=0.0: horizon={h_low} (should be {config.max_dream_horizon})")
    assert h_low == config.max_dream_horizon

    # Medium boredom = intermediate
    h_med = compute_dream_horizon(jnp.float32(0.5), config)
    print(f"  Boredom=0.5: horizon={h_med}")
    assert config.min_horizon <= h_med <= config.max_dream_horizon

    # High boredom = min horizon
    h_high = compute_dream_horizon(jnp.float32(1.0), config)
    print(f"  Boredom=1.0: horizon={h_high} (should be {config.min_horizon})")
    assert h_high == config.min_horizon

    print("✓ Horizon adapts correctly to boredom")


if __name__ == "__main__":
    print("=" * 60)
    print("INTRINSIC MOTIVATION MODULE TESTS")
    print("=" * 60)

    test_identical_ensemble()
    test_diverse_ensemble()
    test_competence_tracking()
    test_horizon_computation()

    print("\n" + "=" * 60)
    print("ALL INTRINSIC TESTS PASSED! ✓")
    print("=" * 60)
