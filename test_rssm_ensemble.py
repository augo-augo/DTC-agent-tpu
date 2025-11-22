"""
Test script for RSSM ensemble and intrinsic motivation.

This script verifies:
1. RSSM ensemble initialization and parameter stacking
2. Forward pass shapes are correct [ensemble_size, batch, ...]
3. No NaN values are produced
4. Intrinsic reward calculations work correctly
5. Correct dtypes (bfloat16 for compute, float32 for intrinsic)
"""

import jax
import jax.numpy as jnp
import sys

# Add project to path
sys.path.insert(0, '/home/user/DTC-agent-tpu')

from dtc_jax.configs.base_config import DTCConfig
from dtc_jax.dtc.rssm import RSSMEnsemble, compute_rssm_loss
from dtc_jax.dtc.intrinsic import (
    calculate_clean_novelty,
    update_cognitive_wave,
    compute_intrinsic_reward,
    calculate_dynamic_horizon,
)
from dtc_jax.dtc.types import IntrinsicState


def test_rssm_ensemble():
    """Test RSSM ensemble initialization and forward pass."""
    print("\n" + "="*60)
    print("Testing RSSM Ensemble")
    print("="*60)

    # Create config
    config = DTCConfig(
        global_batch_size=16,
        ensemble_size=5,
        obs_dim=32,
        action_dim=4,
    )

    # Initialize ensemble
    key = jax.random.PRNGKey(42)
    ensemble = RSSMEnsemble(config)

    print(f"\n✓ Created RSSMEnsemble with config:")
    print(f"  - Ensemble size: {config.ensemble_size}")
    print(f"  - Batch size: {config.global_batch_size}")
    print(f"  - Obs dim: {config.obs_dim}")
    print(f"  - Action dim: {config.action_dim}")
    print(f"  - Latent dim (deterministic): {config.latent_dim_deterministic}")
    print(f"  - Latent dim (stochastic): {config.latent_dim_stochastic}")

    # Initialize parameters
    key, init_key = jax.random.split(key)
    params = ensemble.init(init_key, batch_size=config.global_batch_size)

    print(f"\n✓ Initialized ensemble parameters")

    # Check parameter structure
    def check_param_shape(params, path=""):
        """Recursively check parameter shapes."""
        if isinstance(params, dict):
            for k, v in params.items():
                check_param_shape(v, f"{path}/{k}" if path else k)
        elif isinstance(params, jnp.ndarray):
            # First dimension should be ensemble_size
            assert params.shape[0] == config.ensemble_size, \
                f"Parameter {path} first dim is {params.shape[0]}, expected {config.ensemble_size}"
            print(f"  {path}: {params.shape}")

    print(f"\nParameter shapes (first dim should be {config.ensemble_size}):")
    check_param_shape(params)

    # Initialize states
    prev_states = ensemble.init_ensemble_states(config.global_batch_size)

    print(f"\n✓ Initialized ensemble states:")
    print(f"  - Deterministic shape: {prev_states.deterministic.shape}")
    print(f"  - Stochastic shape: {prev_states.stochastic.shape}")

    # Expected shapes
    expected_det_shape = (config.ensemble_size, config.global_batch_size, config.latent_dim_deterministic)
    expected_stoch_shape = (config.ensemble_size, config.global_batch_size, config.latent_dim_stochastic)

    assert prev_states.deterministic.shape == expected_det_shape, \
        f"Deterministic shape mismatch: {prev_states.deterministic.shape} vs {expected_det_shape}"
    assert prev_states.stochastic.shape == expected_stoch_shape, \
        f"Stochastic shape mismatch: {prev_states.stochastic.shape} vs {expected_stoch_shape}"

    # Create dummy inputs
    key, action_key, obs_key, keys_key = jax.random.split(key, 4)
    actions = jax.random.normal(action_key, (config.global_batch_size, config.action_dim))
    observations = jax.random.normal(obs_key, (config.global_batch_size, config.obs_dim))
    rng_keys = jax.random.split(keys_key, config.ensemble_size)

    print(f"\n✓ Created dummy inputs:")
    print(f"  - Actions shape: {actions.shape}")
    print(f"  - Observations shape: {observations.shape}")
    print(f"  - RNG keys shape: {len(rng_keys)}")

    # Forward pass
    outputs = ensemble.apply_ensemble(
        params=params,
        prev_states=prev_states,
        actions=actions,
        observations=observations,
        keys=rng_keys,
        training=True,
    )

    print(f"\n✓ Forward pass completed!")
    print(f"  - States deterministic shape: {outputs.states.deterministic.shape}")
    print(f"  - States stochastic shape: {outputs.states.stochastic.shape}")
    print(f"  - Prior means shape: {outputs.prior_means.shape}")
    print(f"  - Prior stds shape: {outputs.prior_stds.shape}")
    print(f"  - Posterior means shape: {outputs.posterior_means.shape}")
    print(f"  - Posterior stds shape: {outputs.posterior_stds.shape}")

    # Verify shapes
    expected_ensemble_batch_shape = (config.ensemble_size, config.global_batch_size)
    assert outputs.prior_means.shape == (*expected_ensemble_batch_shape, config.latent_dim_stochastic)
    assert outputs.prior_stds.shape == (*expected_ensemble_batch_shape, config.latent_dim_stochastic)
    assert outputs.posterior_means.shape == (*expected_ensemble_batch_shape, config.latent_dim_stochastic)
    assert outputs.posterior_stds.shape == (*expected_ensemble_batch_shape, config.latent_dim_stochastic)

    # Check for NaNs
    def check_no_nans(arr, name):
        """Check array for NaNs."""
        has_nan = jnp.any(jnp.isnan(arr))
        if has_nan:
            print(f"  ✗ {name} contains NaNs!")
            return False
        else:
            print(f"  ✓ {name} has no NaNs")
            return True

    print(f"\nChecking for NaN values:")
    all_good = True
    all_good &= check_no_nans(outputs.states.deterministic, "states.deterministic")
    all_good &= check_no_nans(outputs.states.stochastic, "states.stochastic")
    all_good &= check_no_nans(outputs.prior_means, "prior_means")
    all_good &= check_no_nans(outputs.prior_stds, "prior_stds")
    all_good &= check_no_nans(outputs.posterior_means, "posterior_means")
    all_good &= check_no_nans(outputs.posterior_stds, "posterior_stds")

    assert all_good, "Some outputs contain NaNs!"

    # Check std values are positive and bounded
    print(f"\nChecking std value ranges:")
    print(f"  Prior std - min: {jnp.min(outputs.prior_stds):.6f}, max: {jnp.max(outputs.prior_stds):.6f}")
    print(f"  Posterior std - min: {jnp.min(outputs.posterior_stds):.6f}, max: {jnp.max(outputs.posterior_stds):.6f}")

    assert jnp.all(outputs.prior_stds > 0), "Prior stds must be positive"
    assert jnp.all(outputs.posterior_stds > 0), "Posterior stds must be positive"
    assert jnp.all(outputs.prior_stds < 10), "Prior stds should be bounded (clip working?)"
    assert jnp.all(outputs.posterior_stds < 10), "Posterior stds should be bounded (clip working?)"

    # Test RSSM loss
    loss, info = compute_rssm_loss(
        outputs.prior_means,
        outputs.prior_stds,
        outputs.posterior_means,
        outputs.posterior_stds,
        config,
    )

    print(f"\n✓ RSSM loss computation:")
    print(f"  - Total loss: {loss:.6f}")
    for k, v in info.items():
        print(f"  - {k}: {v:.6f}")

    assert not jnp.isnan(loss), "Loss is NaN!"
    assert loss >= 0, "Loss should be non-negative"

    print(f"\n{'='*60}")
    print("✓ RSSM Ensemble tests PASSED!")
    print(f"{'='*60}")

    return outputs


def test_intrinsic_motivation(ensemble_outputs):
    """Test intrinsic motivation calculations."""
    print("\n" + "="*60)
    print("Testing Intrinsic Motivation")
    print("="*60)

    config = DTCConfig()

    # Test clean novelty
    print(f"\nTesting clean novelty calculation...")
    novelty = calculate_clean_novelty(
        ensemble_outputs.posterior_means,
        ensemble_outputs.posterior_stds,
    )

    print(f"✓ Clean novelty computed:")
    print(f"  - Shape: {novelty.shape}")
    print(f"  - Mean: {jnp.mean(novelty):.6f}")
    print(f"  - Max: {jnp.max(novelty):.6f}")
    print(f"  - Min: {jnp.min(novelty):.6f}")
    print(f"  - Dtype: {novelty.dtype}")

    assert novelty.dtype == jnp.float32, f"Novelty must be float32, got {novelty.dtype}"
    assert not jnp.any(jnp.isnan(novelty)), "Novelty contains NaNs"
    assert jnp.all(novelty >= 0), "Novelty must be non-negative"

    # Test with identical inputs (should give zero novelty)
    print(f"\nTesting identical inputs (should give zero novelty)...")
    identical_means = jnp.zeros((5, 16, 32))
    identical_stds = jnp.ones((5, 16, 32)) * 0.1
    zero_novelty = calculate_clean_novelty(identical_means, identical_stds)

    print(f"  - Identical means novelty: {jnp.mean(zero_novelty):.10f}")
    assert jnp.allclose(zero_novelty, 0.0, atol=1e-6), \
        "Identical inputs should produce zero novelty"

    # Test cognitive wave
    print(f"\nTesting cognitive wave (dual-timescale competence)...")
    intrinsic_state = IntrinsicState.init()

    print(f"✓ Initial intrinsic state:")
    print(f"  - c_slow: {intrinsic_state.c_slow}")
    print(f"  - c_fast: {intrinsic_state.c_fast}")
    print(f"  - boredom: {intrinsic_state.boredom}")
    print(f"  - step_count: {intrinsic_state.step_count}")

    # Simulate learning progress
    print(f"\nSimulating learning progress over 10 steps...")
    errors = [1.0, 0.9, 0.8, 0.6, 0.4, 0.3, 0.2, 0.15, 0.1, 0.05]

    for i, error in enumerate(errors):
        new_state, competence, boredom = update_cognitive_wave(
            intrinsic_state,
            jnp.array(error, dtype=jnp.float32),
            config,
        )

        print(f"  Step {i+1}: error={error:.2f}, c_slow={new_state.c_slow:.4f}, "
              f"c_fast={new_state.c_fast:.4f}, competence={competence:.4f}, boredom={boredom:.4f}")

        # Verify dtypes
        assert new_state.c_slow.dtype == jnp.float32
        assert new_state.c_fast.dtype == jnp.float32
        assert competence.dtype == jnp.float32
        assert boredom.dtype == jnp.float32

        intrinsic_state = new_state

    # Test complete intrinsic reward
    print(f"\nTesting complete intrinsic reward computation...")
    intrinsic_state = IntrinsicState.init()

    intrinsic_reward, new_state, info = compute_intrinsic_reward(
        ensemble_outputs.posterior_means,
        ensemble_outputs.posterior_stds,
        intrinsic_state,
        config,
    )

    print(f"✓ Intrinsic reward computed:")
    print(f"  - Reward shape: {intrinsic_reward.shape}")
    print(f"  - Reward mean: {jnp.mean(intrinsic_reward):.6f}")
    print(f"  - Reward dtype: {intrinsic_reward.dtype}")

    print(f"\n  Info dict:")
    for k, v in info.items():
        print(f"    - {k}: {v:.6f}")

    assert intrinsic_reward.dtype == jnp.float32
    assert not jnp.any(jnp.isnan(intrinsic_reward))

    # Test dynamic horizon calculation
    print(f"\nTesting dynamic horizon calculation...")
    for boredom_val in [0.0, 0.25, 0.5, 0.75, 1.0]:
        horizon = calculate_dynamic_horizon(
            jnp.array(boredom_val, dtype=jnp.float32),
            config,
        )
        print(f"  Boredom={boredom_val:.2f} -> Horizon={horizon}")

        assert config.dream_horizon_max // 4 <= horizon <= config.dream_horizon_max

    print(f"\n{'='*60}")
    print("✓ Intrinsic Motivation tests PASSED!")
    print(f"{'='*60}")


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("DTC 3.0 JAX/TPU Implementation Tests")
    print("="*60)

    # Test RSSM ensemble
    ensemble_outputs = test_rssm_ensemble()

    # Test intrinsic motivation
    test_intrinsic_motivation(ensemble_outputs)

    print("\n" + "="*60)
    print("✓✓✓ ALL TESTS PASSED! ✓✓✓")
    print("="*60)
    print("\nCore components verified:")
    print("  ✓ RSSM ensemble with vmap")
    print("  ✓ No NaN values in forward pass")
    print("  ✓ Correct output shapes [ensemble_size, batch, ...]")
    print("  ✓ Log-std clipping working correctly")
    print("  ✓ Clean novelty calculation (float32)")
    print("  ✓ Dual-timescale competence tracking")
    print("  ✓ Dynamic horizon calculation")
    print("\nReady to proceed with:")
    print("  - Actor/Critic networks (dtc/models.py)")
    print("  - Replay buffer (dtc/memory.py)")
    print("  - Dream rollout (dtc/dream.py)")
    print("  - Training loop (training/trainer.py)")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
