"""
Test script for RSSM ensemble implementation.

Verifies:
  1. Correct output shapes with ensemble dimension
  2. No NaN outputs (log_std clipping works)
  3. Ensemble vmap produces different outputs per member
  4. Loss computation is numerically stable
"""

import jax
import jax.numpy as jnp
from dtc_jax.configs.base_config import DTCConfig
from dtc_jax.dtc.rssm import (
    create_ensemble_params,
    ensemble_forward,
    compute_ensemble_predictions,
    compute_rssm_loss,
)
from dtc_jax.dtc.dtc_types import RSSMState


def test_ensemble_creation():
    """Test that ensemble parameters are properly stacked."""
    print("=" * 60)
    print("TEST 1: Ensemble Parameter Creation")
    print("=" * 60)

    config = DTCConfig()
    key = jax.random.PRNGKey(0)
    batch_size = 4

    ensemble_params, init_state = create_ensemble_params(config, key, batch_size)

    # Check that params have ensemble dimension
    def check_shape(path, param):
        print(f"  {path}: {param.shape}")
        assert param.shape[0] == config.ensemble_size, \
            f"Expected first dim to be {config.ensemble_size}, got {param.shape[0]}"

    print(f"\nEnsemble size: {config.ensemble_size}")
    print("Checking parameter shapes (first dimension should be ensemble_size):")
    jax.tree_util.tree_map_with_path(check_shape, ensemble_params)

    print("\n✓ Ensemble parameters correctly created with ensemble dimension")
    return ensemble_params, init_state, config


def test_forward_pass(ensemble_params, init_state, config):
    """Test forward pass produces correct shapes and no NaNs."""
    print("\n" + "=" * 60)
    print("TEST 2: Forward Pass Shape and Stability")
    print("=" * 60)

    batch_size = 4
    key = jax.random.PRNGKey(42)

    # Create dummy inputs
    action = jax.random.normal(key, (batch_size, config.action_dim))
    observation = jax.random.normal(key, (batch_size, config.obs_dim))

    # Forward pass with observation (posterior)
    print("\nRunning ensemble forward pass with observation (posterior mode)...")
    ensemble_states, ensemble_info = ensemble_forward(
        params=ensemble_params,
        config=config,
        prev_state=init_state,
        action=action,
        observation=observation,
        key=key,
        use_sample=True,
    )

    # Check shapes
    print(f"\nOutput shapes:")
    print(f"  Deterministic state: {ensemble_states.deterministic.shape}")
    print(f"  Stochastic state: {ensemble_states.stochastic.shape}")
    print(f"  Prior mean: {ensemble_info['prior_mean'].shape}")
    print(f"  Prior std: {ensemble_info['prior_std'].shape}")
    print(f"  Posterior mean: {ensemble_info['posterior_mean'].shape}")
    print(f"  Reconstructed obs: {ensemble_info['reconstructed_obs'].shape}")

    # Expected shapes
    expected_det_shape = (config.ensemble_size, batch_size, config.gru_hidden_dim)
    expected_stoch_shape = (config.ensemble_size, batch_size, config.latent_dim_stochastic)
    expected_obs_shape = (config.ensemble_size, batch_size, config.obs_dim)

    assert ensemble_states.deterministic.shape == expected_det_shape, \
        f"Expected {expected_det_shape}, got {ensemble_states.deterministic.shape}"
    assert ensemble_states.stochastic.shape == expected_stoch_shape, \
        f"Expected {expected_stoch_shape}, got {ensemble_states.stochastic.shape}"
    assert ensemble_info['reconstructed_obs'].shape == expected_obs_shape, \
        f"Expected {expected_obs_shape}, got {ensemble_info['reconstructed_obs'].shape}"

    print("\n✓ All shapes correct!")

    # Check for NaNs
    print("\nChecking for NaN values...")
    has_nan = False

    def check_nan(name, arr):
        nonlocal has_nan
        if jnp.any(jnp.isnan(arr)):
            print(f"  ✗ Found NaN in {name}")
            has_nan = True
        else:
            print(f"  ✓ No NaN in {name}")

    check_nan("deterministic state", ensemble_states.deterministic)
    check_nan("stochastic state", ensemble_states.stochastic)
    check_nan("prior mean", ensemble_info['prior_mean'])
    check_nan("prior std", ensemble_info['prior_std'])
    check_nan("reconstructed obs", ensemble_info['reconstructed_obs'])

    assert not has_nan, "Found NaN values in outputs!"

    print("\n✓ No NaN values found - numerical stability verified!")

    # Check that ensemble members produce different outputs
    print("\nChecking ensemble diversity...")
    stoch_variance = jnp.var(ensemble_states.stochastic, axis=0).mean()
    print(f"  Mean variance across ensemble members: {stoch_variance:.6f}")

    assert stoch_variance > 1e-6, "Ensemble members are producing identical outputs!"
    print("✓ Ensemble members are producing diverse outputs!")

    return ensemble_states, ensemble_info


def test_prior_only(ensemble_params, init_state, config):
    """Test imagination mode (prior only, no observation)."""
    print("\n" + "=" * 60)
    print("TEST 3: Imagination Mode (Prior Only)")
    print("=" * 60)

    batch_size = 4
    key = jax.random.PRNGKey(99)

    # Create dummy action
    action = jax.random.normal(key, (batch_size, config.action_dim))

    print("\nRunning ensemble forward pass without observation (prior mode)...")
    ensemble_states, ensemble_info = ensemble_forward(
        params=ensemble_params,
        config=config,
        prev_state=init_state,
        action=action,
        observation=None,  # No observation = prior mode
        key=key,
        use_sample=True,
    )

    print(f"\nOutput shapes:")
    print(f"  Deterministic state: {ensemble_states.deterministic.shape}")
    print(f"  Stochastic state: {ensemble_states.stochastic.shape}")
    print(f"  Prior mean: {ensemble_info['prior_mean'].shape}")

    # In prior mode, posterior should be None
    assert ensemble_info['posterior_mean'] is None, "Posterior should be None in prior mode"
    print("\n✓ Posterior correctly set to None in prior mode")

    # Check for NaNs
    assert not jnp.any(jnp.isnan(ensemble_states.stochastic)), "NaN in stochastic state"
    assert not jnp.any(jnp.isnan(ensemble_info['prior_mean'])), "NaN in prior mean"

    print("✓ Prior-only mode works correctly!")


def test_uncertainty_computation(ensemble_params, init_state, config):
    """Test ensemble uncertainty quantification."""
    print("\n" + "=" * 60)
    print("TEST 4: Uncertainty Quantification")
    print("=" * 60)

    batch_size = 4
    key = jax.random.PRNGKey(123)

    action = jax.random.normal(key, (batch_size, config.action_dim))
    observation = jax.random.normal(key, (batch_size, config.obs_dim))

    # Forward pass
    ensemble_states, ensemble_info = ensemble_forward(
        params=ensemble_params,
        config=config,
        prev_state=init_state,
        action=action,
        observation=observation,
        key=key,
        use_sample=True,
    )

    # Compute predictions
    ensemble_means, ensemble_stds = compute_ensemble_predictions(ensemble_info)

    print(f"\nEnsemble predictions shape:")
    print(f"  Means: {ensemble_means.shape}")
    print(f"  Stds: {ensemble_stds.shape}")

    # Compute epistemic uncertainty (disagreement)
    epistemic_uncertainty = jnp.var(ensemble_means, axis=0).mean()
    print(f"\nEpistemic uncertainty (ensemble disagreement): {epistemic_uncertainty:.6f}")

    # Compute aleatoric uncertainty (expected noise)
    aleatoric_uncertainty = jnp.mean(jnp.square(ensemble_stds))
    print(f"Aleatoric uncertainty (expected noise): {aleatoric_uncertainty:.6f}")

    assert epistemic_uncertainty > 0, "Epistemic uncertainty should be positive"
    assert aleatoric_uncertainty > 0, "Aleatoric uncertainty should be positive"

    print("\n✓ Uncertainty quantification working correctly!")


def test_loss_computation(ensemble_params, config):
    """Test RSSM loss computation on a sequence."""
    print("\n" + "=" * 60)
    print("TEST 5: Loss Computation on Sequence")
    print("=" * 60)

    batch_size = 4
    seq_len = 10
    key = jax.random.PRNGKey(456)

    # Create dummy sequence data
    observations = jax.random.normal(key, (batch_size, seq_len, config.obs_dim))
    actions = jax.random.normal(key, (batch_size, seq_len, config.action_dim))

    print(f"\nSequence shapes:")
    print(f"  Observations: {observations.shape}")
    print(f"  Actions: {actions.shape}")

    # Compute loss
    print("\nComputing RSSM loss...")
    loss, metrics = compute_rssm_loss(
        params=ensemble_params,
        config=config,
        observations=observations,
        actions=actions,
        key=key,
    )

    print(f"\nLoss components:")
    print(f"  Total loss: {metrics['total_loss']:.6f}")
    print(f"  Reconstruction loss: {metrics['reconstruction_loss']:.6f}")
    print(f"  KL loss: {metrics['kl_loss']:.6f}")

    # Check for NaNs
    assert not jnp.isnan(loss), "Loss is NaN!"
    assert not jnp.isnan(metrics['reconstruction_loss']), "Reconstruction loss is NaN!"
    assert not jnp.isnan(metrics['kl_loss']), "KL loss is NaN!"

    # Check that losses are positive
    assert metrics['reconstruction_loss'] >= 0, "Reconstruction loss should be non-negative"
    assert metrics['kl_loss'] >= 0, "KL loss should be non-negative"

    print("\n✓ Loss computation is numerically stable!")

    # Test gradient computation
    print("\nTesting gradient computation...")
    grad_fn = jax.grad(lambda p: compute_rssm_loss(p, config, observations, actions, key)[0])
    grads = grad_fn(ensemble_params)

    # Check that gradients exist and are not NaN
    has_nan_grad = False
    def check_grad(path, grad):
        nonlocal has_nan_grad
        if jnp.any(jnp.isnan(grad)):
            print(f"  ✗ NaN gradient in {path}")
            has_nan_grad = True

    jax.tree_util.tree_map_with_path(check_grad, grads)

    assert not has_nan_grad, "Found NaN gradients!"
    print("✓ Gradients computed successfully without NaN!")


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("RSSM ENSEMBLE VERIFICATION TESTS")
    print("=" * 60)

    # Test 1: Ensemble creation
    ensemble_params, init_state, config = test_ensemble_creation()

    # Test 2: Forward pass
    ensemble_states, ensemble_info = test_forward_pass(ensemble_params, init_state, config)

    # Test 3: Prior-only mode
    test_prior_only(ensemble_params, init_state, config)

    # Test 4: Uncertainty
    test_uncertainty_computation(ensemble_params, init_state, config)

    # Test 5: Loss computation
    test_loss_computation(ensemble_params, config)

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED! ✓")
    print("=" * 60)
    print("\nSummary:")
    print("  ✓ Ensemble parameters correctly stacked")
    print("  ✓ Forward pass produces correct shapes")
    print("  ✓ No NaN values (numerical stability verified)")
    print("  ✓ Ensemble members produce diverse outputs")
    print("  ✓ Prior-only mode (imagination) works")
    print("  ✓ Uncertainty quantification works")
    print("  ✓ Loss computation is stable")
    print("  ✓ Gradients computed without NaN")
    print("\nRSSM implementation is ready for training! 🚀")


if __name__ == "__main__":
    main()
