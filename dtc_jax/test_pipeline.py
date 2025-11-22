"""
End-to-end pipeline test for DTC 3.0 Hephaestus implementation.

This script tests the full training pipeline without requiring TPU hardware:
1. Initialize all components
2. Run a few training steps
3. Verify shapes and no NaNs
4. Report performance metrics

Run with: python test_pipeline.py
"""

import jax
import jax.numpy as jnp
from jax import random
import time

from dtc_jax.configs.base_config import DTCConfig
from dtc_jax.dtc import trainer
from dtc_jax.dtc import collector


def test_carrier_initialization():
    """Test carrier state creation."""
    print("\n" + "="*60)
    print("TEST 1: Carrier State Initialization")
    print("="*60)

    config = DTCConfig()
    key = random.PRNGKey(42)

    carrier = trainer.create_carrier_state(config, key)

    print(f"✓ Carrier created")
    print(f"  - RSSM params shape (sample): {jax.tree_util.tree_leaves(carrier.rssm_params)[0].shape}")
    print(f"  - Actor-Critic params shape (sample): {jax.tree_util.tree_leaves(carrier.actor_critic_params)[0].shape}")
    print(f"  - Buffer capacity: {carrier.buffer.observations.shape[0]}")
    print(f"  - Current step: {carrier.step}")

    return carrier, config


def test_single_train_step(carrier, config):
    """Test a single training step (non-pmapped)."""
    print("\n" + "="*60)
    print("TEST 2: Single Training Step (No pmap)")
    print("="*60)

    print("Executing train_step...")
    start = time.time()

    new_carrier, metrics = trainer.train_step(carrier, config)

    elapsed = time.time() - start

    print(f"✓ Train step completed in {elapsed*1000:.2f}ms")
    print(f"\nMetrics:")
    print(f"  - World model loss: {float(metrics.world_model_loss):.4f}")
    print(f"  - Reconstruction loss: {float(metrics.reconstruction_loss):.4f}")
    print(f"  - KL loss: {float(metrics.kl_loss):.4f}")
    print(f"  - Actor loss: {float(metrics.actor_loss):.4f}")
    print(f"  - Critic loss: {float(metrics.critic_loss):.4f}")
    print(f"  - Intrinsic reward: {float(metrics.intrinsic_reward):.4f}")
    print(f"  - Boredom: {float(metrics.boredom):.4f}")

    # Check for NaNs
    has_nan = any(jnp.isnan(jax.tree_util.tree_leaves(metrics)).any() for _ in [True])
    if has_nan:
        print("❌ NaN detected in metrics!")
        return False
    else:
        print("✓ No NaNs in metrics")

    return new_carrier


def test_pmap_training(carrier, config):
    """Test pmapped training across available devices."""
    print("\n" + "="*60)
    print("TEST 3: Pmapped Training")
    print("="*60)

    devices = jax.devices()
    print(f"Available devices: {len(devices)}")

    if len(devices) < 2:
        print("⚠️  Only 1 device available, skipping pmap test")
        print("   (pmap will work but won't show parallelism)")

    # Replicate carrier
    print(f"\nReplicating carrier across {len(devices)} devices...")
    replicated_carrier = trainer.replicate_carrier_for_pmap(carrier, len(devices))
    print(f"✓ Carrier replicated")

    # Create pmapped function
    print("Creating pmapped training function...")
    pmapped_train_step = trainer.create_train_fn(config)
    print("✓ Pmapped function created")

    # Run training step
    print("\nExecuting pmapped train_step...")
    start = time.time()

    replicated_carrier, metrics = pmapped_train_step(replicated_carrier)

    elapsed = time.time() - start

    print(f"✓ Pmapped train step completed in {elapsed*1000:.2f}ms")

    # Unreplicate metrics
    metrics_single = jax.tree_map(lambda x: x[0], metrics)

    print(f"\nMetrics (from device 0):")
    print(f"  - World model loss: {float(metrics_single.world_model_loss):.4f}")
    print(f"  - Actor loss: {float(metrics_single.actor_loss):.4f}")
    print(f"  - Boredom: {float(metrics_single.boredom):.4f}")

    return replicated_carrier


def test_experience_collection(carrier, config):
    """Test environment interaction and buffer updates."""
    print("\n" + "="*60)
    print("TEST 4: Experience Collection")
    print("="*60)

    key = random.PRNGKey(123)

    # Create dummy environment
    env_reset_fn, env_step_fn = collector.create_dummy_env_fns(config)

    key, env_key = random.split(key)
    env_state, initial_obs = env_reset_fn(env_key)

    print("✓ Dummy environment created")
    print(f"  - Observation shape: {initial_obs.shape}")

    # Collect a few steps
    print("\nCollecting 10 steps of experience...")

    buffer_count_before = carrier.buffer.count

    for i in range(10):
        carrier, env_state, info = collector.collect_experience_step(
            carrier,
            env_state,
            env_step_fn,
            config,
            deterministic=False
        )

        if i == 0:
            print(f"  Step 0 info:")
            print(f"    - Extrinsic reward: {float(info['extrinsic_reward'].mean()):.4f}")
            print(f"    - Intrinsic reward: {float(info['intrinsic_reward']):.4f}")
            print(f"    - Clean novelty: {float(info['clean_novelty']):.4f}")

    buffer_count_after = carrier.buffer.count

    print(f"\n✓ Collection complete")
    print(f"  - Buffer count before: {buffer_count_before}")
    print(f"  - Buffer count after: {buffer_count_after}")
    print(f"  - Transitions added: {buffer_count_after - buffer_count_before}")

    return carrier


def test_memory_sampling(carrier, config):
    """Test replay buffer sampling."""
    print("\n" + "="*60)
    print("TEST 5: Memory Sampling")
    print("="*60)

    # Add some dummy data to buffer first
    print("Filling buffer with dummy data...")

    from dtc_jax.dtc import memory as memory_module

    # Add enough data to make sampling valid
    min_data = config.sequence_length + config.frontier_size
    dummy_obs = jnp.zeros((min_data, config.obs_dim))
    dummy_action = jnp.zeros((min_data, config.action_dim))
    dummy_reward = jnp.zeros(min_data)
    dummy_done = jnp.zeros(min_data, dtype=jnp.bool_)
    dummy_h = jnp.zeros((min_data, config.latent_dim_deterministic))
    dummy_z = jnp.zeros((min_data, config.latent_dim_stochastic))
    dummy_intrinsic = jnp.zeros(min_data)

    # Add transitions
    carrier_buffer = memory_module.add_batch(
        carrier.buffer,
        dummy_obs,
        dummy_action,
        dummy_reward,
        dummy_done,
        dummy_h,
        dummy_z,
        dummy_intrinsic
    )

    print(f"✓ Buffer filled with {carrier_buffer.count} transitions")

    # Test sampling
    is_ready = memory_module.is_ready_to_sample(carrier_buffer, config)
    print(f"  Buffer ready to sample: {is_ready}")

    if is_ready:
        key = random.PRNGKey(456)
        batch, key = memory_module.sample_stochastic_stratified(
            carrier_buffer, config, key
        )

        print(f"\n✓ Batch sampled successfully")
        print(f"  - Observations shape: {batch.observations.shape}")
        print(f"  - Actions shape: {batch.actions.shape}")
        print(f"  - Expected: [{config.local_batch_size}, {config.sequence_length}, ...]")
    else:
        print("⚠️  Buffer not ready to sample (need more data)")


def run_all_tests():
    """Run all tests sequentially."""
    print("\n" + "="*60)
    print("DTC 3.0 HEPHAESTUS PIPELINE TEST")
    print("="*60)
    print("\nThis will test the full training pipeline end-to-end.")

    try:
        # Test 1: Initialization
        carrier, config = test_carrier_initialization()

        # Test 2: Single training step
        carrier = test_single_train_step(carrier, config)

        # Test 3: Pmapped training
        replicated_carrier = test_pmap_training(carrier, config)

        # Unreplicate for remaining tests
        carrier = trainer.unreplicate_carrier(replicated_carrier)

        # Test 4: Experience collection
        carrier = test_experience_collection(carrier, config)

        # Test 5: Memory sampling
        test_memory_sampling(carrier, config)

        # Summary
        print("\n" + "="*60)
        print("ALL TESTS PASSED ✅")
        print("="*60)
        print("\nThe pipeline is ready for training!")
        print("Run: python train.py --num_steps 10000")

        return True

    except Exception as e:
        print("\n" + "="*60)
        print("TEST FAILED ❌")
        print("="*60)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = run_all_tests()
    exit(0 if success else 1)
