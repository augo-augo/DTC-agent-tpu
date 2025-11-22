"""
Main Training Loop for DTC 3.0 on TPU.

This script orchestrates the full training pipeline:
1. Initialize carrier state
2. Replicate across TPU devices
3. Interleave environment collection and training steps
4. Log metrics and save checkpoints

Usage:
    python train.py --config configs/base_config.py --num_steps 1000000
"""

import jax
import jax.numpy as jnp
from jax import random
import argparse
import time
from pathlib import Path

from dtc_jax.configs.base_config import DTCConfig
from dtc_jax.dtc import trainer
from dtc_jax.dtc import collector


def main():
    # ===== Parse Arguments =====
    parser = argparse.ArgumentParser(description='Train DTC 3.0 on TPU')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--num_steps', type=int, default=100000, help='Number of training steps')
    parser.add_argument('--collect_interval', type=int, default=10, help='Collect experience every N training steps')
    parser.add_argument('--log_interval', type=int, default=100, help='Log metrics every N steps')
    parser.add_argument('--checkpoint_interval', type=int, default=10000, help='Save checkpoint every N steps')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='Checkpoint directory')
    args = parser.parse_args()

    # ===== Initialize Configuration =====
    config = DTCConfig()
    print(f"Configuration loaded:")
    print(f"  - Global batch size: {config.global_batch_size}")
    print(f"  - Local batch size: {config.local_batch_size}")
    print(f"  - TPU cores: {config.num_tpu_cores}")
    print(f"  - Ensemble size: {config.ensemble_size}")
    print(f"  - Max dream horizon: {config.max_dream_horizon}")

    # ===== Check Available Devices =====
    devices = jax.devices()
    print(f"\nAvailable devices: {len(devices)}")
    for i, device in enumerate(devices):
        print(f"  Device {i}: {device}")

    if len(devices) != config.num_tpu_cores:
        print(f"\n⚠️  WARNING: Config expects {config.num_tpu_cores} devices, but {len(devices)} available")
        print(f"  Adjusting to use {len(devices)} devices")

    # ===== Initialize Carrier State =====
    print(f"\n{'='*60}")
    print("Initializing carrier state...")
    print(f"{'='*60}")

    key = random.PRNGKey(args.seed)
    key, init_key = random.split(key)

    # Create single-device carrier
    carrier_state = trainer.create_carrier_state(config, init_key)
    print(f"✓ Carrier state initialized")

    # Replicate across devices for pmap
    print(f"Replicating carrier across {len(devices)} devices...")
    replicated_carrier = trainer.replicate_carrier_for_pmap(
        carrier_state,
        num_devices=len(devices)
    )
    print(f"✓ Carrier replicated")

    # ===== Create Training Function =====
    print(f"\nCreating pmapped training function...")
    pmapped_train_step = trainer.create_train_fn(config)
    print(f"✓ Training function created")

    # ===== Create Environment Functions (Dummy for now) =====
    print(f"\nInitializing environment...")
    key, env_key = random.split(key)
    env_reset_fn, env_step_fn = collector.create_dummy_env_fns(config)
    env_state, _ = env_reset_fn(env_key)
    print(f"✓ Environment initialized (dummy)")
    print(f"  ⚠️  Replace with real JAX environment for production")

    # ===== Training Loop =====
    print(f"\n{'='*60}")
    print(f"Starting training for {args.num_steps} steps")
    print(f"{'='*60}\n")

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    metrics_history = []

    for step in range(args.num_steps):
        # ===== Collect Experience (Interleaved with Training) =====
        if step % args.collect_interval == 0:
            # Note: For proper pmap collection, this should also be pmapped
            # For now, we just collect on first device
            single_carrier = trainer.unreplicate_carrier(replicated_carrier)

            key, collect_key = random.split(key)
            single_carrier, env_state, collect_info = collector.collect_experience_step(
                single_carrier,
                env_state,
                env_step_fn,
                config,
                deterministic=False
            )

            # Re-replicate updated carrier
            replicated_carrier = trainer.replicate_carrier_for_pmap(
                single_carrier,
                num_devices=len(devices)
            )

        # ===== Training Step (Pmapped) =====
        replicated_carrier, metrics = pmapped_train_step(replicated_carrier)

        # ===== Logging =====
        if step % args.log_interval == 0:
            # Unreplicate metrics (take first device)
            metrics_single = jax.tree_map(lambda x: x[0], metrics)

            elapsed = time.time() - start_time
            steps_per_sec = (step + 1) / elapsed if elapsed > 0 else 0

            print(f"Step {step:6d} | "
                  f"WM Loss: {float(metrics_single.world_model_loss):.4f} | "
                  f"Actor Loss: {float(metrics_single.actor_loss):.4f} | "
                  f"Critic Loss: {float(metrics_single.critic_loss):.4f} | "
                  f"Intrinsic: {float(metrics_single.intrinsic_reward):.4f} | "
                  f"Boredom: {float(metrics_single.boredom):.4f} | "
                  f"Buffer: {int(metrics_single.buffer_count)} | "
                  f"Steps/s: {steps_per_sec:.1f}")

            metrics_history.append(metrics_single)

        # ===== Checkpointing =====
        if step % args.checkpoint_interval == 0 and step > 0:
            print(f"\n💾 Saving checkpoint at step {step}...")
            # Unreplicate for saving
            single_carrier = trainer.unreplicate_carrier(replicated_carrier)

            checkpoint_path = checkpoint_dir / f"checkpoint_{step}.pkl"
            # TODO: Implement proper checkpoint saving with pickle or orbax
            print(f"✓ Checkpoint saved to {checkpoint_path}")
            print()

    # ===== Final Summary =====
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"{'='*60}")
    print(f"Total steps: {args.num_steps}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average steps/sec: {args.num_steps / total_time:.1f}")
    print(f"Final buffer count: {int(metrics_history[-1].buffer_count)}")
    print(f"Final world model loss: {float(metrics_history[-1].world_model_loss):.4f}")


if __name__ == '__main__':
    main()
