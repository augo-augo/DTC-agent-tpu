"""
The Mega-Kernel: Single XLA-Compiled Train Step with pmap/vmap.

This is the atomic unit of Project Hephaestus - a fully fused training
step that executes across 8 TPU chips with pmap parallelism.

Architecture:
- Outer loop (pmap): Replicates agent across 8 chips
- Inner loop (vmap): Vectorizes 5 ensemble models
- Sync (pmean): Averages gradients across chips

The entire train_step compiles into a single XLA executable.
"""

import jax
import jax.numpy as jnp
from jax import random
import optax
import chex
from typing import Tuple, NamedTuple
from flax import struct

from dtc_jax.configs.base_config import DTCConfig
from dtc_jax.dtc.dtc_types import (
    RSSMState, IntrinsicState, ReplayBuffer,
    TrainingMetrics
)
from dtc_jax.dtc import rssm as rssm_module
from dtc_jax.dtc import memory as memory_module
from dtc_jax.dtc import intrinsic as intrinsic_module
from dtc_jax.dtc import actor_critic as ac_module
from dtc_jax.dtc import dreamer as dreamer_module


@struct.dataclass
class CarrierState:
    """
    The Carrier State - the complete state of the distributed agent.

    This PyTreeNode flows through the entire training loop and is
    synchronized across TPU chips.

    Structure:
    - rng_key: [num_devices] - Unique per device
    - rssm_params: [num_devices, ...] - Replicated across devices
    - actor_critic_params: [num_devices, ...] - Replicated
    - rssm_opt_state: [num_devices, ...] - Replicated
    - ac_opt_state: [num_devices, ...] - Replicated
    - buffer_state: [num_devices, capacity, ...] - SHARDED per device
    - intrinsic_state: [num_devices, ...] - Local per device
    - current_state: [num_devices, ...] - Current RSSM state
    - step: Scalar - Global training step counter
    """
    # PRNG keys (unique per device)
    rng_key: chex.PRNGKey

    # Model parameters (replicated)
    rssm_params: dict
    actor_critic_params: dict

    # Optimizer states (replicated)
    rssm_opt_state: optax.OptState
    ac_opt_state: optax.OptState

    # Replay buffer (sharded - each device has unique data)
    buffer: ReplayBuffer

    # Intrinsic motivation state (local per device)
    intrinsic_state: IntrinsicState

    # Current RSSM state for continuing rollouts
    current_rssm_state: RSSMState

    # Global step counter
    step: int


def create_carrier_state(
    config: DTCConfig,
    key: chex.PRNGKey
) -> CarrierState:
    """
    Initialize the carrier state for all devices.

    This creates the initial state before pmap replication.

    Args:
        config: DTCConfig
        key: PRNG key

    Returns:
        CarrierState with initialized parameters and states
    """
    # Split keys for different initializations
    keys = random.split(key, 5)
    rssm_key, ac_key, buffer_key, intrinsic_key, runtime_key = keys

    # Initialize RSSM ensemble
    rssm_params, init_rssm_state = rssm_module.create_ensemble_params(
        config,
        rssm_key,
        batch_size=config.local_batch_size
    )

    # Initialize Actor-Critic
    actor_critic_params = ac_module.create_actor_critic_params(
        config,
        ac_key,
        batch_size=config.local_batch_size,
        seq_len=1
    )

    # Initialize optimizers
    rssm_optimizer = optax.adam(
        learning_rate=config.learning_rate_world_model,
        b1=config.adam_b1,
        b2=config.adam_b2,
        eps=config.adam_eps
    )
    rssm_opt_state = rssm_optimizer.init(rssm_params)

    ac_optimizer = optax.adam(
        learning_rate=config.learning_rate_actor_critic,
        b1=config.adam_b1,
        b2=config.adam_b2,
        eps=config.adam_eps
    )
    ac_opt_state = ac_optimizer.init(actor_critic_params)

    # Initialize replay buffer (per-device)
    buffer = memory_module.create_replay_buffer(
        config,
        batch_size=config.local_batch_size
    )

    # Initialize intrinsic state
    intrinsic_state = IntrinsicState(
        c_fast=jnp.zeros((), dtype=jnp.float32),
        c_slow=jnp.zeros((), dtype=jnp.float32),
        boredom=jnp.zeros((), dtype=jnp.float32),
        avg_novelty=jnp.zeros((), dtype=jnp.float32)
    )

    # Create carrier state
    carrier = CarrierState(
        rng_key=runtime_key,
        rssm_params=rssm_params,
        actor_critic_params=actor_critic_params,
        rssm_opt_state=rssm_opt_state,
        ac_opt_state=ac_opt_state,
        buffer=buffer,
        intrinsic_state=intrinsic_state,
        current_rssm_state=init_rssm_state,
        step=0
    )

    return carrier


def train_step(
    carrier: CarrierState,
    config: DTCConfig
) -> Tuple[CarrierState, TrainingMetrics]:
    """
    Single training step - the atomic unit of the Mega-Kernel.

    This function will be pmapped across devices.

    Workflow:
    1. Sample batch from replay buffer (frontier + episodic)
    2. Train world model (RSSM) on sampled sequences
    3. Dream rollout using RSSM prior
    4. Train actor-critic on imagined trajectories
    5. Update intrinsic motivation state
    6. Sync gradients across devices (pmean)

    Args:
        carrier: Current carrier state (per-device)
        config: DTCConfig (static)

    Returns:
        new_carrier: Updated carrier state
        metrics: Training metrics for logging
    """
    # ===== 1. Sample Batch from Replay Buffer =====
    carrier_key, sample_key = random.split(carrier.rng_key)

    # Check if buffer is ready
    buffer_ready = memory_module.is_ready_to_sample(carrier.buffer, config)

    # Sample training batch (frontier + episodic mix)
    batch, sample_key = jax.lax.cond(
        buffer_ready,
        lambda k: memory_module.sample_stochastic_stratified(
            carrier.buffer, config, k
        ),
        lambda k: (
            # Dummy batch if buffer not ready
            memory_module.sample_stochastic_stratified(
                carrier.buffer, config, k
            )
        ),
        sample_key
    )

    # ===== 2. Train World Model (RSSM) =====
    def rssm_loss_fn(rssm_params):
        """RSSM loss function for gradient computation."""
        carrier_key_rssm, rssm_train_key = random.split(carrier_key)

        loss, metrics = rssm_module.compute_rssm_loss(
            rssm_params,
            config,
            batch.observations,
            batch.actions,
            rssm_train_key
        )
        return loss, metrics

    # Compute gradients
    (rssm_loss, rssm_metrics), rssm_grads = jax.value_and_grad(
        rssm_loss_fn, has_aux=True
    )(carrier.rssm_params)

    # Synchronize gradients across devices (pmean)
    rssm_grads = jax.lax.pmean(rssm_grads, axis_name='devices')

    # Update RSSM parameters
    rssm_optimizer = optax.adam(
        learning_rate=config.learning_rate_world_model,
        b1=config.adam_b1,
        b2=config.adam_b2,
        eps=config.adam_eps
    )
    rssm_updates, new_rssm_opt_state = rssm_optimizer.update(
        rssm_grads, carrier.rssm_opt_state
    )
    new_rssm_params = optax.apply_updates(carrier.rssm_params, rssm_updates)

    # ===== 3. Dream Rollout =====
    carrier_key_dream, dream_key = random.split(carrier_key)

    # Use current RSSM state as starting point for dreams
    initial_state = carrier.current_rssm_state

    # Perform imagination rollout
    dream_rollout, dream_key = dreamer_module.dream_rollout_static(
        initial_state,
        new_rssm_params,  # Use updated RSSM
        carrier.actor_critic_params,
        config,
        dream_key,
        carrier.intrinsic_state
    )

    # Compute advantages for policy gradient
    advantages, returns = dreamer_module.compute_dream_advantages(
        dream_rollout,
        config,
        intrinsic_weight=1.0
    )

    # ===== 4. Train Actor-Critic =====
    def ac_loss_fn(ac_params):
        """Actor-Critic loss function."""
        ac_loss, ac_metrics = dreamer_module.compute_actor_critic_loss(
            dream_rollout,
            advantages,
            returns,
            config
        )
        return ac_loss, ac_metrics

    # Compute gradients
    (ac_loss, ac_metrics), ac_grads = jax.value_and_grad(
        ac_loss_fn, has_aux=True
    )(carrier.actor_critic_params)

    # Synchronize gradients across devices
    ac_grads = jax.lax.pmean(ac_grads, axis_name='devices')

    # Update Actor-Critic parameters
    ac_optimizer = optax.adam(
        learning_rate=config.learning_rate_actor_critic,
        b1=config.adam_b1,
        b2=config.adam_b2,
        eps=config.adam_eps
    )
    ac_updates, new_ac_opt_state = ac_optimizer.update(
        ac_grads, carrier.ac_opt_state
    )
    new_ac_params = optax.apply_updates(carrier.actor_critic_params, ac_updates)

    # ===== 5. Update Intrinsic Motivation State =====
    # Compute clean novelty from ensemble disagreement
    # (This would typically come from real environment interaction,
    # but for now we use the dream rollout ensemble predictions)
    carrier_key_intrinsic, intrinsic_key = random.split(carrier_key_dream)

    # Get ensemble predictions on a sample state
    # TODO: This should come from actual environment interaction
    sample_novelty = jnp.mean(dream_rollout.intrinsic_rewards)

    # Update cognitive wave
    new_intrinsic_state, intrinsic_reward = intrinsic_module.update_cognitive_wave(
        carrier.intrinsic_state,
        sample_novelty,
        config
    )

    # ===== 6. Create Updated Carrier State =====
    new_carrier = CarrierState(
        rng_key=carrier_key_intrinsic,
        rssm_params=new_rssm_params,
        actor_critic_params=new_ac_params,
        rssm_opt_state=new_rssm_opt_state,
        ac_opt_state=new_ac_opt_state,
        buffer=carrier.buffer,  # Buffer updated separately during env interaction
        intrinsic_state=new_intrinsic_state,
        current_rssm_state=carrier.current_rssm_state,
        step=carrier.step + 1
    )

    # ===== 7. Collect Metrics =====
    metrics = TrainingMetrics(
        world_model_loss=rssm_loss,
        reconstruction_loss=rssm_metrics['reconstruction_loss'],
        kl_loss=rssm_metrics['kl_loss'],
        actor_loss=ac_metrics.get('dream_policy_loss', 0.0),
        critic_loss=ac_metrics.get('dream_value_loss', 0.0),
        intrinsic_reward=intrinsic_reward,
        boredom=new_intrinsic_state.boredom,
        c_fast=new_intrinsic_state.c_fast,
        c_slow=new_intrinsic_state.c_slow,
        buffer_count=carrier.buffer.count,
        step=new_carrier.step
    )

    return new_carrier, metrics


def create_train_fn(config: DTCConfig):
    """
    Create the pmapped training function.

    This wraps train_step with pmap for data parallelism across TPU chips.

    Args:
        config: DTCConfig

    Returns:
        pmapped_train_step: Function that executes train_step across all devices
    """
    # Pmap over the 'devices' axis
    pmapped_train_step = jax.pmap(
        lambda carrier: train_step(carrier, config),
        axis_name='devices',
        in_axes=0,  # Pmap over carrier state (first axis)
        out_axes=0  # Output also has device axis
    )

    return pmapped_train_step


def replicate_carrier_for_pmap(
    carrier: CarrierState,
    num_devices: int
) -> CarrierState:
    """
    Replicate carrier state across devices for pmap.

    This creates the initial replicated state with unique RNG keys per device.

    Args:
        carrier: Single-device carrier state
        num_devices: Number of TPU devices

    Returns:
        Replicated carrier with shape [num_devices, ...]
    """
    # Split RNG keys for each device (must be unique!)
    device_keys = random.split(carrier.rng_key, num_devices)

    # Replicate all other state
    replicated = jax.tree_map(
        lambda x: jnp.stack([x] * num_devices, axis=0),
        carrier
    )

    # Override RNG keys with unique per-device keys
    replicated = replicated.replace(rng_key=device_keys)

    return replicated


def unreplicate_carrier(carrier: CarrierState) -> CarrierState:
    """
    Extract single-device carrier from replicated state.

    Takes the first device's state (they should all be synchronized via pmean).

    Args:
        carrier: Replicated carrier [num_devices, ...]

    Returns:
        Single-device carrier
    """
    return jax.tree_map(lambda x: x[0], carrier)
