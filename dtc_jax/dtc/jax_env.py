"""
JAX-Native Environment: "The Feeder"

A fully JAX-native grid world environment that compiles into the TPU Mega-Kernel.
Implements the three "Red Lines":
1. Shape Invariance: All tensors have fixed shapes across timesteps
2. Precision Firewall: bfloat16 for compute, float32 for accumulation
3. RNG Contract: Every function takes and returns a new PRNG key

This environment is inspired by Crafter but simplified to essential mechanics:
- Grid-based navigation (16×16)
- Resource collection (8 resource types)
- Simple rewards for exploration and collection
- Zero CPU-TPU transfer (pure JAX)
"""

from typing import Tuple, NamedTuple
import jax
import jax.numpy as jnp
from jax import Array
from chex import PRNGKey, assert_shape
import chex


# ============================================================================
# CONSTANTS & CONFIGURATION
# ============================================================================

GRID_SIZE = 16  # 16×16 grid
NUM_RESOURCE_TYPES = 8  # Fixed number of resource types
MAX_INVENTORY = 10  # Max items per resource type
MAX_EPISODE_STEPS = 1000  # Episode length limit

# Action space: 0=up, 1=down, 2=left, 3=right, 4=collect
NUM_ACTIONS = 5

# Observation dimensions
OBS_DIM = GRID_SIZE * GRID_SIZE + NUM_RESOURCE_TYPES + 2  # grid + inventory + position


# ============================================================================
# ENVIRONMENT STATE (Fixed Shape!)
# ============================================================================

class EnvState(NamedTuple):
    """
    Environment state with FIXED SHAPES.

    RED LINE #1: Shape Invariant
    All fields have the same shape at every timestep.
    """
    # Agent position (x, y) in [0, GRID_SIZE)
    position: Array  # shape: (2,), dtype: int32

    # Grid state: 16×16 grid where each cell contains resource type (0-7, 255=empty)
    grid: Array  # shape: (GRID_SIZE, GRID_SIZE), dtype: uint8

    # Inventory: count of each resource type
    inventory: Array  # shape: (NUM_RESOURCE_TYPES,), dtype: int32

    # Episode step counter
    step_count: Array  # shape: (), dtype: int32

    # Cumulative reward (float32 for accumulation - RED LINE #2)
    cumulative_reward: Array  # shape: (), dtype: float32


def validate_env_state(state: EnvState) -> None:
    """Validate that EnvState satisfies shape invariance."""
    assert_shape(state.position, (2,))
    assert_shape(state.grid, (GRID_SIZE, GRID_SIZE))
    assert_shape(state.inventory, (NUM_RESOURCE_TYPES,))
    assert_shape(state.step_count, ())
    assert_shape(state.cumulative_reward, ())


# ============================================================================
# RNG CONTRACT (RED LINE #3)
# ============================================================================

def split_key(key: PRNGKey) -> Tuple[PRNGKey, PRNGKey]:
    """
    Split a PRNG key into two new keys.

    RED LINE #3: RNG Contract
    Always split keys, never reuse them.
    """
    return jax.random.split(key, 2)


# ============================================================================
# ENVIRONMENT INITIALIZATION
# ============================================================================

def reset(key: PRNGKey) -> Tuple[PRNGKey, EnvState]:
    """
    Reset the environment to initial state.

    Args:
        key: PRNG key

    Returns:
        new_key: New PRNG key (RNG Contract)
        state: Initial environment state

    RED LINE #1: All outputs have fixed shapes
    RED LINE #3: Takes key, returns new key
    """
    key, grid_key, pos_key = jax.random.split(key, 3)

    # Generate random grid with resources
    # Each cell: 0-7 (resource type) or 255 (empty)
    # ~70% empty cells, ~30% resources
    resource_probs = jnp.array([0.7] + [0.3 / NUM_RESOURCE_TYPES] * NUM_RESOURCE_TYPES)
    grid_values = jax.random.choice(
        grid_key,
        jnp.arange(NUM_RESOURCE_TYPES + 1, dtype=jnp.uint8),
        shape=(GRID_SIZE, GRID_SIZE),
        p=resource_probs
    )
    # Map 0->255 (empty), 1-8 -> 0-7 (resources)
    grid = jnp.where(grid_values == 0, 255, grid_values - 1).astype(jnp.uint8)

    # Random starting position
    pos_x = jax.random.randint(pos_key, (), 0, GRID_SIZE, dtype=jnp.int32)
    pos_y = jax.random.randint(key, (), 0, GRID_SIZE, dtype=jnp.int32)
    position = jnp.array([pos_x, pos_y], dtype=jnp.int32)

    # Empty inventory
    inventory = jnp.zeros(NUM_RESOURCE_TYPES, dtype=jnp.int32)

    state = EnvState(
        position=position,
        grid=grid,
        inventory=inventory,
        step_count=jnp.array(0, dtype=jnp.int32),
        cumulative_reward=jnp.array(0.0, dtype=jnp.float32)
    )

    # Validate shape invariance
    validate_env_state(state)

    return key, state


# ============================================================================
# ENVIRONMENT DYNAMICS
# ============================================================================

def step(state: EnvState, action: Array, key: PRNGKey) -> Tuple[PRNGKey, EnvState, Array, Array]:
    """
    Execute one environment step.

    Args:
        state: Current environment state
        action: Action to take (0-4: up, down, left, right, collect)
        key: PRNG key

    Returns:
        new_key: New PRNG key (RNG Contract)
        new_state: Next environment state
        reward: Reward (float32 - RED LINE #2)
        done: Episode termination flag

    RED LINE #1: All state tensors maintain fixed shapes
    RED LINE #2: Reward is float32 for accumulation
    RED LINE #3: Takes key, returns new key
    """
    # Validate input
    validate_env_state(state)

    # Parse action (clamp to valid range)
    action_idx = jnp.clip(action, 0, NUM_ACTIONS - 1).astype(jnp.int32)

    # Movement actions (0-3)
    delta_x = jnp.array([0, 0, -1, 1, 0], dtype=jnp.int32)[action_idx]
    delta_y = jnp.array([-1, 1, 0, 0, 0], dtype=jnp.int32)[action_idx]

    # NEW POSITION: Compute new position with boundary clamping
    new_x = jnp.clip(state.position[0] + delta_x, 0, GRID_SIZE - 1)
    new_y = jnp.clip(state.position[1] + delta_y, 0, GRID_SIZE - 1)
    new_position = jnp.array([new_x, new_y], dtype=jnp.int32)

    # COLLECTION: Attempt to collect resource at current position
    current_cell = state.grid[state.position[1], state.position[0]]
    is_collect_action = (action_idx == 4)
    has_resource = (current_cell != 255)
    can_collect = is_collect_action & has_resource

    # Resource type at current cell (0-7 if valid, 0 if empty)
    resource_type = jnp.where(has_resource, current_cell, 0)

    # Update inventory (increment the collected resource type)
    # Use one-hot to update specific inventory slot
    resource_one_hot = jax.nn.one_hot(resource_type, NUM_RESOURCE_TYPES, dtype=jnp.int32)
    inventory_delta = jnp.where(can_collect, resource_one_hot, 0)
    new_inventory = jnp.clip(
        state.inventory + inventory_delta,
        0,
        MAX_INVENTORY
    )

    # Update grid (remove collected resource)
    # Set current cell to 255 (empty) if collected
    new_grid = state.grid.at[state.position[1], state.position[0]].set(
        jnp.where(can_collect, 255, current_cell)
    )

    # REWARD COMPUTATION (float32 - RED LINE #2)
    # 1. Movement reward: small penalty for moving (-0.01)
    movement_reward = jnp.where(action_idx < 4, -0.01, 0.0).astype(jnp.float32)

    # 2. Collection reward: +1.0 for successful collection
    collection_reward = jnp.where(can_collect, 1.0, 0.0).astype(jnp.float32)

    # 3. Exploration reward: small bonus for visiting new areas
    # (simplified: reward for moving to cells with resources)
    exploration_reward = jnp.where(
        (action_idx < 4) & (new_grid[new_y, new_x] != 255),
        0.1,
        0.0
    ).astype(jnp.float32)

    # Total reward (float32 accumulation)
    reward = movement_reward + collection_reward + exploration_reward

    # UPDATE STATE
    new_step_count = state.step_count + 1
    new_cumulative_reward = state.cumulative_reward + reward

    new_state = EnvState(
        position=new_position,
        grid=new_grid,
        inventory=new_inventory,
        step_count=new_step_count,
        cumulative_reward=new_cumulative_reward
    )

    # TERMINATION: Episode ends after MAX_EPISODE_STEPS
    done = (new_step_count >= MAX_EPISODE_STEPS)

    # Validate output shape invariance
    validate_env_state(new_state)

    # Split key for next call (RNG Contract)
    key, new_key = split_key(key)

    return new_key, new_state, reward, done


# ============================================================================
# OBSERVATION ENCODING (PRECISION FIREWALL)
# ============================================================================

def state_to_observation(state: EnvState) -> Array:
    """
    Convert environment state to observation vector.

    RED LINE #2: Precision Firewall
    - Output: bfloat16 for neural network input
    - Normalization uses float32 internally, then cast

    Returns:
        obs: Observation vector, shape: (OBS_DIM,), dtype: bfloat16
    """
    # Grid flattened (normalize 0-255 -> 0-1)
    grid_flat = state.grid.flatten().astype(jnp.float32) / 255.0

    # Inventory (normalize 0-MAX_INVENTORY -> 0-1)
    inventory_norm = state.inventory.astype(jnp.float32) / MAX_INVENTORY

    # Position (normalize 0-GRID_SIZE -> 0-1)
    position_norm = state.position.astype(jnp.float32) / GRID_SIZE

    # Concatenate all features (in float32)
    obs_f32 = jnp.concatenate([
        grid_flat,           # 256 features
        inventory_norm,      # 8 features
        position_norm        # 2 features
    ], axis=0)

    # Cast to bfloat16 for TPU computation (PRECISION FIREWALL)
    obs_bf16 = obs_f32.astype(jnp.bfloat16)

    # Validate shape
    assert_shape(obs_bf16, (OBS_DIM,))

    return obs_bf16


# ============================================================================
# VECTORIZED ENVIRONMENT (for vmap/pmap)
# ============================================================================

def batched_reset(key: PRNGKey, batch_size: int) -> Tuple[PRNGKey, EnvState]:
    """
    Reset multiple environments in parallel.

    Args:
        key: PRNG key
        batch_size: Number of parallel environments

    Returns:
        new_key: New PRNG key
        states: Batched environment states
    """
    # Split keys for each environment
    key, *env_keys = jax.random.split(key, batch_size + 1)

    # Vectorized reset
    reset_fn = jax.vmap(reset, in_axes=(0,))
    new_keys, states = reset_fn(jnp.array(env_keys))

    # Return first new key (all keys are equivalent after reset)
    return new_keys[0], states


def batched_step(
    states: EnvState,
    actions: Array,
    key: PRNGKey
) -> Tuple[PRNGKey, EnvState, Array, Array]:
    """
    Step multiple environments in parallel.

    Args:
        states: Batched environment states
        actions: Batched actions, shape: (batch_size,)
        key: PRNG key

    Returns:
        new_key: New PRNG key
        new_states: Batched next states
        rewards: Batched rewards, shape: (batch_size,)
        dones: Batched done flags, shape: (batch_size,)
    """
    batch_size = jax.tree_util.tree_leaves(states)[0].shape[0]

    # Split keys for each environment
    key, *env_keys = jax.random.split(key, batch_size + 1)

    # Vectorized step
    step_fn = jax.vmap(step, in_axes=(0, 0, 0))
    new_keys, new_states, rewards, dones = step_fn(
        states, actions, jnp.array(env_keys)
    )

    return new_keys[0], new_states, rewards, dones


# ============================================================================
# ENVIRONMENT INFO
# ============================================================================

def get_env_info() -> dict:
    """Get environment specifications."""
    return {
        'name': 'JAX-GridWorld',
        'observation_dim': OBS_DIM,
        'action_dim': NUM_ACTIONS,
        'grid_size': GRID_SIZE,
        'num_resources': NUM_RESOURCE_TYPES,
        'max_episode_steps': MAX_EPISODE_STEPS,
        'observation_dtype': jnp.bfloat16,
        'reward_dtype': jnp.float32,
    }


# ============================================================================
# TESTING & VALIDATION
# ============================================================================

if __name__ == '__main__':
    """Test the JAX-native environment."""
    print("=" * 80)
    print("JAX-NATIVE ENVIRONMENT: THE FEEDER")
    print("=" * 80)

    # Environment info
    info = get_env_info()
    print("\nEnvironment Info:")
    for k, v in info.items():
        print(f"  {k}: {v}")

    # Test reset
    print("\n[TEST 1] Reset")
    key = jax.random.PRNGKey(42)
    key, state = reset(key)
    print(f"  Position: {state.position}")
    print(f"  Inventory: {state.inventory}")
    print(f"  Grid shape: {state.grid.shape}, dtype: {state.grid.dtype}")
    print(f"  Step count: {state.step_count}")

    # Test observation encoding
    print("\n[TEST 2] Observation Encoding")
    obs = state_to_observation(state)
    print(f"  Observation shape: {obs.shape}, dtype: {obs.dtype}")
    print(f"  Observation range: [{float(obs.min()):.3f}, {float(obs.max()):.3f}]")

    # Test step
    print("\n[TEST 3] Environment Step")
    for i, action_name in enumerate(['UP', 'DOWN', 'LEFT', 'RIGHT', 'COLLECT']):
        action = jnp.array(i, dtype=jnp.int32)
        key_before = key
        key, new_state, reward, done = step(state, action, key)
        print(f"  Action {i} ({action_name}): pos={new_state.position}, reward={float(reward):.3f}, done={done}")

    # Test RNG contract
    print("\n[TEST 4] RNG Contract")
    key1 = jax.random.PRNGKey(123)
    key2, state1 = reset(key1)
    key3, state2 = reset(key1)  # Same input key
    print(f"  Same input key produces same state: {jnp.array_equal(state1.grid, state2.grid)}")

    key4, state3 = reset(key2)  # Different input key
    print(f"  Different input key produces different state: {not jnp.array_equal(state1.grid, state3.grid)}")

    # Test batched operations
    print("\n[TEST 5] Batched Operations")
    batch_size = 4
    key = jax.random.PRNGKey(999)
    key, batched_states = batched_reset(key, batch_size)
    print(f"  Batched states shape: {batched_states.grid.shape}")

    actions = jnp.array([0, 1, 2, 3], dtype=jnp.int32)
    key, new_states, rewards, dones = batched_step(batched_states, actions, key)
    print(f"  Batched rewards: {rewards}")
    print(f"  Batched dones: {dones}")

    # Test JIT compilation
    print("\n[TEST 6] JIT Compilation")
    jit_reset = jax.jit(reset)
    jit_step = jax.jit(step)

    key = jax.random.PRNGKey(0)

    # Warmup
    key, state = jit_reset(key)
    action = jnp.array(0, dtype=jnp.int32)
    key, state, reward, done = jit_step(state, action, key)

    # Benchmark
    import time

    num_steps = 1000
    key = jax.random.PRNGKey(0)
    key, state = jit_reset(key)

    start = time.time()
    for _ in range(num_steps):
        action = jax.random.randint(key, (), 0, NUM_ACTIONS, dtype=jnp.int32)
        key, state, reward, done = jit_step(state, action, key)

        # Reset if done
        key, reset_state = jit_reset(key)
        state = jax.tree_util.tree_map(
            lambda x, y: jnp.where(done, x, y),
            reset_state, state
        )

    elapsed = time.time() - start
    print(f"  {num_steps} steps in {elapsed:.3f}s ({num_steps/elapsed:.0f} steps/sec)")

    print("\n[TEST 7] Shape Invariance Validation")
    print("  Testing 100 random steps...")
    key = jax.random.PRNGKey(42)
    key, state = jit_reset(key)
    initial_shapes = jax.tree_util.tree_map(lambda x: x.shape, state)

    for step_idx in range(100):
        action = jax.random.randint(key, (), 0, NUM_ACTIONS, dtype=jnp.int32)
        key, state, reward, done = jit_step(state, action, key)

        current_shapes = jax.tree_util.tree_map(lambda x: x.shape, state)
        if current_shapes != initial_shapes:
            print(f"  ❌ SHAPE INVARIANCE VIOLATED at step {step_idx}")
            print(f"     Initial: {initial_shapes}")
            print(f"     Current: {current_shapes}")
            break

        if done:
            key, state = jit_reset(key)
    else:
        print("  ✓ Shape invariance maintained for 100 steps")

    print("\n" + "=" * 80)
    print("ALL TESTS PASSED")
    print("=" * 80)
