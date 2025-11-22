# Project Hephaestus: DTC 3.0 JAX/TPU Architecture

**A single-graph, XLA-compiled cognitive architecture designed for >50,000 steps/second on TPU v4-8.**

---

## Overview

This implementation follows the Hephaestus technical specification for building a high-performance model-based reinforcement learning agent on TPU using JAX. The system treats the TPU Pod as a **Data-Parallel Cluster** with sharded replay buffers and synchronized gradient updates.

### Key Performance Characteristics

- **Global Batch:** 256 (32 per chip on TPU v4-8)
- **Parallelism:**
  - Outer loop (`pmap`): 8-way data parallelism across TPU chips
  - Inner loop (`vmap`): 5-model ensemble vectorization
  - Gradient sync (`pmean`): High-speed ICI synchronization
- **Target:** >50,000 training steps per second

---

## System Topology: The Sharded Brain

```
┌─────────────────────────────────────────────────────────────┐
│                      TPU Pod (8 Chips)                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Chip 0           Chip 1           ...          Chip 7      │
│  ┌─────────┐     ┌─────────┐                  ┌─────────┐  │
│  │ Agent   │     │ Agent   │                  │ Agent   │  │
│  │ Batch=32│     │ Batch=32│       ...        │ Batch=32│  │
│  ├─────────┤     ├─────────┤                  ├─────────┤  │
│  │ Buffer  │     │ Buffer  │                  │ Buffer  │  │
│  │ (HBM)   │     │ (HBM)   │       ...        │ (HBM)   │  │
│  └────┬────┘     └────┬────┘                  └────┬────┘  │
│       │               │                            │        │
│       └───────────────┴────── pmean sync ──────────┘        │
│                    (gradient averaging)                     │
└─────────────────────────────────────────────────────────────┘
```

Each chip:
- Runs its own agent replica
- Maintains a unique slice of the replay buffer in HBM
- Only communicates gradients (not data) via high-speed interconnects

---

## Architecture Components

### 1. The Mega-Kernel (`trainer.py`)

The atomic unit of training - a single XLA-compiled function that:

1. Samples from sharded replay buffers (frontier + episodic mix)
2. Trains the RSSM world model on real data
3. Performs imagination rollouts in latent space
4. Updates actor-critic via policy gradients
5. Synchronizes gradients across chips (`pmean`)

**Location:** `dtc_jax/dtc/trainer.py`

**Key Functions:**
- `train_step()`: Single training iteration (pmapped)
- `create_carrier_state()`: Initialize complete system state
- `create_train_fn()`: Wrap train_step with pmap

### 2. RSSM World Model (`rssm.py`)

Vectorized ensemble of recurrent state-space models:

```
h_t = GRU(h_{t-1}, [z_{t-1}, a_{t-1}])     # Deterministic state
z_t ~ p(z_t | h_t)                          # Prior (imagination)
z_t ~ q(z_t | h_t, o_t)                     # Posterior (training)
o_t = decoder(h_t, z_t)                     # Reconstruction
r_t = reward_predictor(h_t, z_t)            # Reward model
```

**Ensemble Implementation:**
- Single module definition
- Parameters stacked 5 times
- `jax.vmap` over ensemble dimension
- All members run in parallel on MXUs

**Location:** `dtc_jax/dtc/rssm.py`

### 3. Memory: Stochastic Stratified Sampling (`memory.py`)

**O(1) sampling** that balances recency and coverage:

- **Frontier Batch (50%):** `randint(ptr - frontier_size, ptr)` - Recent discoveries
- **Episodic Batch (50%):** `randint(0, count)` - Historical coverage

**Why this works:** Forces the model to constantly verify against its entire history ("rehearsal") while integrating recent findings ("exploration"), without expensive similarity search.

**Location:** `dtc_jax/dtc/memory.py`

### 4. Actor-Critic with Salience Pooling (`actor_critic.py`)

**Salience Mechanism:**
```
Z ∈ ℝ^[batch, seq_len, dim]  →  Salience Net  →  s ∈ ℝ^[batch, seq_len, 1]
W = softmax(s)
z_global = Σ(Z ⊙ W)
```

This creates a learned bottleneck that forces the agent to compress the scene into what's "interesting" based on novelty.

**Networks:**
- **Actor:** Gaussian policy with tanh squashing
- **Critic:** Value function for advantage estimation
- **Both** operate on salience-pooled latent representations

**Location:** `dtc_jax/dtc/actor_critic.py`

### 5. Dream Rollout Engine (`dreamer.py`)

Model-based imagination using RSSM prior:

1. Start from current latent state
2. Sample actions from policy
3. Predict next states using RSSM (no observations!)
4. Compute intrinsic rewards from ensemble disagreement
5. Estimate values and advantages (GAE)
6. Train policy via imagined trajectories

**Adaptive Horizon:** Length modulated by boredom (8-64 steps)

**Location:** `dtc_jax/dtc/dreamer.py`

### 6. Intrinsic Motivation (`intrinsic.py`)

**Dual-Timescale Competence:**

```
Clean Novelty = Var(ensemble_means) - Mean(ensemble_vars²)
C_fast = EMA(novelty, α=0.1)     # Quick response
C_slow = EMA(novelty, α=0.005)   # Long-term trend
Competence = ReLU(C_slow - C_fast)
```

**Boredom Mechanism:**
- Accumulates when competence < threshold
- Triggers horizon reduction to save compute
- Decays when learning resumes

**Location:** `dtc_jax/dtc/intrinsic.py`

---

## Data Flow

```
Environment → Observation → RSSM Encoder → Latent State
                                              ↓
                                         Replay Buffer
                                          (sharded)
                                              ↓
                           Sample ← Frontier + Episodic ← O(1)
                              ↓
                         RSSM Training
                          (posterior)
                              ↓
                      Updated World Model
                              ↓
                       Dream Rollout ←──────┐
                        (prior only)        │
                              ↓              │
                    Actor-Critic Training   │
                    (policy gradient)       │
                              ↓              │
                      Intrinsic Motivation  │
                              │              │
                              └─ Boredom ────┘
                            (adaptive horizon)
```

---

## File Structure

```
dtc_jax/
├── configs/
│   └── base_config.py          # Frozen config for XLA compilation
├── dtc/
│   ├── dtc_types.py            # JAX-compatible type definitions
│   ├── rssm.py                 # World model with ensemble
│   ├── intrinsic.py            # Intrinsic motivation & boredom
│   ├── memory.py               # Replay buffer with stratified sampling
│   ├── actor_critic.py         # Policy & value with salience pooling
│   ├── dreamer.py              # Imagination rollouts
│   ├── trainer.py              # Mega-kernel & pmap training
│   └── collector.py            # Environment interaction
├── train.py                    # Main training loop
└── test_rssm.py                # Comprehensive RSSM tests
```

---

## Usage

### Quick Start (Dummy Environment)

```bash
cd dtc_jax
python train.py --num_steps 10000 --log_interval 100
```

This runs with a dummy environment for testing the pipeline.

### Production Setup (JAX Environment)

For maximum TPU performance, replace the dummy environment with a JAX-native environment:

**Option 1: Jumanji (Discrete)**
```python
import jumanji
env = jumanji.make('Snake-v0')
```

**Option 2: Brax (Continuous)**
```python
import brax.envs
env = brax.envs.create('ant')
```

**Option 3: Custom JAX Environment**

Implement your environment with JAX operations:

```python
class MyJAXEnv:
    def reset(self, key):
        # Return state, observation (JAX arrays)
        pass

    def step(self, state, action):
        # Return new_state, obs, reward, done (JAX arrays)
        pass
```

Update `collector.py` to use your environment.

### Arguments

```bash
python train.py \
  --seed 42 \
  --num_steps 1000000 \
  --collect_interval 10 \      # Collect experience every N steps
  --log_interval 100 \          # Log metrics every N steps
  --checkpoint_interval 10000 \ # Save checkpoint every N steps
  --checkpoint_dir ./checkpoints
```

---

## Configuration

Edit `configs/base_config.py` to adjust hyperparameters:

```python
@chex.dataclass(frozen=True)
class DTCConfig:
    # Environment
    obs_dim: int = 64
    action_dim: int = 8

    # Training
    global_batch_size: int = 256
    num_tpu_cores: int = 8
    local_batch_size: int = 32  # global / num_cores

    # World Model
    ensemble_size: int = 5
    latent_dim_deterministic: int = 1024
    latent_dim_stochastic: int = 32

    # Dream
    max_dream_horizon: int = 64
    min_horizon: int = 8

    # Learning Rates
    learning_rate_world_model: float = 3e-4
    learning_rate_actor_critic: float = 1e-4

    # ... (see file for all options)
```

**⚠️ Important:** Config must be frozen for XLA compilation. All shapes must be static.

---

## The "Red Lines" (Do Not Cross)

### 1. Shape Invariance

Tensors at step T must have identical shapes to step T+1. No growing lists.

**Bad:**
```python
trajectory.append(state)  # Dynamic shape!
```

**Good:**
```python
trajectory[t] = state  # Pre-allocated array
```

### 2. Precision Firewall

- **Computation (matmuls, convolutions):** `bfloat16` for throughput
- **Accumulation (reductions, variances, EMAs):** `float32` for precision

**Why:** Prevents "zero-variance" bug where noise appears as signal.

### 3. RNG Contract

Every function takes a key and returns a new key. Never reuse keys.

**Bad:**
```python
x = random.normal(key, shape)
y = random.normal(key, shape)  # Correlated!
```

**Good:**
```python
key, x_key = random.split(key)
x = random.normal(x_key, shape)
key, y_key = random.split(key)
y = random.normal(y_key, shape)
```

---

## Performance Optimization

### Current Status

- ✅ Single-graph XLA compilation
- ✅ pmap parallelism across 8 chips
- ✅ vmap ensemble vectorization
- ✅ O(1) memory sampling
- ✅ Static shapes throughout

### Expected Bottlenecks

1. **CPU Environment:** If using non-JAX environment, CPU-TPU transfer dominates
   - **Solution:** Port environment logic to JAX
2. **Gradient Synchronization:** Large models may have pmean overhead
   - **Solution:** Gradient compression or async updates (advanced)
3. **Buffer I/O:** If buffer operations are not pure JAX
   - **Solution:** Already handled - all buffer ops are JAX

### Profiling

```python
# Add to train.py
with jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=True):
    for step in range(100):
        replicated_carrier, metrics = pmapped_train_step(replicated_carrier)
```

Open the generated Perfetto link to visualize XLA graph execution.

---

## Testing

### Run RSSM Tests

```bash
cd dtc_jax
python -m pytest test_rssm.py -v
```

Tests include:
- ✅ Ensemble parameter shapes
- ✅ Forward pass correctness
- ✅ Ensemble diversity
- ✅ Prior-only mode
- ✅ Uncertainty quantification
- ✅ Loss computation & gradients

### Quick Sanity Check

```bash
python train.py --num_steps 100 --log_interval 10
```

Should complete in <30s without errors.

---

## Troubleshooting

### OOM on TPU

**Symptom:** "Out of memory" during buffer allocation

**Solutions:**
1. Reduce `replay_capacity` in config
2. Reduce `global_batch_size`
3. Reduce `max_dream_horizon`

### Slow Training (<<50k steps/s)

**Symptom:** Steps/sec far below target

**Likely Causes:**
1. Using CPU environment → Port to JAX
2. XLA not compiling → Check for Python control flow in train_step
3. Frequent host-device sync → Remove `.block_until_ready()` calls

### NaN Losses

**Symptom:** Losses become NaN after N steps

**Likely Causes:**
1. Log-std not clamped → Check `log_std_min/max` in config
2. Division by zero → Add `epsilon` to variance computations
3. Gradient explosion → Reduce learning rates or add gradient clipping

### PMean Axis Errors

**Symptom:** "NameError: axis name 'devices' not in scope"

**Solution:** Ensure `pmean` only called inside pmapped function

---

## Implementation Phases (Completed)

- ✅ **Phase 1:** Carrier state and type system
- ✅ **Phase 2:** RSSM with ensemble (tested)
- ✅ **Phase 3:** Intrinsic motivation (tested)
- ✅ **Phase 4:** Replay buffer with stratified sampling
- ✅ **Phase 5:** Actor-critic with salience pooling
- ✅ **Phase 6:** Dream rollout engine
- ✅ **Phase 7:** Mega-kernel with pmap
- ✅ **Phase 8:** Main training loop

---

## Next Steps

1. **Integrate Real Environment:** Replace dummy env with Jumanji/Brax
2. **Checkpoint Management:** Implement save/load with Orbax
3. **Distributed Logging:** Add TensorBoard or WandB integration
4. **Hyperparameter Tuning:** Sweep learning rates, horizon, ensemble size
5. **Benchmark:** Measure actual steps/sec on TPU v4-8

---

## Citation

This implementation follows the technical specification:

```
Project Hephaestus: DTC 3.0 JAX/TPU Architecture
Technical Specification Document (Release Candidate 1)
Objective: >50,000 steps/sec on TPU v4-8
```

Core architectural principles:
- Hafner et al. (2023) - Dreamer V3
- Burda et al. (2018) - Exploration by Random Network Distillation (epistemic uncertainty)
- Schulman et al. (2017) - Proximal Policy Optimization (GAE)
- Pathak et al. (2017) - Curiosity-driven Exploration (intrinsic motivation)

---

## License

See `LICENSE` file in repository root.

---

## Contact

For questions about this implementation, please open an issue in the repository.

For TPU access and optimization, see: https://cloud.google.com/tpu

---

**Status:** ✅ Full pipeline implemented and ready for testing on TPU hardware.
