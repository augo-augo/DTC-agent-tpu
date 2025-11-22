# Installation Guide - DTC 3.0 Hephaestus

## Prerequisites

- Python 3.9 or higher
- TPU v4-8 pod (for production) OR CPU/GPU (for development/testing)
- 50GB+ disk space for checkpoints and logs

---

## Installation Steps

### 1. Clone Repository

```bash
git clone <your-repo-url>
cd DTC-agent-tpu/dtc_jax
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Linux/Mac
# OR
venv\Scripts\activate  # On Windows
```

### 3. Install Dependencies

#### For TPU (Production)

```bash
pip install -U pip
pip install -r requirements.txt
```

**Note:** JAX TPU installation requires Google Cloud TPU runtime. See: https://cloud.google.com/tpu/docs/run-calculation-jax

#### For CPU/GPU (Development/Testing)

```bash
# CPU-only (lightweight)
pip install "jax[cpu]>=0.4.20"
pip install jaxlib>=0.4.20 flax>=0.7.5 optax>=0.1.7 distrax>=0.1.3 chex>=0.1.83 numpy>=1.24.0

# OR GPU (CUDA 12)
pip install "jax[cuda12]>=0.4.20" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install jaxlib>=0.4.20 flax>=0.7.5 optax>=0.1.7 distrax>=0.1.3 chex>=0.1.83 numpy>=1.24.0
```

### 4. Verify Installation

```bash
python -c "import jax; print(f'JAX version: {jax.__version__}'); print(f'Devices: {jax.devices()}')"
```

Expected output:
```
JAX version: 0.4.20 (or higher)
Devices: [CpuDevice(id=0)] or [TpuDevice(id=0), TpuDevice(id=1), ...]
```

---

## Testing

### Quick Test (No dependencies required)

The following tests check code structure without running:

```bash
# Check Python syntax
python -m py_compile dtc/*.py

# Check imports (requires dependencies)
python -c "from dtc_jax.configs.base_config import DTCConfig; print('✓ Config OK')"
python -c "from dtc_jax.dtc import rssm; print('✓ RSSM OK')"
```

### Unit Tests (Requires JAX)

```bash
# Run RSSM tests
pytest test_rssm.py -v

# Expected: 6 tests passed
```

### Pipeline Test (End-to-End)

```bash
# Full pipeline test with dummy environment
python test_pipeline.py
```

This will test:
1. ✅ Carrier state initialization
2. ✅ Single training step (no pmap)
3. ✅ Pmapped training across devices
4. ✅ Experience collection
5. ✅ Memory sampling

Expected runtime: ~30 seconds on CPU, <5 seconds on TPU

### Full Training Run (Minimal)

```bash
# Train for 1000 steps with dummy environment
python train.py --num_steps 1000 --log_interval 100

# Should complete in <2 minutes on CPU
```

---

## Troubleshooting

### Import Error: "No module named 'jax'"

**Solution:** Install dependencies (see step 3 above)

### JAX Cannot Find TPU

**Symptom:**
```
Devices: [CpuDevice(id=0)]
```

**Solutions:**
1. Verify you're running on TPU VM: `gcloud compute tpus list`
2. Check TPU configuration: `echo $TPU_NAME`
3. Restart TPU runtime: `sudo systemctl restart tpu-runtime`

### Import Error: "cannot import name 'PRNGKey' from 'dtc_jax.dtc.dtc_types'"

**Solution:** Ensure you're in the correct directory:
```bash
cd /path/to/DTC-agent-tpu/dtc_jax
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### OOM During Test

**Symptom:** "Out of memory" during buffer allocation

**Solution:** Reduce buffer size in `test_pipeline.py` or increase system RAM

---

## Running on Google Cloud TPU

### 1. Create TPU VM

```bash
gcloud compute tpus tpu-vm create dtc-tpu \
  --zone=us-central2-b \
  --accelerator-type=v4-8 \
  --version=tpu-ubuntu2204-base \
  --scopes=cloud-platform
```

### 2. SSH into TPU VM

```bash
gcloud compute tpus tpu-vm ssh dtc-tpu --zone=us-central2-b
```

### 3. Setup Environment

```bash
# Install Python 3.10
sudo apt update
sudo apt install python3.10 python3.10-venv python3-pip -y

# Clone repo
git clone <your-repo-url>
cd DTC-agent-tpu/dtc_jax

# Create venv
python3.10 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 4. Verify TPU Access

```bash
python -c "import jax; print(jax.devices())"

# Expected: [TpuDevice(id=0), TpuDevice(id=1), ..., TpuDevice(id=7)]
```

### 5. Run Training

```bash
python train.py --num_steps 1000000 --log_interval 100
```

Monitor output for steps/sec (target: >50,000)

---

## Development Workflow

### Code Changes

After modifying code:

```bash
# Re-run tests
python test_pipeline.py

# If tests pass, run short training
python train.py --num_steps 100 --log_interval 10
```

### Profiling

To profile TPU performance:

```python
# Add to train.py before training loop:
with jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=True):
    for step in range(100):
        replicated_carrier, metrics = pmapped_train_step(replicated_carrier)

# Open the generated Perfetto link in browser
```

### Debugging

Enable JAX debug mode:

```bash
export JAX_DEBUG_NANS=True
export JAX_DISABLE_JIT=1  # For debugging (slow!)

python test_pipeline.py
```

---

## Next Steps

1. ✅ Verify installation: `python test_pipeline.py`
2. ✅ Run short training: `python train.py --num_steps 1000`
3. 🔄 Replace dummy environment with real env (Jumanji/Brax)
4. 🔄 Run full training: `python train.py --num_steps 1000000`
5. 🔄 Analyze checkpoints and logs
6. 🔄 Hyperparameter tuning

---

## Support

For JAX/TPU issues:
- JAX docs: https://jax.readthedocs.io
- TPU docs: https://cloud.google.com/tpu/docs

For DTC 3.0 implementation questions:
- See `README_HEPHAESTUS.md`
- Review code comments in `dtc/` modules

---

**Status:** Ready for testing on TPU hardware
