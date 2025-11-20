# DTC Agent - TPU Training Guide

This guide explains how to run the DTC (Dual-Timescale Competence) agent on Google Cloud TPU, specifically optimized for Google Colab.

## Quick Start with Google Colab

The easiest way to get started is using our pre-configured Colab notebook:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/DTC-agent-tpu/blob/main/DTC_TPU_Colab.ipynb)

Simply click the button above and follow the notebook instructions!

## What's New in TPU Version

This TPU-optimized version includes:

### ✅ TPU Compatibility
- **Pure PyTorch episodic memory** - Replaces FAISS with TPU-compatible k-NN
- **XLA-safe logging** - Buffered logging to avoid graph breaks
- **Threading disabled** - Removes threading locks incompatible with Colab
- **Headless environment** - Automatic Colab/headless mode detection

### 🚀 Performance Optimizations
- **Large batch sizes** - Utilizes all 8 TPU cores efficiently (effective batch: 1024)
- **Optimized memory usage** - Configured for Colab's free TPU tier
- **XLA graph compilation** - Native XLA compilation (torch.compile disabled)

### 📊 Key Features Preserved
- Intrinsic motivation (competence, curiosity, empowerment)
- Slot attention for object-centric representations
- World model ensemble (5 models)
- Episodic memory with k-NN recall
- Dreaming/imagination for planning
- Global workspace cognitive routing

## Manual Setup (Advanced)

If you want to run outside of Colab:

### Prerequisites
```bash
# Install PyTorch XLA
pip install torch torch-xla[tpu] -f https://storage.googleapis.com/libtpu-releases/index.html

# Install dependencies
pip install -r requirements-colab.txt

# Install DTC agent
pip install -e .
```

### Running Training
```bash
# Using TPU configuration
python -m dtc_agent.training \
    --config configs/tpu.yaml \
    --max-episodes 100 \
    --log-to-wandb false
```

## Configuration

The TPU configuration (`configs/tpu.yaml`) is optimized for:
- **Google Colab TPU v2/v3** (8 cores, 128GB total HBM)
- **Batch size**: 128 per core (1024 effective total)
- **Memory capacity**: 8000 episodes (fits in Colab free tier)
- **No FAISS**: Uses pure PyTorch k-NN on TPU

### Key Settings

```yaml
device: xla                    # Use XLA/TPU device
batch_size: 128                # Per-core batch size
rollout_capacity: 8192         # Rollout buffer size
compile_modules: false         # XLA handles compilation
world_model_ensemble: 5        # Ensemble size
episodic_memory:
  capacity: 8000               # Memory capacity
  key_dim: 128                 # Key dimension
```

### Tuning for Your Needs

**If you run out of memory:**
- Reduce `batch_size` (e.g., 64)
- Reduce `episodic_memory.capacity` (e.g., 5000)
- Reduce `world_model_ensemble` (e.g., 3)

**If training is slow:**
- Increase `batch_size` (TPUs love large batches)
- Reduce `max_dream_horizon` (currently 30)
- Increase `dream_every_n_steps` (currently 4)

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     DTC Agent on TPU                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐      ┌──────────────┐                    │
│  │   Encoder    │──────▶│ Slot Attn.   │                    │
│  │ (Vision CNN) │      │ (8 slots)    │                    │
│  └──────────────┘      └──────┬───────┘                    │
│                               │                             │
│                               ▼                             │
│  ┌──────────────────────────────────────┐                  │
│  │      Global Workspace Router         │                  │
│  │   (Attention-based broadcast)        │                  │
│  └─────────────────┬────────────────────┘                  │
│                    │                                        │
│         ┌──────────┼──────────┐                            │
│         ▼          ▼          ▼                            │
│  ┌──────────┐ ┌────────┐ ┌──────────┐                     │
│  │  Actor   │ │ Critic │ │  Memory  │ (TPU k-NN)         │
│  └──────────┘ └────────┘ └──────────┘                     │
│                                                             │
│  ┌──────────────────────────────────────┐                  │
│  │      World Model Ensemble (5)        │                  │
│  │  Encoder → Dynamics → Decoder        │                  │
│  └──────────────────────────────────────┘                  │
│                                                             │
│  ┌──────────────────────────────────────┐                  │
│  │      Intrinsic Motivation            │                  │
│  │  • Competence (learning progress)    │                  │
│  │  • Curiosity (epistemic novelty)     │                  │
│  │  • Empowerment (control)             │                  │
│  │  • Survival (safety)                 │                  │
│  └──────────────────────────────────────┘                  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## TPU-Specific Implementation Details

### Episodic Memory
The original FAISS-based episodic memory is replaced with a pure PyTorch implementation:

```python
# Old (FAISS - CPU only)
self._cpu_index = faiss.IndexFlatL2(key_dim)

# New (PyTorch - TPU compatible)
distances = torch.cdist(query, keys, p=2.0) ** 2
topk_distances, topk_indices = torch.topk(distances, k=k, largest=False)
```

**Benefits:**
- ✅ Runs entirely on TPU (no CPU transfers)
- ✅ Batched operations
- ✅ XLA graph compatible
- ⚠️ Slightly higher memory usage for large capacities

### XLA Logging
Print statements cause XLA graph breaks, so we use buffered logging:

```python
# Old (causes graph breaks)
print(f"Loss: {loss.item()}")

# New (XLA-safe)
from dtc_agent.utils import xla_print, flush_xla_logs

xla_print(f"Loss: {loss.item()}")
# ... training ...
flush_xla_logs()  # Print all buffered logs at once
```

### Threading
Colab's TPU environment doesn't support full threading, so we use no-op locks:

```python
from dtc_agent.utils import set_threading_enabled, get_lock

set_threading_enabled(False)  # Automatic on XLA device
lock = get_lock()  # Returns NoOpLock on TPU
```

## Performance Tips

### Maximizing TPU Utilization
1. **Use large batches** - TPUs are designed for large matrix operations
2. **Minimize host-device transfers** - Keep data on TPU
3. **Avoid dynamic shapes** - XLA needs static graphs
4. **Batch operations** - Use batched tensor ops instead of loops

### Expected Performance
On Colab TPU v2:
- **Training speed**: ~50-100 steps/second (depending on dream horizon)
- **Memory usage**: ~80GB HBM (out of 128GB available)
- **Episode throughput**: ~5-10 episodes/minute

## Monitoring Training

### Using Weights & Biases (Recommended)
```python
# In Colab notebook
import wandb
wandb.login()

# Then run training with W&B enabled
!python -m dtc_agent.training \
    --config configs/tpu.yaml \
    --log-to-wandb true
```

### Manual Log Inspection
```python
from dtc_agent.utils import flush_xla_logs

# Flush buffered XLA logs
flush_xla_logs()
```

## Troubleshooting

### Common Issues

**1. "TPU not found"**
- Make sure Runtime → Change runtime type → TPU is selected
- Restart runtime if needed

**2. "Out of memory"**
```yaml
# In configs/tpu.yaml, reduce:
batch_size: 64                 # Down from 128
episodic_memory:
  capacity: 5000               # Down from 8000
world_model_ensemble: 3        # Down from 5
```

**3. "Training is slow"**
- Check that `compile_modules: false` (XLA compiles automatically)
- Increase batch size for better TPU utilization
- Reduce dream horizon if too long

**4. "Graph recompilation"**
- Usually happens on first few steps (normal)
- If frequent, check for dynamic shapes or print statements
- Our code handles this automatically with XLA-safe logging

### Debugging

**Enable debug logging:**
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

**Check XLA metrics:**
```python
import torch_xla.debug.metrics as met
print(met.metrics_report())
```

## Advanced Topics

### Custom Configurations
Create your own config by extending `configs/tpu.yaml`:

```yaml
# configs/my_tpu_config.yaml
defaults:
  - tpu

# Override specific settings
batch_size: 64
world_model_ensemble: 3
max_dream_horizon: 20
```

### Distributed TPU (Multi-host)
For multi-host TPU pods, use `torch_xla.distributed`:

```python
import torch_xla.distributed.xla_multiprocessing as xmp

def train_fn(rank):
    # Training code here
    pass

xmp.spawn(train_fn, args=(), nprocs=8)
```

(Note: Colab only provides single-host TPU)

### Checkpointing
Checkpoints are automatically saved during training:

```python
# Load checkpoint
checkpoint = torch.load('dtc_agent_checkpoint_step_1000.pt')
loop.world_model.load_state_dict(checkpoint['world_model'])
loop.actor.load_state_dict(checkpoint['actor'])
```

## Limitations

### Colab Free Tier
- **Time limit**: 12 hours max session
- **Disconnection**: May disconnect after inactivity
- **Resource limits**: Shared TPU resources

**Workarounds:**
- Use checkpointing frequently
- Download checkpoints to Google Drive
- Consider Colab Pro for longer sessions

### TPU vs GPU Differences
| Feature | GPU (CUDA) | TPU (XLA) |
|---------|-----------|-----------|
| Threading | Full support | Limited (disabled) |
| FAISS | Native | Not supported (PyTorch alternative) |
| Dynamic shapes | Supported | Avoided (graph breaks) |
| Batch size | 32-128 | 512-2048 (optimal) |
| Print statements | Anytime | Buffered (XLA-safe) |

## Citation

If you use this code, please cite:

```bibtex
@software{dtc_agent_tpu,
  title = {DTC Agent - TPU Optimized},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/YOUR_USERNAME/DTC-agent-tpu}
}
```

## License

MIT License - see LICENSE file for details.

## Support

- **Issues**: [GitHub Issues](https://github.com/YOUR_USERNAME/DTC-agent-tpu/issues)
- **Discussions**: [GitHub Discussions](https://github.com/YOUR_USERNAME/DTC-agent-tpu/discussions)
- **Email**: your.email@example.com

## Acknowledgments

- PyTorch XLA team for excellent TPU support
- Google Colab for free TPU access
- Crafter environment authors
- Original DTC architecture researchers
