#!/usr/bin/env python3
"""
Colab TPU Setup Script for DTC Agent

This script initializes the Google Colab environment for TPU training.
Run this at the beginning of your Colab notebook.
"""

import os
import sys
from pathlib import Path


def setup_colab_tpu():
    """Configure Colab environment for TPU training."""

    print("=" * 60)
    print("DTC Agent - Colab TPU Setup")
    print("=" * 60)

    # Check if running in Colab
    try:
        import google.colab
        in_colab = True
        print("✓ Running in Google Colab")
    except ImportError:
        in_colab = False
        print("✗ Not running in Google Colab")
        return False

    # Install TPU dependencies
    print("\n1. Installing TPU dependencies...")
    os.system("pip install -q torch torch-xla[tpu] -f https://storage.googleapis.com/libtpu-releases/index.html")

    # Install other requirements
    print("\n2. Installing DTC agent dependencies...")
    if Path("requirements-colab.txt").exists():
        os.system("pip install -q -r requirements-colab.txt")
    else:
        print("   ⚠ requirements-colab.txt not found, installing manually...")
        os.system("pip install -q numpy einops tqdm omegaconf wandb gymnasium crafter matplotlib imageio pillow")

    # Install the DTC agent package
    print("\n3. Installing DTC agent package...")
    os.system("pip install -q -e .")

    # Verify TPU availability
    print("\n4. Verifying TPU availability...")
    try:
        import torch_xla
        import torch_xla.core.xla_model as xm

        device = xm.xla_device()
        print(f"✓ TPU device available: {device}")

        # Test basic operation
        test_tensor = torch.ones(2, 2, device=device)
        xm.mark_step()
        print("✓ TPU test operation successful")

        # Get TPU configuration
        num_devices = xm.xrt_world_size()
        print(f"✓ Number of TPU cores: {num_devices}")

    except Exception as e:
        print(f"✗ TPU verification failed: {e}")
        print("\nPlease ensure:")
        print("  1. Runtime is set to TPU (Runtime -> Change runtime type)")
        print("  2. TPU is available in your region")
        return False

    # Set environment variables for optimal performance
    print("\n5. Setting environment variables...")
    os.environ["XLA_USE_BF16"] = "0"  # Use FP32 for stability
    os.environ["XLA_TENSOR_ALLOCATOR_MAXSIZE"] = "100000000"  # 100MB chunks
    print("✓ Environment variables configured")

    # Disable threading for FAISS compatibility (if using CPU fallback)
    try:
        import faiss
        faiss.omp_set_num_threads(1)
        print("✓ FAISS threading disabled for compatibility")
    except ImportError:
        print("  (FAISS not installed - using PyTorch-based memory)")

    print("\n" + "=" * 60)
    print("✓ Setup complete! Ready to train on TPU.")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Load your configuration: cfg = OmegaConf.load('configs/tpu.yaml')")
    print("  2. Run training: python -m dtc_agent.training")
    print("\n")

    return True


def check_gpu_memory():
    """Check available GPU/TPU memory."""
    try:
        import torch_xla.core.xla_model as xm
        device = xm.xla_device()
        print(f"TPU device: {device}")
    except Exception as e:
        print(f"Could not check TPU memory: {e}")


if __name__ == "__main__":
    success = setup_colab_tpu()
    sys.exit(0 if success else 1)
