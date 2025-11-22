"""
DTC 3.0 JAX/TPU Implementation - Project Hephaestus

Dual-Timescale Competence Agent with intrinsic motivation.
Designed for >50,000 steps/second on TPU v4-8.
"""

from dtc_jax.dtc import (
    dtc_types,
    rssm,
    intrinsic,
    memory,
    actor_critic,
    dreamer,
    trainer,
    collector
)

__all__ = [
    'dtc_types',
    'rssm',
    'intrinsic',
    'memory',
    'actor_critic',
    'dreamer',
    'trainer',
    'collector'
]
