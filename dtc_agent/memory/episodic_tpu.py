"""TPU-compatible episodic memory implementation.

This module provides a pure PyTorch implementation of episodic memory
that works efficiently on TPU without requiring FAISS or CPU transfers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch


@dataclass
class EpisodicEntry:
    key: torch.Tensor
    context: torch.Tensor
    z_self: torch.Tensor
    slots: torch.Tensor
    action: torch.Tensor | None = None
    target_latent: torch.Tensor | None = None


@dataclass
class EpisodicSnapshot:
    z_self: torch.Tensor
    slots: torch.Tensor
    action: torch.Tensor | None = None
    target_latent: torch.Tensor | None = None


@dataclass
class EpisodicBufferConfig:
    """Configuration for the episodic recall buffer.

    Attributes:
        capacity: Maximum number of entries retained in memory.
        key_dim: Dimensionality of the key vectors.
    """

    capacity: int
    key_dim: int


class EpisodicBufferTPU:
    """Pure PyTorch episodic buffer for TPU compatibility.

    This implementation avoids FAISS and threading to work efficiently on TPU.
    All operations are performed in PyTorch on the device where data resides.
    """

    def __init__(self, config: EpisodicBufferConfig, device: torch.device | None = None) -> None:
        """Initialize the buffer.

        Args:
            config: Parameters governing capacity and index dimensionality.
            device: Device to store the buffer on (defaults to CPU, set to TPU device for best performance).
        """
        self.config = config
        self.device = device if device is not None else torch.device("cpu")

        # Store all keys in a tensor for efficient batched distance computation
        # Shape: [capacity, key_dim]
        self.keys = torch.zeros(
            (config.capacity, config.key_dim),
            dtype=torch.float32,
            device=self.device
        )

        # Store entries in a dictionary (same as original)
        self.values: Dict[int, EpisodicEntry] = {}
        self.next_id = 0
        self.num_entries = 0  # Track how many entries are valid

    def __len__(self) -> int:
        """Return the number of stored episodic entries."""
        return self.num_entries

    def write(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        snapshot: EpisodicSnapshot | None = None,
    ) -> None:
        """Insert key/value pairs into the episodic memory.

        Args:
            key: Tensor of shape ``[batch, key_dim]`` used for indexing.
            value: Tensor payload associated with each key.
            snapshot: Optional latent/action metadata retained for replay.

        Raises:
            ValueError: If ``key`` is not a 2-D tensor with ``key_dim`` columns.
        """
        if key.ndim != 2:
            raise ValueError("key must be shape [batch, key_dim]")
        if key.shape[1] != self.config.key_dim:
            raise ValueError("key dimension mismatch")

        batch = key.shape[0]

        # Move to buffer device
        key_device = key.to(dtype=torch.float32, device=self.device)
        value_device = value.to(device=self.device)

        if snapshot is not None:
            z_self_device = snapshot.z_self.to(device=self.device)
            slots_device = snapshot.slots.to(device=self.device)
            action_device = (
                snapshot.action.to(device=self.device)
                if snapshot.action is not None
                else None
            )
            target_device = (
                snapshot.target_latent.to(device=self.device)
                if snapshot.target_latent is not None
                else None
            )
        else:
            z_self_device = None
            slots_device = None
            action_device = None
            target_device = None

        for idx in range(batch):
            # Handle capacity overflow with circular buffer
            if self.num_entries >= self.config.capacity:
                self._evict_oldest()

            buffer_idx = self.next_id % self.config.capacity

            # Update key tensor
            self.keys[buffer_idx] = key_device[idx]

            # Create and store entry
            context_tensor = value_device[idx]
            entry = EpisodicEntry(
                key=key_device[idx],
                context=context_tensor,
                z_self=z_self_device[idx] if z_self_device is not None else context_tensor.clone(),
                slots=slots_device[idx] if slots_device is not None else context_tensor.unsqueeze(0),
                action=action_device[idx] if action_device is not None else None,
                target_latent=target_device[idx] if target_device is not None else None,
            )

            self.values[self.next_id] = entry
            self.next_id += 1
            self.num_entries = min(self.num_entries + 1, self.config.capacity)

    def read(self, query: torch.Tensor, k: int = 5) -> Tuple[torch.Tensor, torch.Tensor]:
        """Retrieve approximate nearest neighbours for each query vector.

        Args:
            query: Tensor of shape ``[batch, key_dim]`` to match against the index.
            k: Number of neighbours to retrieve for each query.

        Returns:
            Tuple ``(distances, values)`` where ``distances`` contains squared
            L2 distances and ``values`` contains the recalled payloads.

        Raises:
            ValueError: If ``query`` is not a 2-D tensor with ``key_dim`` columns.
        """
        if query.ndim != 2:
            raise ValueError("query must be shape [batch, key_dim]")

        target_device = query.device
        batch_size = query.shape[0]

        if self.num_entries == 0:
            # Return zeros if buffer is empty
            fallback_values = torch.zeros(
                (batch_size, k, self.config.key_dim),
                device=target_device
            )
            fallback_distances = torch.full(
                (batch_size, k),
                float('inf'),
                device=target_device
            )
            return fallback_distances, fallback_values

        # Move query to buffer device for computation
        query_device = query.to(dtype=torch.float32, device=self.device)

        # Get valid keys (only up to num_entries)
        valid_keys = self.keys[:self.num_entries]  # Shape: [num_entries, key_dim]

        # Compute pairwise L2 distances
        # Shape: [batch, num_entries]
        distances = torch.cdist(query_device, valid_keys, p=2.0) ** 2

        # Get top-k nearest neighbors
        k_actual = min(k, self.num_entries)
        topk_distances, topk_indices = torch.topk(
            distances,
            k=k_actual,
            dim=1,
            largest=False,  # Get smallest distances
            sorted=True
        )

        # Retrieve values for the top-k indices
        # Get the actual IDs from our values dict
        valid_ids = sorted(self.values.keys())[-self.num_entries:]

        retrieved_values = []
        for batch_idx in range(batch_size):
            batch_values = []
            for k_idx in range(k_actual):
                idx_in_buffer = topk_indices[batch_idx, k_idx].item()
                actual_id = valid_ids[idx_in_buffer]
                entry = self.values[actual_id]
                batch_values.append(entry.context)

            # Pad if k_actual < k
            while len(batch_values) < k:
                batch_values.append(torch.zeros_like(entry.context))

            retrieved_values.append(torch.stack(batch_values))

        stacked_values = torch.stack(retrieved_values)  # Shape: [batch, k, value_dim]

        # Pad distances if needed
        if k_actual < k:
            padding = torch.full(
                (batch_size, k - k_actual),
                float('inf'),
                device=self.device
            )
            topk_distances = torch.cat([topk_distances, padding], dim=1)

        # Move back to target device
        values_out = stacked_values.to(target_device)
        distances_out = topk_distances.to(target_device)

        return distances_out, values_out

    def sample_uniform(
        self, batch_size: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample latent snapshots uniformly from stored entries."""
        if not self.values:
            raise ValueError("Episodic buffer is empty")

        ids = list(self.values.keys())[-self.num_entries:]  # Get valid IDs

        if len(ids) < batch_size:
            # Sample with replacement if needed
            sampled_indices = torch.randint(0, len(ids), (batch_size,))
        else:
            # Sample without replacement
            sampled_indices = torch.randperm(len(ids))[:batch_size]

        z_self_list = []
        slots_list = []

        for idx in sampled_indices:
            entry = self.values[ids[idx.item()]]
            z_self_list.append(entry.z_self)
            slots_list.append(entry.slots)

        z_self_tensor = torch.stack(z_self_list).to(device)
        slots_tensor = torch.stack(slots_list).to(device)

        return z_self_tensor, slots_tensor

    def sample_transitions(
        self, batch_size: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample latent transitions containing (state, action, next_state) tuples."""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        valid_ids = [
            idx
            for idx, entry in self.values.items()
            if entry.action is not None and entry.target_latent is not None
        ]

        if not valid_ids:
            raise ValueError("No episodic transitions with stored actions are available")

        if len(valid_ids) < batch_size:
            # Sample with replacement
            sampled_indices = torch.randint(0, len(valid_ids), (batch_size,))
        else:
            # Sample without replacement
            sampled_indices = torch.randperm(len(valid_ids))[:batch_size]

        latent_state_list: List[torch.Tensor] = []
        action_list: List[torch.Tensor] = []
        target_list: List[torch.Tensor] = []

        for idx in sampled_indices:
            entry = self.values[valid_ids[idx.item()]]
            assert entry.action is not None and entry.target_latent is not None
            latent_state_list.append(entry.context)
            action_list.append(entry.action)
            target_list.append(entry.target_latent)

        latent_tensor = torch.stack(latent_state_list).to(device)
        action_tensor = torch.stack(action_list).to(device)
        target_tensor = torch.stack(target_list).to(device)

        return latent_tensor, action_tensor, target_tensor

    def _evict_oldest(self) -> None:
        """Evict the oldest entry from the buffer."""
        if not self.values:
            return

        oldest_idx = min(self.values.keys())
        del self.values[oldest_idx]
        self.num_entries = len(self.values)
