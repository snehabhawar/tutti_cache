# Author: Sneha Bhawar (github.com/snehabhawar)
"""
TuttiKVCache — custom KV cache implementing Tutti's
layerwise async prefetching with slack-aware scheduling.

Drop-in replacement for HuggingFace DynamicCache.
Works with any model that uses the cache.update() interface.
"""

import time
from typing import Any, Dict, Optional, Tuple

import torch

from .store import KVStore, LayerState
from .engine import AsyncTransferEngine
from .scheduler import SlackAwareScheduler


class TuttiKVCache:
    """
    Custom KV cache implementing Tutti's layerwise async prefetching.

    Architecture:
        - KV cache lives in CPU RAM between tokens
        - Comes to GPU only during attention computation
        - Returns to CPU RAM immediately after
        - Next layer's cache prefetched while current layer computes
        - Slack-aware scheduler optimizes prefetch timing

    Usage:
        from tutti_cache import create_tutti_cache
        cache = create_tutti_cache(model)
        outputs = model.generate(input_ids, past_key_values=cache)
    """

    def __init__(self,
                 kv_store: KVStore,
                 transfer_engine: AsyncTransferEngine,
                 scheduler: SlackAwareScheduler,
                 num_layers: int = 32,
                 device: str = "cuda:0"):

        self.kv_store = kv_store
        self.transfer_engine = transfer_engine
        self.scheduler = scheduler
        self.num_layers = num_layers
        self.device = device
        self._seq_lengths: Dict[int, int] = {}
        self._layer_devices: Dict[int, str] = {}
        self._is_prefill = True
        self._prefill_done = False
        self.update_count = 0
        self.prefetch_hits = 0
        self.prefetch_misses = 0

    def update(self,
               key_states: torch.Tensor,
               value_states: torch.Tensor,
               layer_idx: int,
               cache_kwargs: Optional[Dict[str, Any]] = None,
               ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Called by model attention for every layer.
        Stores KV cache in CPU RAM and manages prefetching.
        """
        self.update_count += 1

        # Track which device this layer uses (multi-GPU support)
        layer_device = str(key_states.device)
        self._layer_devices[layer_idx] = layer_device

        # PREFILL — seq_len > 1 means processing all input tokens
        if key_states.shape[2] > 1:
            self._is_prefill = True
            t0 = time.perf_counter()
            self.kv_store.offload(layer_idx, key_states, value_states)
            t1 = time.perf_counter()
            self.scheduler.on_transfer_complete(layer_idx, t1 - t0)
            self._seq_lengths[layer_idx] = key_states.shape[2]
            if layer_idx + 1 < self.num_layers:
                self.scheduler.on_layer_start(
                    layer_idx, target_device=layer_device)
            return key_states, value_states

        # DECODE — one new token at a time
        self._is_prefill = False
        self._prefill_done = True

        existing = self._get_from_store(layer_idx, layer_device)

        if existing is not None:
            existing_keys, existing_values = existing
            if str(existing_keys.device) != layer_device:
                existing_keys = existing_keys.to(layer_device)
                existing_values = existing_values.to(layer_device)
            full_keys = torch.cat([existing_keys, key_states], dim=2)
            full_values = torch.cat([existing_values, value_states], dim=2)
        else:
            full_keys = key_states
            full_values = value_states

        self._seq_lengths[layer_idx] = full_keys.shape[2]

        t0 = time.perf_counter()
        self.kv_store.offload(layer_idx, full_keys, full_values)
        t1 = time.perf_counter()
        self.scheduler.on_transfer_complete(layer_idx, t1 - t0)

        # Schedule prefetch for next layer while this one computes
        if layer_idx + 1 < self.num_layers:
            next_state = self.kv_store.get_state(layer_idx + 1)
            if next_state == LayerState.ON_CPU:
                next_device = self._layer_devices.get(
                    layer_idx + 1, layer_device)
                self.transfer_engine.request_prefetch(
                    layer_idx + 1, target_device=next_device)
                self.scheduler.total_decisions += 1

        return full_keys, full_values

    def _get_from_store(self,
                        layer_idx: int,
                        target_device: str = None
                        ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Get KV cache. Fast path: prefetch ready. Slow path: sync fetch."""
        with self.transfer_engine.results_lock:
            if layer_idx in self.transfer_engine.prefetch_results:
                result = self.transfer_engine.prefetch_results.pop(layer_idx)
                self.transfer_engine.transfer_events.pop(layer_idx, None)
                self.prefetch_hits += 1
                return result

        state = self.kv_store.get_state(layer_idx)
        if state == LayerState.ON_CPU:
            self.prefetch_misses += 1
            return self.kv_store.prefetch(
                layer_idx, target_device=target_device)

        return None

    # ── HuggingFace Cache Interface ──

    def get_seq_length(self, layer_idx: int = 0) -> int:
        if isinstance(layer_idx, torch.Tensor):
            layer_idx = int(layer_idx.item())
        return self._seq_lengths.get(layer_idx, 0)

    def get_max_cache_shape(self) -> Optional[int]:
        if not self._seq_lengths:
            return None
        return max(self._seq_lengths.values())

    def get_mask_sizes(self, cache_position=None,
                       layer_idx=None, **kwargs):
        if layer_idx is None:
            idx = 0
        elif isinstance(layer_idx, torch.Tensor):
            idx = int(layer_idx.item())
        else:
            idx = int(layer_idx)
        seq_len = self.get_seq_length(idx)
        if seq_len == 0 and cache_position is not None:
            seq_len = int(cache_position[-1].item()) + 1
        if seq_len == 0:
            return None, None
        return seq_len, 0

    def __len__(self) -> int:
        return self.num_layers

    def reorder_cache(self, beam_idx) -> None:
        pass

    def batch_repeat_interleave(self, *args, **kwargs) -> None:
        pass

    def batch_select_indices(self, *args, **kwargs) -> None:
        pass

    @property
    def has_previous_state(self) -> bool:
        return self._prefill_done

    @property
    def is_compileable(self) -> bool:
        return False

    @property
    def is_sliding(self):
        return [False] * self.num_layers

    @property
    def is_initialized(self) -> bool:
        return True

    def reset(self) -> None:
        """Reset cache for new generation."""
        self.kv_store.clear()
        self.transfer_engine.clear()
        self._seq_lengths.clear()
        self._layer_devices.clear()
        self._is_prefill = True
        self._prefill_done = False
        self.update_count = 0
        self.prefetch_hits = 0
        self.prefetch_misses = 0

    def print_stats(self) -> None:
        total = self.prefetch_hits + self.prefetch_misses
        hit_rate = self.prefetch_hits / total if total > 0 else 0
        print(f"\nTuttiKVCache Statistics:")
        print(f"  Total updates:    {self.update_count}")
        print(f"  Prefetch hits:    {self.prefetch_hits}")
        print(f"  Prefetch misses:  {self.prefetch_misses}")
        print(f"  Hit rate:         {hit_rate:.1%}")
