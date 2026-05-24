# Author: Sneha Bhawar (github.com/snehabhawar)
"""
KVStore — CPU RAM storage for KV cache tensors.
Tracks state of each layer's cache.
"""

import threading
import time
import torch
from enum import Enum
from typing import Dict, Tuple, Optional


class LayerState(Enum):
    ON_GPU      = "on_gpu"
    OFFLOADING  = "offloading"
    ON_CPU      = "on_cpu"
    PREFETCHING = "prefetching"


class KVStore:
    """
    Stores KV cache tensors in CPU RAM.
    Tracks state of every layer.
    Thread-safe for async operations.
    """

    def __init__(self, num_layers: int, device: str = "cuda:0"):
        self.num_layers = num_layers
        self.device = device
        self.cpu_store: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        self.states: Dict[int, LayerState] = {}
        self.lock = threading.Lock()
        self.compute_times: Dict[int, float] = {}
        self.transfer_times: Dict[int, float] = {}
        for i in range(num_layers):
            self.states[i] = LayerState.ON_GPU

    def offload(self, layer_idx: int,
                key: torch.Tensor,
                value: torch.Tensor) -> None:
        """Move KV tensors from GPU to CPU RAM."""
        start = time.perf_counter()
        key_cpu = key.detach().to("cpu", non_blocking=True)
        value_cpu = value.detach().to("cpu", non_blocking=True)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        with self.lock:
            self.cpu_store[layer_idx] = (key_cpu, value_cpu)
            self.states[layer_idx] = LayerState.ON_CPU
            self.transfer_times[layer_idx] = elapsed

    def prefetch(self, layer_idx: int,
                 target_device: str = None
                 ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Move KV tensors from CPU RAM back to GPU."""
        with self.lock:
            if layer_idx not in self.cpu_store:
                return None
            if self.states[layer_idx] != LayerState.ON_CPU:
                return None
            self.states[layer_idx] = LayerState.PREFETCHING
            key_cpu, value_cpu = self.cpu_store[layer_idx]
        device = target_device or self.device
        start = time.perf_counter()
        key_gpu = key_cpu.to(device, non_blocking=True)
        value_gpu = value_cpu.to(device, non_blocking=True)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        with self.lock:
            self.states[layer_idx] = LayerState.ON_GPU
            self.transfer_times[layer_idx] = (
                self.transfer_times.get(layer_idx, elapsed) + elapsed
            ) / 2
        return key_gpu, value_gpu

    def get_state(self, layer_idx: int) -> LayerState:
        with self.lock:
            return self.states.get(layer_idx, LayerState.ON_GPU)

    def clear(self) -> None:
        with self.lock:
            self.cpu_store.clear()
            for i in range(self.num_layers):
                self.states[i] = LayerState.ON_GPU

    def memory_usage_gb(self) -> float:
        total = 0
        with self.lock:
            for key, value in self.cpu_store.values():
                total += key.nbytes + value.nbytes
        return total / 1024**3
