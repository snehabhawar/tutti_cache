# Author: Sneha Bhawar (github.com/snehabhawar)
"""
SlackAwareScheduler — timing-based prefetch decisions.
Implements Tutti's slack-aware scheduling algorithm.
"""

import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch

from .store import KVStore, LayerState
from .engine import AsyncTransferEngine


@dataclass
class LayerProfile:
    """Per-layer timing profile for scheduling decisions."""
    layer_idx: int
    compute_times: List[float] = field(default_factory=list)
    transfer_times: List[float] = field(default_factory=list)

    @property
    def avg_compute(self) -> float:
        if not self.compute_times:
            return 0.01
        return sum(self.compute_times[-10:]) / len(self.compute_times[-10:])

    @property
    def avg_transfer(self) -> float:
        if not self.transfer_times:
            return 0.005
        return sum(self.transfer_times[-10:]) / len(self.transfer_times[-10:])

    @property
    def slack(self) -> float:
        """
        Time available before this layer is needed
        minus time to transfer it.
        Positive = safe to prefetch.
        Negative = urgent, fetch immediately.
        """
        return self.avg_compute - self.avg_transfer

    def record_compute(self, elapsed: float) -> None:
        self.compute_times.append(elapsed)
        if len(self.compute_times) > 20:
            self.compute_times.pop(0)

    def record_transfer(self, elapsed: float) -> None:
        self.transfer_times.append(elapsed)
        if len(self.transfer_times) > 20:
            self.transfer_times.pop(0)


class SlackAwareScheduler:
    """
    Decides WHAT to prefetch and WHEN.

    Uses per-layer timing profiles to calculate slack:
        slack = compute_time - transfer_time

    Positive slack = safe, prefetch ahead of time.
    Negative slack = urgent, fetch immediately.

    Adapts over time as timing profiles update with
    rolling averages from real measurements.
    """

    def __init__(self,
                 num_layers: int,
                 transfer_engine: AsyncTransferEngine,
                 kv_store: KVStore,
                 lookahead: int = 2):
        self.num_layers = num_layers
        self.transfer_engine = transfer_engine
        self.kv_store = kv_store
        self.lookahead = lookahead
        self.profiles: Dict[int, LayerProfile] = {
            i: LayerProfile(i) for i in range(num_layers)}
        self.scheduled: set = set()
        self.scheduled_lock = threading.Lock()
        self.total_decisions = 0
        self.early_prefetches = 0
        self.late_prefetches = 0
        self.hit_rate = 0.0

    def on_layer_start(self, current_layer: int,
                       target_device: str = None) -> None:
        """Called when a layer starts computing. Schedules prefetches."""
        for ahead in range(1, self.lookahead + 1):
            next_layer = current_layer + ahead
            if next_layer >= self.num_layers:
                break
            state = self.kv_store.get_state(next_layer)
            with self.scheduled_lock:
                already_scheduled = next_layer in self.scheduled
            if state == LayerState.ON_CPU and not already_scheduled:
                profile = self.profiles[next_layer]
                urgency = self.lookahead - ahead + 1
                if profile.slack > 0 or urgency >= self.lookahead:
                    self.transfer_engine.request_prefetch(
                        next_layer, target_device=target_device)
                    with self.scheduled_lock:
                        self.scheduled.add(next_layer)
                    self.total_decisions += 1

    def on_layer_end(self, layer_idx: int,
                     compute_elapsed: float) -> None:
        """Records compute timing for scheduler."""
        self.profiles[layer_idx].record_compute(compute_elapsed)
        with self.scheduled_lock:
            self.scheduled.discard(layer_idx)

    def on_transfer_complete(self, layer_idx: int,
                             transfer_elapsed: float) -> None:
        """Records transfer timing for scheduler."""
        self.profiles[layer_idx].record_transfer(transfer_elapsed)

    def get_layer_kv(self, layer_idx: int,
                     target_device: str = None
                     ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Get KV cache for layer. Tracks hit/miss rate."""
        with self.transfer_engine.results_lock:
            already_ready = layer_idx in self.transfer_engine.prefetch_results
        if already_ready:
            self.early_prefetches += 1
        else:
            self.late_prefetches += 1
        result = self.transfer_engine.get_prefetched(
            layer_idx, target_device=target_device)
        total = self.early_prefetches + self.late_prefetches
        if total > 0:
            self.hit_rate = self.early_prefetches / total
        return result

    def reset_for_new_token(self) -> None:
        with self.scheduled_lock:
            self.scheduled.clear()

    def print_stats(self) -> None:
        print(f"\nScheduler Statistics:")
        print(f"  Total decisions:  {self.total_decisions}")
        print(f"  Early prefetches: {self.early_prefetches}")
        print(f"  Late prefetches:  {self.late_prefetches}")
        print(
