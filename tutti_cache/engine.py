# Author: Sneha Bhawar (github.com/snehabhawar)
"""
AsyncTransferEngine — CUDA stream based async KV cache transfers.
Compute and transfer overlap on separate streams.
"""

import threading
import time
import torch
from queue import Queue, Empty
from typing import Dict, Tuple, Optional

from .store import KVStore, LayerState


class AsyncTransferEngine:
    """
    Manages async KV cache transfers using CUDA streams.
    Two streams run simultaneously:
      compute_stream: model attention computation
      transfer_stream: CPU <-> GPU tensor movement
    """

    def __init__(self, kv_store: KVStore, device: str = "cuda:0"):
        self.kv_store = kv_store
        self.device = device
        self.compute_stream = torch.cuda.Stream(device=device)
        self.transfer_stream = torch.cuda.Stream(device=device)
        self.prefetch_queue = Queue()
        self.prefetch_results: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        self.results_lock = threading.Lock()
        self.transfer_events: Dict[int, torch.cuda.Event] = {}
        self.running = False
        self.transfer_thread = None

    def start(self) -> None:
        """Start the background transfer thread."""
        self.running = True
        self.transfer_thread = threading.Thread(
            target=self._transfer_worker, daemon=True)
        self.transfer_thread.start()

    def stop(self) -> None:
        """Stop the background transfer thread."""
        self.running = False
        self.prefetch_queue.put(None)
        if self.transfer_thread:
            self.transfer_thread.join(timeout=5)

    def _transfer_worker(self) -> None:
        while self.running:
            try:
                item = self.prefetch_queue.get(timeout=0.1)
                if item is None:
                    break
                layer_idx, target_device = item
                with torch.cuda.stream(self.transfer_stream):
                    result = self.kv_store.prefetch(
                        layer_idx, target_device=target_device)
                    if result is not None:
                        key_gpu, val_gpu = result
                        event = torch.cuda.Event()
                        event.record(self.transfer_stream)
                        with self.results_lock:
                            self.prefetch_results[layer_idx] = (key_gpu, val_gpu)
                            self.transfer_events[layer_idx] = event
            except Empty:
                continue
            except Exception as e:
                print(f"Transfer worker error: {e}")

    def request_prefetch(self, layer_idx: int,
                         target_device: str = None) -> None:
        """Queue a prefetch request. Non-blocking."""
        state = self.kv_store.get_state(layer_idx)
        if state == LayerState.ON_CPU:
            self.prefetch_queue.put(
                (layer_idx, target_device or self.device))

    def get_prefetched(self, layer_idx: int,
                       timeout: float = 0.1,
                       target_device: str = None
                       ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Get prefetched result. Waits up to timeout seconds."""
        deadline = time.perf_counter() + timeout
        while time.perf_counter() < deadline:
            with self.results_lock:
                if layer_idx in self.prefetch_results:
                    result = self.prefetch_results.pop(layer_idx)
                    event = self.transfer_events.pop(layer_idx, None)
                    if event is not None:
                        self.compute_stream.wait_event(event)
                    return result
            time.sleep(0.001)
        return self.kv_store.prefetch(
            layer_idx, target_device=target_device or self.device)

    def clear(self) -> None:
        while not self.prefetch_queue.empty():
            try:
                self.prefetch_queue.get_nowait()
            except Empty:
                break
        with self.results_lock:
            self.prefetch_results.clear()
            self.transfer_events.clear()
        self.kv_store.clear()
