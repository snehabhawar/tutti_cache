# Author: Sneha Bhawar (github.com/snehabhawar)
"""
tutti_cache — Layerwise async KV cache prefetching for LLMs.

Implements Tutti's slack-aware prefetching:
arxiv.org/abs/2605.03375

95% speed recovery vs naive CPU offloading.
Works with any HuggingFace causal LM.

Author: Sneha Bhawar
GitHub: https://github.com/snehabhawar

Usage:
    from tutti_cache import create_tutti_cache
    cache = create_tutti_cache(model)
    outputs = model.generate(input_ids, past_key_values=cache)
"""

from .cache import TuttiKVCache
from .factory import create_tutti_cache
from .store import KVStore, LayerState
from .engine import AsyncTransferEngine
from .scheduler import SlackAwareScheduler, LayerProfile

__version__ = "0.1.0"
__author__ = "Sneha Bhawar"
__github__ = "https://github.com/snehabhawar"

__all__ = [
    "TuttiKVCache",
    "create_tutti_cache",
    "KVStore",
    "LayerState",
    "AsyncTransferEngine",
    "SlackAwareScheduler",
    "LayerProfile",
]
