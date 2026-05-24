# Author: Sneha Bhawar (github.com/snehabhawar)
"""
Factory function for creating TuttiKVCache instances.
Auto-detects model configuration.
"""

from typing import Optional
from .store import KVStore
from .engine import AsyncTransferEngine
from .scheduler import SlackAwareScheduler
from .cache import TuttiKVCache


def create_tutti_cache(model,
                       device: Optional[str] = None,
                       lookahead: int = 2) -> TuttiKVCache:
    """
    Create a TuttiKVCache for any HuggingFace causal LM.

    Auto-detects:
        - Number of transformer layers
        - Primary GPU device
        - Multi-GPU configurations

    Args:
        model:      Any HuggingFace AutoModelForCausalLM
        device:     Target device (auto-detected if None)
        lookahead:  Layers ahead to prefetch (default: 2)

    Returns:
        TuttiKVCache ready to use as past_key_values

    Example:
        from tutti_cache import create_tutti_cache

        cache = create_tutti_cache(model)
        outputs = model.generate(
            input_ids,
            past_key_values=cache,
            max_new_tokens=500
        )
    """
    config = model.config

    # Auto-detect number of layers
    if hasattr(config, 'num_hidden_layers'):
        num_layers = config.num_hidden_layers
    elif hasattr(config, 'n_layer'):
        num_layers = config.n_layer
    elif hasattr(config, 'num_layers'):
        num_layers = config.num_layers
    else:
        raise ValueError(
            f"Cannot auto-detect num_layers from "
            f"{config.model_type}. "
            f"Please pass num_layers manually."
        )

    # Auto-detect device
    if device is None:
        try:
            device = str(next(model.parameters()).device)
        except StopIteration:
            device = "cuda:0"

    print(f"TuttiKVCache: {num_layers} layers | "
          f"{config.model_type} | {device} | "
          f"lookahead={lookahead}")

    kv_store = KVStore(num_layers=num_layers, device=device)
    transfer_engine = AsyncTransferEngine(kv_store, device=device)
    transfer_engine.start()
    scheduler = SlackAwareScheduler(
        num_layers=num_layers,
        transfer_engine=transfer_engine,
        kv_store=kv_store,
        lookahead=lookahead
    )

    return TuttiKVCache(
        kv_store=kv_store,
        transfer_engine=transfer_engine,
        scheduler=scheduler,
        num_layers=num_layers,
        device=device
    )
