# tutti_cache

**By [Sneha Bhawar](https://github.com/snehabhawar)**

Layerwise async KV cache prefetching for LLMs — independent implementation of Tutti's slack-aware prefetching algorithm.

> Paper: [Tutti: GPU-Centric KV Cache Management for LLM Inference](https://arxiv.org/abs/2605.03375) (May 2026)  
> Status: First public implementation. Tutti's official code not yet released.

---

## What this does

During LLM inference, the KV cache grows with every token generated. Storing it on GPU is expensive. Moving it to CPU RAM cuts speed by 53%.

tutti_cache recovers 95% of that speed penalty using layerwise async prefetching:

- While layer N computes attention, layer N+1's cache is already being fetched from CPU RAM in the background
- GPU never waits for data
- Slack-aware scheduler measures actual compute and transfer times per layer and adapts in real time
- Works with any HuggingFace causal LM — zero code changes between models

---

## Results

Tested on Kaggle T4 x2 (30GB VRAM), Phi-2 2.7B, 1200 token generation:

| Metric | Run 1 Baseline | Run 2 Naive Offload | tutti_cache |
|--------|----------------|---------------------|-------------|
| Tokens/sec | 19.41 | 9.10 | **18.89** |
| Speed penalty | 0% | 53% | **2.7%** |
| GPU KV cache | 0.255 GB | 0.235 GB | **0.062 GB** |
| CPU RAM | 0 GB | 0.398 GB | **0.078 GB** |
| Prefetch hit rate | — | 0% | **96.9%** |
| Speed recovered | — | 0% | **95.0%** |

### Model agnosticism — zero code changes between models

| Model | Size | Layers | Hit Rate | Tok/s |
|-------|------|--------|----------|-------|
| Phi-2 | 2.7B | 32 | 96.9% | 18.89 |
| OPT-6.7B | 6.7B | 32 | 96.8% | 14.10 |
| Mistral 7B | 7B | 32 | 96.9% | 11.76 |

---

## Installation

```bash
git clone https://github.com/snehabhawar/tutti_cache
cd tutti_cache
pip install -e .
```

---

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from tutti_cache import create_tutti_cache
import torch

# Load any HuggingFace causal LM
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/phi-2",
    torch_dtype=torch.float16,
    device_map="cuda:0",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(
    "microsoft/phi-2",
    trust_remote_code=True
)

# Two lines to enable Tutti prefetching
cache = create_tutti_cache(model)

# Generate exactly as normal
inputs = tokenizer("Your prompt here", return_tensors="pt").to("cuda:0")
outputs = model.generate(
    **inputs,
    past_key_values=cache,
    max_new_tokens=500
)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

---

## Architecture
tutti_cache/
├── KVStore                 CPU RAM tensor storage
│   ├── offload()           GPU → CPU (async)
│   ├── prefetch()          CPU → GPU (async)
│   └── LayerState          ON_GPU | ON_CPU | OFFLOADING | PREFETCHING
│
├── AsyncTransferEngine     CUDA stream management
│   ├── compute_stream      model attention runs here
│   ├── transfer_stream     transfers run here (parallel)
│   └── _transfer_worker    background thread
│
├── SlackAwareScheduler     timing-based decisions
│   ├── LayerProfile        rolling avg compute + transfer times
│   ├── slack calculation   compute_time - transfer_time
│   └── lookahead=2         prefetches N+1 and N+2 layers ahead
│
└── TuttiKVCache            drop-in cache object
├── update()            called by model attention every layer
├── prefill phase       offload all layers after input processing
├── decode phase        fetch → concat → offload → prefetch next
└── HuggingFace interface

### How it works

**Naive offloading (53% penalty):**
Layer 0: [wait for fetch][compute][offload]
Layer 1:                          [wait for fetch][compute][offload]
GPU idles during every transfer

**tutti_cache (2.7% penalty):**
compute_stream:  [L0 compute][L1 compute][L2 compute]
transfer_stream: [fetch L1──][fetch L2──][fetch L3──]
GPU never waits

### Slack-aware scheduling
avg_compute  = 10.0ms
avg_transfer = 0.35ms
slack        = 9.65ms
Positive slack → safe to prefetch ahead
Negative slack → fetch immediately
Updates via rolling average — adapts as sequence grows

---

## Hardware context
T4 PCIe 3.0:   ~16 GB/s bandwidth → 95.0% recovery (our result)
A100 PCIe 4.0: ~32 GB/s bandwidth → ~98.3% recovery (Tutti paper)

The gap between our result and Tutti's 98.3% is explained entirely
by PCIe bandwidth difference between T4 and A100.

---

## Requirements
torch >= 2.0
transformers >= 4.40
CUDA GPU required
Python >= 3.8

---

## Citation

```bibtex
@article{tutti2026,
  title={Tutti: GPU-Centric I/O-Aware KV Cache Management
         for Large Language Model Inference},
  author={Shi Qiu et al.},
  journal={arXiv preprint arXiv:2605.03375},
  year={2026}
}
```

---

## Author

**Sneha Bhawar**  
GitHub: [@snehabhawar](https://github.com/snehabhawar)

Built as part of LLM inference optimization research.  
Independent implementation — Tutti's official code not yet publicly available.

---

## License

MIT
