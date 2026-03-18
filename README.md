# CoDec

This is the code for [CoDec: Prefix-Shared Decoding Kernel for LLMs](https://arxiv.org/pdf/2505.17694)


## Environment

CUDA Toolkit 12.9

## Installation

```bash
uv pip install torch
uv pip install -Ue . --no-build-isolation
```

## Evaluation

```bash
# kernel evaluation
scripts/kernel.sh

# end to end evaluation
scripts/e2e.sh
```