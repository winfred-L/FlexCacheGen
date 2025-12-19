# FlexCacheGen

VLM generation framework with a flexible KV cache manager.

## Installation

```bash
conda create -n flexcachegen python=3.12 -y
conda activate flexcachegen
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
MAX_JOBS=4 pip install flash-attn --no-build-isolation
pip install -e .
```