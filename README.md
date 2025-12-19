# FlexCacheGen

VLM generation framework with a flexible KV cache manager.

## Installation

```bash
conda create -n flexcachegen python=3.12 -y
conda activate flexcachegen

# install pytorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130

# install flash-attn
MAX_JOBS=4 pip install flash-attn --no-build-isolation

# install torchcodec
conda install "ffmpeg"
pip install torchcodec

# install other packages
pip install -e .
```