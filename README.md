# FlexCacheGen

A VLM generation framework with a flexible KV cache manager.

## Introduction

Combine [FlexGen](https://github.com/FMInference/FlexLLMGen) and [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm), support latest VLM [Qwen3-VL](https://huggingface.co/collections/Qwen/qwen3-vl).

Features:
- Sparse attention implememt with sparse KV cache management.
- KV cache offloading to memory after prefill stage, only load important part in decoding stage.
- Overlapping attention computation with KV cache IO.
- Paged KV cache management with head granularity, instead of token granularity.


## Architechture

```
Engine 负责计算流程控制
ModelRunner 负责提供对应计算组件的接口
KVCacheManager 负责 KV Cache 管理

```


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
pip install torchcodec --index-url https://download.pytorch.org/whl/cu130

# install other packages
pip install -e .
```