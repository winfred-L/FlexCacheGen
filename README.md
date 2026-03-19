# FlexCacheGen

A VLM generation framework with a flexible KV cache manager.

## Introduction

Combine Offloading([FlexGen])(https://github.com/FMInference/FlexLLMGen) and PagedAttention([nano-vllm])(https://github.com/GeeeekExplorer/nano-vllm), support latest open-sourced VLM [Qwen3-VL](https://huggingface.co/collections/Qwen/qwen3-vl).


Features:
- Modality-aware KV cache sparsity, more efficient for multi-modal LLM.
- Dynamic KV Selection without permanent eviction, better accuracy kept. (supported by multi-tier storage)
- Paged KV cache management, less wasted memory.
- Overlapping attention computation with KV cache IO, faster inference speed.


## Architechture

```
VLMEngine controls the generation process.
Model(Qwen3VLModel) provides computing APIs for VLMEngine.
KVCacheManager manages KV cache movement and sparsity.
```


## Environment

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

## Model and Dataset Download

```bash
# models
modelscope download --model Qwen/Qwen3-VL-8B-Instruct --local_dir /data/lyc/models/Qwen3-VL-8B-Instruct
modelscope download --model ZhipuAI/GLM-4.6V-Flash --local_dir /data/lyc/models/GLM-4.6V-Flash

# MLVU summary
modelscope download --dataset AI-ModelScope/MLVU --local_dir /data1/lyc/datasets/MLVU --include 'MLVU/json/9_summary/*'
modelscope download --dataset AI-ModelScope/MLVU --local_dir /data1/lyc/datasets/MLVU --include 'MLVU/video/9_summary/*'
```