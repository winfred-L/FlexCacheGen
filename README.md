# FlexCacheGen

## Introduction

FlexCacheGen is a **memory-efficient VLM inference framework** with a flexible KV cache manager, designed for long-context multi-modal LLM generation tasks.

FlexCacheGen combines Offloading([FlexGen](https://github.com/FMInference/FlexLLMGen)) and PagedAttention([nano-vllm](https://github.com/GeeeekExplorer/nano-vllm)), exploits novel modality-aware KV cache sparsity, supports latest open-sourced VLM [Qwen3-VL](https://huggingface.co/collections/Qwen/qwen3-vl).


### Why FlexCacheGen?
| Challenge | Traditional Approach | FlexCacheGen Solution |
|-----------|---------------------|----------------------|
| Long video context OOM | Static KV cache | Multi-tiered offloading + Dynamic KV selection |
| Multi-modal sparsity ignored | Uniform caching | Modality-aware algorithm + Spatial data locality |
| Memory fragmentation | Pre-allocated cache | Paged KV management |
| IO bottleneck | Sequential execution | Overlapping pipeline |




### Features
- Modality-aware KV cache sparsity, more efficient for multi-modal LLM.
- Dynamic KV Selection without permanent eviction, better accuracy kept. (supported by multi-tier storage)
- Paged KV cache management, less wasted memory.
- Overlapping attention computation with KV cache IO, faster inference speed.


### Supported Models
- [Qwen3-VL](https://huggingface.co/collections/Qwen/qwen3-vl)



### Architecture

```
VLMEngine controls the generation process.
Model(Qwen3VLModel) provides computing APIs for VLMEngine.
KVCacheManager manages KV cache movement and sparsity.
```


## Quick Start

### Environment

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
pip install -e . # -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### Model and Dataset Download

```bash
# models
modelscope download --model Qwen/Qwen3-VL-8B-Instruct --local_dir /path/to/models/Qwen3-VL-8B-Instruct
modelscope download --model ZhipuAI/GLM-4.6V-Flash --local_dir /path/to/models/GLM-4.6V-Flash

# MLVU Sub-Scene Captioning
modelscope download --dataset AI-ModelScope/MLVU --local_dir /path/to/datasets/MLVU --include 'MLVU/json/8_sub_scene.json'
modelscope download --dataset AI-ModelScope/MLVU --local_dir /path/to/datasets/MLVU --include 'MLVU/video/8_sub_scene/*'

# MLVU Summary
modelscope download --dataset AI-ModelScope/MLVU --local_dir /path/to/datasets/MLVU --include 'MLVU/json/9_summary.json'
modelscope download --dataset AI-ModelScope/MLVU --local_dir /path/to/datasets/MLVU --include 'MLVU/video/9_summary/*'
```

### Setup env

create `.env` from `.env.example` and setup model/dataset path.

### Run Example

```bash
python ./scripts/example.sh
```

