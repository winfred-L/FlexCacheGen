export CUDA_VISIBLE_DEVICES=1
nsys profile -t cuda,nvtx -o after-prefill_0.2_0.8_offloading_pipeline \
    python main.py